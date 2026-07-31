"""Compare mini-deepspeed ZeRO-0/1/2/3 directly against DeepSpeed."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mini_deepspeed as mds


class ReferenceRegressor(nn.Module):
    """A small, non-divisible parameter layout shared by both engines."""

    def __init__(self) -> None:
        super().__init__()
        # 4*9 + 9 + 9*2 + 2 = 65 elements; 65 is not divisible by 2 or 4.
        self.layers = nn.Sequential(nn.Linear(4, 9), nn.Tanh(), nn.Linear(9, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--reduce-bucket-size", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    if args.steps <= 0 or args.microbatches <= 0:
        parser.error("--steps and --microbatches must be positive")
    return args


def _require_deepspeed() -> Any:
    try:
        import deepspeed
    except ImportError as error:
        raise SystemExit(
            "DeepSpeed is intentionally not a mini-deepspeed dependency. Install it in an isolated "
            "environment, then launch this program with that environment's torchrun."
        ) from error
    # This numerical reference does not collect profiling data. Disabling
    # annotations also avoids a DeepSpeed 0.19.3/NVTX-domain API mismatch on
    # newer CUDA Python environments; it does not change model execution,
    # collectives, or optimizer semantics.
    import deepspeed.utils.nvtx as nvtx

    nvtx.enable_nvtx = False
    return deepspeed


def _batch(step: int, microbatch: int, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(100 + step * 20 + microbatch * 5 + rank)
    return (
        torch.randn(6, 4, generator=generator, device=device),
        torch.randn(6, 2, generator=generator, device=device),
    )


def _flatten(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _flatten_deepspeed(deepspeed: Any, module: nn.Module, stage: int) -> torch.Tensor:
    """Gather Stage-3 parameter shards only while inspecting the reference."""
    if stage != 3:
        return _flatten(module)
    with deepspeed.zero.GatheredParameters(list(module.parameters()), modifier_rank=None):
        return _flatten(module).clone()


def _assert_replicated(flat: torch.Tensor, rtol: float, atol: float) -> None:
    replicas = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, flat)
    for replica in replicas[1:]:
        torch.testing.assert_close(replica, replicas[0], rtol=rtol, atol=atol)


def _run_mini(
    stage: int, args: argparse.Namespace, rank: int, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(314)
    engine = mds.initialize(
        ReferenceRegressor().to(device),
        {
            "zero_stage": stage,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "reduce_bucket_size": args.reduce_bucket_size,
        },
    )
    initial = engine.parameter_vector()
    _assert_replicated(initial, args.rtol, args.atol)
    history: list[torch.Tensor] = []
    for step in range(args.steps):
        engine.zero_grad()
        for microbatch in range(args.microbatches):
            inputs, targets = _batch(step, microbatch, rank, device)
            engine.backward(F.mse_loss(engine(inputs), targets))
        engine.step()
        snapshot = engine.parameter_vector()
        _assert_replicated(snapshot, args.rtol, args.atol)
        history.append(snapshot)
    return initial, history


def _deepspeed_config(stage: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": 6,
        "gradient_accumulation_steps": args.microbatches,
        # DeepSpeed's default gradient clipping would change the optimizer
        # algorithm relative to mini-deepspeed, which intentionally does not
        # expose clipping in its small teaching API.
        "gradient_clipping": 0.0,
        # DeepSpeed 0.19.3 performs ``global_steps % steps_per_print`` during
        # each update, so zero is not a valid way to suppress its periodic
        # logging. Keep this above the short reference run's step count.
        "steps_per_print": 1_000_000,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": args.weight_decay,
                # Demand PyTorch AdamW so the reference optimizer has the same
                # public semantics as mini-deepspeed's explicit AdamW update.
                "torch_adam": True,
            },
        },
        "zero_optimization": {
            "stage": stage,
            "reduce_scatter": True,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": args.reduce_bucket_size,
        },
    }


def _run_deepspeed(
    deepspeed: Any, stage: int, args: argparse.Namespace, rank: int, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(314)
    module = ReferenceRegressor().to(device)
    engine, _, _, _ = deepspeed.initialize(
        model=module,
        model_parameters=module.parameters(),
        config=_deepspeed_config(stage, args),
        dist_init_required=False,
    )
    initial = _flatten_deepspeed(deepspeed, engine.module, stage)
    _assert_replicated(initial, args.rtol, args.atol)
    history: list[torch.Tensor] = []
    for step in range(args.steps):
        engine.zero_grad()
        for microbatch in range(args.microbatches):
            inputs, targets = _batch(step, microbatch, rank, device)
            # DeepSpeed advances its accumulation boundary in ``step()``, not
            # ``backward()``. Disable its default loss/GAS scaling: mini's
            # engine deliberately accumulates the same unscaled gradient sum.
            engine.backward(F.mse_loss(engine(inputs), targets), scale_wrt_gas=False)
            expected_boundary = microbatch + 1 == args.microbatches
            if engine.is_gradient_accumulation_boundary() != expected_boundary:
                raise RuntimeError(
                    "DeepSpeed accumulation boundary did not match gradient_accumulation_steps"
                )
            previous_global_steps = engine.global_steps
            engine.step()
            expected_global_steps = previous_global_steps + int(expected_boundary)
            if engine.global_steps != expected_global_steps:
                raise RuntimeError("DeepSpeed step did not follow its documented accumulation boundary")
            if expected_boundary:
                snapshot = _flatten_deepspeed(deepspeed, engine.module, stage)
                _assert_replicated(snapshot, args.rtol, args.atol)
                history.append(snapshot)
    return initial, history


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("compare_deepspeed.py requires CUDA/NCCL for the reference run")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    deepspeed = _require_deepspeed()
    try:
        if rank == 0:
            print(
                f"reference: DeepSpeed {deepspeed.__version__}; PyTorch {torch.__version__}; "
                f"world_size={dist.get_world_size()}; dtype=float32"
            )
        for stage in (0, 1, 2, 3):
            mini_initial, mini_history = _run_mini(stage, args, rank, device)
            dist.barrier()
            deepspeed_initial, deepspeed_history = _run_deepspeed(deepspeed, stage, args, rank, device)
            torch.testing.assert_close(deepspeed_initial, mini_initial, rtol=args.rtol, atol=args.atol)
            if len(deepspeed_history) != len(mini_history):
                raise RuntimeError("mini-deepspeed and DeepSpeed produced a different number of updates")
            max_error = 0.0
            for deepspeed_parameters, mini_parameters in zip(deepspeed_history, mini_history, strict=True):
                torch.testing.assert_close(deepspeed_parameters, mini_parameters, rtol=args.rtol, atol=args.atol)
                max_error = max(max_error, _max_abs_error(deepspeed_parameters, mini_parameters))
            if rank == 0:
                print(
                    f"PASS: mini ZeRO-{stage} initial and {args.steps} post-step parameter snapshots "
                    f"== DeepSpeed; max_abs_error={max_error:.3e}"
                )
        if rank == 0:
            print("PASS: all DeepSpeed ZeRO-0/1/2/3 reference comparisons matched")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
