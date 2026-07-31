"""Prove that ZeRO-0/1/2 produce equal two-rank updates on one device type."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mini_deepspeed as mds
from train_toy import ToyRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--reduce-bucket-size",
        type=int,
        default=1_048_576,
        help="ZeRO-2 bucket target in parameter elements",
    )
    return parser.parse_args()


def run_stage(
    stage: int, device: torch.device, rank: int, steps: int, reduce_bucket_size: int
) -> tuple[torch.Tensor, object]:
    torch.manual_seed(314)
    engine = mds.initialize(
        ToyRegressor().to(device),
        {"zero_stage": stage, "lr": 1e-3, "reduce_bucket_size": reduce_bucket_size},
    )
    for step in range(steps):
        generator = torch.Generator(device=device).manual_seed(9000 + step * 97 + rank)
        inputs = torch.randn(16, 32, generator=generator, device=device)
        targets = torch.randn(16, 8, generator=generator, device=device)
        loss = F.mse_loss(engine(inputs), targets)
        engine.backward(loss)
        engine.step()
        engine.zero_grad()
    flat = torch.cat([parameter.detach().reshape(-1) for parameter in engine.module.parameters()])
    return flat, engine.report()


def assert_replicas(flat: torch.Tensor) -> None:
    """Confirm the update all-gather left every data-parallel rank identical."""
    replicas = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, flat)
    for replica in replicas[1:]:
        torch.testing.assert_close(replica, replicas[0], rtol=1e-6, atol=1e-7)


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()

    baseline, baseline_report = run_stage(0, device, rank, args.steps, args.reduce_bucket_size)
    assert_replicas(baseline)
    for stage in (1, 2):
        actual, report = run_stage(stage, device, rank, args.steps, args.reduce_bucket_size)
        assert_replicas(actual)
        torch.testing.assert_close(actual, baseline, rtol=1e-6, atol=1e-7)
        if rank == 0:
            print(
                f"ZeRO-{stage} == ZeRO-0 after {args.steps} steps; "
                f"state={report.model_state_elements} vs {baseline_report.model_state_elements}; "
                f"world_size={dist.get_world_size()}; buckets={report.gradient_bucket_count}; "
                f"sync={report.synchronization}"
            )
    if rank == 0:
        print("PASS: every rank's ZeRO-0/1/2 parameter vectors are numerically equal")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
