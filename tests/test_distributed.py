"""CPU/Gloo coverage for ZeRO ownership, bucketing, and numeric equivalence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

import mini_deepspeed as mds

_REDUCE_BUCKET_SIZE = 10


class _DivisibleRegressor(nn.Module):
    """64 trainable elements, each parameter tensor divisible by four."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 8))
        self.offset = nn.Parameter(torch.zeros(4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs) + self.offset.repeat(2)


def _make_model(numel: int = 65) -> nn.Module:
    # 4*9 + 9 + 9*2 + 2 = 65 trainable elements. It is deliberately not
    # divisible by the two or four ranks exercised below.
    if numel == 65:
        return nn.Sequential(nn.Linear(4, 9), nn.Tanh(), nn.Linear(9, 2))
    if numel == 64:
        # 4*4 + 4 + 4*8 + 8 + 4 = 64. Individual tensors (16, 4,
        # 32, 8, 4) and therefore every bucket are divisible by 2 and 4.
        return _DivisibleRegressor()
    raise ValueError(f"unsupported test parameter count: {numel}")


def _worker(
    rank: int, world_size: int, init_file: str, stage: int, result_file: str, model_numel: int = 65
) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        # Different construction seeds verify that initialization is broadcast
        # from rank 0, as DDP and DeepSpeed require for data-parallel replicas.
        torch.manual_seed(22 + rank)
        model = _make_model(model_numel)
        engine = mds.initialize(
            model,
            {"zero_stage": stage, "lr": 5e-3, "reduce_bucket_size": _REDUCE_BUCKET_SIZE},
        )

        gradients_released = True
        # Two backwards before each step validate gradient accumulation. Each
        # rank receives different data, so collectives must average real DP gradients.
        for step in range(3):
            engine.zero_grad()
            for microbatch in range(2):
                torch.manual_seed(100 + step * 20 + microbatch * 5 + rank)
                inputs = torch.randn(6, 4)
                targets = torch.randn(6, 2 if model_numel == 65 else 8)
                loss = F.mse_loss(engine(inputs), targets)
                engine.backward(loss)
                if stage == 2:
                    gradients_released &= all(parameter.grad is None for parameter in model.parameters())
            engine.step()

        flat = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        replicas = [torch.empty_like(flat) for _ in range(world_size)]
        dist.all_gather(replicas, flat)
        for replica in replicas[1:]:
            torch.testing.assert_close(replica, replicas[0], rtol=1e-6, atol=1e-7)

        if rank == 0:
            torch.save(
                {
                    "parameters": flat,
                    "report": engine.report(),
                    "gradients_released": gradients_released,
                },
                result_file,
            )
    finally:
        dist.destroy_process_group()


def _run_stage(stage: int, world_size: int, root: Path, model_numel: int = 65) -> dict[str, object]:
    init_file = root / f"gloo-init-{world_size}-{stage}"
    result_file = root / f"stage-{world_size}-{stage}.pt"
    mp.spawn(
        _worker,
        args=(world_size, str(init_file), stage, str(result_file), model_numel),
        nprocs=world_size,
        join=True,
    )
    return torch.load(result_file, weights_only=False)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_zero_stages_match_stage_zero_for_one_two_and_four_ranks(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        stage0 = _run_stage(0, world_size, root)
        stage1 = _run_stage(1, world_size, root)
        stage2 = _run_stage(2, world_size, root)

    total = stage0["parameters"].numel()
    reports = {stage: result["report"] for stage, result in enumerate((stage0, stage1, stage2))}
    assert total == 65
    assert all(report.parameter_elements == total for report in reports.values())

    if world_size == 1:
        assert reports[0].optimizer_state_elements == reports[1].optimizer_state_elements
        assert reports[1].optimizer_state_elements == reports[2].optimizer_state_elements
        assert reports[0].gradient_elements == reports[2].gradient_elements
    else:
        assert reports[1].optimizer_state_elements < reports[0].optimizer_state_elements
        assert reports[2].optimizer_state_elements < reports[0].optimizer_state_elements
        assert reports[2].gradient_elements < reports[0].gradient_elements

    # The 36, 9, 18, and 2 element tensors form four buckets with a 10 element
    # target. Per-bucket padding is intentionally reflected in the report.
    assert reports[2].gradient_bucket_count == 4
    assert reports[2].gradient_elements == sum((size + world_size - 1) // world_size for size in (36, 9, 18, 2))
    assert stage2["gradients_released"]
    assert "Gloo fallback" in reports[2].synchronization

    for result in (stage1, stage2):
        torch.testing.assert_close(result["parameters"], stage0["parameters"], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("world_size", [2, 4])
def test_zero_stages_match_with_even_parameter_shards(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        stage0 = _run_stage(0, world_size, root, model_numel=64)
        stage1 = _run_stage(1, world_size, root, model_numel=64)
        stage2 = _run_stage(2, world_size, root, model_numel=64)

    assert stage0["parameters"].numel() == 64
    assert stage2["report"].gradient_elements == 64 // world_size
    for result in (stage1, stage2):
        torch.testing.assert_close(result["parameters"], stage0["parameters"], rtol=1e-6, atol=1e-7)
