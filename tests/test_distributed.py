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


def _worker(rank: int, world_size: int, init_file: str, stage: int, result_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    # Different local construction seeds exercise the engine's rank-0
    # initialization broadcast, the same invariant DDP establishes.
    torch.manual_seed(22 + rank)
    model = nn.Sequential(nn.Linear(4, 9), nn.Tanh(), nn.Linear(9, 2))
    engine = mds.initialize(model, {"zero_stage": stage, "lr": 5e-3})

    for step in range(3):
        torch.manual_seed(100 + step * 10 + rank)
        inputs = torch.randn(6, 4)
        targets = torch.randn(6, 2)
        loss = F.mse_loss(engine(inputs), targets)
        engine.backward(loss)
        engine.step()
        engine.zero_grad()

    flat = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    replicas = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(replicas, flat)
    for replica in replicas[1:]:
        torch.testing.assert_close(replica, replicas[0], rtol=1e-6, atol=1e-7)

    if rank == 0:
        torch.save({"parameters": flat, "report": engine.report()}, result_file)
    dist.destroy_process_group()


def _run_stage(stage: int, root: Path) -> dict[str, object]:
    init_file = root / f"gloo-init-{stage}"
    result_file = root / f"stage-{stage}.pt"
    mp.spawn(_worker, args=(2, str(init_file), stage, str(result_file)), nprocs=2, join=True)
    return torch.load(result_file, weights_only=False)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_two_rank_zero_stages_remain_replicated(stage: int) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        result = _run_stage(stage, root)

    report = result["report"]
    total = result["parameters"].numel()
    assert report.parameter_elements == total
    if stage == 0:
        assert report.optimizer_state_elements == 2 * total
    else:
        assert report.optimizer_state_elements < 2 * total
    if stage == 2:
        assert report.gradient_elements < total
        assert "Gloo fallback" in report.synchronization


def test_two_rank_zero_stages_match_the_ddp_baseline() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        stage0 = _run_stage(0, root)
        stage1 = _run_stage(1, root)
        stage2 = _run_stage(2, root)

    for result in (stage1, stage2):
        torch.testing.assert_close(result["parameters"], stage0["parameters"], rtol=1e-6, atol=1e-7)
