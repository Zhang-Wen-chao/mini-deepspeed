"""Train an ordinary MLP with a selected ZeRO stage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# Allow `python examples/train_toy.py` from the project checkout without a
# package installation, which is convenient for the first distributed run.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mini_deepspeed as mds


class ToyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 8),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")

    rank = dist.get_rank() if distributed else 0
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    torch.manual_seed(7)
    engine = mds.initialize(ToyRegressor().to(device), {"zero_stage": args.zero_stage, "lr": 1e-3})
    for step in range(args.steps):
        generator = torch.Generator(device=device).manual_seed(1000 + step * 31 + rank)
        inputs = torch.randn(16, 32, generator=generator, device=device)
        targets = torch.randn(16, 8, generator=generator, device=device)
        loss = F.mse_loss(engine(inputs), targets)
        engine.backward(loss)
        engine.step()
        engine.zero_grad()
        if rank == 0 and (step == 0 or step == args.steps - 1):
            print(f"step={step + 1:>3} loss={loss.item():.6f}")

    report = engine.report()
    if rank == 0:
        print(
            f"ZeRO-{report.stage}: params={report.parameter_elements}, "
            f"grads={report.gradient_elements}, optimizer={report.optimizer_state_elements}, "
            f"total={report.model_state_elements}, sync={report.synchronization}"
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
