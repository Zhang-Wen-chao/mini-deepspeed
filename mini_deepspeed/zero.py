"""A compact, explicit implementation of ZeRO stages 0, 1 and 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn

from .layout import FlatParameterLayout
from .reports import ZeroReport


@dataclass(frozen=True)
class ZeroConfig:
    """Optimizer options for the educational AdamW-based ZeRO engine."""

    stage: int = 0
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.stage not in (0, 1, 2):
            raise ValueError("only ZeRO stages 0, 1 and 2 are implemented")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


class ZeroOptimizer:
    """Synchronously update identical data-parallel model replicas.

    Stage 0 is a flat AdamW and gradient all-reduce baseline. Stage 1 stores
    Adam moments only for a rank's parameter shard. Stage 2 replaces full
    gradient all-reduce with a reduce-scatter to that shard. Updated parameter
    shards are all-gathered so every rank retains a complete forward model.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], config: ZeroConfig):
        self.config = config
        self._distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self._distributed else 0
        self.world_size = dist.get_world_size() if self._distributed else 1
        self.layout = FlatParameterLayout(parameters, self.world_size)
        self._synchronize_initial_parameters()
        self.step_count = 0

        # Only partitioned stages need the layout's communication padding.
        # Stage 0 deliberately mirrors ordinary AdamW exactly.
        state_numel = self.layout.numel if config.stage == 0 else self.layout.shard_numel
        self.exp_avg = torch.zeros(state_numel, device=self.layout.device, dtype=self.layout.dtype)
        self.exp_avg_sq = torch.zeros_like(self.exp_avg)
        self._last_sync = "none (single process)"
        self._last_collective_elements = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.layout.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def _synchronize_initial_parameters(self) -> None:
        """Adopt rank 0's model once, matching ordinary DDP initialization.

        ZeRO assumes data-parallel replicas begin from the same weights. Doing
        the broadcast here makes that invariant explicit rather than relying
        on every caller to seed and construct their model identically.
        """
        if not self._distributed:
            return
        parameters = self.layout.flatten_parameters()
        dist.broadcast(parameters, src=0)
        self.layout.assign(self.layout.pad(parameters))

    def step(self) -> None:
        if self.config.stage == 0:
            # DDP's all-reduce does not require equal-size parameter shards.
            # Keeping this vector unpadded also makes Stage 0 a literal AdamW
            # baseline: exactly two optimizer values per model parameter.
            gradients = self.layout.flatten_gradients()
            parameters = self.layout.flatten_parameters()
            self._adamw(parameters, self._all_reduce_mean(gradients))
            self.layout.assign(self.layout.pad(parameters))
            self.step_count += 1
            return

        gradients = self.layout.pad(self.layout.flatten_gradients())
        if self.config.stage == 2:
            gradient_shard = self._reduce_scatter_mean(gradients)
        else:
            mean_gradients = self._all_reduce_mean(gradients)
            gradient_shard = self.layout.local_shard(mean_gradients, self.rank)

        parameters = self.layout.pad(self.layout.flatten_parameters())
        parameter_shard = self.layout.local_shard(parameters, self.rank).clone()
        self._adamw(parameter_shard, gradient_shard)
        self.layout.assign(self._all_gather_shards(parameter_shard))
        self.step_count += 1

    def report(self) -> ZeroReport:
        full = self.layout.numel
        gradients = full if self.config.stage in (0, 1) else self.layout.shard_numel
        return ZeroReport(
            stage=self.config.stage,
            rank=self.rank,
            world_size=self.world_size,
            parameter_elements=full,
            gradient_elements=gradients,
            optimizer_state_elements=self.exp_avg.numel() + self.exp_avg_sq.numel(),
            synchronization=self._last_sync,
            logical_collective_elements=self._last_collective_elements,
        )

    def _adamw(self, parameters: torch.Tensor, gradients: torch.Tensor) -> None:
        beta1, beta2 = self.config.betas
        self.exp_avg.lerp_(gradients, 1 - beta1)
        self.exp_avg_sq.lerp_(gradients.square(), 1 - beta2)
        next_step = self.step_count + 1
        bias_correction1 = 1 - beta1**next_step
        bias_correction2 = 1 - beta2**next_step
        denominator = self.exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.config.eps)
        update = self.exp_avg.div(bias_correction1).div(denominator)
        if self.config.weight_decay:
            parameters.mul_(1 - self.config.lr * self.config.weight_decay)
        parameters.add_(update, alpha=-self.config.lr)

    def _all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._distributed:
            self._last_sync = "none (single process)"
            self._last_collective_elements = 0
            return tensor
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result.div_(self.world_size)
        self._last_sync = "all-reduce gradients"
        self._last_collective_elements = tensor.numel()
        return result

    def _reduce_scatter_mean(self, padded_gradients: torch.Tensor) -> torch.Tensor:
        if not self._distributed:
            self._last_sync = "none (single process)"
            self._last_collective_elements = 0
            return padded_gradients
        output = torch.empty_like(self.layout.local_shard(padded_gradients, self.rank))
        if dist.get_backend() == "nccl":
            dist.reduce_scatter_tensor(output, padded_gradients, op=dist.ReduceOp.SUM)
            self._last_sync = "reduce-scatter gradients (NCCL)"
        else:
            # Gloo builds often omit reduce_scatter. This preserves exact
            # ZeRO-2 math for CPU tests while labelling the non-efficient path.
            reduced = padded_gradients.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            output.copy_(self.layout.local_shard(reduced, self.rank))
            self._last_sync = "all-reduce + slice gradients (Gloo fallback)"
        output.div_(self.world_size)
        self._last_collective_elements = padded_gradients.numel()
        return output

    def _all_gather_shards(self, shard: torch.Tensor) -> torch.Tensor:
        if not self._distributed:
            return shard
        shards = [torch.empty_like(shard) for _ in range(self.world_size)]
        dist.all_gather(shards, shard)
        self._last_collective_elements += self.layout.padded_numel
        self._last_sync += " + all-gather updated parameters"
        return torch.cat(shards)
