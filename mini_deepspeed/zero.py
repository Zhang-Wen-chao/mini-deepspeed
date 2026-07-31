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
    """Optimizer options for the educational AdamW-based ZeRO engine.

    ``reduce_bucket_size`` is measured in parameter elements. ZeRO-2 groups
    whole, consecutive parameters into buckets no smaller than a parameter,
    then reduce-scatters a bucket as soon as autograd finishes it.
    """

    stage: int = 0
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    reduce_bucket_size: int = 1_048_576

    def __post_init__(self) -> None:
        if self.stage not in (0, 1, 2):
            raise ValueError("only ZeRO stages 0, 1 and 2 are implemented")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.reduce_bucket_size <= 0:
            raise ValueError("reduce_bucket_size must be positive")


@dataclass
class GradientBucket:
    """One ZeRO-2 communication and optimizer-state unit."""

    layout: FlatParameterLayout
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    ready_parameters: int = 0
    reduced_this_backward: bool = False
    gradient_shard: torch.Tensor | None = None


class ZeroOptimizer:
    """Synchronously update identical data-parallel model replicas.

    Stage 0 is a flat AdamW and gradient all-reduce baseline. Stage 1 stores
    Adam moments only for a rank's parameter shard. Stage 2 registers
    post-accumulate gradient hooks: once every parameter in a bucket is ready,
    it reduce-scatters that bucket and immediately releases the full grads.
    Updated parameter shards are all-gathered so every rank retains a complete
    forward model.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], config: ZeroConfig):
        self.config = config
        self._distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self._distributed else 0
        self.world_size = dist.get_world_size() if self._distributed else 1
        self.layout = FlatParameterLayout(parameters, self.world_size)
        self._synchronize_initial_parameters()
        self.step_count = 0
        self._last_sync = "none (single process)"
        self._last_collective_elements = 0

        self._buckets: tuple[GradientBucket, ...] = ()
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._backward_active = False
        self._stage2_backward_calls = 0
        self._stage2_collective_elements = 0

        if config.stage == 2:
            self.exp_avg = None
            self.exp_avg_sq = None
            self._buckets = self._build_gradient_buckets()
            self._register_gradient_hooks()
        else:
            # Only partitioned stages need the layout's communication padding.
            # Stage 0 deliberately mirrors ordinary AdamW exactly.
            state_numel = self.layout.numel if config.stage == 0 else self.layout.shard_numel
            self.exp_avg = torch.zeros(state_numel, device=self.layout.device, dtype=self.layout.dtype)
            self.exp_avg_sq = torch.zeros_like(self.exp_avg)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.layout.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward, using hooks to reduce and release ZeRO-2 buckets."""
        if self.config.stage != 2:
            loss.backward()
            return

        self._begin_stage2_backward()
        try:
            loss.backward()
        except BaseException:
            self._abort_stage2_backward()
            raise
        self._finish_stage2_backward()

    def _synchronize_initial_parameters(self) -> None:
        """Adopt rank 0's model once, matching ordinary DDP initialization."""
        if not self._distributed:
            return
        parameters = self.layout.flatten_parameters()
        dist.broadcast(parameters, src=0)
        self.layout.assign(self.layout.pad(parameters))

    def _build_gradient_buckets(self) -> tuple[GradientBucket, ...]:
        groups: list[list[nn.Parameter]] = []
        current: list[nn.Parameter] = []
        current_numel = 0
        for parameter in self.layout.parameters:
            if current and current_numel + parameter.numel() > self.config.reduce_bucket_size:
                groups.append(current)
                current = []
                current_numel = 0
            current.append(parameter)
            current_numel += parameter.numel()
            if current_numel >= self.config.reduce_bucket_size:
                groups.append(current)
                current = []
                current_numel = 0
        if current:
            groups.append(current)

        buckets: list[GradientBucket] = []
        for parameters in groups:
            layout = FlatParameterLayout(parameters, self.world_size)
            exp_avg = torch.zeros(layout.shard_numel, device=layout.device, dtype=layout.dtype)
            buckets.append(GradientBucket(layout, exp_avg, torch.zeros_like(exp_avg)))
        return tuple(buckets)

    def _register_gradient_hooks(self) -> None:
        for bucket in self._buckets:
            for parameter in bucket.layout.parameters:
                if not hasattr(parameter, "register_post_accumulate_grad_hook"):
                    raise RuntimeError("ZeRO-2 bucket hooks require PyTorch 2.1 or newer")
                handle = parameter.register_post_accumulate_grad_hook(
                    lambda _parameter, bucket=bucket: self._reduce_ready_bucket(bucket)
                )
                self._hook_handles.append(handle)

    def _begin_stage2_backward(self) -> None:
        if self._backward_active:
            raise RuntimeError("cannot begin a new backward while ZeRO-2 hooks are active")
        if self._stage2_backward_calls == 0:
            self._stage2_collective_elements = 0
        self._backward_active = True
        for bucket in self._buckets:
            bucket.reduced_this_backward = False

    def _finish_stage2_backward(self) -> None:
        self._backward_active = False
        missing = [index for index, bucket in enumerate(self._buckets) if not bucket.reduced_this_backward]
        incomplete = [
            index for index, bucket in enumerate(self._buckets) if bucket.ready_parameters != 0
        ]
        if missing or incomplete:
            self._abort_stage2_backward()
            raise RuntimeError(
                "ZeRO-2 bucket hooks require every trainable parameter to participate in each backward; "
                f"missing buckets={missing}, incomplete buckets={incomplete}"
            )
        self._stage2_backward_calls += 1

    def _abort_stage2_backward(self) -> None:
        self._backward_active = False
        self._stage2_backward_calls = 0
        self._stage2_collective_elements = 0
        for bucket in self._buckets:
            bucket.ready_parameters = 0
            bucket.reduced_this_backward = False
            bucket.gradient_shard = None

    def _reduce_ready_bucket(self, bucket: GradientBucket) -> None:
        if not self._backward_active:
            raise RuntimeError("ZeRO-2 requires engine.backward(loss), not loss.backward() directly")
        bucket.ready_parameters += 1
        if bucket.ready_parameters > len(bucket.layout.parameters):
            raise RuntimeError("a ZeRO-2 bucket was reduced more than once in one backward")
        if bucket.ready_parameters != len(bucket.layout.parameters):
            return

        padded_gradients = bucket.layout.pad(bucket.layout.flatten_gradients())
        gradient_shard = self._reduce_scatter_mean(padded_gradients, bucket.layout)
        if bucket.gradient_shard is None:
            bucket.gradient_shard = gradient_shard
        else:
            bucket.gradient_shard.add_(gradient_shard)

        # After reduce-scatter only this rank's shard is needed for the AdamW
        # update. Releasing these complete parameter gradients is the physical
        # lifecycle difference from the first, step-boundary-only version.
        for parameter in bucket.layout.parameters:
            parameter.grad = None
        bucket.ready_parameters = 0
        bucket.reduced_this_backward = True

    def step(self) -> None:
        if self.config.stage == 0:
            # DDP's all-reduce does not require equal-size parameter shards.
            # Keeping this vector unpadded also makes Stage 0 a literal AdamW
            # baseline: exactly two optimizer values per model parameter.
            gradients = self.layout.flatten_gradients()
            parameters = self.layout.flatten_parameters()
            self._adamw(parameters, self._all_reduce_mean(gradients), self.exp_avg, self.exp_avg_sq)
            self.layout.assign(self.layout.pad(parameters))
            self.step_count += 1
            return

        if self.config.stage == 1:
            gradients = self.layout.pad(self.layout.flatten_gradients())
            mean_gradients = self._all_reduce_mean(gradients)
            gradient_shard = self.layout.local_shard(mean_gradients, self.rank)
            parameters = self.layout.pad(self.layout.flatten_parameters())
            parameter_shard = self.layout.local_shard(parameters, self.rank).clone()
            self._adamw(parameter_shard, gradient_shard, self.exp_avg, self.exp_avg_sq)
            self.layout.assign(self._all_gather_shards(parameter_shard, self.layout))
            if self._distributed:
                self._last_collective_elements += self.layout.padded_numel
                self._last_sync += " + all-gather updated parameters"
            self.step_count += 1
            return

        self._step_stage2()

    def _step_stage2(self) -> None:
        if self._backward_active:
            raise RuntimeError("cannot call step while ZeRO-2 backward hooks are active")
        if self._stage2_backward_calls == 0:
            raise RuntimeError("ZeRO-2 step requires at least one engine.backward(loss) call")
        missing = [index for index, bucket in enumerate(self._buckets) if bucket.gradient_shard is None]
        if missing:
            raise RuntimeError(f"ZeRO-2 is missing reduced gradient shards for buckets {missing}")

        for bucket in self._buckets:
            parameters = bucket.layout.pad(bucket.layout.flatten_parameters())
            parameter_shard = bucket.layout.local_shard(parameters, self.rank).clone()
            self._adamw(parameter_shard, bucket.gradient_shard, bucket.exp_avg, bucket.exp_avg_sq)
            bucket.layout.assign(self._all_gather_shards(parameter_shard, bucket.layout))
            if self._distributed:
                self._stage2_collective_elements += bucket.layout.padded_numel

        if self._distributed:
            backend = dist.get_backend()
            reduction = "reduce-scatter gradients (NCCL)" if backend == "nccl" else "all-reduce + slice gradients (Gloo fallback)"
            self._last_sync = f"{reduction} + all-gather updated parameters ({len(self._buckets)} buckets)"
            self._last_collective_elements = self._stage2_collective_elements
        else:
            self._last_sync = "none (single process)"
            self._last_collective_elements = 0

        for bucket in self._buckets:
            bucket.gradient_shard = None
        self._stage2_backward_calls = 0
        self.step_count += 1

    def report(self) -> ZeroReport:
        full = self.layout.numel
        if self.config.stage == 2:
            gradients = sum(bucket.layout.shard_numel for bucket in self._buckets)
            optimizer_states = sum(bucket.exp_avg.numel() + bucket.exp_avg_sq.numel() for bucket in self._buckets)
            bucket_count = len(self._buckets)
        else:
            gradients = full
            optimizer_states = self.exp_avg.numel() + self.exp_avg_sq.numel()
            bucket_count = 1
        return ZeroReport(
            stage=self.config.stage,
            rank=self.rank,
            world_size=self.world_size,
            parameter_elements=full,
            gradient_elements=gradients,
            optimizer_state_elements=optimizer_states,
            synchronization=self._last_sync,
            logical_collective_elements=self._last_collective_elements,
            gradient_bucket_count=bucket_count,
        )

    def _adamw(
        self,
        parameters: torch.Tensor,
        gradients: torch.Tensor,
        exp_avg: torch.Tensor | None,
        exp_avg_sq: torch.Tensor | None,
    ) -> None:
        if exp_avg is None or exp_avg_sq is None:
            raise RuntimeError("AdamW state is unavailable for this ZeRO stage")
        beta1, beta2 = self.config.betas
        exp_avg.lerp_(gradients, 1 - beta1)
        exp_avg_sq.lerp_(gradients.square(), 1 - beta2)
        next_step = self.step_count + 1
        bias_correction1 = 1 - beta1**next_step
        bias_correction2 = 1 - beta2**next_step
        denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.config.eps)
        update = exp_avg.div(bias_correction1).div(denominator)
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

    def _reduce_scatter_mean(
        self, padded_gradients: torch.Tensor, layout: FlatParameterLayout
    ) -> torch.Tensor:
        if not self._distributed:
            return layout.local_shard(padded_gradients, self.rank)
        output = torch.empty_like(layout.local_shard(padded_gradients, self.rank))
        if dist.get_backend() == "nccl":
            dist.reduce_scatter_tensor(output, padded_gradients, op=dist.ReduceOp.SUM)
        else:
            # Gloo builds often omit reduce_scatter. This preserves exact
            # ZeRO-2 math for CPU tests while labelling the non-efficient path.
            reduced = padded_gradients.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            output.copy_(layout.local_shard(reduced, self.rank))
        output.div_(self.world_size)
        self._stage2_collective_elements += layout.padded_numel
        return output

    def _all_gather_shards(self, shard: torch.Tensor, layout: FlatParameterLayout) -> torch.Tensor:
        if not self._distributed:
            return shard
        shards = [torch.empty_like(shard) for _ in range(self.world_size)]
        dist.all_gather(shards, shard)
        return torch.cat(shards)
