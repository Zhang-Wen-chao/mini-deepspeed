"""The intentionally small user-facing training-engine wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .reports import ZeroReport
from .zero import ZeroConfig, ZeroOptimizer


class DeepSpeedEngine(nn.Module):
    """Wrap an ordinary PyTorch module with the ZeRO optimizer lifecycle."""

    def __init__(self, module: nn.Module, config: ZeroConfig):
        super().__init__()
        self.module = module
        self.optimizer = ZeroOptimizer(module.parameters(), config)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        self.optimizer.prepare_forward()
        try:
            return self.module(*args, **kwargs)
        except BaseException:
            self.optimizer.abort_forward()
            raise

    def backward(self, loss: Tensor) -> None:
        self.optimizer.backward(loss)

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def abort_forward(self) -> None:
        """Discard a ZeRO-3 materialization whose loss will not be backpropagated."""
        self.optimizer.abort_forward()

    def report(self) -> ZeroReport:
        return self.optimizer.report()

    def parameter_vector(self) -> Tensor:
        """Return a detached full trainable parameter vector for inspection."""
        return self.optimizer.parameter_vector()


def initialize(module: nn.Module, config: ZeroConfig | Mapping[str, Any] | None = None) -> DeepSpeedEngine:
    """Create an engine from a dataclass or a small DeepSpeed-like mapping."""
    if config is None:
        zero_config = ZeroConfig()
    elif isinstance(config, ZeroConfig):
        zero_config = config
    else:
        values = dict(config)
        if "zero_stage" in values:
            values["stage"] = values.pop("zero_stage")
        zero_config = ZeroConfig(**values)
    return DeepSpeedEngine(module, zero_config)
