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
        if config.stage == 3:
            # `module.state_dict()` fires pre-hooks only on the module it is
            # called on, so register on every submodule: a direct call on a
            # child (for example `engine.module[0].state_dict()`) must also
            # fail loudly instead of serializing empty placeholders.
            for submodule in self.module.modules():
                submodule.register_state_dict_pre_hook(self._reject_stage3_module_state_dict)
                submodule.register_load_state_dict_pre_hook(self._reject_stage3_module_load_state_dict)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        self.optimizer.prepare_forward()
        try:
            output = self.module(*args, **kwargs)
        except BaseException as error:
            self.optimizer.finish_forward(error)
            raise
        self.optimizer.finish_forward(None)
        return output

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

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Reject Stage-3 checkpoints until a dedicated sharded format exists."""
        if self.optimizer.config.stage == 3:
            raise RuntimeError(
                "ZeRO-3 checkpointing is not implemented; engine.state_dict() cannot safely serialize "
                "sharded parameters or optimizer state."
            )
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Reject unsupported Stage-3 checkpoint restoration."""
        if self.optimizer.config.stage == 3:
            raise RuntimeError(
                "ZeRO-3 checkpointing is not implemented; engine.load_state_dict() cannot safely restore "
                "sharded parameters or optimizer state."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _reject_stage3_module_state_dict(self, module: nn.Module, prefix: str, keep_vars: bool) -> None:
        raise RuntimeError(
            "ZeRO-3 checkpointing is not implemented; module.state_dict() would expose invalid "
            "parameter storage."
        )

    def _reject_stage3_module_load_state_dict(
        self,
        module: nn.Module,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        raise RuntimeError("ZeRO-3 checkpointing is not implemented; module.load_state_dict() is unavailable.")


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
