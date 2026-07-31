"""A deterministic flat view over a model's trainable parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterSpan:
    """Where one parameter lives in the unpadded flat parameter vector."""

    start: int
    end: int
    shape: torch.Size


class FlatParameterLayout:
    """Flatten model parameters and split them into equal-sized DP shards.

    Padding exists only for collective communication. `assign` discards it, so
    a model never receives artificial parameter values.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], world_size: int):
        source = tuple(parameter for parameter in parameters if parameter.requires_grad)
        self._validate_parameter_storage(source)
        self.parameters = source
        if not self.parameters:
            raise ValueError("ZeRO requires at least one trainable parameter")
        if world_size < 1:
            raise ValueError("world_size must be positive")

        offset = 0
        spans: list[ParameterSpan] = []
        for parameter in self.parameters:
            next_offset = offset + parameter.numel()
            spans.append(ParameterSpan(offset, next_offset, parameter.shape))
            offset = next_offset
        self.spans = tuple(spans)
        self.numel = offset
        self.world_size = world_size
        self.shard_numel = (self.numel + world_size - 1) // world_size
        self.padded_numel = self.shard_numel * world_size

    @staticmethod
    def _validate_parameter_storage(parameters: tuple[nn.Parameter, ...]) -> None:
        """Reject aliases that flat ZeRO ownership cannot preserve safely."""
        seen_ids: set[int] = set()
        spans: list[tuple[int, int]] = []
        for parameter in parameters:
            if id(parameter) in seen_ids:
                raise ValueError("ZeRO parameter iterable must not contain the same Parameter twice")
            seen_ids.add(id(parameter))
            if parameter.layout != torch.strided or not parameter.is_contiguous():
                raise ValueError("ZeRO supports only contiguous strided Parameters")
            if parameter.numel() == 0:
                raise ValueError("ZeRO does not support empty Parameters")
            storage = parameter.untyped_storage()
            if parameter.storage_offset() != 0 or storage.nbytes() != parameter.numel() * parameter.element_size():
                raise ValueError("ZeRO does not support Parameter views or shared storage")
            start = storage.data_ptr()
            end = start + storage.nbytes()
            if any(start < other_end and other_start < end for other_start, other_end in spans):
                raise ValueError("ZeRO does not support Parameter views or shared storage")
            spans.append((start, end))

    @property
    def device(self) -> torch.device:
        return self.parameters[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.parameters[0].dtype

    def flatten_parameters(self) -> torch.Tensor:
        return torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters])

    def flatten_gradients(self) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for parameter in self.parameters:
            gradient = parameter.grad
            pieces.append(torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1))
        return torch.cat(pieces)

    def pad(self, flat: torch.Tensor) -> torch.Tensor:
        if flat.numel() != self.numel:
            raise ValueError(f"expected {self.numel} elements, got {flat.numel()}")
        if self.padded_numel == self.numel:
            return flat
        return torch.nn.functional.pad(flat, (0, self.padded_numel - self.numel))

    def local_shard(self, padded: torch.Tensor, rank: int) -> torch.Tensor:
        if padded.numel() != self.padded_numel:
            raise ValueError("tensor must be padded before selecting a shard")
        return padded.narrow(0, rank * self.shard_numel, self.shard_numel)

    def assign(self, padded: torch.Tensor) -> None:
        """Copy an updated padded vector into the model parameters."""
        if padded.numel() != self.padded_numel:
            raise ValueError("tensor must have the layout's padded size")
        self.assign_flat(padded.narrow(0, 0, self.numel))

    def assign_flat(self, flat: torch.Tensor) -> None:
        """Copy an updated unpadded vector into the model parameters."""
        if flat.numel() != self.numel:
            raise ValueError("tensor must have the layout's unpadded size")
        with torch.no_grad():
            for parameter, span in zip(self.parameters, self.spans, strict=True):
                parameter.copy_(flat[span.start : span.end].view(span.shape))

    def materialize(self, padded: torch.Tensor) -> None:
        """Make full parameter tensors view a gathered padded vector."""
        if padded.numel() != self.padded_numel:
            raise ValueError("tensor must have the layout's padded size")
        flat = padded.narrow(0, 0, self.numel)
        with torch.no_grad():
            for parameter, span in zip(self.parameters, self.spans, strict=True):
                parameter.data = flat[span.start : span.end].view(span.shape)

    def release(self) -> None:
        """Drop full parameter storage while retaining parameter identities."""
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.grad = None
                parameter.data = torch.empty(0, device=parameter.device, dtype=parameter.dtype)
