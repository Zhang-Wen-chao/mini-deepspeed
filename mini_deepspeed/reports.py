"""Inspectable logical state and communication accounting for a ZeRO step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZeroReport:
    stage: int
    rank: int
    world_size: int
    parameter_elements: int
    gradient_elements: int
    optimizer_state_elements: int
    synchronization: str
    logical_collective_elements: int
    gradient_bucket_count: int

    @property
    def model_state_elements(self) -> int:
        """Parameters + gradients + two Adam moment tensors on this rank."""
        return self.parameter_elements + self.gradient_elements + self.optimizer_state_elements
