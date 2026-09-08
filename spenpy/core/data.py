"""Named axes and explicit acquisition stages; never infer stage from shape."""

from dataclasses import dataclass, field, replace
from typing import Any

import torch

STAGES = {
    "sorted_samples",
    "uniform_kspace",
    "regridded_kspace",
    "ro_image",
    "coil_image",
    "magnitude_image",
}


@dataclass
class Acquisition:
    data: torch.Tensor
    axes: tuple[str, ...]
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.axes = tuple(self.axes)
        if len(self.axes) != self.data.ndim or len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must uniquely name every tensor dimension")
        if self.stage not in STAGES:
            raise ValueError(f"Unknown acquisition stage: {self.stage}")
        if self.stage != "magnitude_image" and not self.data.is_complex():
            raise ValueError("Signal acquisitions must be complex-valued")

    def select(self, **indices: int):
        """Select slices/volumes without squeezing unrelated singleton axes."""
        data, axes = self.data, list(self.axes)
        for name, index in indices.items():
            if name not in axes:
                raise ValueError(f"Unknown axis {name!r}; axes are {axes}")
            axis = axes.index(name)
            data = data.select(axis, index)
            axes.pop(axis)
        metadata = dict(self.metadata)
        metadata["selection"] = dict(metadata.get("selection", {}), **indices)
        return Acquisition(data, tuple(axes), self.stage, metadata)

    def permute(self, *axes: str):
        if set(axes) != set(self.axes) or len(axes) != len(self.axes):
            raise ValueError("permute must contain every named axis exactly once")
        return replace(
            self,
            data=self.data.permute(*(self.axes.index(a) for a in axes)),
            axes=tuple(axes),
        )

    def to(self, device):
        return replace(self, data=self.data.to(device))
