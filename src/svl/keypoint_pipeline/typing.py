from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class DetectorConfig(ABC):
    name: str


@dataclass
class SuperPointConfig(DetectorConfig):
    name: str = "SuperPoint"
    device: str = "cpu"
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = -1


@dataclass
class MatcherConfig(ABC):
    name: str


@dataclass
class SuperGlueConfig(MatcherConfig):
    name: str = "SuperGlue"
    device: str = "cpu"
    weights: str = "outdoor"
    descriptor_dim: int = 256
    keypoint_encoder: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    GNN_layers: List[str] = field(default_factory=lambda: ["self", "cross"] * 9)
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.2


@dataclass
class ImageKeyPoints:
    """Class to store keypoints, descriptors, and scores for an image.

    Parameters
    ----------
    keypoints : np.ndarray | torch.Tensor
        keypoints tensor of shape (N, 2)
    descriptors : np.ndarray | torch.Tensor
        descriptors tensor of shape (N, D)
    scores : np.ndarray | torch.Tensor, optional
        scores tensor of shape (N,), by default None
    image_size : tuple[int, int], optional
        image size, by default None
    """

    keypoints: np.ndarray | torch.Tensor
    descriptors: np.ndarray | torch.Tensor
    scores: np.ndarray | torch.Tensor | None = None
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.scores is not None:
            assert len(self.keypoints) == len(self.scores)

        self._is_torch = isinstance(self.keypoints, torch.Tensor)

    def __len__(self) -> int:
        return len(self.keypoints)

    def __getitem__(self, index: int) -> ImageKeyPoints:
        return ImageKeyPoints(
            keypoints=self.keypoints[index],
            descriptors=self.descriptors[index],
            scores=self.scores[index] if self.scores is not None else None,
        )

    def attributes(self) -> List[str]:
        return ["keypoints", "descriptors", "scores", "image_size"]

    @property
    def device(self) -> str:
        return self.keypoints.device if self.is_torch else "cpu"

    @property
    def is_torch(self) -> bool:
        return isinstance(self.keypoints, torch.Tensor)

    def to(self, device: str) -> ImageKeyPoints:
        return (
            ImageKeyPoints(
                keypoints=self.keypoints.to(device),
                descriptors=self.descriptors.to(device),
                scores=self.scores.to(device) if self.scores is not None else None,
                image_size=self.image_size,
            )
            if self.is_torch
            else self
        )

    def squeeze(self) -> ImageKeyPoints:
        return ImageKeyPoints(
            keypoints=self.keypoints.squeeze(),
            descriptors=self.descriptors.squeeze(),
            scores=self.scores.squeeze() if self.scores is not None else None,
            image_size=self.image_size,
        )

    def add_batch_dimension(self) -> ImageKeyPoints:
        return ImageKeyPoints(
            keypoints=self.keypoints[None, ...],
            descriptors=self.descriptors[None, ...],
            scores=self.scores[None, ...] if self.scores is not None else None,
            image_size=self.image_size,
        )

    def numpy(self) -> ImageKeyPoints:
        self._is_torch = False
        return (
            ImageKeyPoints(
                keypoints=self.keypoints.numpy(),
                descriptors=self.descriptors.numpy(),
                scores=self.scores.numpy() if self.scores is not None else None,
                image_size=self.image_size,
            )
            if self.is_torch
            else self
        )

    def to_dict(self, suffix_idx: int = None) -> Dict[str, np.ndarray | torch.Tensor]:
        if suffix_idx is not None:
            return {
                f"keypoints{suffix_idx}": self.keypoints,
                f"descriptors{suffix_idx}": self.descriptors,
                f"scores{suffix_idx}": self.scores,
                "image_size": self.image_size,
            }
        else:
            return {
                "keypoints": self.keypoints,
                "descriptors": self.descriptors,
                "scores": self.scores,
                "image_size": self.image_size,
            }

    def torch(self) -> ImageKeyPoints:
        self._is_torch = True
        return (
            ImageKeyPoints(
                keypoints=torch.tensor(self.keypoints),
                descriptors=torch.tensor(self.descriptors),
                scores=torch.tensor(self.scores) if self.scores is not None else None,
                image_size=self.image_size,
            )
            if not self.is_torch
            else self
        )

    def detach(self) -> ImageKeyPoints:
        return (
            ImageKeyPoints(
                keypoints=self.keypoints.detach(),
                descriptors=self.descriptors.detach(),
                scores=self.scores.detach() if self.scores is not None else None,
                image_size=self.image_size,
            )
            if self.is_torch
            else self
        )
