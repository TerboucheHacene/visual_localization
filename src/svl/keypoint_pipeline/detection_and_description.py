from __future__ import annotations

import dataclasses

import numpy as np
import torch

from superglue_lib.models.superpoint import SuperPoint
from superglue_lib.models.utils import frame2tensor
from svl.keypoint_pipeline.base import (
    CombinedKeyPointAlgorithm,
    KeyPointDescriptor,
    KeyPointDetector,
)
from svl.keypoint_pipeline.typing import ImageKeyPoints, SuperPointConfig


@CombinedKeyPointAlgorithm.register
class SuperPointAlgorithm(CombinedKeyPointAlgorithm):
    """SuperPoint Keypoint Algorithm that can be used to detect and describe keypoints.

    Parameters
    ----------
    config : SuperPointConfig
        configuration for SuperPoint
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.config = config
        self.detector = SuperPoint(dataclasses.asdict(config))
        self.detector = self.detector.eval()
        self.detector = self.detector.to(config.device)

    def detect_and_describe_keypoints(self, image: np.ndarray) -> ImageKeyPoints:
        """
        Detect keypoints in an image using SuperPoint.

        Parameters
        ----------
        image : np.ndarray
            image to detect keypoints in

        Returns
        -------
        ImageKeyPoints
            keypoints with their descriptors
        """

        tensor = frame2tensor(image, self.config.device)
        data = {
            "image": tensor,
        }
        with torch.no_grad():
            outputs = self.detector(data)
        outputs = {k: v[0] for k, v in outputs.items()}
        outputs["descriptors"] = outputs["descriptors"].transpose(1, 0)
        outputs["image_size"] = [image.shape[0], image.shape[1]]
        outputs = ImageKeyPoints(**outputs).to("cpu").numpy()
        return outputs

    def detect_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in an image using SuperPoint.

        Parameters
        ----------
        image : np.ndarray
            image to detect keypoints in

        Returns
        -------
        np.ndarray
            keypoints
        """
        tensor = frame2tensor(image, self.config.device)
        data = {
            "image": tensor,
        }
        with torch.no_grad():
            outputs = self.detector(data)
        outputs = {k: v[0] for k, v in outputs.items()}
        outputs["descriptors"] = outputs["descriptors"].transpose(1, 0)
        outputs["image_size"] = [image.shape[0], image.shape[1]]
        outputs = ImageKeyPoints(**outputs).to("cpu").numpy()
        return outputs.keypoints

    def describe_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        raise NotImplementedError("SuperPoint does not support describing keypoints.")


class GeneralKeypointAlgorithm(CombinedKeyPointAlgorithm):
    """General keypoint algorithm that can be used to detect and describe keypoints.

    Parameters
    ----------
    detector : KeyPointDetector
        keypoint detector
    descriptor : KeyPointDescriptor
        keypoint descriptor
    """

    def __init__(
        self, detector: KeyPointDetector, descriptor: KeyPointDescriptor
    ) -> None:
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor

    def detect_and_describe_keypoints(self, image: np.ndarray) -> ImageKeyPoints:
        """
        Detect and describe keypoints in an image using BRISK.

        Parameters
        ----------
        image : np.ndarray
            image to detect keypoints in

        Returns
        -------
        ImageKeyPoints
            keypoints with their descriptors
        """
        keypoints = self.detector.detect_keypoints(image)
        descriptors = self.descriptor.describe_keypoints(image, keypoints)
        return ImageKeyPoints(keypoints=keypoints, descriptors=descriptors)
