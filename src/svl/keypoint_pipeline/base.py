from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np

from svl.keypoint_pipeline.typing import ImageKeyPoints


class KeyPointDescriptor(ABC):
    """Abstract class for keypoint descriptors."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def describe_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Describe keypoints in an image.

        Parameters
        ----------
        image : np.ndarray
            image to describe keypoints in
        keypoints : np.ndarray
            keypoints to describe

        Returns
        -------
        np.ndarray
            descriptors
        """
        pass


class KeyPointDetector(ABC):
    """Abstract class for keypoint detectors."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in an image.

        Parameters
        ----------
        image : np.ndarray
            image to detect keypoints in

        Returns
        -------
        np.ndarray
            keypoints
        """
        pass

    def _keypoints_to_array(self, keypoints: Tuple[cv2.KeyPoint]) -> np.ndarray:
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)


class CombinedKeyPointAlgorithm(ABC):
    """Abstract class for combined keypoint detection and description."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in an image.

        Parameters
        ----------
        image : np.ndarray
            image to detect keypoints in

        Returns
        -------
        np.ndarray
            keypoints
        """
        pass

    @abstractmethod
    def describe_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Describe keypoints in an image.

        Parameters
        ----------
        image : np.ndarray
            image to describe keypoints in
        keypoints : np.ndarray
            keypoints to describe

        Returns
        -------
        np.ndarray
            descriptors
        """
        pass

    @abstractmethod
    def detect_and_describe_keypoints(self, image: np.ndarray) -> ImageKeyPoints:
        """
        Detect and describe keypoints in an image.

        Parameters
        ----------
        image : np.ndarray
            image to detect and describe keypoints in

        Returns
        -------
        ImageKeyPoints
            keypoints and descriptors
        """
        pass


class KeyPointMatcher(ABC):
    """Abstract class for keypoint matchers."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def match_keypoints(
        self, keypoints1: ImageKeyPoints, keypoints2: ImageKeyPoints
    ) -> np.ndarray:
        """
        Match keypoints between two sets of descriptors.

        Parameters
        ----------
        keypoints1 : ImageKeyPoints
            keypoints and descriptors from the first image
        keypoints2 : ImageKeyPoints
            keypoints and descriptors from the second image

        Returns
        -------
        np.ndarray
            matches
        """
        pass
