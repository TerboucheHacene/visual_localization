import numpy as np

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.typing import SuperPointConfig


def test_superpoint_algorithm(image: np.ndarray):
    assert image.shape == (512, 512)

    config = SuperPointConfig()
    algorithm = SuperPointAlgorithm(config)
    keypoints = algorithm.detect_keypoints(image)

    assert isinstance(keypoints, np.ndarray)
