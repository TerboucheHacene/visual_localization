import numpy as np
from conftest import FEATURES

from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import ImageKeyPoints, SuperGlueConfig


def test_super_glue_matcher():
    config = SuperGlueConfig()

    matcher = SuperGlueMatcher(config=config)
    assert matcher.config == config
    assert matcher.device == config.device

    # Test match_keypoints
    keypoints_1 = ImageKeyPoints(
        keypoints=np.array([[1, 2], [3, 4], [5, 6]]),
        descriptors=np.random.rand(3, FEATURES).astype(np.float32),
        scores=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        image_size=(100, 100),
    )
    keypoints_2 = ImageKeyPoints(
        keypoints=np.array([[7, 8], [9, 10], [11, 12]]),
        descriptors=np.random.rand(3, FEATURES).astype(np.float32),
        scores=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        image_size=(100, 100),
    )

    matches, confidence = matcher.match_keypoints(keypoints_1, keypoints_2)
    assert isinstance(matches, np.ndarray)
    assert isinstance(confidence, np.ndarray)
    assert matches.shape[0] == confidence.shape[0]
