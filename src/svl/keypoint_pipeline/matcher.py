import dataclasses
from typing import Tuple

import numpy as np
import torch

from superglue_lib.models.superglue import SuperGlue
from svl.keypoint_pipeline.base import KeyPointMatcher
from svl.keypoint_pipeline.typing import ImageKeyPoints, SuperGlueConfig


class SuperGlueMatcher(KeyPointMatcher):
    """SuperGlue keypoint matcher.

    Parameters
    ----------
    config : SuperGlueConfig
        configuration for SuperGlue
    """

    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        self.config = config
        self.device = config.device
        self.matcher = SuperGlue(dataclasses.asdict(config))
        self.matcher = self.matcher.eval()
        self.matcher = self.matcher.to(config.device)

    def match_keypoints(
        self, keypoints_1: ImageKeyPoints, keypoints_2: ImageKeyPoints
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match keypoints between two sets of descriptors using SuperGlue.

        Parameters
        ----------
        keypoints_1 : ImageKeyPoints
            keypoints from the first image
        keypoints_2 : ImageKeyPoints
            keypoints from the second image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            matches and confidence
        """
        inputs = {}
        # SuperGlue matching needs images only to get the image size
        # Workaround: create dummy images with the image_size specified in the keypoints
        inputs["image0"] = torch.randn(1, 1, *keypoints_1.image_size)
        inputs["image1"] = torch.randn(1, 1, *keypoints_2.image_size)

        # Convert keypoints to tensor
        keypoints_1 = (
            keypoints_1.torch()
            .to(self.device)
            .add_batch_dimension()
            .to_dict(suffix_idx=0)
        )
        keypoints_1["descriptors0"] = keypoints_1["descriptors0"].transpose(-2, -1)

        keypoints_2 = (
            keypoints_2.torch()
            .to(self.device)
            .add_batch_dimension()
            .to_dict(suffix_idx=1)
        )
        keypoints_2["descriptors1"] = keypoints_2["descriptors1"].transpose(-2, -1)

        inputs.update(**keypoints_1)
        inputs.update(**keypoints_2)

        with torch.no_grad():
            preds = self.matcher(inputs)
        matches = preds["matches0"].cpu().numpy().squeeze()
        confidence = preds["matching_scores0"].cpu().numpy().squeeze()
        return matches, confidence
