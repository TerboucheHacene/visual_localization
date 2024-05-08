import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

from superglue_lib.models.utils import make_matching_plot_fast
from svl.keypoint_pipeline.base import CombinedKeyPointAlgorithm
from svl.keypoint_pipeline.matcher import KeyPointMatcher
from svl.localization.base import BasePipeline, PipelineConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import DroneImage, GeoSatelliteImage
from svl.tms.geo import haversine_distance
from svl.tms.schemas import GpsCoordinate


class Pipeline(BasePipeline):
    """Class to run the localization pipeline based on keypoint matching.

    The pipeline consists of the following steps:
    1. Detect keypoints in the drone image
    2. Match the keypoints with the keypoints in the satellite image
    3. Estimate the affine transform between the matched keypoints
    4. Compute the center of the affine transform
    5. Compute the predicted GPS coordinates based on the center of the affine transform
    6. Compute the haversine distance between the predicted GPS coordinates and the
       ground truth GPS coordinates
    7. Save the visualization of the matching results


    Parameters
    ----------
    map_reader : SatelliteMapReader
        the map reader to read the satellite images, stored in a database
    drone_streamer : DroneImageStreamer
        the drone image streamer to read the drone images
    detector : CombinedKeyPointAlgorithm
        the keypoint detector to accomplish the detection and description of keypoints
    matcher : KeyPointMatcher
        the keypoint matcher to accomplish the matching of keypoints
    config : PipelineConfig
        the configuration of the pipeline
    query_processor : QueryProcessor
        the query processor to preprocess the query image (resize, warp, etc.)
    logger : logging.Logger
        the logger to use for logging
    """

    def __init__(
        self,
        map_reader: SatelliteMapReader,
        drone_streamer: DroneImageStreamer,
        detector: CombinedKeyPointAlgorithm,
        matcher: KeyPointMatcher,
        config: PipelineConfig,
        query_processor: QueryProcessor,
        logger: logging.Logger,
    ) -> None:

        super().__init__(
            map_reader=map_reader,
            drone_streamer=drone_streamer,
            detector=detector,
            matcher=matcher,
            config=config,
            query_processor=query_processor,
            logger=logger,
        )

    def run_on_image(
        self, drone_image: DroneImage, output_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """Run the pipeline on a single drone image.

        Parameters
        ----------
        drone_image : DroneImage
            the drone image to process
        output_path : Union[str, Path]
            the output path to save the visualization

        Returns
        -------
        Dict[str, Any]
            the prediction results with the following keys:
            - is_match: bool
                whether a match was found
            - gt_coordinate: GpsCoordinate
                the ground truth GPS coordinate
            - predicted_coordinate: GpsCoordinate
                the predicted GPS coordinate
            - center: Tuple[int, int]
                the center of the affine transform
            - best_dst: np.ndarray
                the affine transform matrix
            - matched_image: str
                the name of the matched satellite image
            - distance: float
                the haversine distance between the predicted and ground truth GPS
                coordinates
        """
        self.logger.info(f"Processing image {drone_image.name}")

        max_macthes = -1
        best_dst = None
        matched_image = None
        is_match = False
        predicted_coordinates = None
        center = None
        distance = None
        matched_kpts0 = None
        matched_kpts1 = None
        features_mean = None
        matched_confidence = None
        matched_valid = None
        matched_inliers = None

        drone_image.key_points = self.detector.detect_and_describe_keypoints(
            drone_image.image
        )
        gt_coordinates = GpsCoordinate(
            lat=drone_image.geo_point.latitude,
            long=drone_image.geo_point.longitude,
        )

        for idx in tqdm(
            range(len(self.map_reader)),
            desc="Matching images",
            total=len(self.map_reader),
        ):
            satellite_image: GeoSatelliteImage = self.map_reader[idx]

            # Match the keypoints
            matches, confidence = self.matcher.match_keypoints(
                drone_image.key_points, satellite_image.key_points
            )
            valid = matches > -1
            mkpts0 = drone_image.key_points.keypoints[valid]
            mkpts1 = satellite_image.key_points.keypoints[matches[valid]]

            if len(mkpts0) < 4:
                logging.debug(
                    f"Skipping image {satellite_image.name} not enough matches {len(mkpts0)}"
                )
                continue

            ret, num_inliers, dst = self.estimate_and_apply_geometric_transform(
                mkpts0, mkpts1, drone_image.image.shape[:2]
            )

            if ret and len(mkpts1) > max_macthes:
                try:
                    denormalized_center = self.compute_center(dst)
                except Exception as e:
                    self.logger.error(f"Error computing center: {e}")
                    continue

                max_macthes = len(mkpts1)
                center = self.normalize_center(
                    denormalized_center, satellite_image.image.shape
                )
                if center[0] < 0 or center[0] > 1 or center[1] < 0 or center[1] > 1:
                    continue

                best_dst = dst
                matched_image = satellite_image
                features_mean = np.mean(mkpts0, axis=0)
                matched_kpts0 = mkpts0
                matched_kpts1 = mkpts1
                matched_confidence = confidence
                matched_valid = valid
                matched_inliers = num_inliers

                # viz dron image
                viz_satellite_image = satellite_image.image.copy()
                viz_satellite_image = self.draw_transform_polygon_on_image(
                    viz_satellite_image, dst
                )
                viz_satellite_image = self.draw_center(
                    viz_satellite_image, denormalized_center
                )

                # viz satellite image
                viz_drone_image = drone_image.image.copy()
                viz_drone_image = self.draw_center(
                    viz_drone_image, (int(features_mean[0]), int(features_mean[1]))
                )

        if best_dst is not None:
            predicted_coordinates = self.compute_geo_pose(matched_image, center)

            distance = haversine_distance(gt_coordinates, predicted_coordinates)
            is_match = True
            color = cm.jet(matched_confidence[matched_valid])
            if output_path:
                output_path = (
                    Path(output_path) if isinstance(output_path, str) else output_path
                )
                viz_path = output_path / f"{drone_image.name}_viz.jpg"
                out = make_matching_plot_fast(
                    image0=viz_drone_image,
                    image1=viz_satellite_image,
                    kpts0=drone_image.key_points.keypoints,
                    kpts1=matched_image.key_points.keypoints,
                    mkpts0=matched_kpts0,
                    mkpts1=matched_kpts1,
                    color=color,
                    text="",
                    path=None,
                    show_keypoints=True,
                    small_text=[
                        f"GT: {gt_coordinates}",
                        f"Pred: {predicted_coordinates}",
                    ],
                )
                self.save_viz(out, viz_path)
            self.logger.info(
                f"Predicted coordinates: {predicted_coordinates}, GT coordinates: {gt_coordinates}"
            )
            self.logger.info(f"Haversine distance in meters: {distance * 1000}")
        else:
            self.logger.warning(f"No match found for {drone_image.name}")

        return {
            "is_match": is_match,
            "gt_coordinate": gt_coordinates,
            "predicted_coordinate": predicted_coordinates,
            "center": center,
            "best_dst": best_dst,
            "num_inliers": matched_inliers,
            "matched_image": matched_image,
            "distance": distance * 1000 if distance else None,
        }

    def run(self, output_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """Run the pipeline on all drone images.

        Parameters
        ----------
        output_path : Union[str, Path]
            the output path to save the visualization

        Returns
        -------
        List[Dict[str, Any]]
            the prediction results for all drone images
        """
        self.logger.info(f"Running the pipeline on {len(self.drone_streamer)} images")
        preds = []
        num_matches = 0
        for drone_image in self.drone_streamer:
            query = self.query_processor(drone_image)
            pred = self.run_on_image(query, output_path)
            pred["matched_image"] = (
                pred["matched_image"].name if pred["matched_image"] else None
            )
            preds.append(pred)
            num_matches += pred["is_match"]
            if pred["is_match"] and pred["distance"] > 50:
                self.logger.warning(
                    f"Large distance: {pred['distance']} for {drone_image.name}"
                )
        self.logger.info(
            f"Number of matches: {num_matches} among {len(self.drone_streamer)} images"
        )
        return preds
