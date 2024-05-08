import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

from superglue_lib.models.utils import make_matching_plot_fast
from svl.keypoint_pipeline.base import CombinedKeyPointAlgorithm
from svl.keypoint_pipeline.matcher import KeyPointMatcher
from svl.localization.base import BasePipeline, PipelineConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import TileSatelliteMapReader
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import DroneImage, TileImage
from svl.tms.geo import haversine_distance
from svl.tms.schemas import GpsCoordinate


class TilePipeline(BasePipeline):
    """Pipeline for localizing drone images using satellite map tiles.

    Parameters
    ----------
    map_reader : TileSatelliteMapReader
        reader for the satellite map tiles
    drone_streamer : DroneImageStreamer
        streamer for the drone images
    detector : CombinedKeyPointAlgorithm
        keypoint detector and descriptor
    matcher : KeyPointMatcher
        keypoint matcher for matching keypoints
    config : PipelineConfig
        configuration for the pipeline
    query_processor : QueryProcessor
        query processor for the drone images
    logger : logging.Logger
        logger to use for logging
    """

    def __init__(
        self,
        map_reader: TileSatelliteMapReader,
        drone_streamer: DroneImageStreamer,
        detector: CombinedKeyPointAlgorithm,
        matcher: KeyPointMatcher,
        config: PipelineConfig,
        query_processor: QueryProcessor,
        logger: logging.Logger,
    ):

        super().__init__(
            drone_streamer=drone_streamer,
            map_reader=map_reader,
            detector=detector,
            matcher=matcher,
            config=config,
            query_processor=query_processor,
            logger=logger,
        )

    def compute_geo_pose(
        self, satellite_tile: TileImage, matching_center: Tuple[int, int]
    ) -> GpsCoordinate:
        """Compute the GPS coordinates of the drone image based on the satellite tile.

        Parameters
        ----------
        satellite_tile : TileImage
            satellite tile image
        matching_center : Tuple[int, int]
            center of the matching keypoints

        Returns
        -------
        GpsCoordinate
            GPS coordinates of the drone image
        """

        # Get the top left and bottom right corners of the tile
        top_left_lat = satellite_tile.tile.lat
        top_left_long = satellite_tile.tile.long
        bottom_right_lat = satellite_tile.tile.bottom_right_corner.lat
        bottom_right_long = satellite_tile.tile.bottom_right_corner.long

        latitude = (
            top_left_lat + abs(bottom_right_lat - top_left_lat) * matching_center[1]
        )
        longitude = (
            top_left_long + abs(bottom_right_long - top_left_long) * matching_center[0]
        )
        return GpsCoordinate(lat=latitude, long=longitude)

    def run_on_image(
        self, drone_image: DroneImage, output_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """Run the pipeline on a single drone image.

        Parameters
        ----------
        drone_image : DroneImage
            drone image to process
        output_path : Union[str, Path], optional
            path to save the visualization, by default None

        Returns
        -------
        Dict[str, Any]
            results of the pipeline
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
            satellite_image: TileImage = self.map_reader[idx]

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

                max_macthes = len(mkpts1)
                denormalized_center = self.compute_center(dst)
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
                cv2.imwrite(str(viz_path), out)
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

    def run(self, output_path: Union[str, Path] = None) -> list[Dict[str, Any]]:
        """Run the pipeline on all drone images.

        Parameters
        ----------
        output_path : Union[str, Path], optional
            path to save the visualization, by default None

        Returns
        -------
        list[Dict[str, Any]]
            results of the pipeline for each drone image
        """
        results = []
        for drone_image in tqdm(self.drone_streamer):
            drone_image = self.query_processor(drone_image)
            result = self.run_on_image(drone_image, output_path)
            results.append(result)
        return results
