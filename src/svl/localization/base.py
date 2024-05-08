import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from superglue_lib.models.utils import process_resize
from svl.keypoint_pipeline.base import CombinedKeyPointAlgorithm, KeyPointMatcher
from svl.keypoint_pipeline.typing import ImageKeyPoints
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import DroneImage, GeoSatelliteImage
from svl.tms.schemas import GpsCoordinate


@dataclass
class BaseMapReaderItem:
    image_path: Path
    image: Optional[np.ndarray] = None
    size: Optional[Tuple[int, int]] = None
    key_points: Optional[ImageKeyPoints] = None
    name: str = field(init=False)


@dataclass
class PipelineConfig:
    homography_method: int = cv2.RANSAC
    homography_threshold: float = 5.0
    homography_confidence: float = 0.995
    homography_max_iter: int = 2000


class BaseMapReader(ABC):
    """Base class for reading and processing map images

    Parameters
    ----------
    logger : logging.Logger
        Logger object
    resize_size : Optional[Tuple[int, int]], optional
        Size to resize the images to, by default None
    cv2_read_mode : int, optional
        OpenCV read mode, by default cv2.IMREAD_COLOR

    """

    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    def __init__(
        self,
        db_path: Path,
        logger: logging.Logger,
        resize_size: Optional[Tuple[int, int]] = None,
        cv2_read_mode: int = cv2.IMREAD_COLOR,
    ) -> None:
        db_path = db_path if isinstance(db_path, Path) else Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database path not found at {db_path}")
        if not db_path.is_dir():
            raise NotADirectoryError(f"Database path is not a directory at {db_path}")
        self.db_path = db_path
        self.cv2_read_mode = cv2_read_mode
        self.resize_size = resize_size
        self.logger = logger
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the image database"""
        self._image_db: List[BaseMapReaderItem] = []
        self._num_images: int = 0
        self._is_loaded: bool = False
        self._is_described: bool = False

    @property
    def image_names(self) -> List[str]:
        """List of image names in the database"""
        return [image.name for image in self._image_db]

    def __len__(self) -> int:
        """Number of images in the database"""
        return self._num_images

    def __getitem__(self, key: Union[int, str]) -> BaseMapReaderItem:
        """Get image item from the database

        Parameters
        ----------
        key : Union[int, str]
            Index or name of the image. If key is an integer, it returns the image at
            that index. If key is a string, it returns the image with that name.

        Returns
        -------
        BaseMapReaderItem
            Image item from the database
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self._image_db):
                raise IndexError("Index out of range")
            return self._image_db[key]
        elif isinstance(key, str):
            if key not in self.image_names:
                raise KeyError(f"Image with name {key} not found in the database")
            for img_item in self._image_db:
                if img_item.name == key:
                    return img_item
        else:
            raise KeyError("Key must be either an integer or a string")

    def read(self, image_path: Union[str, Path]) -> np.ndarray:
        """Read image from file

        Parameters
        ----------
        image_path : Union[str, Path]
            Path to the image file

        Returns
        -------
        np.ndarray
            Image array
        """
        image_path = image_path if isinstance(image_path, Path) else Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")

        img = cv2.imread(str(image_path), self.cv2_read_mode)

        return img

    def load_images(self) -> None:
        """Load images from the database"""

        if self._is_loaded:
            self.logger.info("Images already loaded")
            return
        for idx in tqdm(range(len(self)), desc="Loading images", total=len(self)):
            self[idx].image = self.read(self[idx].image_path)
            self[idx].size = self[idx].image.shape[:2]
        self._is_loaded = True
        self.logger.info("Images loaded successfully")

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image given np.ndarray. It first computes the new size based on the
        resize_size attribute and then resizes the image.

        Parameters
        ----------
        image : np.ndarray
            Image array, shape (height, width, channels)

        Returns
        -------
        np.ndarray
            Resized image array
        """
        if self.resize_size is None:
            return image
        height, width = image.shape[:2]
        new_width, new_height = process_resize(width, height, self.resize_size)
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized_image

    def resize_image(self, image_name: str) -> None:
        """Resize image item in the database

        Parameters
        ----------
        image_name : str
            Name of the image to resize
        """
        if self[image_name].image is None:
            self.logger.info(f"Image {image_name} not loaded. Loading image")
            self[image_name].image = self.read(self[image_name].image_path)

        img_item = self[image_name]
        img_item.image = self.resize(img_item.image)
        img_item.size = img_item.image.shape[:2]

    def resize_db_images(self) -> None:
        """Resize all images in the database"""
        self.logger.info("Resizing images in the database")
        for imge_name in tqdm(self.image_names):
            self.resize_image(imge_name)

    def extract_features(
        self, image: np.ndarray, algorithm: CombinedKeyPointAlgorithm
    ) -> ImageKeyPoints:
        """Extract features from an image using the given algorithm

        Parameters
        ----------
        image : np.ndarray
            Image array to extract features from
        algorithm : CombinedKeyPointAlgorithm
            Key point detection and description algorithm
        """
        return algorithm.detect_and_describe_keypoints(image)

    def extract_features_from_image(
        self, image_name: str, algorithm: CombinedKeyPointAlgorithm
    ) -> None:
        """Extract features from an image in the database using the given algorithm

        Parameters
        ----------
        image_name : str
            Name of the image to extract features from
        algorithm : CombinedKeyPointAlgorithm
            Key point detection and description algorithm
        """
        kp: ImageKeyPoints = self.extract_features(self[image_name].image, algorithm)
        self[image_name].key_points = kp

    def describe_db_images(self, algorithm: CombinedKeyPointAlgorithm) -> None:
        """Describe all images in the database using the given algorithm

        Parameters
        ----------
        algorithm : CombinedKeyPointAlgorithm
            Key point detection and description algorithm
        """

        if not self._is_loaded:
            raise ValueError("Images are not loaded, call load_images() first")
        self.logger.info(
            f"Describing images in the database using {algorithm.__class__.__name__}"
        )
        for img_info in tqdm(self._image_db):
            self.extract_features_from_image(img_info.name, algorithm)

        self._is_described = True


class BasePipeline:
    """Base class for the localization pipeline

    Parameters
    ----------
    map_reader : BaseMapReader
        Map reader object  to read and process map images
    drone_streamer : DroneImageStreamer
        Drone image streamer object to stream drone images
    detector : CombinedKeyPointAlgorithm
        Key point detection and description algorithm
    matcher : KeyPointMatcher
        Key point matcher algorithm
    config : PipelineConfig
        Pipeline configuration object
    query_processor : QueryProcessor
        Query processor object
    logger : logging.Logger
        Logger object
    """

    def __init__(
        self,
        map_reader: BaseMapReader,
        drone_streamer: DroneImageStreamer,
        detector: CombinedKeyPointAlgorithm,
        matcher: KeyPointMatcher,
        config: PipelineConfig,
        query_processor: QueryProcessor,
        logger: logging.Logger,
    ) -> None:
        self.map_reader = map_reader
        self.drone_streamer = drone_streamer
        self.detector = detector
        self.matcher = matcher
        self.config = config
        self.query_processor = query_processor
        self.logger = logger

    def estimate_and_apply_geometric_transform(
        self, mkpts0: np.ndarray, mkpts1: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[bool, float, np.ndarray]:
        """Estimate and apply a geometric transform between two sets of matched keypoints.

        Parameters
        ----------
        mkpts0 : np.ndarray
            Matched keypoints from the first image.
        mkpts1 : np.ndarray
            Matched keypoints from the second image.
        image_shape : Tuple[int, int]
            Shape of the image (height, width).

        Returns
        -------
        bool
            Whether the geometric transform was estimated and applied successfully.
        np.ndarray [shape=(4, 1, 2)]
            Transformed corners if the transform was applied successfully, None otherwise.
        float
            Number of inliers.
        """
        success = False
        transformed_corners = None
        num_inliers = 0

        try:
            transformation_matrix, mask = cv2.findHomography(
                mkpts0,
                mkpts1,
                self.config.homography_method,
                self.config.homography_threshold,
                maxIters=self.config.homography_max_iter,
                confidence=self.config.homography_confidence,
            )
            num_inliers = np.sum(mask)
            h, w = image_shape

            if transformation_matrix is not None:
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(
                    pts, transformation_matrix
                )
                success = True
        except cv2.error as e:
            self.logger.error(
                f"Failed to estimate and apply the geometric transform: {e}"
            )

        return success, num_inliers, transformed_corners if success else None

    def compute_geo_pose(
        self, satellite_image: GeoSatelliteImage, matching_center: Tuple[int, int]
    ) -> GpsCoordinate:
        """Compute the GPS coordinates of the drone given the satellite image and the
        matching center.

        The GPS coordinates are computed by interpolating the latitude and longitude
        based on the matching center:
        latitude = top_left.lat + abs(matching_center[1]) * (bottom_right.lat - top_left.lat)
        longitude = top_left.long + abs(matching_center[0]) * (bottom_right.long - top_left.long)

        Parameters
        ----------
        satellite_image : GeoSatelliteImage
            the satellite image
        matching_center : Tuple[int, int]
            the center of the affine transform

        Returns
        -------
        GpsCoordinate
            the GPS coordinates of the drone
        """

        latitude = satellite_image.top_left.lat + abs(matching_center[1]) * (
            satellite_image.bottom_right.lat - satellite_image.top_left.lat
        )
        longitude = satellite_image.top_left.long + abs(matching_center[0]) * (
            satellite_image.bottom_right.long - satellite_image.top_left.long
        )

        return GpsCoordinate(lat=latitude, long=longitude)

    def save_viz(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save the image to the output path.

        Parameters
        ----------
        image : np.ndarray
            the image to save
        output_path : Union[str, Path]
            the output path
        """

        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not output_path.parent.exists():
            self.logger.warning(
                f"Output directory {output_path.parent} does not exist. Creating it."
            )
            output_path.parent.mkdir(parents=True)
        cv2.imwrite(str(output_path), image)

    def run_on_image(
        self, drone_image: DroneImage, output_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError("run_on_image method must be implemented in subclass")

    def run(self, output_path: Union[str, Path] = None):
        raise NotImplementedError("run method must be implemented in subclass")

    def compute_metrics(self, preds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute the metrics for the predictions.

        Parameters
        ----------
        preds : List[Dict[str, Any]]
            the predictions from the pipeline run

        Returns
        -------
        Dict[str, Any]
            the metrics for the predictions with the following keys:
            - num_matches: number of matches
            - mae: mean absolute error of the distances
            - max_distance: maximum distance of the matches
            - min_distance: minimum distance of the matches
            - ratio_matches: ratio of matches to total predictions
        """

        num_matches = 0
        total_distance = 0
        for pred in preds:
            num_matches += pred["is_match"]
            total_distance += pred["distance"] if pred["is_match"] else 0
        return {
            "num_matches": num_matches,
            "mae": total_distance / num_matches if num_matches > 0 else 0,
            "max_distance": (
                max([pred["distance"] for pred in preds if pred["is_match"]])
                if num_matches > 0
                else 0
            ),
            "min_distance": (
                min([pred["distance"] for pred in preds if pred["is_match"]])
                if num_matches > 0
                else 0
            ),
            "ratio_matches": num_matches / len(preds),
        }

    def draw_transform_polygon_on_image(
        self, image: np.ndarray, transformed_corners: np.ndarray
    ) -> np.ndarray:
        """Draw a polygon representing the affine transform on the image.

        Parameters
        ----------
        image : np.ndarray
            An ndarray representing the image.
        transformed_corners : np.ndarray
            A 4x1x2 array representing the transformed polygon corners.

        Returns
        -------
        np.ndarray
            The image with the transform polygon drawn.

        Notes
        -----
        The transformed_corners should contain 4 points (corners) after transformation.
        """
        transformed_corners = np.int32(transformed_corners)
        image_with_polygon = cv2.polylines(
            image,
            [transformed_corners],
            isClosed=True,
            color=(255, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        return image_with_polygon

    def compute_center(self, transformed_corners: np.ndarray) -> Tuple[int, int]:
        """Compute the center of the affine transform.

        The center is computed as the centroid of the transformed polygon corners. It
        is computed as the average of the x and y coordinates of the transformed corners.

        Parameters
        ----------
        transformed_corners : np.ndarray
            A 4x1x2 array representing the transformed polygon corners.

        Returns
        -------
        Tuple[int, int]
            The center coordinates (cx, cy) of the affine transform.
        """

        moments = cv2.moments(transformed_corners)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    def normalize_center(
        self, center: Tuple[int, int], image_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Normalize the center of the transformed polygon.

        Parameters
        ----------
        center : Tuple[int, int]
            the center of the transformed polygon
        image_shape : Tuple[int, int]
            the shape of the image

        Returns
        -------
        Tuple[float, float]
            the normalized center coordinates (cx, cy) of the transformed polygon
        """
        cx, cy = center
        return cx / image_shape[1], cy / image_shape[0]

    def draw_center(self, image: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Draw the center of the affine transform on the image.

        Parameters
        ----------
        image : np.ndarray
            an ndarray representing the image
        center : Tuple[int, int]
            the center of the affine transform

        Returns
        -------
        np.ndarray
            the image with the center of the affine transform drawn
        """
        cx, cy = center
        cv2.circle(
            image,
            (cx, cy),
            radius=10,
            color=(255, 0, 255),
            thickness=5,
        )
        return image
