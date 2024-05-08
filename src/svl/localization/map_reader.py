import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from svl.keypoint_pipeline.base import CombinedKeyPointAlgorithm
from svl.keypoint_pipeline.typing import ImageKeyPoints
from svl.localization.base import BaseMapReader
from svl.tms.data_structures import (
    FlightZone,
    GeoSatelliteImage,
    Mosaic,
    Tile,
    TileImage,
)
from svl.tms.schemas import GpsCoordinate


class SatelliteMapReader(BaseMapReader):
    """Class for reading and processing satellite map images

    This class reads satellite map images and their metadata from a directory.
    The metadata is expected to be in a CSV file with the columns specified in the
    `COLUMN_NAMES` attribute.


    Parameters
    ----------
    db_path : Path
        Path to the directory containing the images
    logger : logging.Logger
        Logger object
    resize_size : Tuple[int, int]
        Size to resize the images to
    cv2_read_mode : int, optional
        OpenCV read mode, by default cv2.IMREAD_GRAYSCALE
    metadata_method : str, optional
        Method to load metadata, by default "CSV"
    """

    COLUMN_NAMES = [
        "Filename",
        "Top_left_lat",
        "Top_left_lon",
        "Bottom_right_lat",
        "Bottom_right_long",
    ]
    METADATA_METHOD = [
        "CSV",
    ]

    def __init__(
        self,
        db_path: Path,
        logger: logging.Logger,
        resize_size: Tuple[int, int],
        cv2_read_mode: int = cv2.IMREAD_GRAYSCALE,
        metadata_method: str = "CSV",
    ) -> None:
        super().__init__(db_path, logger, resize_size, cv2_read_mode)
        if metadata_method not in self.METADATA_METHOD:
            raise ValueError(f"Invalid metadata method {metadata_method}")

    def setup_db(self) -> None:
        """Setup the image database."""
        self._build_image_db()
        self.load_images()
        self._load_csv_metadata()
        self.set_metadata_for_all_images()

    def initialize_db(self) -> None:
        """Initialize the image database."""
        super()._initialize_db()
        self._geo_metadata: pd.DataFrame = None

    def _build_image_db(self) -> None:
        """Build the image database from the images in the directory."""
        self.logger.info(f"Building image database from {self.db_path}")
        img_idx = 0
        for image_path in tqdm(sorted(self.db_path.glob("*"))):
            if image_path.suffix in self.IMAGE_EXTENSIONS:
                self._image_db.append(
                    GeoSatelliteImage(
                        image_path=image_path,
                        index=img_idx,
                    )
                )
                img_idx += 1

        self._num_images = img_idx
        self.logger.info(
            f"Image database built successfully with {self._num_images} images"
        )

    def _load_csv_metadata(self):
        """Load metadata from a CSV file into a DataFrame."""
        csv_files = list(self.db_path.glob("*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.db_path}")
        if len(csv_files) > 1:
            raise ValueError(f"Multiple CSV files found in {self.db_path}")
        csv_file = csv_files[0]
        self.logger.info(f"Loading metadata from {csv_file}")
        df = pd.read_csv(csv_file)
        if not all(col in df.columns for col in self.COLUMN_NAMES):
            raise ValueError(f"Invalid metadata columns in {csv_file}")
        df["Filename"] = df["Filename"].apply(lambda x: x.split(".")[0])
        self._geo_metadata = df
        self.logger.info("Metadata loaded successfully")

    def set_image_metadata(self, image_name: str, metadata: Dict[str, float]) -> None:
        """Set metadata for a specific image."""
        satellite_image: GeoSatelliteImage = self[image_name]
        satellite_image.top_left = GpsCoordinate(
            lat=metadata["Top_left_lat"], long=metadata["Top_left_lon"]
        )
        satellite_image.bottom_right = GpsCoordinate(
            lat=metadata["Bottom_right_lat"], long=metadata["Bottom_right_long"]
        )

    def set_metadata_for_all_images(self) -> None:
        """Set metadata for all images in the database."""
        self.logger.info("Setting metadata for all images")

        for img_info in tqdm(self._image_db):
            img_metadata = self._geo_metadata[
                self._geo_metadata["Filename"] == img_info.name
            ]
            if len(img_metadata) == 1:
                img_metadata = img_metadata.to_dict(orient="records")[0]
                del img_metadata["Filename"]
                self.set_image_metadata(img_info.name, img_metadata)
            elif len(img_metadata) > 1:
                self.logger.warning(
                    f"Multiple metadata entries found for image {img_info.name}"
                )
            else:
                self.logger.warning(f"Metadata not found for image {img_info.name}")

    @property
    def goe_metadata(self) -> pd.DataFrame:
        return self._geo_metadata


class TileSatelliteMapReader(BaseMapReader):
    """Class for reading and processing satellite map images in a tile-based format.

    This class reads satellite map images in a tile-based format from a directory.
    The images are expected to be named in the format `x_y_z.png` where `x`, `y`, and `z`
    are the tile indices and zoom level respectively.

    Parameters
    ----------
    db_path : Path
        Path to the directory containing the images
    logger : logging.Logger
        Logger object
    zoom_level : int
        Zoom level of the tile images
    resize_size : Tuple[int, int]
        Size to resize the images to
    flight_zone : FlightZone
        Flight zone of the images
    cv2_read_mode : int, optional
        OpenCV read mode, by default cv2.IMREAD_COLOR

    """

    def __init__(
        self,
        db_path: Path,
        zoom_level: int,
        logger: logging.Logger,
        flight_zone: FlightZone = None,
        cv2_read_mode: int = cv2.IMREAD_GRAYSCALE,
    ) -> None:
        super().__init__(db_path, logger, None, cv2_read_mode)
        self.zoom_level = zoom_level
        self.flight_zone = flight_zone

    def initialize_db(self) -> None:
        """Initialize the image database."""
        super()._initialize_db()
        self._image_db: List[TileImage] = list()
        self._num_images: int = 0
        self._mosaic: Mosaic = None
        self._is_loaded = False
        self._is_described = False

    def setup_db(self) -> None:
        """Setup the image database."""
        if self.flight_zone is not None:
            self._build_image_db_with_flight_zone()
        else:
            self._build_image_db()
        self.load_images()

    def __getitem__(self, key: Union[int, str, Tuple[int, int]]) -> TileImage:
        """Get an image from the database.

        Parameters
        ----------
        key : Union[int, str, Tuple[int, int]]
            Key to get the image by. Can be:
            - int: Index of the image in the database
            - str: Name of the image
            - Tuple[int, int]: Local indices of the image in the mosaic

        Returns
        -------
        TileImage
            Image object
        """

        if isinstance(key, int):
            if key < 0 or key >= len(self._image_db):
                raise IndexError("Index out of range")
            return self._image_db[key]
        elif isinstance(key, str):
            if key not in self.image_names:
                raise KeyError(f"Image with name {key} not found in the database")
            for img_info in self._image_db:
                if img_info.name == key:
                    return img_info

        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Key must be a tuple of two integers")
            x, y = key
            ret = None
            for img_info in self._image_db:
                if img_info.index == (x, y):
                    ret = img_info
            if ret is None:
                raise KeyError(f"Image with index {key} not found in the database")
            return ret
        else:
            raise KeyError("Key must be either an integer or a string")

    def _build_image_db_with_flight_zone(self) -> None:
        """Build the image database with a flight zone.

        This method builds the image database with a flight zone defined. The tile
        indices are calculated based on the flight zone. The corresponding images
        are loaded from the database.
        """

        self.logger.info(f"Building image database from {self.db_path}")
        img_count = 0

        for tile, local_indices in self.flight_zone.yield_tiles_with_indices(
            zoom_level=self.zoom_level
        ):
            image_path = self.db_path / f"{tile.x}_{tile.y}_{tile.zoom_level}.png"
            if image_path.exists():
                self._image_db.append(
                    TileImage(
                        image_path=image_path,
                        tile=tile,
                        index=local_indices,
                    )
                )
                img_count += 1
            else:
                raise FileNotFoundError(f"Image file not found at {image_path}")
        self._num_images = img_count

    def _build_image_db(self):
        """Build the image database from the images in the directory.

        This method builds the image database from the images in the directory. The
        tile indices are extracted from the image names. The flight zone is calculated
        based on the top-left and bottom-right tiles.
        """

        if self.flight_zone is not None:
            raise ValueError(
                "Flight zone is set, use _build_image_db_with_flight_zone()"
            )

        self.logger.info(f"Building image database from {self.db_path}")
        img_idx = 0
        for image_path in tqdm(sorted(self.db_path.glob("*"))):
            if image_path.suffix in self.IMAGE_EXTENSIONS:
                filename = image_path.stem  # x_y_z.jpg
                image_zoom_level = int(filename.split("_")[2])
                if image_zoom_level != self.zoom_level:
                    raise ValueError(
                        f"Invalid zoom level {image_zoom_level} for image {filename}"
                    )
                tile = Tile(
                    x=int(filename.split("_")[0]),
                    y=int(filename.split("_")[1]),
                    zoom_level=image_zoom_level,
                )
                self._image_db.append(
                    TileImage(
                        image_path=image_path,
                        index=None,
                        tile=tile,
                    )
                )
                img_idx += 1
        self._num_images = img_idx
        self.logger.info(
            f"Image database built successfully with {self.num_images} images"
        )

        top_left_tile = self.get_top_left_tile()
        bottom_right_tile = self.get_bottom_right_tile()
        self.flight_zone = FlightZone(
            top_left_lat=top_left_tile.lat,
            top_left_long=top_left_tile.long,
            bottom_right_lat=bottom_right_tile.lat,
            bottom_right_long=bottom_right_tile.long,
        )

        # set local indices
        for img_info in self._image_db:
            local_index_x = img_info.tile.x - top_left_tile.x
            local_index_y = img_info.tile.y - top_left_tile.y
            img_info.index = (local_index_x, local_index_y)

    def get_top_left_tile(self) -> Tile:
        """Get the top-left tile in the database."""
        x_min = min([img.tile.x for img in self._image_db])
        y_min = min([img.tile.y for img in self._image_db])
        # TODO: ensure both xmin and ymin exist together in the same tile
        return Tile(x=x_min, y=y_min, zoom_level=self.zoom_level)

    def get_bottom_right_tile(self) -> Tile:
        """Get the bottom-right tile in the database."""
        x_max = max([img.tile.x for img in self._image_db])
        y_max = max([img.tile.y for img in self._image_db])
        # TODO: ensure both xmax and ymax exist together in the same tile
        return Tile(x=x_max, y=y_max, zoom_level=self.zoom_level)


class MosaicSatelliteMapReader(TileSatelliteMapReader):
    """Class for reading and processing satellite map images in a mosaic format.

    This class reads a single satellite map image in a mosaic format. The image is
    expected to be a mosaic of multiple tiles. The image is loaded and processed as a
    single image.

    Parameters
    ----------
    db_path : Path
        Path to the tile database
    zoom_level : int
        Zoom level of the map tiles
    logger : logging.Logger
        Logger object for logging messages
    flight_zone : Optional[FlightZone], optional
        Flight zone definition, by default None
    cv2_read_mode : int, optional
        OpenCV read mode for images, by default cv2.IMREAD_GRAYSCALE
    """

    def __init__(
        self,
        db_path: Path,
        zoom_level: int,
        logger: logging.Logger,
        flight_zone: Optional[FlightZone] = None,
        cv2_read_mode: int = cv2.IMREAD_GRAYSCALE,
    ) -> None:
        """
        Initialize the MosaicSatelliteMapReader.

        Parameters
        ----------
        db_path : Path
            Path to the tile database.
        zoom_level : int
            Zoom level of the map tiles.
        logger : logging.Logger
            Logger object for logging messages.
        flight_zone : Optional[FlightZone], optional
            Flight zone definition, by default None.
        cv2_read_mode : int, optional
            OpenCV read mode for images, by default cv2.IMREAD_GRAYSCALE.
        """
        super().__init__(db_path, zoom_level, logger, flight_zone, cv2_read_mode)
        self._mosaic: Optional[Mosaic] = None

    @property
    def mosaic(self) -> Mosaic:
        """Get the constructed mosaic image."""
        if self._mosaic is None:
            raise ValueError("Mosaic not constructed, call construct_mosaic() first")
        return self._mosaic

    def _generate_mosaic_image(self) -> np.ndarray:
        """Generate a mosaic image from the loaded tile images."""
        if not self._is_loaded:
            raise ValueError("Images are not loaded, call load_images() first")

        width_in_pixels, height_in_pixels = self.flight_zone.size_in_pixels(
            zoom_level=self.zoom_level
        )

        mosaic_image = np.zeros((height_in_pixels, width_in_pixels), dtype=np.uint8)

        for image_tile in self._image_db:
            local_indices = image_tile.index
            tile_size = image_tile.tile.tile_size
            image = image_tile.image
            x_offset = local_indices[0] * tile_size
            y_offset = local_indices[1] * tile_size

            mosaic_image[
                y_offset : y_offset + tile_size, x_offset : x_offset + tile_size
            ] = image

        return mosaic_image

    def construct_mosaic(self, algorithm: CombinedKeyPointAlgorithm) -> Mosaic:
        """
        Construct a mosaic from the images in the database using the specified keypoint
        algorithm.

        Parameters
        ----------
        algorithm : CombinedKeyPointAlgorithm
            Key point algorithm to use for feature extraction.

        Returns
        -------
        Mosaic
            Constructed mosaic object.
        """
        mosaic_image = self._generate_mosaic_image()
        key_points: ImageKeyPoints = self.extract_features(mosaic_image, algorithm)
        key_points.image_size = (mosaic_image.shape[1], mosaic_image.shape[0])

        mosaic = Mosaic(
            image=mosaic_image,
            zoom_level=self.zoom_level,
            flight_zone=self.flight_zone,
            key_points=key_points,
        )

        self.logger.info("Mosaic constructed successfully")
        self._mosaic = mosaic

        return mosaic
