from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np

from svl.keypoint_pipeline.typing import ImageKeyPoints
from svl.tms.geo import (
    get_lat_long_from_tile_xy,
    get_tile_xy_from_lat_long,
    haversine_distance,
    resolution_at_zoom_level,
)
from svl.tms.schemas import GeoPoint, GpsCoordinate, Orientation, TileCoordinate
from svl.utils.checks import check_gps_zone, check_zoom_level
from svl.utils.constants import TILE_SIZE
from svl.utils.io import JsonParser, YamlParser


@dataclass
class Tile:
    """Class to represent a tile in a tile map service (TMS).

    A tile is a square image that is part of a larger map. The tile is identified by its
    x, y coordinates and zoom level.

    Parameters
    ----------
    x : int
        x coordinate of the tile
    y : int
        y coordinate of the tile
    zoom_level : int
        zoom level of the tile

    Properties
    ----------
    lat : float
        latitude of the top left corner of the tile
    long : float
        longitude of the top left corner of the tile
    spatial_resolution : float
        spatial resolution of the tile
    file_name : str
        file name of the tile
    bottom_right_corner : GpsCoordinate
        bottom right corner of the tile

    """

    x: int
    y: int
    zoom_level: int
    lat: float = field(init=False)  # top left corner
    long: float = field(init=False)  # top left corner
    spatial_resolution: float = field(init=False)
    file_name: str = field(init=False)
    tile_size: int = TILE_SIZE

    def __post_init__(self) -> None:
        check_zoom_level(self.zoom_level)
        self.file_name = f"{self.x}_{self.y}_{self.zoom_level}"

        self.lat, self.long = get_lat_long_from_tile_xy(
            x=self.x, y=self.y, zoom_level=self.zoom_level
        )
        self.spatial_resolution = self.compute_spatial_resolution(
            tile_size=self.tile_size
        )

    @property
    def bottom_right_corner(self) -> GpsCoordinate:
        """Return the bottom right corner of the tile."""
        lat, long = get_lat_long_from_tile_xy(
            x=self.x + 1, y=self.y + 1, zoom_level=self.zoom_level
        )
        return GpsCoordinate(lat=lat, long=long)

    @staticmethod
    def from_lat_long(lat: float, long: float, zoom_level: int) -> Tile:
        """Create a tile from latitude, longitude, and zoom level."""
        x, y = get_tile_xy_from_lat_long(lat=lat, long=long, zoom_level=zoom_level)
        return Tile(x=x, y=y, zoom_level=zoom_level)

    @staticmethod
    def from_tile_coordinate(tile_coordinate: TileCoordinate) -> Tile:
        """Create a tile from a tile coordinate."""
        return Tile(
            x=tile_coordinate.x,
            y=tile_coordinate.y,
            zoom_level=tile_coordinate.zoom_level,
        )

    def compute_spatial_resolution(self, tile_size: int = 256) -> float:
        """Calculate the spatial resolution of the tile."""
        return resolution_at_zoom_level(
            lat=self.lat, zoom_level=self.zoom_level, tile_size=tile_size
        )


@dataclass
class FlightZone:
    """Class to represent a flight zone, a rectangular area represented by GPS coordinates.

    A flight zone is defined by the top left and bottom right GPS coordinates. The flight
    zone is associated with a width and height in meters.

    Parameters
    ----------
    top_left_lat : float
        latitude of the top left corner of the flight zone
    top_left_long : float
        longitude of the top left corner of the flight zone
    bottom_right_lat : float
        latitude of the bottom right corner of the flight zone
    bottom_right_long : float
        longitude of the bottom right corner of the flight zone

    Properties
    ----------
    width_in_meters : float
        width of the flight zone in meters
    height_in_meters : float
        height of the flight zone in meters

    """

    top_left_lat: float
    top_left_long: float
    bottom_right_lat: float
    bottom_right_long: float
    width_in_meters: float = field(init=False)
    height_in_meters: float = field(init=False)

    def __post_init__(self) -> None:
        check_gps_zone(
            top_left=GpsCoordinate(lat=self.top_left_lat, long=self.top_left_long),
            bottom_right=GpsCoordinate(
                lat=self.bottom_right_lat, long=self.bottom_right_long
            ),
        )

        self.width_in_meters = (
            haversine_distance(
                GpsCoordinate(lat=self.top_left_lat, long=self.top_left_long),
                GpsCoordinate(lat=self.top_left_lat, long=self.bottom_right_long),
            )
            * 1000
        )
        self.height_in_meters = (
            haversine_distance(
                GpsCoordinate(lat=self.top_left_lat, long=self.top_left_long),
                GpsCoordinate(lat=self.bottom_right_lat, long=self.top_left_long),
            )
            * 1000
        )

    @staticmethod
    def from_yaml(yaml_file: str) -> FlightZone:
        data = YamlParser.load_yaml(yaml_file)
        return FlightZone(
            top_left_lat=data["top_left_lat"],
            top_left_long=data["top_left_long"],
            bottom_right_lat=data["bottom_right_lat"],
            bottom_right_long=data["bottom_right_long"],
        )

    @staticmethod
    def from_json(json_file: str) -> FlightZone:
        data = JsonParser.load_json(json_file)
        return FlightZone(
            top_left_lat=data["top_left_lat"],
            top_left_long=data["top_left_long"],
            bottom_right_lat=data["bottom_right_lat"],
            bottom_right_long=data["bottom_right_long"],
        )

    @staticmethod
    def from_gps_coordinates(
        top_left: GpsCoordinate, bottom_right: GpsCoordinate
    ) -> FlightZone:
        return FlightZone(
            top_left_lat=top_left.lat,
            top_left_long=top_left.long,
            bottom_right_lat=bottom_right.lat,
            bottom_right_long=bottom_right.long,
        )

    def top_left_tile(self, zoom_level: int) -> Tile:
        return Tile.from_lat_long(
            lat=self.top_left_lat, long=self.top_left_long, zoom_level=zoom_level
        )

    def bottom_right_tile(self, zoom_level: int) -> Tile:
        return Tile.from_lat_long(
            lat=self.bottom_right_lat, long=self.bottom_right_long, zoom_level=zoom_level
        )

    def tile_coordinates(self, zoom_level: int) -> Tuple[TileCoordinate, TileCoordinate]:
        return (
            TileCoordinate(
                x=self.top_left_tile(zoom_level).x,
                y=self.top_left_tile(zoom_level).y,
                zoom_level=zoom_level,
            ),
            TileCoordinate(
                x=self.bottom_right_tile(zoom_level).x,
                y=self.bottom_right_tile(zoom_level).y,
                zoom_level=zoom_level,
            ),
        )

    def size_in_tiles(self, zoom_level: int) -> Tuple[int, int]:
        """Return the size of the flight zone in tiles at the given zoom level."""
        top_left = self.top_left_tile(zoom_level)
        bottom_right = self.bottom_right_tile(zoom_level)
        x0, x1 = sorted([top_left.x, bottom_right.x])
        y0, y1 = sorted([top_left.y, bottom_right.y])
        return (x1 - x0 + 1, y1 - y0 + 1)

    def size_in_pixels(
        self, zoom_level: int, tile_size: int = TILE_SIZE
    ) -> Tuple[int, int]:
        """Return the size of the flight zone in pixels at the given zoom level."""
        size = self.size_in_tiles(zoom_level)
        return (size[0] * tile_size, size[1] * tile_size)

    def tiles_with_indices(
        self, zoom_level: int
    ) -> Tuple[List[Tile], List[Tuple[int, int]]]:
        """Return a list of tiles that are within the flight zone at the given zoom level.

        parameters
        ----------
        zoom_level : int
            zoom level

        Returns
        -------
        List[Tile]
            tiles within the flight zone
        List[Tuple[int, int]]
            x, y coordinates of the tiles relative to the top left tile
        """
        top_left = self.top_left_tile(zoom_level)
        bottom_right = self.bottom_right_tile(zoom_level)
        x0, x1 = sorted([top_left.x, bottom_right.x])
        y0, y1 = sorted([top_left.y, bottom_right.y])
        tiles = []
        local_indices = []
        for i, x in enumerate(range(x0, x1 + 1)):
            for j, y in enumerate(range(y0, y1 + 1)):
                tiles.append(Tile(x=x, y=y, zoom_level=zoom_level))
                local_indices.append((i, j))
        return tiles, local_indices

    def tiles(self, zoom_level: int) -> List[Tile]:
        """Return a list of tiles that are within the flight zone at the given zoom level.

        Parameters
        ----------
        zoom_level : int
            zoom level

        Returns
        -------
        List[Tile]
            tiles within the flight zone
        """
        top_left = self.top_left_tile(zoom_level)
        bottom_right = self.bottom_right_tile(zoom_level)
        x0, x1 = sorted([top_left.x, bottom_right.x])
        y0, y1 = sorted([top_left.y, bottom_right.y])
        tiles = []
        for x, y in product(range(x0, x1 + 1), range(y0, y1 + 1)):
            tiles.append(Tile(x=x, y=y, zoom_level=zoom_level))
        return tiles

    def yield_tiles(self, zoom_level: int) -> Generator[Tile]:
        """Get a generator of tiles that are within the flight zone at the given zoom level.

        Parameters
        ----------
        zoom_level : int
            zoom level

        Returns
        -------
        Generator[Tile]
            generator of tiles within the flight zone
        """
        top_left = self.top_left_tile(zoom_level)
        bottom_right = self.bottom_right_tile(zoom_level)
        x0, x1 = sorted([top_left.x, bottom_right.x])
        y0, y1 = sorted([top_left.y, bottom_right.y])
        for x, y in product(range(x0, x1 + 1), range(y0, y1 + 1)):
            yield Tile(x=x, y=y, zoom_level=zoom_level)

    def yield_tiles_with_indices(
        self, zoom_level: int
    ) -> Generator[Tuple[Tile, Tuple[int, int]]]:
        """Get a generator of tiles and their indices that are within the flight zone.

        Parameters
        ----------
        zoom_level : int
            zoom level

        Returns
        -------
        Generator[Tuple[Tile, Tuple[int, int]]]
            generator of tiles and their indices within the flight zone
        """
        top_left = self.top_left_tile(zoom_level)
        bottom_right = self.bottom_right_tile(zoom_level)
        x0, x1 = sorted([top_left.x, bottom_right.x])
        y0, y1 = sorted([top_left.y, bottom_right.y])
        for i, x in enumerate(range(x0, x1 + 1)):
            for j, y in enumerate(range(y0, y1 + 1)):
                yield Tile(x=x, y=y, zoom_level=zoom_level), (i, j)


@dataclass
class CameraModel:
    """A CameraModel is a dataclass that represents the intrinsic parameters of a camera.

    Parameters
    ----------
    focal_length : float
        focal length of the camera in millimeters
    resolution_width : int
        width of the image in pixels
    resolution_height : int
        height of the image in pixels
    hfov_deg : float
        horizontal field of view in degrees
    principal_point_x : float
        x coordinate of the principal point
    principal_point_y : float
        y coordinate of the principal point

    Properties
    ----------
    hfov_rad : float
        horizontal field of view in radians
    resolution : Tuple
        resolution of the image
    aspect_ratio : float
        aspect ratio of the image
    focal_length_px : float
        focal length in pixels
    """

    focal_length: float
    resolution_width: int
    resolution_height: int
    hfov_deg: float
    hfov_rad: float = field(init=False)
    resolution: Tuple = field(init=False)
    aspect_ratio: float = field(init=False)
    focal_length_px: float = field(init=False)
    principal_point_x: float = None
    principal_point_y: float = None

    def __post_init__(self) -> None:
        self.hfov_rad = self.hfov_deg * (math.pi / 180)
        self.resolution = (self.resolution_width, self.resolution_height)
        self.aspect_ratio = self.resolution_width / self.resolution_height
        self.focal_length_px = self.resolution_width / (2 * math.tan(self.hfov_rad / 2))
        if self.principal_point_x is None:
            self.principal_point_x = self.resolution_width / 2
        if self.principal_point_y is None:
            self.principal_point_y = self.resolution_height / 2

    @staticmethod
    def from_yaml(yaml_file: str) -> CameraModel:
        data = YamlParser.load_yaml(yaml_file)
        return CameraModel(
            focal_length=data["focal_length"],
            resolution_width=data["resolution_width"],
            resolution_height=data["resolution_height"],
            hfov_deg=data["hfov_deg"],
        )

    @staticmethod
    def from_json(json_file: str) -> CameraModel:
        data = JsonParser.load_json(json_file)
        return CameraModel(
            focal_length=data["focal_length"],
            resolution_width=data["resolution_width"],
            resolution_height=data["resolution_height"],
            hfov_deg=data["hfov_deg"],
        )


@dataclass
class DroneImage:
    """A DroneImage is a dataclass that represents an image captured by a drone.

    Parameters
    ----------
    image_path : Path
        path to the image file
    geo_point : GeoPoint
        geo point of the image (latitude, longitude, altitude), meant to be the ground-
        truth position of the drone when the image was captured
    camera_orientation : Orientation
        orientation of the camera (pitch, roll, yaw)
    drone_orientation : Orientation
        orientation of the drone (pitch, roll, yaw)
    camera_model : CameraModel
        camera model of the drone (focal length, resolution, hfov)
    image : np.ndarray
        image as a numpy array
    key_points : ImageKeyPoints
        key points of the image

    Properties
    ----------
    name : str
        name of the image (file name without extension)
    """

    image_path: Path
    geo_point: GeoPoint = None
    camera_orientation: Orientation = None
    drone_orientation: Orientation = None
    camera_model: CameraModel = None
    image: np.ndarray = None
    key_points: ImageKeyPoints = None
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.image_path.stem


@dataclass
class GeoSatelliteImage:
    """A GeoSatelliteImage is a dataclass that represents an image captured by a satellite.

    Parameters
    ----------
    image_path : Path
        path to the image file
    top_left : GpsCoordinate
        top left corner of the image (latitude, longitude)
    bottom_right : GpsCoordinate
        bottom right corner of the image (latitude, longitude)
    image : np.ndarray
        image as a numpy array
    key_points : ImageKeyPoints
        key points of the image

    Properties
    ----------
    name : str
        name of the image (file name without extension)
    """

    image_path: Path
    top_left: GpsCoordinate = None
    bottom_right: GpsCoordinate = None
    image: np.ndarray = None
    index: int = None
    key_points: ImageKeyPoints = None
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.image_path.stem


@dataclass
class TileImage:
    """A TileImage is an abstraction of a TMS tile with an image and key points.

    Parameters
    ----------
    tile : Tile
        tile of the image (x, y, zoom level)
    image_path : Union[str, Path]
        path to the image file
    image : np.ndarray
        image as a numpy array
    key_points : ImageKeyPoints
        key points of the image

    Properties
    ----------
    name : str
        name of the image (file name without extension)
    """

    tile: Tile
    image_path: Union[str, Path]
    index: Tuple[int, int] = None
    image: np.ndarray = None
    key_points: ImageKeyPoints = None
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.image_path.stem


@dataclass
class Mosaic:
    """A Mosaic is a dataclass that represents a mosaic of tiles.

    Parameters
    ----------
    zoom_level : int
        zoom level of the mosaic
    image : np.ndarray
        image as a numpy array that represents the full flight zone
    key_points : ImageKeyPoints
        key points of the mosaic
    flight_zone : FlightZone
        flight zone of the mosaic
    """

    zoom_level: int
    image: np.ndarray = None
    key_points: ImageKeyPoints = None
    flight_zone: FlightZone = None
