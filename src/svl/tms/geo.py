import math
from typing import Tuple

from svl.tms.schemas import GpsCoordinate, TileCoordinate
from svl.utils.checks import check_gps_coordinate, check_tile_coordinate
from svl.utils.constants import EARTH_RADIUS, EQUTORIAL_CIRCUMFERENCE_METERS, TILE_SIZE


def get_tile_xy_from_lat_long(
    lat: float, long: float, zoom_level: int
) -> Tuple[int, int]:
    """
    Convert lat, long coordinates to x, y tile coordinates.

    Parameters
    ----------
    lat : float
        latitude
    long : float
        longitude
    zoom_level : int
        zoom level

    Returns
    -------
    Tuple[int, int]
        x, y tile coordinates
    """

    x = int((long + 180) / 360 * 2**zoom_level)
    y = int(
        (
            1
            - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
            / math.pi
        )
        / 2
        * 2**zoom_level
    )
    check_tile_coordinate(TileCoordinate(x=x, y=y, zoom_level=zoom_level))
    return x, y


def get_lat_long_from_tile_xy(x: int, y: int, zoom_level: int) -> Tuple[float, float]:
    """
    Convert x, y tile coordinates to lat, long coordinates.

    Parameters
    ----------
    x : int
        x tile coordinate
    y : int
        y tile coordinate
    zoom_level : int
        zoom level

    Returns
    -------
    Tuple[float, float]
        lat, long coordinates
    """

    n = 2.0**zoom_level
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    check_gps_coordinate(GpsCoordinate(lat=lat_deg, long=lon_deg))
    return lat_deg, lon_deg


def get_xy_pixel_from_xy_tile(
    x: int, y: int, origin_x: int, origin_y: int, tile_size: int
) -> Tuple[int, int]:
    """
    Convert x, y tile coordinates to x, y pixel coordinates. The origin is the top left corner of the tile.


    Parameters
    ----------
    x : int
        x tile coordinate
    y : int
        y tile coordinate
    origin_x : int
        x tile origin
    origin_y : int
        y tile origin
    tile_size : int
        size of the tile

    Returns
    -------
    Tuple[int, int]
        x, y pixel coordinates
    """

    x_pixel = (x - origin_x) * tile_size
    y_pixel = (y - origin_y) * tile_size
    return x_pixel, y_pixel


def resolution_at_zoom_level(
    lat: float, zoom_level: int, tile_size: int = TILE_SIZE
) -> float:
    """
    Calculate the spatial resolution at a given latitude and zoom level.


    Parameters
    ----------
    lat : float
        latitude
    zoom_level : int
        zoom level
    tile_size : int
        size of the tile

    Returns
    -------
    float
        spatial resolution
    """

    equator_circumference = EQUTORIAL_CIRCUMFERENCE_METERS
    resolution_at_zoom_0 = equator_circumference / tile_size
    return resolution_at_zoom_0 * math.cos(math.radians(lat)) / 2**zoom_level


def haversine_distance(gps1: GpsCoordinate, gps2: GpsCoordinate) -> float:
    """
    Calculate the haversine distance between two GPS coordinates.

    Parameters
    ----------
    gps1 : GpsCoordinate
        first GPS coordinate
    gps2 : GpsCoordinate
        second GPS coordinate

    Returns
    -------
    float
        haversine distance in kilometers
    """

    lat1 = math.radians(gps1.lat)
    long1 = math.radians(gps1.long)
    lat2 = math.radians(gps2.lat)
    long2 = math.radians(gps2.long)

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlong / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = c * EARTH_RADIUS
    return distance
