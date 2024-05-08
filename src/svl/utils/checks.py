from svl.tms.schemas import GpsCoordinate, TileCoordinate
from svl.utils.constants import (
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MAX_ZOOM_LEVEL,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    MIN_ZOOM_LEVEL,
)
from svl.utils.errors import (
    InvalidLatitudeError,
    InvalidLatitudeZoneError,
    InvalidLongitudeError,
    InvalidLongitudeZoneError,
    InvalidTileIndexError,
    InvalidZoomLevelError,
)


def check_zoom_level(zoom_level: int) -> None:
    """
    Check if the zoom level is valid.

    Parameters
    ----------
    zoom_level : int
        zoom level

    Raises
    ------
    InvalidZoomLevelError
        if the zoom level is not valid
    """
    if zoom_level < MIN_ZOOM_LEVEL or zoom_level > MAX_ZOOM_LEVEL:
        raise InvalidZoomLevelError()


def check_tile_coordinate(tile: TileCoordinate) -> None:
    """
    Check if the tile coordinate is valid.

    Parameters
    ----------
    tile : TileCoordinate
        tile coordinate

    Raises
    ------
    InvalidZoomLevelError
        if the zoom level is not valid
    InvalidTileIndexError
        if the tile index is not valid
    """
    check_zoom_level(tile.zoom_level)
    MAX_XY = 2**tile.zoom_level
    if tile.x < 0 or tile.x >= MAX_XY or tile.y < 0 or tile.y >= MAX_XY:
        raise InvalidTileIndexError()


def check_latitude(latitude: float) -> None:
    """
    Check if the latitude is valid.

    Parameters
    ----------
    latitude : float
        latitude

    Raises
    ------
    InvalidLatitudeError
        if the latitude is not valid
    """
    if latitude < MIN_LATITUDE or latitude > MAX_LATITUDE:
        raise InvalidLatitudeError()


def check_longitude(longitude: float) -> None:
    """
    Check if the longitude is valid.

    Parameters
    ----------
    longitude : float
        longitude

    Raises
    ------
    InvalidLongitudeError
        if the longitude is not valid
    """
    if longitude < MIN_LONGITUDE or longitude > MAX_LONGITUDE:
        raise InvalidLongitudeError()


def check_gps_coordinate(coordinate: GpsCoordinate) -> None:
    """
    Check if the GPS coordinate is valid.

    Parameters
    ----------
    coordinate : GpsCoordinate
        GPS coordinate

    Raises
    ------
    InvalidLatitudeError
        if the latitude is not valid
    InvalidLongitudeError
        if the longitude is not valid
    """
    check_latitude(coordinate.lat)
    check_longitude(coordinate.long)


def check_gps_zone(top_left: GpsCoordinate, bottom_right: GpsCoordinate) -> None:
    """
    Check if the GPS zone is valid.

    Parameters
    ----------
    top_left : GpsCoordinate
        top left GPS coordinate
    bottom_right : GpsCoordinate
        bottom right GPS coordinate

    Raises
    ------
    InvalidLatitudeError
        if the latitude is not valid
    InvalidLongitudeError
        if the longitude is not valid
    InvalidLatitudeZoneError
        if the latitude zone is not valid
    InvalidLongitudeZoneError
        if the longitude zone is not valid
    """
    check_gps_coordinate(top_left)
    check_gps_coordinate(bottom_right)

    if top_left.lat < bottom_right.lat:
        raise InvalidLatitudeZoneError()
    if top_left.long > bottom_right.long:
        raise InvalidLongitudeZoneError()
