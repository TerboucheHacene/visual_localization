from dataclasses import dataclass


@dataclass
class GpsCoordinate:
    lat: float
    long: float


@dataclass
class TileCoordinate:
    x: int
    y: int
    zoom_level: int


@dataclass
class WorldCoordinate:
    x: float
    y: float


@dataclass
class GeoPoint:
    latitude: float
    longitude: float
    altitude: float


@dataclass
class Orientation:
    pitch: float
    roll: float
    yaw: float
