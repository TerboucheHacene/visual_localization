class BaseError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class InvalidZoomLevelError(BaseError):
    def __init__(self, message: str = "The zoom level must be between 1 and 20."):
        super().__init__(message)


class InvalidLatitudeError(BaseError):
    def __init__(self, message: str = "The latitude must be between -90 and 90."):
        super().__init__(message)


class InvalidLongitudeError(BaseError):
    def __init__(self, message: str = "The longitude must be between -180 and 180."):
        super().__init__(message)


class InvalidCoordinateError(BaseError):
    def __init__(self, message: str = "The coordinate must be a valid GpsCoordinate."):
        super().__init__(message)


class InvalidLatitudeZoneError(BaseError):
    def __init__(
        self,
        message: str = "The top left latitude must be greater than the bottom right latitude.",
    ):
        super().__init__(message)


class InvalidLongitudeZoneError(BaseError):
    def __init__(
        self,
        message: str = "The top left longitude must be less than the bottom right longitude.",
    ):
        super().__init__(message)


class InvalidTileIndexError(BaseError):
    def __init__(
        self, message: str = "The tile index must be between 0 and 2^zoom_level."
    ):
        super().__init__(message)
