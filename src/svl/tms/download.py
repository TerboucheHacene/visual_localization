import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rasterio
import requests
from rasterio.transform import Affine
from tqdm import tqdm

from svl.tms.data_structures import FlightZone, Tile


class TileDownloader:
    """Class for downloading tiles from a TMS server.

    Parameters
    ----------
    url : str
        URL for downloading the tiles. The URL should contain the following placeholders:
        - {x}: x coordinate of the tile
        - {y}: y coordinate of the tile
        - {z}: zoom level of the tile
        - {api_key}: API key for accessing the server
    channels : int
        number of channels in the image (3 for RGB, 1 for grayscale)
    api_key : str
        API key for accessing the server
    headers : Dict
        headers to include in the request, default is None
    img_format : str
        image format (png, jpg, etc.), default is "png"
    """

    def __init__(
        self,
        url: str,
        channels: int = 3,
        api_key: str = None,
        headers: Dict = None,
        img_format: str = "png",
    ):
        self.url = url
        self.api_key = api_key
        self.headers = headers if headers else {}
        self.channels = channels
        self.img_format = img_format

    def download_tile(self, tile: Tile, output_path: str) -> None:
        """Download the tile from the server and save it to the output path.

        Parameters
        ----------
        tile : Tile
            tile object to be downloaded
        output_path : str
            path to save the downloaded tile

        Raises
        ------
        ValueError
            if the request fails
        """
        url = self.format_url(tile)
        file_path = self.format_file_path(tile, output_path)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
        else:
            raise ValueError(f"Failed to download tile: {response.status_code}")

    def download_tile_as_image(self, tile: Tile) -> np.ndarray:
        """Download the tile from the server and return the image as a numpy array.

        Parameters
        ----------
        tile : Tile
            tile to download

        Returns
        -------
        np.ndarray
            an array representing the image

        Raises
        ------
        ValueError
            if the request fails
        """
        url = self.format_url(tile)
        response = requests.get(url, stream=True, headers=self.headers)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(
                image, cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE
            )
            return image
        else:
            raise ValueError(f"Failed to download tile: {response.status_code}")

    def _format_file_path(self, tile: Tile, output_path: str) -> str:
        """Format the file path for saving the tile.

        Parameters
        ----------
        tile : Tile
            Tile to download
        output_path : str
            Path to save the downloaded tile

        Returns
        -------
        str
            File path
        """
        return os.path.join(output_path, f"{tile.file_name}.{self.img_format}")

    def load_image(self, tile: Tile, output_path: str) -> np.ndarray:
        """Load a tile image from the specified output path.

        Parameters
        ----------
        tile : Tile
            Tile to load
        output_path : str
            Path where the tile is saved

        Returns
        -------
        np.ndarray
            An array representing the loaded image
        """
        file_path = self._format_file_path(tile, output_path)
        image = cv2.imread(
            file_path, cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE
        )
        return image

    def download_tile_and_save_as_image(
        self, tile: Tile, output_path: str
    ) -> np.ndarray:
        """Download a tile from the server and save it to the output path, returning the image.

        Parameters
        ----------
        tile : Tile
            Tile object to be downloaded
        output_path : str
            Path to save the downloaded tile

        Returns
        -------
        np.ndarray
            An array representing the downloaded image

        Raises
        ------
        ValueError
            If the request fails
        """
        self.download_tile(tile, output_path)
        return self.load_image(tile, output_path)

    def format_url(self, tile: Tile) -> str:
        """Format the URL for downloading the tile.

        Parameters
        ----------
        tile : Tile
            tile to download

        Returns
        -------
        str
            formatted URL
        """
        return self.url.format(
            x=tile.x, y=tile.y, z=tile.zoom_level, api_key=self.api_key
        )

    def format_file_path(self, tile: Tile, output_path: str) -> str:
        """
        Format the file path for saving the tile.

        Parameters
        ----------
        tile : Tile
            tile to download

        Returns
        -------
        str
            file path
        """
        return os.path.join(output_path, f"{tile.file_name}.{self.img_format}")

    def download_tiles(self, tiles: List[Tile], output_path: str) -> None:
        for tile in tqdm(tiles, desc="Downloading", total=len(tiles)):
            self.download_tile(tile, output_path)


class FlightZoneDownloader:
    """Class for downloading a flight zone from a TMS server.

    Parameters
    ----------
    tile_downloader : TileDownloader
        tile downloader object able to download tiles
    flight_zone : FlightZone
        flight zone to download
    """

    AVAILABE_FORMATS = ["tiff", "tif", "png", "jpg", "jpeg"]

    def __init__(self, tile_downloader: TileDownloader, flight_zone: FlightZone):
        self.tile_downloader = tile_downloader
        self.flight_zone = flight_zone

    def download_flight_zone(self, zoom_level: int, output_path: str) -> None:
        """Download the tiles for the flight zone.

        Parameters
        ----------
        zoom_level : int
            zoom level of the tiles
        output_path : str
            path to save the downloaded tiles
        """
        tiles = self.flight_zone.tiles(zoom_level)
        self.tile_downloader.download_tiles(tiles, output_path)

    def download_tiles_and_save_as_mosaic(
        self, zoom_level: int, output_path: str, mosaic_format: str = "png"
    ) -> None:
        """Download the tiles for the flight zone and save them as a mosaic.

        This will download the tiles as a set of images and then combine them into a
        single mosaic image. The output path will have the following structure:
        |- output_path
            |- tiles
                |- x_y_z.png
                |- ...
                |- x_y_z.png
            |- mosaic.mosaic_format


        Parameters
        ----------
        zoom_level : int
            zoom level of the tiles
        output_path : str
            path to save the mosaic
        mosaic_format : str
            format of the mosaic (png, jpg, tiff, etc.), default is "png"
        """
        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        tiles_path: Path = output_path / "tiles"
        tiles_path.mkdir(parents=True, exist_ok=True)

        tiles, indices = self.flight_zone.tiles_with_indices(zoom_level)
        tile_size = tiles[0].tile_size
        # Calculate the size of the mosaic
        width_in_pixels, height_in_pixels = self.flight_zone.size_in_pixels(
            zoom_level, tile_size
        )
        mosaic = np.zeros((height_in_pixels, width_in_pixels, 3), dtype=np.uint8)

        for tile, index in tqdm(
            zip(tiles, indices), total=len(tiles), desc="Downloading"
        ):
            image = self.tile_downloader.download_tile_and_save_as_image(
                tile, tiles_path
            )
            i, j = index  # i is the x coordinate, j is the y coordinate
            mosaic[
                j * tile_size : (j + 1) * tile_size, i * tile_size : (i + 1) * tile_size
            ] = image

        self.save_mosaic(mosaic, str(output_path / f"mosaic.{mosaic_format}"))

    def download_flight_zone_as_mosaic(self, zoom_level: int) -> np.ndarray:
        """Download the tiles for the flight zone as a mosaic.

        Parameters
        ----------
        zoom_level : int
            zoom level of the tiles
        Returns
        -------
        np.ndarray
            mosaic image as a numpy array of the flight zone
        """

        tiles, indices = self.flight_zone.tiles_with_indices(zoom_level)
        tile_size = tiles[0].tile_size
        # Calculate the size of the mosaic
        width_in_pixels, height_in_pixels = self.flight_zone.size_in_pixels(
            zoom_level, tile_size
        )
        mosaic = np.zeros((height_in_pixels, width_in_pixels, 3), dtype=np.uint8)

        for tile, index in tqdm(
            zip(tiles, indices), total=len(tiles), desc="Downloading"
        ):
            image = self.tile_downloader.download_tile_as_image(tile)
            i, j = index  # i is the x coordinate, j is the y coordinate
            mosaic[
                j * tile_size : (j + 1) * tile_size, i * tile_size : (i + 1) * tile_size
            ] = image

        return mosaic

    def calculate_mosaic_size(self, zoom_level: int) -> Tuple[int, int]:
        """Calculate the size of the mosaic in tiles.

        When the flight zone is not aligned with the tiles, the mosaic size will be larger
        than the flight zone size.

        Parameters
        ----------
        zoom_level : int
            zoom level of the tiles

        Returns
        -------
        Tuple[int, int]
            mosaic size (width in tiles, height in tiles)
        """
        top_left = Tile(
            lat=self.flight_zone.top_left_lat,
            long=self.flight_zone.top_left_long,
            zoom_level=zoom_level,
        )
        bottom_right = Tile(
            lat=self.flight_zone.bottom_right_lat,
            long=self.flight_zone.bottom_right_long,
            zoom_level=zoom_level,
        )
        x0, x1 = sorted([top_left.x, bottom_right.x])
        y0, y1 = sorted([top_left.y, bottom_right.y])
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        return width, height

    def save_mosaic(self, mosaic: np.ndarray, output_path: str) -> None:
        """Save the mosaic to a file.

        Parameters
        ----------
        mosaic : np.ndarray
            mosaic np array to save
        output_path : str
            output path where the mosaic will be saved including the format
        """

        format = output_path.split(".")[-1]
        if format not in self.AVAILABE_FORMATS:
            raise ValueError(
                f"Invalid format: {format}, must be one of {self.AVAILABE_FORMATS}"
            )
        if format == "tiff" or format == "tif":
            self.save_mosaic_as_geo_tiff(mosaic, output_path)
        else:
            self.save_mosaic_as_image(mosaic, output_path)

    def save_mosaic_as_image(self, mosaic: np.ndarray, output_path: str) -> None:
        """Save the mosaic as an image.

        Parameters
        ----------
        mosaic : np.ndarray
            mosaic
        output_path : str
            output path
        """
        if Path(output_path).parent.exists() is False:
            raise ValueError(f"Output path {output_path} does not exist")
        cv2.imwrite(output_path, mosaic)

    def save_mosaic_as_geo_tiff(self, mosaic: np.ndarray, output_path: str) -> None:
        """Save the mosaic as a geotiff file.

        Parameters
        ----------
        mosaic : np.ndarray
            mosaic
        output_path : str
            output path
        """
        if Path(output_path).parent.exists() is False:
            raise ValueError(f"Output path {output_path} does not exist")

        mosaic = np.transpose(mosaic, (2, 0, 1))  # HWC -> CHW
        xres = (self.flight_zone.bottom_right_long - self.flight_zone.top_left_long) / (
            mosaic.shape[2]
        )
        yres = (self.flight_zone.top_left_lat - self.flight_zone.bottom_right_lat) / (
            mosaic.shape[1]
        )

        transform = Affine.translation(
            self.flight_zone.top_left_long, self.flight_zone.bottom_right_lat
        ) * Affine.scale(xres, yres)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            width=mosaic.shape[2],
            height=mosaic.shape[1],
            count=3,
            dtype=mosaic.dtype,
            crs="+proj=latlong",
            transform=transform,
        ) as dst:
            dst.write(mosaic, indexes=[1, 2, 3])
