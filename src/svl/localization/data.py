import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from svl.localization.map_reader import TileSatelliteMapReader
from svl.tms.data_structures import FlightZone, TileImage


class GeoDataProcessor:
    """Class to process a flight zone and extract cells of satellite map images.

    A cell is a composite image of multiple satellite map tiles, forming a grid.
    The cell size is defined in terms of the number of tiles in the grid. The stride
    is the number of tiles to skip between consecutive cells. The extracted cells are
    saved as images in the output folder.
    This allows for the creation of a dataset of satellite map images to be used for
    localization tasks.

    Parameters
    ----------
    database_path : Path
        Path to the tile database.
    zoom_level : int
        Zoom level of the map tiles.
    flight_zone : FlightZone
        Flight zone definition.


    Examples
    --------
    >>> from svl.localization.data import GeoDataProcessor
    >>> from svl.tms.data_structures import FlightZone
    >>> from pathlib import Path
    >>> database_path = Path("path/to/database")
    >>> zoom_level = 18
    >>> flight_zone = FlightZone(
    ...     top_left_lat=37.422,
    ...     top_left_long=-122.084,
    ...     bottom_right_lat=37.421,
    ...     bottom_right_long=-122.083,
    ... )
    >>> processor = GeoDataProcessor(database_path, zoom_level, flight_zone)
    >>> cell_size = 3
    >>> cell_stride = 1
    >>> output_folder = Path("path/to/output")
    >>> processor.extract_and_save_cells(cell_size, cell_stride, output_folder)
    """

    def __init__(
        self, database_path: Path, zoom_level: int, flight_zone: FlightZone
    ) -> None:
        """Initialize GeoDataProcessor.

        Parameters
        ----------
        database_path : Path
            Path to the tile database.
        zoom_level : int
            Zoom level of the map tiles.
        flight_zone : FlightZone
            Flight zone definition.
        """
        self.database_path = database_path
        self.zoom_level = zoom_level
        self.flight_zone = flight_zone

        # Initialize tile reader
        self.tile_reader = TileSatelliteMapReader(
            db_path=database_path,
            logger=logging.getLogger(f"{__name__}.TileSatelliteMapReader"),
            zoom_level=zoom_level,
            resize_size=None,
            flight_zone=flight_zone,
        )
        self.tile_reader.initialize_db()
        self.tile_reader.load_images()

    def extract_and_save_cells(
        self, cell_size: int, cell_stride: int, output_folder: Path
    ) -> None:
        """Extract sections of the flight zone and save them as images.

        Parameters
        ----------
        cell_size : int
            Size of the cell in tiles.
        cell_stride : int
            Stride of the cell in tiles.
        output_folder : Path
            Path to the output folder.
        """
        cell_images, metadata = self.generate_cell_images(
            grid_size=cell_size, stride=cell_stride
        )

        if not output_folder.exists():
            output_folder.mkdir(parents=True)

        images_path = output_folder / "images"
        if not images_path.exists():
            images_path.mkdir(parents=True)

        for idx, cell_image in enumerate(cell_images):
            cv2.imwrite(str(images_path / metadata[idx]["Filename"]), cell_image)

        df = pd.DataFrame(metadata)
        df.to_csv(output_folder / "metadata.csv", index=False)

    def partition_flight_zone(
        self,
        zone_size: Tuple[int, int],
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> List[Dict[str, int]]:
        """Partition the flight zone into cells based on the window size and stride.

        Parameters
        ----------
        zone_size : Tuple[int, int]
            Size of the flight zone in tiles.
        window_size : Tuple[int, int]
            Size of the window for each cell in tiles.
        stride : Tuple[int, int]
            Stride of the window in tiles.

        Returns
        -------
        List[Dict[str, int]]
            List of cells with start and end coordinates.
        """
        zone_width, zone_height = zone_size
        window_width, window_height = window_size
        stride_width, stride_height = stride

        cells = []

        for y in range(0, zone_height, stride_height):
            for x in range(0, zone_width, stride_width):
                # Define cell boundaries
                cell_start_x = x
                cell_start_y = y
                cell_end_x = min(x + window_width, zone_width)
                cell_end_y = min(y + window_height, zone_height)

                # Create cell dictionary
                cell = {
                    "start_x": cell_start_x,
                    "start_y": cell_start_y,
                    "end_x": cell_end_x,
                    "end_y": cell_end_y,
                }
                cells.append(cell)

        return cells

    def get_tile_size(self) -> int:
        """Get the size of the tiles in the database."""
        return self.tile_reader[0].tile.tile_size

    def generate_cell_images(
        self, grid_size: int, stride: int = None
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        """Generate composite images for each cell.

        Parameters
        ----------
        grid_size : int
            Size of the grid for each cell.
        stride : int, optional
            Stride between consecutive cells.

        Returns
        -------
        Tuple[List[np.ndarray], List[Dict[str, float]]]
            List of composite cell images and their metadata.
        """
        width_in_tiles, height_in_tiles = self.flight_zone.size_in_tiles(
            zoom_level=self.zoom_level
        )

        if stride is None:
            stride = grid_size

        cells = self.partition_flight_zone(
            zone_size=(width_in_tiles, height_in_tiles),
            window_size=(grid_size, grid_size),
            stride=(stride, stride),
        )

        cell_images = []
        all_metadata = []

        for cell in cells:
            super_image = np.zeros(
                (
                    grid_size * self.get_tile_size(),
                    grid_size * self.get_tile_size(),
                ),
                dtype=np.uint8,
            )

            for i in range(cell["start_x"], cell["end_x"]):
                for j in range(cell["start_y"], cell["end_y"]):
                    local_indices = (i, j)
                    image_tile: TileImage = self.tile_reader[local_indices]

                    tile_size = image_tile.tile.tile_size
                    x_offset = (i - cell["start_x"]) * tile_size
                    y_offset = (j - cell["start_y"]) * tile_size
                    super_image[
                        y_offset : y_offset + tile_size, x_offset : x_offset + tile_size
                    ] = image_tile.image

            cell_images.append(super_image)

            # Metadata for the cell
            top_left_tile = self.tile_reader[(cell["start_x"], cell["start_y"])].tile
            bottom_right_tile = self.tile_reader[
                (cell["end_x"] - 1, cell["end_y"] - 1)
            ].tile

            metadata = {
                "Filename": f"grid_{cell['start_x']}_{cell['end_x']}_{cell['start_y']}_{cell['end_y']}_{self.zoom_level}.png",  # noqa: E501
                "Top_left_lat": top_left_tile.lat,
                "Top_left_lon": top_left_tile.long,
                "Bottom_right_lat": bottom_right_tile.lat,
                "Bottom_right_long": bottom_right_tile.long,
            }
            all_metadata.append(metadata)

        return cell_images, all_metadata
