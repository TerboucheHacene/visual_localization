from typing import List, Tuple

import cv2
import numpy as np

from superglue_lib.models.utils import process_resize
from svl.tms.data_structures import CameraModel, DroneImage


def get_intrinsics(camera_model: CameraModel, scale: float = 1.0) -> np.ndarray:
    """
    Get the intrinsics matrix of a camera model.

    Parameters
    ----------
    camera_model : CameraModel
        the camera model
    scale : float
        the scale factor to apply to the focal length, default is 1.0

    Returns
    -------
    np.ndarray
        the intrinsics matrix
    """
    intrinsics = np.array(
        [
            [camera_model.focal_length_px / scale, 0, camera_model.principal_point_x],
            [0, camera_model.focal_length_px / scale, camera_model.principal_point_y],
            [0, 0, 1],
        ]
    )
    return intrinsics


def rotation_matrix_from_angles(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute the rotation matrix from the roll, pitch, and yaw angles.

    Parameters
    ----------
    roll : float
        the roll angle
    pitch : float
        the pitch angle
    yaw : float
        the yaw angle

    Returns
    -------
    np.ndarray
        the rotation matrix
    """
    from scipy.spatial.transform import Rotation

    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    return r


class QueryProcessor:
    """Class to process the query image.

    Parameters
    ----------
    processings : List[str], optional
        the list of processings to apply to the query image, by default None
    size : Tuple[int, int], optional
        the size of the query image, by default None
    camera_model : CameraModel, optional
        the camera model of the query image, by default None
    satellite_resolution : float, optional
        the resolution of the satellite image, by default None
    """

    def __init__(
        self,
        processings: List[str] = None,
        size: Tuple[int, int] = None,
        camera_model: CameraModel = None,
        satellite_resolution: float = None,
    ) -> None:
        self.size = size
        self.camera_model = camera_model
        self.satellite_resolution = satellite_resolution
        self.fcts = {
            "resize": self.resize_image,
            "warp": self.warp_image,
        }
        self.processings = processings

    def __call__(self, query: DroneImage) -> DroneImage:
        """
        Process the query image.

        Parameters
        ----------
        query : DroneImage
            the query image

        Returns
        -------
        DroneImage
            the processed query image
        """
        if self.processings is None:
            return query
        for processing in self.processings:
            if processing in self.fcts:
                query = self.fcts[processing](query)
        return query

    def resize_image(self, query: DroneImage) -> DroneImage:
        """Resize the query image.

        If the size is provided, the query image is resized to the given size. If the
        camera model and the satellite resolution are provided, the query image is resized
        to the same meters per pixel as the satellite image.

        Parameters
        ----------
        query : DroneImage
            the query image

        Returns
        -------
        DroneImage
            the resized query image
        """

        image = query.image
        if self.size is not None:
            height, width = image.shape[:2]
            new_width, new_height = process_resize(width, height, self.size)
            resized_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            query.image = resized_image

        elif self.camera_model is not None and self.satellite_resolution is not None:
            new_size = self.compute_resize_shape(
                self.camera_model, query.geo_point.altitude, self.satellite_resolution
            )
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            query.image = resized_image

        return query

    def compute_resize_scale(
        self,
        camera_model: CameraModel,
        altitude: float,
        satellite_resolution: float,
    ) -> float:
        """Compute the resize scale to resize the query image to the same resolution
        as the satellite image.

        Parameters
        ----------
        camera_model : CameraModel
            the camera model of the query image
        altitude : float
            the altitude of the query image
        satellite_resolution : float
            the resolution of the satellite image

        Returns
        -------
        float
            the resize scale
        """

        # hvof_m = 2 * altitude * np.tan(hfov / 2)
        hvof_m = 2 * altitude * np.tan(camera_model.hfov_rad / 2)
        # resolution = hvof_m / width
        drone_resolution = hvof_m / camera_model.resolution_width
        # Compute the resize scale
        resize_scale = drone_resolution / satellite_resolution
        return resize_scale

    def compute_resize_shape(
        self,
        camera_model: CameraModel,
        altitude: float,
        satellite_resolution: float,
    ) -> Tuple[int, int]:
        """Resize the query image to the same resolution as the satellite image.

        Parameters
        ----------
        camera_model : CameraModel
            the camera model of the query image
        altitude : float
            the altitude of the query image
        satellite_resolution : float
            the resolution of the satellite image

        Returns
        -------
        Tuple[int, int]
            the new size of the query image
        """
        scale = self.compute_resize_scale(camera_model, altitude, satellite_resolution)
        resize_shape = (
            int(camera_model.resolution_height * scale),
            int(camera_model.resolution_width * scale),
        )
        return resize_shape

    def warp_image(self, query: DroneImage) -> DroneImage:
        """Warp the query image so that it is aligned with the satellite image.

        Parameters
        ----------
        query : DroneImage
            the drone image to be warped

        Returns
        -------
        DroneImage
            the warped query image
        """
        if query.camera_model is None and self.camera_model is None:
            raise ValueError(
                "Camera model is missing in the query and in the processor."
            )
        query_camera_model: CameraModel = query.camera_model
        if query_camera_model is None:
            query.camera_model = self.camera_model

        K = get_intrinsics(query.camera_model)
        R_gimbal = rotation_matrix_from_angles(
            # roll=query.camera_orientation.roll,
            # pitch=query.camera_orientation.pitch,
            roll=0,
            pitch=0,
            yaw=query.camera_orientation.yaw,
        )
        R_drone = rotation_matrix_from_angles(
            # roll=query.drone_orientation.roll,
            # pitch=query.drone_orientation.pitch,
            roll=0,
            pitch=0,
            yaw=query.drone_orientation.yaw + 15,
        )
        R_target = rotation_matrix_from_angles(0, 0, 0)

        # Compute the scale factor
        if self.satellite_resolution is not None:
            scale = self.compute_resize_scale(
                query.camera_model, query.geo_point.altitude, self.satellite_resolution
            )
        else:
            scale = 1.0
        K_scale = get_intrinsics(query.camera_model, scale)

        transformation_matrix = (
            K
            @ np.linalg.inv(R_gimbal)
            @ np.linalg.inv(R_target)
            @ R_drone
            @ R_gimbal
            @ np.linalg.inv(K_scale)
        )
        height, width = query.image.shape[:2]

        # define homogeneous coordinates of the image corners
        corners = np.array(
            [
                [0, 0, 1],
                [width - 1, 0, 1],
                [width - 1, height - 1, 1],
                [0, height - 1, 1],
            ]
        ).T

        # apply the transformation to the corners
        warped_corners = transformation_matrix @ corners

        # normalize the coordinates by the third component
        warped_corners /= warped_corners[2]

        # remove the third component
        warped_corners = warped_corners[:2].T

        # move corners to the origin
        warped_corners -= warped_corners.min(axis=0)

        # remove the third component from corners
        corners = corners[:2].T

        # cast to float32
        corners = corners.astype(np.float32)
        warped_corners = warped_corners.astype(np.float32)

        # compute the transformation matrix
        dst = cv2.getPerspectiveTransform(corners, warped_corners)

        # compute new image size
        new_size = warped_corners.max(axis=0).astype(np.int32)

        # warp the image
        warped_image = cv2.warpPerspective(query.image, dst, tuple(new_size))

        query.image = warped_image
        return query
