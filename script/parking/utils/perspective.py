"""Convert a camera view to the parking-track layout (bird's-eye view)"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Coordinate systems used when the matching points were marked.
VIEW_SIZE = (640, 360)       # (width, height)
LAYOUT_SIZE = (1462, 1949)   # (width, height)

VIEW_POINTS = np.array(
    [
        (69, 349),
        (136, 219),
        (219, 156),
        (105, 153),
        (252, 129),
        (173, 126),
        (272, 114),
        (210, 112),
        (286, 104),
        (233, 103),
        (553, 219),
        (486, 171),
        (452, 146),
        (620, 145),
        (429, 130),
        (565, 129),
        (415, 119),
    ],
    dtype=np.float32,
)

LAYOUT_POINTS = np.array(
    [
        (557, 1478),
        (526, 1356),
        (526, 1168),
        (417, 1168),
        (526, 979),
        (417, 979),
        (526, 790),
        (417, 790),
        (526, 602),
        (417, 602),
        (762, 1349),
        (762, 1227),
        (762, 1105),
        (943, 1105),
        (762, 984),
        (943, 984),
        (762, 862),
    ],
    dtype=np.float32,
)

CROP_TOP = 463
CROP_BOTTOM = 468
CROP_RIGHT = 248

@dataclass(frozen=True)
class NewPerspectiveConfig:
    transform: np.ndarray
    output_size: tuple[int, int] = LAYOUT_SIZE
    source_points: np.ndarray | None = None
    destination_points: np.ndarray | None = None


def _validate_image_shape(image_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must contain height and width: {image_shape}")

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive: {image_shape}")
    return int(height), int(width)


def make_new_perspective_config(
    image_shape: tuple[int, ...],
) -> NewPerspectiveConfig:
    height, width = _validate_image_shape(image_shape)
    view_width, view_height = VIEW_SIZE

    source_points = VIEW_POINTS.copy()
    source_points[:, 0] *= width / view_width
    source_points[:, 1] *= height / view_height

    transform, _ = cv2.findHomography(
        source_points,
        LAYOUT_POINTS,
        method=0,
    )
    if transform is None or not np.all(np.isfinite(transform)):
        raise RuntimeError("Could not calculate the view-to-layout homography")

    return NewPerspectiveConfig(
        transform=transform,
        output_size=LAYOUT_SIZE,
        source_points=source_points,
        destination_points=LAYOUT_POINTS.copy(),
    )


def apply_new_perspective(
    image: np.ndarray,
    config: NewPerspectiveConfig | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int | tuple[int, ...] = 0,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if config is None:
        config = make_new_perspective_config(image.shape)

    warped = cv2.warpPerspective(
        image,
        config.transform,
        config.output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    if (
        warped.shape[0] <= CROP_TOP + CROP_BOTTOM
        or warped.shape[1] <= CROP_RIGHT
    ):
        raise ValueError(
            f"Perspective output {warped.shape[1]}x{warped.shape[0]} is too "
            f"small for top={CROP_TOP}, bottom={CROP_BOTTOM}, "
            f"right={CROP_RIGHT} cropping"
        )
    return warped[CROP_TOP:-CROP_BOTTOM, :-CROP_RIGHT]


def transform_points(
    points: np.ndarray,
    config: NewPerspectiveConfig,
) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.shape == (2,):
        points_array = points_array[None, :]
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points must have shape (2,) or (N, 2)")

    return cv2.perspectiveTransform(
        points_array[None, :, :],
        config.transform,
    )[0]


def calibration_errors(config: NewPerspectiveConfig) -> np.ndarray:
    source_points = config.source_points
    destination_points = config.destination_points
    if source_points is None or destination_points is None:
        raise ValueError("config does not contain calibration points")

    projected = transform_points(source_points, config)
    return np.linalg.norm(projected - destination_points, axis=1)
