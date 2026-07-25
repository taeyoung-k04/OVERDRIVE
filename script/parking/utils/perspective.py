from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CALIBRATION_IMAGE_SIZE = (1280, 720)
MAP_OUTPUT_SIZE = (1462, 1949)
OUTPUT_CROP_TOP_RATIO = 0.10
OUTPUT_CROP_LEFT_RATIO = 0.10
OUTPUT_CROP_RIGHT_RATIO = 0.10
OUTPUT_CROP_RIGHT_EXTRA_PIXELS = 125

# Placement of Right_Front_/frame_000000s.jpg in map.svg.
SVG_FRAME_ORIGIN = np.array([471.5, 1234.22], dtype=np.float64)
SVG_FRAME_SIZE = np.array([327.0, 184.0], dtype=np.float64)
SVG_FRAME_ROTATION_DEGREES = -1.00286

# The first two red markers are already coincident in map.svg. For every
# other color, the first point is on the frame and the second is on the map.
SVG_FRAME_POINTS = np.array(
    [
        [523.0, 1353.0],  # red, left
        [758.0, 1353.0],  # red, right
        [719.0, 1325.0],  # orange
        [572.0, 1314.0],  # yellow
        [508.0, 1312.0],  # olive
        [592.0, 1298.0],  # green
        [547.0, 1297.0],  # light green
        [698.0, 1310.0],  # mint
        [686.0, 1301.0],  # cyan
        [678.0, 1295.0],  # blue
        [603.0, 1290.0],  # indigo
        [568.0, 1289.0],  # purple
    ],
    dtype=np.float64,
)

MAP_POINTS = np.array(
    [
        [523.0, 1353.0],  # red, left
        [758.0, 1353.0],  # red, right
        [760.0, 1228.0],  # orange
        [526.0, 1167.0],  # yellow
        [417.0, 1167.0],  # olive
        [526.0, 979.0],   # green
        [417.0, 979.0],   # light green
        [760.0, 1106.0],  # mint
        [760.0, 984.0],   # cyan
        [760.0, 862.0],   # blue
        [526.0, 790.0],   # indigo
        [417.0, 790.0],   # purple
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class NewPerspectiveConfig:
    transform: np.ndarray
    output_size: tuple[int, int]


def _svg_points_to_image_points(points: np.ndarray) -> np.ndarray:
    inverse_angle = np.deg2rad(-SVG_FRAME_ROTATION_DEGREES)
    inverse_rotation = np.array(
        [
            [np.cos(inverse_angle), -np.sin(inverse_angle)],
            [np.sin(inverse_angle), np.cos(inverse_angle)],
        ],
        dtype=np.float64,
    )
    local_points = (points - SVG_FRAME_ORIGIN) @ inverse_rotation.T
    image_scale = np.array(CALIBRATION_IMAGE_SIZE, dtype=np.float64) / SVG_FRAME_SIZE
    return local_points * image_scale


CALIBRATION_FRAME_POINTS = _svg_points_to_image_points(SVG_FRAME_POINTS)


def make_new_perspective_config(image_shape: tuple[int, ...]) -> NewPerspectiveConfig:
    height, width = image_shape[:2]
    calibration_width, calibration_height = CALIBRATION_IMAGE_SIZE
    frame_points = CALIBRATION_FRAME_POINTS.copy()
    frame_points[:, 0] *= width / calibration_width
    frame_points[:, 1] *= height / calibration_height

    transform, _ = cv2.findHomography(frame_points, MAP_POINTS, method=0)
    if transform is None:
        raise RuntimeError("Could not calculate the map-to-frame homography")

    return NewPerspectiveConfig(transform=transform, output_size=MAP_OUTPUT_SIZE)


def apply_new_perspective(
    image: np.ndarray,
    config: NewPerspectiveConfig,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    warped = cv2.warpPerspective(
        image,
        config.transform,
        config.output_size,
        flags=interpolation,
    )
    valid_source = np.full(image.shape[:2], 255, dtype=np.uint8)
    valid_output = cv2.warpPerspective(
        valid_source,
        config.transform,
        config.output_size,
        flags=cv2.INTER_NEAREST,
    )
    valid_rows = np.flatnonzero(np.any(valid_output != 0, axis=1))
    if valid_rows.size == 0:
        raise RuntimeError("Perspective result contains no valid image area")

    content_bottom = int(valid_rows[-1]) + 1
    top = int(round(content_bottom * OUTPUT_CROP_TOP_RATIO))
    left = int(round(warped.shape[1] * OUTPUT_CROP_LEFT_RATIO))
    right = warped.shape[1] - int(
        round(warped.shape[1] * OUTPUT_CROP_RIGHT_RATIO)
    ) - OUTPUT_CROP_RIGHT_EXTRA_PIXELS
    return warped[top:content_bottom, left:right]


def calibration_errors(config: NewPerspectiveConfig) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        CALIBRATION_FRAME_POINTS.astype(np.float32)[None, :, :],
        config.transform,
    )[0]
    return np.linalg.norm(projected - MAP_POINTS, axis=1)
