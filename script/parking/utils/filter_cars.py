"""Filter parking car detections using fitted parking boundaries."""

from __future__ import annotations

import cv2
import numpy as np

from infer_sem_class import CLASS_TO_ID
from utils.lane_detect import (
    Line,
    ParkingLineDetection,
    ReferenceLineDetector,
)


def filter_cars_in_parking_region(
    class_map: np.ndarray,
    parking_dot_line: Line,
    parking_lines: ParkingLineDetection,
    *,
    car_class_id: int = CLASS_TO_ID["car"],
) -> np.ndarray:
    """Remove cars that do not overlap the bottom-right parking region.

    The region is right of ``parking_dot_line`` and below the topmost detected
    parking line.  The top-right and bottom-right image corners select those
    respective half-planes without depending on line direction.
    """
    ReferenceLineDetector._validate_class_map(class_map)
    if (
        not parking_dot_line.valid
        or parking_dot_line.point is None
        or parking_dot_line.direction is None
        or not parking_lines.lines
    ):
        return class_map

    top_line = min(
        parking_lines.lines,
        key=lambda line: (
            sum(point[1] for point in line.segment) / 2.0
            if line.segment is not None
            else float(line.point[1])
        ),
    )
    if top_line.point is None or top_line.direction is None:
        return class_map

    height, width = class_map.shape
    ys, xs = np.indices((height, width), dtype=np.float64)
    top_right = np.asarray([width - 1.0, 0.0], dtype=np.float64)
    bottom_right = np.asarray(
        [width - 1.0, height - 1.0],
        dtype=np.float64,
    )

    def same_side_mask(
        line: Line,
        reference_point: np.ndarray,
    ) -> np.ndarray:
        origin = np.asarray(line.point, dtype=np.float64)
        direction = np.asarray(line.direction, dtype=np.float64)
        reference_delta = reference_point - origin
        reference_side = float(
            direction[0] * reference_delta[1]
            - direction[1] * reference_delta[0]
        )
        pixel_side = (
            direction[0] * (ys - origin[1])
            - direction[1] * (xs - origin[0])
        )
        if reference_side >= 0.0:
            return pixel_side >= 0.0
        return pixel_side <= 0.0

    parking_region = (
        same_side_mask(parking_dot_line, top_right)
        & same_side_mask(top_line, bottom_right)
    )
    car_mask = (class_map == int(car_class_id)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        car_mask,
        connectivity=8,
    )
    filtered = class_map.copy()

    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        component = labels == label
        if not np.any(component & parking_region):
            filtered[component] = CLASS_TO_ID["background"]

    return filtered


def remove_car_detections(
    class_map: np.ndarray,
    *,
    car_class_id: int = CLASS_TO_ID["car"],
) -> np.ndarray:
    """Return a class map with every car pixel changed to background."""
    ReferenceLineDetector._validate_class_map(class_map)
    filtered = class_map.copy()
    filtered[filtered == int(car_class_id)] = CLASS_TO_ID["background"]
    return filtered


__all__ = [
    "filter_cars_in_parking_region",
    "remove_car_detections",
]
