"""Parking phase transition control."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from script.parking.infer_sem_class import CLASS_TO_ID
    from script.parking.utils.lane_detect import (
        Line,
        ParkingLineDetection,
    )
except ModuleNotFoundError:
    from infer_sem_class import CLASS_TO_ID
    from utils.lane_detect import Line, ParkingLineDetection


@dataclass
class PhaseController:
    """Advance from phase 0 when one car has no parking line below it."""

    phase: int = 0

    def update(
        self,
        class_map: np.ndarray,
        parking_lines: ParkingLineDetection,
        parking_dot_line: Line,
        *,
        car_class_id: int = CLASS_TO_ID["car"],
    ) -> int:
        if self.phase != 0:
            return self.phase

        dot_count = (
            0
            if parking_dot_line.points is None
            else len(parking_dot_line.points)
        )
        rejected_count = (
            0
            if parking_dot_line.rejected_points is None
            else len(parking_dot_line.rejected_points)
        )
        if dot_count + rejected_count <= 1:
            self.phase = 1
            return self.phase

        car_mask = (class_map == int(car_class_id)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            car_mask,
            connectivity=8,
        )
        car_labels = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) > 0
        ]
        if len(car_labels) != 1:
            return self.phase

        label = car_labels[0]
        car_x = int(stats[label, cv2.CC_STAT_LEFT])
        car_y = int(stats[label, cv2.CC_STAT_TOP])
        car_width = int(stats[label, cv2.CC_STAT_WIDTH])
        car_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        car_right_x = car_x + car_width
        car_bottom_y = car_y + car_height

        line_below_car = False
        for line in parking_lines.lines:
            if (
                line.point is None
                or line.direction is None
                or line.segment is None
            ):
                continue
            segment_left = min(point[0] for point in line.segment)
            segment_right = max(point[0] for point in line.segment)
            overlap_left = max(float(car_x), float(segment_left))
            overlap_right = min(float(car_right_x), float(segment_right))
            if overlap_left > overlap_right:
                continue

            sample_x = (overlap_left + overlap_right) * 0.5
            direction_x = float(line.direction[0])
            if abs(direction_x) < 1e-6:
                continue
            scale = (sample_x - float(line.point[0])) / direction_x
            line_y = float(line.point[1] + scale * line.direction[1])
            if line_y >= car_bottom_y:
                line_below_car = True
                break

        if not line_below_car:
            self.phase = 1
        return self.phase


__all__ = ["PhaseController"]
