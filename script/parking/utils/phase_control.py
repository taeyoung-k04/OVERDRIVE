"""Parking phase transition control."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import cv2
import numpy as np

from infer_sem_class import CLASS_TO_ID
from utils.lane_detect import Line, ParkingLineDetection


@dataclass
class PhaseController:
    """Control parking phase transitions from fitted scene geometry."""

    phase: int = 0
    horizontal_tolerance_deg: float = 1.0
    phase_3_reverse_seconds: float = 0.5
    phase_3_stop_seconds: float = 4.0
    phase_started_at: Optional[float] = None
    previous_reference_direction_y: Optional[float] = None

    def update(
        self,
        class_map: np.ndarray,
        parking_lines: Optional[ParkingLineDetection] = None,
        parking_dot_line: Optional[Line] = None,
        reference_line: Optional[Line] = None,
        *,
        car_class_id: int = CLASS_TO_ID["car"],
        out_class_id: int = CLASS_TO_ID["out"],
        now: Optional[float] = None,
    ) -> int:
        current_time = time.perf_counter() if now is None else float(now)

        if self.phase >= 5:
            return self.phase

        if self.phase == 4:
            if np.any(class_map == int(out_class_id)):
                self.phase = 5
                self.phase_started_at = current_time
            return self.phase

        if self.phase == 3:
            if self.phase_started_at is None:
                self.phase_started_at = current_time
            if (
                current_time - self.phase_started_at
                >= (
                    self.phase_3_reverse_seconds
                    + self.phase_3_stop_seconds
                )
            ):
                self.phase = 4
                self.phase_started_at = current_time
            return self.phase

        if self.phase == 2:
            if (
                reference_line is None
                or not reference_line.valid
                or reference_line.direction is None
            ):
                return self.phase

            direction_y = float(reference_line.direction[1])
            direction_x = float(reference_line.direction[0])
            angle = abs(math.atan2(direction_y, direction_x))
            horizontal_error = min(angle, abs(math.pi - angle))
            crossed_horizontal = (
                self.previous_reference_direction_y is not None
                and self.previous_reference_direction_y * direction_y < 0.0
            )
            if (
                horizontal_error <= math.radians(
                    self.horizontal_tolerance_deg
                )
                or crossed_horizontal
            ):
                self.phase = 3
                self.phase_started_at = current_time
                self.previous_reference_direction_y = None
            else:
                self.previous_reference_direction_y = direction_y
            return self.phase

        if self.phase == 1:
            if (
                reference_line is None
                or not reference_line.valid
            ):
                return self.phase

            height, width = class_map.shape
            bottom_x = reference_line.x_at(float(height - 1))
            if (
                bottom_x is not None
                and width * 0.5 <= bottom_x <= width - 1
            ):
                self.phase = 2
                self.phase_started_at = current_time
                self.previous_reference_direction_y = (
                    None
                    if reference_line.direction is None
                    else float(reference_line.direction[1])
                )
            return self.phase

        if parking_lines is None or parking_dot_line is None:
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
            self.phase_started_at = current_time
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
            self.phase_started_at = current_time
        return self.phase


__all__ = ["PhaseController"]
