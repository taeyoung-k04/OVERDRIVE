"""Detect the parking ``reference`` class and fit an orientation-free line."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from script.parking.infer_sem_class import CLASS_TO_ID
except ModuleNotFoundError:
    from infer_sem_class import CLASS_TO_ID


@dataclass(frozen=True)
class ReferenceLine:
    """One fitted reference line in image pixel coordinates."""

    valid: bool
    point: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    angle_deg: float = 0.0
    slope: float = 0.0
    intercept: float = 0.0
    confidence: float = 0.0
    points: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    reason: str = ""

    def x_at(self, y: float) -> Optional[float]:
        """Return x at pixel row ``y``; horizontal lines have no unique x."""
        if (
            not self.valid
            or self.point is None
            or self.direction is None
            or abs(float(self.direction[1])) < 1e-6
        ):
            return None
        scale = (float(y) - float(self.point[1])) / float(self.direction[1])
        return float(self.point[0] + scale * self.direction[0])

    def endpoints(
        self,
        image_shape: tuple[int, ...],
        *,
        top_ratio: float = 0.0,
        bottom_ratio: float = 1.0,
    ) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        """Return image-clipped endpoints suitable for ``cv2.line``."""
        del top_ratio, bottom_ratio  # Retained for API compatibility.
        if not self.valid or self.point is None or self.direction is None:
            return None

        height, width = image_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image shape: {image_shape}")

        extent = float(math.hypot(width, height) * 2.0)
        start = np.rint(self.point - self.direction * extent).astype(int)
        end = np.rint(self.point + self.direction * extent).astype(int)
        visible, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, width, height),
            tuple(map(int, start)),
            tuple(map(int, end)),
        )
        if not visible:
            return None
        return clipped_start, clipped_end


class ReferenceLineDetector:
    """Fit and temporally stabilize vertical, diagonal, or horizontal lines.

    The internal geometry is ``point + t * direction``.  This representation
    has no infinite-slope singularity, so a reference can rotate rapidly from
    vertical to horizontal without the fit becoming invalid.
    """

    def __init__(
        self,
        *,
        class_id: int = CLASS_TO_ID["reference"],
        roi_top_ratio: float = 0.0,
        min_component_area: int = 80,
        min_line_points: int = 12,
        row_step: int = 3,
        residual_threshold_px: float = 8.0,
        min_inlier_ratio: float = 0.55,
        line_new_weight: float = 0.90,
        large_angle_threshold_deg: float = 5.0,
        large_angle_new_weight: float = 0.75,
        position_new_weight: float = 0.90,
        smoothing_anchor_y_ratio: float = 0.85,
        history_seconds: float = 0.5,
    ) -> None:
        if not 0.0 <= roi_top_ratio < 1.0:
            raise ValueError("roi_top_ratio must be in [0, 1)")
        if min_component_area <= 0:
            raise ValueError("min_component_area must be positive")
        if min_line_points < 2:
            raise ValueError("min_line_points must be at least 2")
        if row_step <= 0:
            raise ValueError("row_step must be positive")
        if residual_threshold_px <= 0:
            raise ValueError("residual_threshold_px must be positive")
        if not 0.0 <= min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0 and 1")
        for name, value in (
            ("line_new_weight", line_new_weight),
            ("large_angle_new_weight", large_angle_new_weight),
            ("position_new_weight", position_new_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if large_angle_threshold_deg < 0:
            raise ValueError("large_angle_threshold_deg must be non-negative")
        if not 0.0 <= smoothing_anchor_y_ratio <= 1.0:
            raise ValueError("smoothing_anchor_y_ratio must be between 0 and 1")
        if history_seconds < 0:
            raise ValueError("history_seconds must be non-negative")

        self.class_id = int(class_id)
        self.roi_top_ratio = float(roi_top_ratio)
        self.min_component_area = int(min_component_area)
        self.min_line_points = int(min_line_points)
        self.point_step = int(row_step)
        self.residual_threshold_px = float(residual_threshold_px)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.line_new_weight = float(line_new_weight)
        self.large_angle_threshold_rad = float(
            np.deg2rad(large_angle_threshold_deg)
        )
        self.large_angle_new_weight = float(large_angle_new_weight)
        self.position_new_weight = float(position_new_weight)
        # Kept so existing callers do not break. General-line smoothing uses
        # the fitted center point rather than a y-only anchor.
        self.smoothing_anchor_y_ratio = float(smoothing_anchor_y_ratio)
        self.history_seconds = float(history_seconds)

        self._previous_line: Optional[np.ndarray] = None
        self._previous_time: Optional[float] = None

    def reset(self) -> None:
        self._previous_line = None
        self._previous_time = None

    @staticmethod
    def _validate_class_map(class_map: np.ndarray) -> None:
        if not isinstance(class_map, np.ndarray):
            raise TypeError("class_map must be a numpy.ndarray")
        if class_map.ndim != 2:
            raise ValueError(
                f"class_map must have shape (H, W), got {class_map.shape}"
            )
        if class_map.size == 0:
            raise ValueError("class_map must not be empty")

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    def _active_history(self, now: float) -> Optional[np.ndarray]:
        if self._previous_line is None or self._previous_time is None:
            return None
        if now - self._previous_time > self.history_seconds:
            return None
        return self._previous_line

    @staticmethod
    def _point_line_distance(point: np.ndarray, line: np.ndarray) -> float:
        direction = line[:2]
        origin = line[2:]
        delta = point - origin
        return abs(float(direction[0] * delta[1] - direction[1] * delta[0]))

    def _select_component(
        self,
        mask: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates: list[tuple[float, int]] = []
        height, width = mask.shape
        diagonal = max(math.hypot(width, height), 1.0)

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            span = math.hypot(component_width, component_height) / diagonal
            score = float(area) * (0.5 + span)

            if previous is not None:
                distance_ratio = self._point_line_distance(
                    np.asarray(centroids[label], dtype=np.float64),
                    previous,
                ) / diagonal
                score *= max(0.1, 1.0 - 2.0 * distance_ratio)

            candidates.append((score, label))

        if not candidates:
            return None
        selected_label = max(candidates, key=lambda item: item[0])[1]
        return (labels == selected_label).astype(np.uint8)

    def _collect_points(self, component_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(component_mask)
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        points = np.column_stack((xs, ys)).astype(np.float64)
        return points[:: self.point_step]

    @staticmethod
    def _canonical_line(line: np.ndarray) -> np.ndarray:
        result = np.asarray(line, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(result[:2]))
        if norm <= 1e-9:
            raise ValueError("line direction must be non-zero")
        result[:2] /= norm
        if result[1] < 0.0 or (
            abs(result[1]) < 1e-9 and result[0] < 0.0
        ):
            result[:2] *= -1.0
        return result

    def _fit_line(self, points: np.ndarray) -> np.ndarray:
        fitted = cv2.fitLine(
            points.astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        )
        return self._canonical_line(fitted)

    def _robust_fit(
        self,
        points: np.ndarray,
    ) -> tuple[Optional[np.ndarray], np.ndarray, float]:
        if len(points) < self.min_line_points:
            return None, np.zeros(len(points), dtype=bool), 0.0

        inliers = np.ones(len(points), dtype=bool)
        line: Optional[np.ndarray] = None

        for _ in range(4):
            if int(np.count_nonzero(inliers)) < self.min_line_points:
                return None, inliers, 0.0
            line = self._fit_line(points[inliers])
            direction = line[:2]
            origin = line[2:]
            delta = points - origin
            residuals = np.abs(
                direction[0] * delta[:, 1]
                - direction[1] * delta[:, 0]
            )
            median = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - median)))
            threshold = max(
                self.residual_threshold_px,
                median + 2.5 * max(mad, 1.0),
            )
            updated = residuals <= threshold
            if np.array_equal(updated, inliers):
                break
            inliers = updated

        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / max(len(points), 1)
        if (
            line is None
            or inlier_count < self.min_line_points
            or inlier_ratio < self.min_inlier_ratio
        ):
            return None, inliers, inlier_ratio
        return self._fit_line(points[inliers]), inliers, inlier_ratio

    @staticmethod
    def _shortest_axis_angle_delta(current: float, previous: float) -> float:
        """Return the shortest difference between two undirected line angles."""
        return float((current - previous + math.pi / 2.0) % math.pi - math.pi / 2.0)

    def _smooth_line(
        self,
        previous: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        previous_angle = math.atan2(previous[1], previous[0])
        current_angle = math.atan2(current[1], current[0])
        angle_delta = self._shortest_axis_angle_delta(
            current_angle,
            previous_angle,
        )
        angle_weight = (
            self.large_angle_new_weight
            if abs(angle_delta) >= self.large_angle_threshold_rad
            else self.line_new_weight
        )
        smoothed_angle = previous_angle + angle_weight * angle_delta
        smoothed_point = (
            self.position_new_weight * current[2:]
            + (1.0 - self.position_new_weight) * previous[2:]
        )
        return self._canonical_line(
            np.asarray(
                [
                    math.cos(smoothed_angle),
                    math.sin(smoothed_angle),
                    smoothed_point[0],
                    smoothed_point[1],
                ],
                dtype=np.float64,
            )
        )

    def detect(self, class_map: np.ndarray) -> ReferenceLine:
        self._validate_class_map(class_map)
        height, width = class_map.shape
        now = time.perf_counter()
        previous = self._active_history(now)

        roi_top = int(np.clip(round(height * self.roi_top_ratio), 0, height - 1))
        mask = (class_map == self.class_id).astype(np.uint8)
        mask[:roi_top] = 0
        mask = self._clean_mask(mask)

        component = self._select_component(mask, previous)
        if component is None:
            return ReferenceLine(
                valid=False,
                mask=mask,
                reason="no reference component passed the area threshold",
            )

        points = self._collect_points(component)
        line, inliers, inlier_ratio = self._robust_fit(points)
        if line is None:
            return ReferenceLine(
                valid=False,
                confidence=float(inlier_ratio),
                points=points,
                mask=component,
                reason=(
                    f"straight-line fit rejected: "
                    f"{int(np.count_nonzero(inliers))}/{len(points)} inliers"
                ),
            )

        if previous is not None:
            line = self._smooth_line(previous, line)

        inlier_points = points[inliers]
        projections = inlier_points @ line[:2]
        line_coverage = (
            float(np.ptp(projections)) / max(math.hypot(width, height), 1.0)
            if len(projections) > 1
            else 0.0
        )
        confidence = float(
            np.clip(0.65 * inlier_ratio + 0.35 * line_coverage, 0.0, 1.0)
        )

        direction = line[:2].copy()
        point = line[2:].copy()
        angle_deg = float(np.degrees(math.atan2(direction[1], direction[0])))
        if abs(direction[1]) >= 1e-6:
            slope = float(direction[0] / direction[1])
            intercept = float(point[0] - slope * point[1])
        else:
            slope = math.inf
            intercept = math.nan

        self._previous_line = line.copy()
        self._previous_time = now
        return ReferenceLine(
            valid=True,
            point=point,
            direction=direction,
            angle_deg=angle_deg,
            slope=slope,
            intercept=intercept,
            confidence=confidence,
            points=inlier_points,
            mask=component,
            reason="reference line fitted",
        )


def draw_reference_line(
    image: np.ndarray,
    line: ReferenceLine,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a detected reference line on ``image`` in place."""
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")
    endpoints = line.endpoints(image.shape)
    if endpoints is not None:
        cv2.line(image, endpoints[0], endpoints[1], color, thickness, cv2.LINE_AA)
    return image


__all__ = [
    "ReferenceLine",
    "ReferenceLineDetector",
    "draw_reference_line",
]
