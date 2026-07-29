"""Detect the parking ``reference`` class and fit it as a straight line.

The detector consumes the two-dimensional semantic class map produced by
``script.parking.infer_sem_class``.  Unlike the main driving lane detector,
the fitted geometry is always a line of the form ``x = slope * y + intercept``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from script.parking.infer_sem_class import CLASS_TO_ID
except ModuleNotFoundError:
    # Support direct execution such as
    # ``python script/parking/realtime_sem_class_camera.py``.
    from infer_sem_class import CLASS_TO_ID


@dataclass(frozen=True)
class ReferenceLine:
    """One reference-line observation in image pixel coordinates."""

    valid: bool
    slope: float = 0.0
    intercept: float = 0.0
    confidence: float = 0.0
    points: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    reason: str = ""

    def x_at(self, y: float) -> Optional[float]:
        """Return the fitted x coordinate at pixel row ``y``."""
        if not self.valid:
            return None
        return float(self.slope * float(y) + self.intercept)

    def endpoints(
        self,
        image_shape: tuple[int, ...],
        *,
        top_ratio: float = 0.0,
        bottom_ratio: float = 1.0,
    ) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        """Return clipped endpoints suitable for ``cv2.line``."""
        if not self.valid:
            return None

        height, width = image_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image shape: {image_shape}")

        top_y = int(np.clip(round((height - 1) * top_ratio), 0, height - 1))
        bottom_y = int(
            np.clip(round((height - 1) * bottom_ratio), 0, height - 1)
        )
        top_x = int(np.clip(round(self.slope * top_y + self.intercept), 0, width - 1))
        bottom_x = int(
            np.clip(round(self.slope * bottom_y + self.intercept), 0, width - 1)
        )
        return (top_x, top_y), (bottom_x, bottom_y)


class ReferenceLineDetector:
    """Extract and temporally stabilize the parking reference line.

    A representative x coordinate is collected from each sampled image row,
    then a robust least-squares fit estimates ``x = slope * y + intercept``.
    Fitting x as a function of y remains stable for the near-vertical reference
    lines normally seen by the parking camera.
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
        max_line_jump_ratio: float = 0.20,
        line_new_weight: float = 0.65,
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
        if max_line_jump_ratio < 0:
            raise ValueError("max_line_jump_ratio must be non-negative")
        if not 0.0 <= line_new_weight <= 1.0:
            raise ValueError("line_new_weight must be between 0 and 1")
        if history_seconds < 0:
            raise ValueError("history_seconds must be non-negative")

        self.class_id = int(class_id)
        self.roi_top_ratio = float(roi_top_ratio)
        self.min_component_area = int(min_component_area)
        self.min_line_points = int(min_line_points)
        self.row_step = int(row_step)
        self.residual_threshold_px = float(residual_threshold_px)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_line_jump_ratio = float(max_line_jump_ratio)
        self.line_new_weight = float(line_new_weight)
        self.history_seconds = float(history_seconds)

        self._previous_coefficients: Optional[np.ndarray] = None
        self._previous_time: Optional[float] = None

    def reset(self) -> None:
        """Clear the line used for temporal selection and smoothing."""
        self._previous_coefficients = None
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
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    def _active_history(self, now: float) -> Optional[np.ndarray]:
        if self._previous_coefficients is None or self._previous_time is None:
            return None
        if now - self._previous_time > self.history_seconds:
            return None
        return self._previous_coefficients

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

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            vertical_span = component_height / max(height, 1)
            score = float(area) * (0.5 + vertical_span)

            if previous is not None:
                center_x, center_y = centroids[label]
                expected_x = previous[0] * center_y + previous[1]
                distance_ratio = abs(center_x - expected_x) / max(width, 1)
                score *= max(0.1, 1.0 - 2.0 * distance_ratio)

            candidates.append((score, label))

        if not candidates:
            return None

        selected_label = max(candidates, key=lambda item: item[0])[1]
        return (labels == selected_label).astype(np.uint8)

    def _collect_points(
        self,
        component_mask: np.ndarray,
        roi_top: int,
    ) -> np.ndarray:
        height = component_mask.shape[0]
        points: list[tuple[float, float]] = []

        for y in range(roi_top, height, self.row_step):
            xs = np.flatnonzero(component_mask[y])
            if xs.size:
                points.append((float(np.median(xs)), float(y)))

        if not points:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    def _robust_fit(
        self,
        points: np.ndarray,
    ) -> tuple[Optional[np.ndarray], np.ndarray, float]:
        if len(points) < self.min_line_points:
            return None, np.zeros(len(points), dtype=bool), 0.0

        inliers = np.ones(len(points), dtype=bool)
        coefficients: Optional[np.ndarray] = None

        for _ in range(4):
            if int(np.count_nonzero(inliers)) < self.min_line_points:
                return None, inliers, 0.0

            coefficients = np.polyfit(
                points[inliers, 1],
                points[inliers, 0],
                deg=1,
            )
            residuals = np.abs(
                points[:, 0]
                - np.polyval(coefficients, points[:, 1])
            )
            median = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - median)))
            adaptive_threshold = max(
                self.residual_threshold_px,
                median + 2.5 * max(mad, 1.0),
            )
            updated = residuals <= adaptive_threshold
            if np.array_equal(updated, inliers):
                break
            inliers = updated

        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / max(len(points), 1)
        if (
            coefficients is None
            or inlier_count < self.min_line_points
            or inlier_ratio < self.min_inlier_ratio
        ):
            return None, inliers, inlier_ratio

        coefficients = np.polyfit(
            points[inliers, 1],
            points[inliers, 0],
            deg=1,
        )
        return coefficients.astype(np.float64), inliers, inlier_ratio

    @staticmethod
    def _jump_ratio(
        previous: np.ndarray,
        current: np.ndarray,
        height: int,
        width: int,
    ) -> float:
        sample_y = np.asarray(
            [height * 0.35, height * 0.60, height * 0.85],
            dtype=np.float64,
        )
        difference = np.abs(
            np.polyval(previous, sample_y)
            - np.polyval(current, sample_y)
        )
        return float(np.max(difference) / max(width, 1))

    def detect(self, class_map: np.ndarray) -> ReferenceLine:
        """Detect the current parking reference line."""
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

        points = self._collect_points(component, roi_top)
        coefficients, inliers, inlier_ratio = self._robust_fit(points)
        if coefficients is None:
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
            jump = self._jump_ratio(previous, coefficients, height, width)
            if jump > self.max_line_jump_ratio:
                return ReferenceLine(
                    valid=False,
                    confidence=float(inlier_ratio),
                    points=points[inliers],
                    mask=component,
                    reason=f"line jump {jump:.3f} exceeds limit",
                )
            weight = self.line_new_weight
            coefficients = weight * coefficients + (1.0 - weight) * previous

        inlier_points = points[inliers]
        vertical_coverage = (
            float(np.ptp(inlier_points[:, 1])) / max(height - roi_top, 1)
            if len(inlier_points) > 1
            else 0.0
        )
        confidence = float(
            np.clip(0.65 * inlier_ratio + 0.35 * vertical_coverage, 0.0, 1.0)
        )

        self._previous_coefficients = coefficients.copy()
        self._previous_time = now
        return ReferenceLine(
            valid=True,
            slope=float(coefficients[0]),
            intercept=float(coefficients[1]),
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
