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
class Line:
    """One fitted reference line in image pixel coordinates."""

    valid: bool
    point: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    angle_deg: float = 0.0
    slope: float = 0.0
    intercept: float = 0.0
    confidence: float = 0.0
    points: Optional[np.ndarray] = None
    rejected_points: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    segment: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
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

        if self.segment is not None:
            return self.segment

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
        """Smooth line angle and signed normal offset (general intercept)."""
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

        # A line is represented as n . p = rho, where n is perpendicular to
        # its direction.  rho is a stable intercept for every orientation,
        # including a horizontal line where x = slope*y + intercept fails.
        previous_normal = np.asarray(
            [-math.sin(previous_angle), math.cos(previous_angle)],
            dtype=np.float64,
        )
        current_normal = np.asarray(
            [-math.sin(current_angle), math.cos(current_angle)],
            dtype=np.float64,
        )
        previous_rho = float(np.dot(previous_normal, previous[2:]))
        current_rho = float(np.dot(current_normal, current[2:]))

        # The shortest undirected angle may represent the current direction
        # shifted by pi.  In that case its normal and signed rho must flip.
        adjusted_current_angle = previous_angle + angle_delta
        half_turns = int(round(
            (adjusted_current_angle - current_angle) / math.pi
        ))
        if half_turns % 2:
            current_rho *= -1.0

        smoothed_rho = (
            self.position_new_weight * current_rho
            + (1.0 - self.position_new_weight) * previous_rho
        )
        smoothed_normal = np.asarray(
            [-math.sin(smoothed_angle), math.cos(smoothed_angle)],
            dtype=np.float64,
        )
        smoothed_point = smoothed_normal * smoothed_rho
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

    def detect(self, class_map: np.ndarray) -> Line:
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
            return Line(
                valid=False,
                mask=mask,
                reason="no reference component passed the area threshold",
            )

        points = self._collect_points(component)
        line, inliers, inlier_ratio = self._robust_fit(points)
        if line is None:
            return Line(
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
        return Line(
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


class ParkingDotLineDetector(ReferenceLineDetector):
    """Fit a line through the leftmost points of ``parking_line`` blobs.

    Two or more connected components produce a new fit.  Zero or one valid
    component reports an invalid observation.  The fitted line is returned
    directly without temporal smoothing.
    """

    def __init__(
        self,
        *,
        class_id: int = CLASS_TO_ID["parking_line"],
        min_component_area: int = 20,
        outlier_distance_ratio: float = 0.02,
        **kwargs,
    ) -> None:
        # General reference-line fitting uses many mask pixels, whereas this
        # detector fits one point per connected component.  Two points are
        # therefore sufficient.
        kwargs.setdefault("min_line_points", 2)
        if outlier_distance_ratio <= 0:
            raise ValueError("outlier_distance_ratio must be positive")
        super().__init__(
            class_id=class_id,
            min_component_area=min_component_area,
            **kwargs,
        )
        self.outlier_distance_ratio = float(outlier_distance_ratio)

    @staticmethod
    def _clean_dot_mask(mask: np.ndarray) -> np.ndarray:
        """Remove isolated noise without joining separate parking dots."""
        binary = (mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def _leftmost_component_points(
        self,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return one leftmost point for every valid connected component."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        points: list[tuple[float, float]] = []
        selected_mask = np.zeros_like(mask, dtype=np.uint8)

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            component_ys, component_xs = np.nonzero(labels == label)
            if component_xs.size == 0:
                continue

            left_x = int(np.min(component_xs))
            left_ys = component_ys[component_xs == left_x]
            left_y = float(np.median(left_ys))
            points.append((float(left_x), left_y))
            selected_mask[labels == label] = 1

        if not points:
            return (
                np.empty((0, 2), dtype=np.float64),
                selected_mask,
                0,
            )

        # Stable ordering makes debug output and fitting deterministic.
        point_array = np.asarray(points, dtype=np.float64)
        order = np.lexsort((point_array[:, 0], point_array[:, 1]))
        return point_array[order], selected_mask, len(points)

    @staticmethod
    def _orthogonal_residuals(
        points: np.ndarray,
        line: np.ndarray,
    ) -> np.ndarray:
        """Return perpendicular pixel distances from points to ``line``."""
        direction = line[:2]
        origin = line[2:]
        delta = points - origin
        return np.abs(
            direction[0] * delta[:, 1]
            - direction[1] * delta[:, 0]
        )

    def _fit_remove_outliers_refit(
        self,
        points: np.ndarray,
        image_width: int,
    ) -> tuple[Optional[np.ndarray], np.ndarray, float]:
        """Fit all points, reject distant points, and refit the inliers."""
        point_count = len(points)
        if point_count < 2:
            return None, np.zeros(point_count, dtype=bool), 0.0

        initial_line = self._fit_line(points)
        residuals = self._orthogonal_residuals(points, initial_line)
        threshold_px = max(
            2.0,
            float(image_width) * self.outlier_distance_ratio,
        )
        inliers = residuals <= threshold_px
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / point_count

        if inlier_count < 2 or inlier_ratio < self.min_inlier_ratio:
            return None, inliers, inlier_ratio

        refined_line = self._fit_line(points[inliers])
        return refined_line, inliers, inlier_ratio

    @staticmethod
    def _line_observation(
        line: np.ndarray,
        *,
        confidence: float,
        points: np.ndarray,
        rejected_points: np.ndarray,
        mask: np.ndarray,
        reason: str,
    ) -> Line:
        direction = line[:2].copy()
        point = line[2:].copy()
        angle_deg = float(np.degrees(math.atan2(direction[1], direction[0])))

        if abs(direction[1]) >= 1e-6:
            slope = float(direction[0] / direction[1])
            intercept = float(point[0] - slope * point[1])
        else:
            slope = math.inf
            intercept = math.nan

        return Line(
            valid=True,
            point=point,
            direction=direction,
            angle_deg=angle_deg,
            slope=slope,
            intercept=intercept,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            points=points,
            rejected_points=rejected_points,
            mask=mask,
            reason=reason,
        )

    def detect(self, class_map: np.ndarray) -> Line:
        """Detect ``parking_dot_line`` from parking-line components."""
        self._validate_class_map(class_map)
        height, width = class_map.shape

        roi_top = int(np.clip(round(height * self.roi_top_ratio), 0, height - 1))
        mask = (class_map == self.class_id).astype(np.uint8)
        mask[:roi_top] = 0
        mask = self._clean_dot_mask(mask)

        points, selected_mask, component_count = (
            self._leftmost_component_points(mask)
        )

        if component_count == 0:
            return Line(
                valid=False,
                points=points,
                mask=selected_mask,
                reason="not detected: no components",
            )

        if component_count == 1:
            return Line(
                valid=False,
                points=points,
                mask=selected_mask,
                reason=(
                    "not detected: only one component"
                ),
            )

        line, inliers, inlier_ratio = self._fit_remove_outliers_refit(
            points,
            width,
        )
        if line is None:
            return Line(
                valid=False,
                confidence=float(inlier_ratio),
                points=points[inliers],
                rejected_points=points[~inliers],
                mask=selected_mask,
                reason=(
                    f"parking_dot_line fit rejected: "
                    f"{int(np.count_nonzero(inliers))}/{component_count} inliers"
                ),
            )

        inlier_points = points[inliers]
        projections = inlier_points @ line[:2]
        coverage = (
            float(np.ptp(projections)) / max(math.hypot(width, height), 1.0)
            if len(projections) > 1
            else 0.0
        )
        confidence = float(
            np.clip(0.75 * inlier_ratio + 0.25 * coverage, 0.0, 1.0)
        )

        return self._line_observation(
            line,
            confidence=confidence,
            points=inlier_points,
            rejected_points=points[~inliers],
            mask=selected_mask,
            reason=(
                f"parking_dot_line fitted from "
                f"{component_count} components"
            ),
        )


@dataclass(frozen=True)
class ParkingLineDetection:
    """Horizontal parking-line segments detected in one frame."""

    lines: tuple[Line, ...]
    component_count: int
    rejected_count: int

    @property
    def valid(self) -> bool:
        return bool(self.lines)


class ParkingLineDetector(ReferenceLineDetector):
    """Fit each ``parking_line`` component as a near-horizontal segment."""

    def __init__(
        self,
        *,
        class_id: int = CLASS_TO_ID["parking_line"],
        min_component_area: int = 20,
        max_horizontal_angle_deg: float = 20.0,
        **kwargs,
    ) -> None:
        if not 0.0 <= max_horizontal_angle_deg < 90.0:
            raise ValueError(
                "max_horizontal_angle_deg must be in [0, 90)"
            )
        kwargs.setdefault("min_line_points", 2)
        super().__init__(
            class_id=class_id,
            min_component_area=min_component_area,
            **kwargs,
        )
        self.max_horizontal_angle_rad = float(
            np.deg2rad(max_horizontal_angle_deg)
        )

    @staticmethod
    def _horizontal_angle(direction: np.ndarray) -> float:
        angle = abs(float(math.atan2(direction[1], direction[0])))
        return min(angle, abs(math.pi - angle))

    def detect(
        self,
        class_map: np.ndarray,
        *,
        excluded_points: Optional[np.ndarray] = None,
    ) -> ParkingLineDetection:
        """Return one fitted segment for each horizontal parking component."""
        self._validate_class_map(class_map)
        height, width = class_map.shape
        roi_top = int(np.clip(round(height * self.roi_top_ratio), 0, height - 1))
        mask = (class_map == self.class_id).astype(np.uint8)
        mask[:roi_top] = 0
        mask = ParkingDotLineDetector._clean_dot_mask(mask)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        lines: list[Line] = []
        component_count = 0
        rejected_count = 0
        excluded = (
            np.asarray(excluded_points, dtype=np.float64).reshape(-1, 2)
            if excluded_points is not None
            else np.empty((0, 2), dtype=np.float64)
        )

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue
            component_count += 1

            ys, xs = np.nonzero(labels == label)
            points = np.column_stack((xs, ys)).astype(np.float64)

            # ParkingDotLineDetector marks an outlier at the component's
            # leftmost point.  Do not create a parking-line segment for the
            # component represented by that red point.
            left_x = float(np.min(xs))
            left_ys = ys[xs == int(left_x)]
            left_point = np.asarray(
                [left_x, float(np.median(left_ys))],
                dtype=np.float64,
            )
            if (
                len(excluded)
                and float(np.min(np.linalg.norm(
                    excluded - left_point,
                    axis=1,
                ))) <= 2.0
            ):
                rejected_count += 1
                continue

            sampled = points[:: self.point_step]
            if len(sampled) < self.min_line_points:
                rejected_count += 1
                continue

            line, inliers, inlier_ratio = self._robust_fit(sampled)
            if (
                line is None
                or self._horizontal_angle(line[:2])
                > self.max_horizontal_angle_rad
            ):
                rejected_count += 1
                continue

            inlier_points = sampled[inliers]
            direction_x = float(line[0])
            if abs(direction_x) < 1e-6:
                rejected_count += 1
                continue

            # Stop at the component's left edge, then extend the fitted line
            # all the way to the right edge of the image.
            start_x = float(np.min(inlier_points[:, 0]))
            end_x = float(width - 1)
            start_scale = (start_x - float(line[2])) / direction_x
            end_scale = (end_x - float(line[2])) / direction_x
            start = line[2:] + line[:2] * start_scale
            end = line[2:] + line[:2] * end_scale
            segment = (
                (
                    int(np.clip(round(start[0]), 0, width - 1)),
                    int(np.clip(round(start[1]), 0, height - 1)),
                ),
                (
                    int(np.clip(round(end[0]), 0, width - 1)),
                    int(np.clip(round(end[1]), 0, height - 1)),
                ),
            )
            direction = line[:2].copy()
            origin = line[2:].copy()
            angle_deg = float(
                np.degrees(math.atan2(direction[1], direction[0]))
            )
            lines.append(
                Line(
                    valid=True,
                    point=origin,
                    direction=direction,
                    angle_deg=angle_deg,
                    confidence=float(inlier_ratio),
                    points=inlier_points,
                    mask=(labels == label).astype(np.uint8),
                    segment=segment,
                    reason="horizontal parking line fitted",
                )
            )

        return ParkingLineDetection(
            lines=tuple(lines),
            component_count=component_count,
            rejected_count=rejected_count,
        )


def draw_parking_lines(
    image: np.ndarray,
    detection: ParkingLineDetection,
    *,
    color: tuple[int, int, int] = (0, 140, 255),
    thickness: int = 3,
) -> np.ndarray:
    """Draw all detected parking-line segments."""
    for line in detection.lines:
        draw_line(
            image,
            line,
            color=color,
            thickness=thickness,
        )
    return image


def draw_line_points(
    image: np.ndarray,
    line: Line,
    *,
    color: tuple[int, int, int] = (0, 200, 255),
    rejected_color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 6,
) -> np.ndarray:
    """Draw the component points used for fitting ``line``."""
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")
    if line.points is None:
        return image

    height, width = image.shape[:2]
    for x, y in np.asarray(line.points).reshape(-1, 2):
        center = (
            int(np.clip(round(float(x)), 0, width - 1)),
            int(np.clip(round(float(y)), 0, height - 1)),
        )
        cv2.circle(
            image,
            center,
            max(1, int(radius)),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    if line.rejected_points is not None:
        for x, y in np.asarray(line.rejected_points).reshape(-1, 2):
            center = (
                int(np.clip(round(float(x)), 0, width - 1)),
                int(np.clip(round(float(y)), 0, height - 1)),
            )
            cv2.circle(
                image,
                center,
                max(1, int(radius)),
                rejected_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    return image


def draw_line(
    image: np.ndarray,
    line: Line,
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
    "ParkingLineDetection",
    "ParkingLineDetector",
    "ParkingDotLineDetector",
    "Line",
    "ReferenceLineDetector",
    "draw_line_points",
    "draw_line",
    "draw_parking_lines",
]
