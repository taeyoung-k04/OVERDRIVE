#!/usr/bin/env python3
"""Temporal lane-boundary detector for OVERDRIVE.

Design goals
------------
1. Keep the proven single-boundary following strategy.
2. Detect the semantic center boundary and right boundary separately.
3. Treat the center boundary as a potentially dashed/fragmented marking.
4. Use temporal continuity strongly and fixed image position only weakly.
5. Return LaneCurve objects that can be passed to the existing
   RightLaneFollower.compute() method.

Coordinate convention
---------------------
The fitted curve is:

    x = a * y_norm**2 + b * y_norm + c

where y_norm is y / (image_height - 1).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from perception.infer_sem_class import CLASS_TO_ID


class LaneBoundary(str, Enum):
    """Semantic boundary names used by the driving logic."""

    CENTER = "center"
    RIGHT = "right"


@dataclass
class LaneCurve:
    """One detected lane-boundary curve.

    The field names intentionally match the existing RightLaneObservation
    fields so this object can be passed to the current RightLaneFollower.
    """

    boundary: LaneBoundary
    valid: bool
    coefficients: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    residual_px: float = math.inf
    reason: str = ""

    def x_at(self, y_ratio: float) -> Optional[float]:
        """Evaluate this curve at a normalized image y-coordinate."""
        if not self.valid or self.coefficients is None:
            return None
        return float(np.polyval(self.coefficients, float(y_ratio)))


@dataclass
class LaneObservation:
    """Center and right boundaries detected from one semantic class map."""

    center: LaneCurve
    right: LaneCurve

    def reference_for_lane(self, lane_number: int) -> LaneCurve:
        """Return the right-side reference boundary for lane 1 or lane 2.

        Lane 1 follows the semantic center marking as its right boundary.
        Lane 2 follows the semantic right marking as its right boundary.
        """
        if lane_number == 1:
            return self.center
        if lane_number == 2:
            return self.right
        raise ValueError(f"lane_number must be 1 or 2, got {lane_number}")


@dataclass
class _ComponentCandidate:
    """Internal connected-component description."""

    label: int
    mask: np.ndarray
    area: int
    x_median: float
    y_min: float
    y_max: float
    vertical_span: float
    bottom_ratio: float
    span_ratio: float
    position_score: float
    continuity_score: float
    total_score: float


class LaneDetector:
    """Detect temporally stable center/right semantic lane boundaries.

    Notes
    -----
    * RIGHT normally selects one strong continuous component.
    * CENTER may merge several compatible components because it can be dashed.
    * Previous-frame curves are used only to select and smooth the current
      observation. A missing current detection is still returned as invalid;
      the existing follower remains responsible for short lane-loss recovery.
    """

    def __init__(
        self,
        *,
        roi_top_ratio: float = 0.38,
        min_lane_points: int = 18,
        right_min_component_area: int = 90,
        right_fragment_min_area: int = 12,
        center_min_component_area: int = 18,
        center_fragment_min_area: int = 8,
        sample_row_step: int = 2,
        max_row_cluster_gap: int = 4,
        max_lane_jump_ratio: float = 0.18,
        curve_new_weight: float = 0.38,
        history_max_age_seconds: float = 0.80,
        continuity_scale_ratio: float = 0.10,
        center_merge_distance_ratio: float = 0.16,
        right_fallback_min_x_ratio: float = 0.25,
        expected_center_x_ratio: float = 0.50,
        expected_right_x_ratio: float = 0.78,
    ) -> None:
        if not 0.0 <= roi_top_ratio < 1.0:
            raise ValueError("roi_top_ratio must be in [0, 1)")
        if min_lane_points < 3:
            raise ValueError("min_lane_points must be at least 3")
        if sample_row_step < 1:
            raise ValueError("sample_row_step must be at least 1")

        self.roi_top_ratio = float(roi_top_ratio)
        self.min_lane_points = int(min_lane_points)

        self.right_min_component_area = int(right_min_component_area)
        self.right_fragment_min_area = int(right_fragment_min_area)
        self.center_min_component_area = int(center_min_component_area)
        self.center_fragment_min_area = int(center_fragment_min_area)

        self.sample_row_step = int(sample_row_step)
        self.max_row_cluster_gap = int(max_row_cluster_gap)

        self.max_lane_jump_ratio = float(max_lane_jump_ratio)
        self.curve_new_weight = float(np.clip(curve_new_weight, 0.0, 1.0))
        self.history_max_age_seconds = max(0.0, float(history_max_age_seconds))
        self.continuity_scale_ratio = max(1e-3, float(continuity_scale_ratio))
        self.center_merge_distance_ratio = max(
            1e-3,
            float(center_merge_distance_ratio),
        )
        self.right_fallback_min_x_ratio = float(
            np.clip(right_fallback_min_x_ratio, 0.0, 1.0)
        )

        self.expected_x_ratio = {
            LaneBoundary.CENTER: float(
                np.clip(expected_center_x_ratio, 0.0, 1.0)
            ),
            LaneBoundary.RIGHT: float(
                np.clip(expected_right_x_ratio, 0.0, 1.0)
            ),
        }

        self._previous_coefficients: dict[
            LaneBoundary, Optional[np.ndarray]
        ] = {
            LaneBoundary.CENTER: None,
            LaneBoundary.RIGHT: None,
        }
        self._previous_confidence: dict[LaneBoundary, float] = {
            LaneBoundary.CENTER: 0.0,
            LaneBoundary.RIGHT: 0.0,
        }
        self._previous_time: dict[LaneBoundary, Optional[float]] = {
            LaneBoundary.CENTER: None,
            LaneBoundary.RIGHT: None,
        }

    def reset(self) -> None:
        """Clear all temporal detector state."""
        for boundary in LaneBoundary:
            self._previous_coefficients[boundary] = None
            self._previous_confidence[boundary] = 0.0
            self._previous_time[boundary] = None

    def detect(self, class_map: np.ndarray) -> LaneObservation:
        """Detect center and right boundaries from one class map."""
        self._validate_class_map(class_map)

        center = self._extract_curve(
            class_map=class_map,
            boundary=LaneBoundary.CENTER,
            class_id=int(CLASS_TO_ID["lane_center"]),
        )
        right = self._extract_curve(
            class_map=class_map,
            boundary=LaneBoundary.RIGHT,
            class_id=int(CLASS_TO_ID["lane_right"]),
        )

        return LaneObservation(center=center, right=right)

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

    def _history_coefficients(
        self,
        boundary: LaneBoundary,
        now: float,
    ) -> Optional[np.ndarray]:
        previous = self._previous_coefficients[boundary]
        previous_time = self._previous_time[boundary]

        if previous is None or previous_time is None:
            return None

        if now - previous_time > self.history_max_age_seconds:
            return None

        return previous

    @staticmethod
    def _clean_mask(
        mask: np.ndarray,
        boundary: LaneBoundary,
    ) -> np.ndarray:
        """Apply boundary-specific morphology."""
        binary = (mask > 0).astype(np.uint8)

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            open_kernel,
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        if boundary == LaneBoundary.CENTER:
            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 9),
            )
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                vertical_kernel,
            )

        return cleaned

    @staticmethod
    def _split_x_clusters(
        xs: np.ndarray,
        max_gap: int,
    ) -> list[np.ndarray]:
        if xs.size == 0:
            return []

        split_indices = np.flatnonzero(np.diff(xs) > max_gap) + 1
        return [
            cluster
            for cluster in np.split(xs, split_indices)
            if cluster.size > 0
        ]

    def _expected_x(
        self,
        *,
        boundary: LaneBoundary,
        y_ratio: float,
        width: int,
        previous: Optional[np.ndarray],
    ) -> float:
        if previous is not None:
            return float(np.polyval(previous, y_ratio))

        return width * self.expected_x_ratio[boundary]

    def _collect_row_points(
        self,
        *,
        component_mask: np.ndarray,
        boundary: LaneBoundary,
        roi_top: int,
        previous: Optional[np.ndarray],
    ) -> list[tuple[float, float]]:
        """Collect one representative lane x-coordinate per sampled row."""
        height, width = component_mask.shape
        points: list[tuple[float, float]] = []

        for y in range(roi_top, height, self.sample_row_step):
            xs = np.flatnonzero(component_mask[y])
            if xs.size == 0:
                continue

            clusters = self._split_x_clusters(
                xs,
                self.max_row_cluster_gap,
            )
            if not clusters:
                continue

            y_ratio = y / max(height - 1, 1)
            expected_x = self._expected_x(
                boundary=boundary,
                y_ratio=y_ratio,
                width=width,
                previous=previous,
            )

            best_x: Optional[float] = None
            best_score = -math.inf

            for cluster in clusters:
                cluster_x = float(np.median(cluster))
                cluster_size = float(cluster.size)

                distance_ratio = abs(cluster_x - expected_x) / max(width, 1)
                continuity = math.exp(
                    -distance_ratio / self.continuity_scale_ratio
                )

                score = math.log1p(cluster_size) * (0.55 + 0.45 * continuity)

                if score > best_score:
                    best_score = score
                    best_x = cluster_x

            if best_x is not None:
                points.append((best_x, float(y)))

        return points

    def _candidate_continuity_score(
        self,
        *,
        candidate_mask: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> float:
        if previous is None:
            return 0.55

        height, width = candidate_mask.shape
        ys, xs = np.nonzero(candidate_mask)
        if xs.size == 0:
            return 0.0

        if xs.size > 300:
            sample_indices = np.linspace(
                0,
                xs.size - 1,
                300,
                dtype=np.int32,
            )
            xs = xs[sample_indices]
            ys = ys[sample_indices]

        y_norm = ys.astype(np.float64) / max(height - 1, 1)
        predicted_x = np.polyval(previous, y_norm)
        residual_ratio = np.abs(xs - predicted_x) / max(width, 1)

        median_residual_ratio = float(np.median(residual_ratio))

        return float(
            math.exp(
                -median_residual_ratio / self.continuity_scale_ratio
            )
        )

    def _build_candidates(
        self,
        *,
        mask: np.ndarray,
        boundary: LaneBoundary,
        previous: Optional[np.ndarray],
    ) -> list[_ComponentCandidate]:
        height, width = mask.shape

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        minimum_area = (
            self.right_fragment_min_area
            if boundary == LaneBoundary.RIGHT
            else self.center_fragment_min_area
        )

        candidates: list[_ComponentCandidate] = []

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue

            component_mask = (labels == label).astype(np.uint8)
            ys, xs = np.nonzero(component_mask)

            if xs.size == 0:
                continue

            x_median = float(np.median(xs))
            y_min = float(ys.min())
            y_max = float(ys.max())
            vertical_span = max(1.0, y_max - y_min)

            bottom_ratio = float(
                np.clip(
                    y_max / max(height - 1, 1),
                    0.0,
                    1.0,
                )
            )
            span_ratio = float(
                np.clip(
                    vertical_span / max(height * 0.45, 1.0),
                    0.0,
                    1.0,
                )
            )

            expected_x = self._expected_x(
                boundary=boundary,
                y_ratio=0.78,
                width=width,
                previous=previous,
            )
            position_distance_ratio = abs(x_median - expected_x) / max(
                width,
                1,
            )
            position_score = float(
                np.clip(
                    1.0 - position_distance_ratio,
                    0.0,
                    1.0,
                )
            )

            continuity_score = self._candidate_continuity_score(
                candidate_mask=component_mask,
                previous=previous,
            )

            area_score = math.log1p(area)

            if boundary == LaneBoundary.RIGHT:
                total_score = (
                    area_score
                    * (0.30 + 0.70 * bottom_ratio)
                    * (0.30 + 0.70 * span_ratio)
                    * (0.40 + 0.60 * continuity_score)
                    * (0.78 + 0.22 * position_score)
                )
            else:
                total_score = (
                    area_score
                    * (0.35 + 0.65 * bottom_ratio)
                    * (0.35 + 0.65 * span_ratio)
                    * (0.35 + 0.65 * continuity_score)
                    * (0.92 + 0.08 * position_score)
                )

            candidates.append(
                _ComponentCandidate(
                    label=label,
                    mask=component_mask,
                    area=area,
                    x_median=x_median,
                    y_min=y_min,
                    y_max=y_max,
                    vertical_span=vertical_span,
                    bottom_ratio=bottom_ratio,
                    span_ratio=span_ratio,
                    position_score=position_score,
                    continuity_score=continuity_score,
                    total_score=total_score,
                )
            )

        candidates.sort(
            key=lambda candidate: candidate.total_score,
            reverse=True,
        )
        return candidates

    def _component_distance_to_curve(
        self,
        *,
        component_mask: np.ndarray,
        coefficients: np.ndarray,
    ) -> float:
        height, width = component_mask.shape
        ys, xs = np.nonzero(component_mask)

        if xs.size == 0:
            return math.inf

        if xs.size > 250:
            sample_indices = np.linspace(
                0,
                xs.size - 1,
                250,
                dtype=np.int32,
            )
            xs = xs[sample_indices]
            ys = ys[sample_indices]

        y_norm = ys.astype(np.float64) / max(height - 1, 1)
        predicted_x = np.polyval(coefficients, y_norm)

        return float(
            np.median(np.abs(xs - predicted_x)) / max(width, 1)
        )

    def _quick_fit_candidate(
        self,
        candidate: _ComponentCandidate,
        roi_top: int,
        previous: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        height = candidate.mask.shape[0]
        points = self._collect_row_points(
            component_mask=candidate.mask,
            boundary=LaneBoundary.CENTER,
            roi_top=roi_top,
            previous=previous,
        )

        if len(points) < 3:
            return None

        point_array = np.asarray(points, dtype=np.float64)
        xs = point_array[:, 0]
        ys = point_array[:, 1]
        y_norm = ys / max(height - 1, 1)

        degree = 2 if len(points) >= 6 else 1

        try:
            coefficients = np.polyfit(y_norm, xs, degree)
        except (np.linalg.LinAlgError, ValueError):
            return None

        if degree == 1:
            coefficients = np.asarray(
                [0.0, coefficients[0], coefficients[1]],
                dtype=np.float64,
            )

        return coefficients

    def _select_right_mask(
        self,
        *,
        candidates: list[_ComponentCandidate],
        full_mask: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], bool, str]:
        height, width = full_mask.shape

        strong_candidates = [
            candidate
            for candidate in candidates
            if candidate.area >= self.right_min_component_area
        ]

        if strong_candidates:
            return strong_candidates[0].mask.copy(), False, "best right component"

        merged = np.zeros_like(full_mask)
        merged_count = 0

        for candidate in candidates:
            x_ratio = candidate.x_median / max(width - 1, 1)

            history_compatible = (
                previous is not None
                and candidate.continuity_score >= 0.42
            )
            broad_right_region = (
                x_ratio >= self.right_fallback_min_x_ratio
            )

            if history_compatible or broad_right_region:
                merged = cv2.bitwise_or(merged, candidate.mask)
                merged_count += 1

        if merged_count == 0 or np.count_nonzero(merged) < max(
            12,
            self.right_min_component_area // 2,
        ):
            return None, False, "no right-lane component"

        return merged, True, f"merged {merged_count} right fragments"

    def _select_center_mask(
        self,
        *,
        candidates: list[_ComponentCandidate],
        full_mask: np.ndarray,
        roi_top: int,
        previous: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], bool, str]:
        if not candidates:
            return None, False, "no center-lane component"

        seed = candidates[0]

        reference_curve = previous
        if reference_curve is None:
            reference_curve = self._quick_fit_candidate(
                candidate=seed,
                roi_top=roi_top,
                previous=None,
            )

        merged = np.zeros_like(full_mask)
        merged_count = 0

        for candidate in candidates:
            include = False

            if candidate is seed:
                include = True
            elif reference_curve is not None:
                distance_ratio = self._component_distance_to_curve(
                    component_mask=candidate.mask,
                    coefficients=reference_curve,
                )
                include = (
                    distance_ratio
                    <= self.center_merge_distance_ratio
                )
            else:
                include = (
                    candidate.area >= self.center_min_component_area
                    and candidate.span_ratio >= 0.10
                )

            if candidate.continuity_score >= 0.62:
                include = True

            if include:
                merged = cv2.bitwise_or(merged, candidate.mask)
                merged_count += 1

        if merged_count == 0:
            return None, False, "no compatible center fragments"

        minimum_pixels = max(
            10,
            self.center_min_component_area,
        )
        if int(np.count_nonzero(merged)) < minimum_pixels:
            return None, False, "center fragments too small"

        used_merge = merged_count > 1
        return (
            merged,
            used_merge,
            f"merged {merged_count} center fragments",
        )

    def _robust_polyfit(
        self,
        *,
        points: np.ndarray,
        height: int,
    ) -> tuple[
        Optional[np.ndarray],
        np.ndarray,
        float,
    ]:
        xs = points[:, 0].astype(np.float64)
        ys = points[:, 1].astype(np.float64)
        y_norm = ys / max(height - 1, 1)

        try:
            coefficients = np.polyfit(y_norm, xs, 2)
        except (np.linalg.LinAlgError, ValueError):
            return None, np.zeros(len(points), dtype=bool), math.inf

        inliers = np.ones(len(points), dtype=bool)

        for _ in range(3):
            predicted_x = np.polyval(coefficients, y_norm)
            residuals = np.abs(xs - predicted_x)

            active_residuals = residuals[inliers]
            if active_residuals.size == 0:
                break

            median = float(np.median(active_residuals))
            mad = float(
                np.median(
                    np.abs(active_residuals - median)
                )
            )

            robust_sigma = 1.4826 * mad
            threshold = float(
                np.clip(
                    median + 2.8 * robust_sigma + 2.0,
                    5.0,
                    24.0,
                )
            )

            new_inliers = residuals <= threshold
            if int(np.count_nonzero(new_inliers)) < self.min_lane_points:
                break

            if np.array_equal(new_inliers, inliers):
                inliers = new_inliers
                break

            inliers = new_inliers

            try:
                coefficients = np.polyfit(
                    y_norm[inliers],
                    xs[inliers],
                    2,
                )
            except (np.linalg.LinAlgError, ValueError):
                break

        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < self.min_lane_points:
            return coefficients, inliers, math.inf

        predicted_x = np.polyval(
            coefficients,
            y_norm[inliers],
        )
        residual_px = float(
            np.mean(
                np.abs(
                    xs[inliers] - predicted_x
                )
            )
        )

        return coefficients, inliers, residual_px

    def _curve_jump_ratio(
        self,
        *,
        previous: np.ndarray,
        current: np.ndarray,
        width: int,
    ) -> float:
        sample_y = np.asarray(
            [0.52, 0.64, 0.76, 0.86],
            dtype=np.float64,
        )
        previous_x = np.polyval(previous, sample_y)
        current_x = np.polyval(current, sample_y)

        return float(
            np.max(np.abs(current_x - previous_x))
            / max(width, 1)
        )

    def _smooth_coefficients(
        self,
        *,
        previous: Optional[np.ndarray],
        current: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        if previous is None:
            return current.astype(np.float64, copy=True)

        adaptive_weight = self.curve_new_weight + 0.22 * (
            float(np.clip(confidence, 0.0, 1.0)) - 0.5
        )
        adaptive_weight = float(
            np.clip(
                adaptive_weight,
                0.18,
                0.68,
            )
        )

        return (
            adaptive_weight * current
            + (1.0 - adaptive_weight) * previous
        )

    def _confidence(
        self,
        *,
        points: np.ndarray,
        residual_px: float,
        height: int,
        boundary: LaneBoundary,
        continuity_score: float,
        used_fragment_merge: bool,
    ) -> float:
        ys = points[:, 1]

        roi_height = max(
            height * (1.0 - self.roi_top_ratio),
            1.0,
        )
        vertical_coverage = float(
            np.clip(
                (ys.max() - ys.min()) / roi_height,
                0.0,
                1.0,
            )
        )
        bottom_reach = float(
            np.clip(
                ys.max() / max(height - 1, 1),
                0.0,
                1.0,
            )
        )
        point_score = float(
            np.clip(
                len(points) / max(height * 0.22, 1.0),
                0.0,
                1.0,
            )
        )
        residual_score = float(
            np.clip(
                1.0 - residual_px / 22.0,
                0.0,
                1.0,
            )
        )

        confidence = (
            0.28 * vertical_coverage
            + 0.18 * bottom_reach
            + 0.22 * point_score
            + 0.20 * residual_score
            + 0.12 * continuity_score
        )

        if used_fragment_merge:
            confidence *= (
                0.96
                if boundary == LaneBoundary.CENTER
                else 0.86
            )

        return float(np.clip(confidence, 0.0, 1.0))

    def _extract_curve(
        self,
        *,
        class_map: np.ndarray,
        boundary: LaneBoundary,
        class_id: int,
    ) -> LaneCurve:
        height, width = class_map.shape
        now = time.perf_counter()

        roi_top = int(
            np.clip(
                round(height * self.roi_top_ratio),
                0,
                height - 1,
            )
        )

        previous = self._history_coefficients(
            boundary,
            now,
        )

        raw_mask = (class_map == class_id).astype(np.uint8)
        raw_mask[:roi_top, :] = 0

        cleaned_mask = self._clean_mask(
            raw_mask,
            boundary,
        )

        candidates = self._build_candidates(
            mask=cleaned_mask,
            boundary=boundary,
            previous=previous,
        )

        if boundary == LaneBoundary.RIGHT:
            selected_mask, used_merge, selection_reason = (
                self._select_right_mask(
                    candidates=candidates,
                    full_mask=cleaned_mask,
                    previous=previous,
                )
            )
        else:
            selected_mask, used_merge, selection_reason = (
                self._select_center_mask(
                    candidates=candidates,
                    full_mask=cleaned_mask,
                    roi_top=roi_top,
                    previous=previous,
                )
            )

        if selected_mask is None:
            return LaneCurve(
                boundary=boundary,
                valid=False,
                mask=cleaned_mask,
                reason=selection_reason,
            )

        row_points = self._collect_row_points(
            component_mask=selected_mask,
            boundary=boundary,
            roi_top=roi_top,
            previous=previous,
        )

        if len(row_points) < self.min_lane_points:
            return LaneCurve(
                boundary=boundary,
                valid=False,
                points=(
                    np.asarray(
                        row_points,
                        dtype=np.float32,
                    )
                    if row_points
                    else None
                ),
                mask=selected_mask,
                reason=(
                    f"too few points: {len(row_points)}; "
                    f"{selection_reason}"
                ),
            )

        points = np.asarray(
            row_points,
            dtype=np.float32,
        )

        coefficients, inliers, residual_px = self._robust_polyfit(
            points=points,
            height=height,
        )

        if coefficients is None:
            return LaneCurve(
                boundary=boundary,
                valid=False,
                points=points,
                mask=selected_mask,
                reason=f"polyfit failed; {selection_reason}",
            )

        inlier_count = int(np.count_nonzero(inliers))
        if (
            inlier_count < self.min_lane_points
            or not math.isfinite(residual_px)
        ):
            return LaneCurve(
                boundary=boundary,
                valid=False,
                coefficients=coefficients,
                points=points,
                mask=selected_mask,
                reason=(
                    f"too few inliers: {inlier_count}; "
                    f"{selection_reason}"
                ),
            )

        inlier_points = points[inliers]

        for y_ratio in (0.52, 0.58, 0.72, 0.84):
            x = float(np.polyval(coefficients, y_ratio))
            if x < -width * 0.20 or x > width * 1.20:
                return LaneCurve(
                    boundary=boundary,
                    valid=False,
                    coefficients=coefficients,
                    points=inlier_points,
                    mask=selected_mask,
                    residual_px=residual_px,
                    reason=(
                        "curve extrapolated outside frame; "
                        f"{selection_reason}"
                    ),
                )

        if previous is None:
            temporal_score = 0.55
        else:
            jump_ratio = self._curve_jump_ratio(
                previous=previous,
                current=coefficients,
                width=width,
            )
            temporal_score = float(
                math.exp(
                    -jump_ratio
                    / self.continuity_scale_ratio
                )
            )

            provisional_confidence = self._confidence(
                points=inlier_points,
                residual_px=residual_px,
                height=height,
                boundary=boundary,
                continuity_score=temporal_score,
                used_fragment_merge=used_merge,
            )
            if (
                jump_ratio > self.max_lane_jump_ratio
                and provisional_confidence < 0.78
            ):
                return LaneCurve(
                    boundary=boundary,
                    valid=False,
                    coefficients=coefficients,
                    points=inlier_points,
                    mask=selected_mask,
                    confidence=provisional_confidence,
                    residual_px=residual_px,
                    reason=(
                        f"sudden curve jump rejected: "
                        f"{jump_ratio:.3f}; {selection_reason}"
                    ),
                )

        confidence = self._confidence(
            points=inlier_points,
            residual_px=residual_px,
            height=height,
            boundary=boundary,
            continuity_score=temporal_score,
            used_fragment_merge=used_merge,
        )

        smoothed_coefficients = self._smooth_coefficients(
            previous=previous,
            current=coefficients,
            confidence=confidence,
        )

        self._previous_coefficients[boundary] = (
            smoothed_coefficients.copy()
        )
        self._previous_confidence[boundary] = confidence
        self._previous_time[boundary] = now

        return LaneCurve(
            boundary=boundary,
            valid=True,
            coefficients=smoothed_coefficients,
            points=inlier_points,
            mask=selected_mask,
            confidence=confidence,
            residual_px=residual_px,
            reason=selection_reason,
        )


__all__ = [
    "LaneBoundary",
    "LaneCurve",
    "LaneDetector",
    "LaneObservation",
]
