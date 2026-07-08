#!/usr/bin/env python3
"""Robust right-lane follower using all three semantic lane boundaries.

Target lane:
    midpoint between lane_center and lane_right

Why all three boundaries are used:
- lane_center + lane_right are the primary right-lane measurement.
- lane_left provides a geometric consistency check and an additional fallback.
- lane widths are learned over time and used to reject or reconstruct noisy lines.
- near and look-ahead targets are combined so the controller reacts to both
  lateral offset and upcoming curvature.

This module is intentionally lightweight. The extra OpenCV/polyfit work is
normally much cheaper than neural-network inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from lane_following import (
    LaneCurve,
    remove_outlier_points,
    safe_curve_x_at,
)


@dataclass
class ThreeLaneFollowingResult:
    """Result returned by :class:`ThreeLaneRightFollower`."""

    left_curve: LaneCurve | None
    center_curve: LaneCurve | None
    right_curve: LaneCurve | None

    lane_center_x: float | None
    camera_center_x: float
    offset_px: float | None
    offset_norm: float | None
    steering: float | None

    eval_y: int
    valid: bool
    reason: str

    lookahead_y: int
    near_target_x: float | None
    far_target_x: float | None
    confidence: float
    detected_lanes: tuple[str, ...]

    left_lane_width_px: float | None = None
    right_lane_width_px: float | None = None
    departure_risk: bool = False
    used_prediction: bool = False


@dataclass
class _Candidate:
    curve: LaneCurve
    score: float


@dataclass
class _TargetEstimate:
    x: float
    confidence: float
    source: str
    direct_width: float | None = None
    left_width: float | None = None


def _extract_component_points(
    labels: np.ndarray,
    label: int,
    y_offset: int,
    row_step: int = 3,
) -> np.ndarray:
    """Extract one median center point per small y-bin without rescanning rows."""

    ys, xs = np.where(labels == label)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    bins = ys // max(1, row_step)
    unique_bins = np.unique(bins)
    points: list[tuple[float, float]] = []

    for bin_id in unique_bins:
        keep = bins == bin_id
        points.append((float(np.median(xs[keep])), float(y_offset + np.median(ys[keep]))))

    return np.asarray(points, dtype=np.float32)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Return a robust weighted median."""

    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]

    total = float(np.sum(sorted_weights))
    if total <= 1e-9:
        return float(np.median(sorted_values))

    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, total * 0.5, side="left"))
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def _fit_points(
    points: np.ndarray,
    class_name: str,
    image_width: int,
    min_points: int,
    relaxed_min_points: int,
) -> LaneCurve | None:
    """Fit a robust x=f(y) curve to already selected points."""

    if len(points) < relaxed_min_points:
        return None

    points = remove_outlier_points(points, min_keep=relaxed_min_points)
    if len(points) < relaxed_min_points:
        return None

    xs = points[:, 0]
    ys = points[:, 1]
    y_span = float(np.ptp(ys)) if len(ys) else 0.0

    degree = 2 if len(points) >= min_points and y_span >= 45.0 else 1

    if y_span > 1e-6:
        y_norm = (ys - np.min(ys)) / y_span
    else:
        y_norm = np.zeros_like(ys)

    # The lower image area matters more for immediate lateral safety.
    weights = 1.0 + 2.2 * y_norm

    try:
        coeffs = np.polyfit(ys, xs, deg=degree, w=weights)
    except np.linalg.LinAlgError:
        if degree != 2:
            return None
        try:
            coeffs = np.polyfit(ys, xs, deg=1, w=weights)
        except np.linalg.LinAlgError:
            return None

    pred_xs = np.polyval(coeffs, ys)
    mean_residual = float(np.mean(np.abs(pred_xs - xs)))

    point_score = min(1.0, len(points) / 22.0)
    span_score = float(np.clip(y_span / 120.0, 0.15, 1.0))
    residual_score = float(
        np.clip(1.0 - mean_residual / max(8.0, image_width * 0.10), 0.05, 1.0)
    )
    confidence = point_score * (0.35 + 0.65 * span_score) * residual_score

    return LaneCurve(
        coeffs=coeffs,
        points=points,
        class_name=class_name,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _curve_distance(
    curve_a: LaneCurve,
    curve_b: LaneCurve,
    sample_ys: Iterable[int],
) -> float:
    diffs = [abs(curve_a.x_at(float(y)) - curve_b.x_at(float(y))) for y in sample_ys]
    return float(np.mean(diffs)) if diffs else float("inf")


def fit_lane_curve_guided(
    class_map: np.ndarray,
    class_id: int,
    class_name: str,
    roi_top_ratio: float,
    previous_curve: LaneCurve | None,
    eval_y: int,
    lookahead_y: int,
    max_candidates: int = 4,
    min_points: int = 6,
    relaxed_min_points: int = 4,
) -> LaneCurve | None:
    """Fit a lane while rejecting disconnected semantic-mask noise.

    The old fitter takes a median over every surviving component in a row. If a
    false blob exists beside the real lane, that median can jump. Here each
    connected component is fitted separately, then the best component is chosen
    using vertical coverage, residual, and temporal proximity.
    """

    h, w = class_map.shape[:2]
    start_y = int(np.clip(h * roi_top_ratio, 0, h - 1))

    # Work only inside the driving ROI and run connected-components once.
    # This avoids the old clean_mask -> connectedComponents -> second
    # connectedComponents path, which is unnecessarily expensive.
    mask = (class_map[start_y:h, :] == class_id).astype(np.uint8) * 255
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, square_kernel, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[_Candidate] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if area < 12 or component_height < 12:
            continue

        points = _extract_component_points(
            labels=labels,
            label=label,
            y_offset=start_y,
            row_step=3,
        )
        curve = _fit_points(
            points=points,
            class_name=class_name,
            image_width=w,
            min_points=min_points,
            relaxed_min_points=relaxed_min_points,
        )
        if curve is None:
            continue

        area_score = float(np.clip(area / max(20.0, h * 0.30), 0.1, 1.0))
        score = curve.confidence * (0.70 + 0.30 * area_score)

        if previous_curve is not None:
            distance = _curve_distance(
                curve,
                previous_curve,
                sample_ys=(lookahead_y, (lookahead_y + eval_y) // 2, eval_y),
            )
            temporal_score = float(np.exp(-distance / max(8.0, w * 0.10)))
            score *= 0.60 + 0.40 * temporal_score

        candidates.append(_Candidate(curve=curve, score=score))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = candidates[:max_candidates]

    # Merge fragments that lie on the same geometric curve as the best seed.
    seed = candidates[0].curve
    aligned_points = [seed.points]
    align_threshold = max(12.0, w * 0.055)

    for item in candidates[1:]:
        distance = _curve_distance(
            seed,
            item.curve,
            sample_ys=(lookahead_y, (lookahead_y + eval_y) // 2, eval_y),
        )
        if distance <= align_threshold:
            aligned_points.append(item.curve.points)

    if len(aligned_points) > 1:
        merged = _fit_points(
            points=np.concatenate(aligned_points, axis=0),
            class_name=class_name,
            image_width=w,
            min_points=min_points,
            relaxed_min_points=relaxed_min_points,
        )
        if merged is not None:
            return merged

    return seed


class ThreeLaneRightFollower:
    """Stateful three-boundary controller for the right lane."""

    def __init__(
        self,
        left_class_id: int,
        center_class_id: int,
        right_class_id: int,
        roi_top_ratio: float = 0.30,
        eval_y_ratio: float = 0.82,
        lookahead_y_ratio: float = 0.58,
        camera_center_offset_px: float = 0.0,
        steering_kp: float = 1.0,
        lookahead_weight: float = 0.28,
        heading_weight: float = 0.32,
        max_abs_steering: float = 1.0,
        fallback_lane_width_px: float | None = None,
        target_smoothing: float = 0.45,
        prediction_horizon: float = 0.30,
        width_ema_alpha: float = 0.22,
        min_confidence: float = 0.18,
        departure_margin_ratio: float = 0.16,
        departure_gain: float = 0.55,
    ):
        self.left_class_id = left_class_id
        self.center_class_id = center_class_id
        self.right_class_id = right_class_id

        self.roi_top_ratio = roi_top_ratio
        self.eval_y_ratio = eval_y_ratio
        self.lookahead_y_ratio = lookahead_y_ratio
        self.camera_center_offset_px = camera_center_offset_px

        self.steering_kp = steering_kp
        self.lookahead_weight = lookahead_weight
        self.heading_weight = heading_weight
        self.max_abs_steering = max_abs_steering
        self.fallback_lane_width_px = fallback_lane_width_px

        self.target_smoothing = target_smoothing
        self.prediction_horizon = prediction_horizon
        self.width_ema_alpha = width_ema_alpha
        self.min_confidence = min_confidence
        self.departure_margin_ratio = departure_margin_ratio
        self.departure_gain = departure_gain

        self.previous_curves: dict[str, LaneCurve | None] = {
            "left": None,
            "center": None,
            "right": None,
        }

        self.width_near: dict[str, float | None] = {"left": None, "right": None}
        self.width_far: dict[str, float | None] = {"left": None, "right": None}
        self.width_update_count = 0

        self.filtered_near_target: float | None = None
        self.filtered_far_target: float | None = None
        self.near_velocity = 0.0
        self.far_velocity = 0.0

    @staticmethod
    def _ema(previous: float | None, value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return float(previous + alpha * (value - previous))

    def _default_widths(self, image_width: int, far_scale: float) -> tuple[float, float]:
        near = self.fallback_lane_width_px
        if near is None or near <= 0:
            near = image_width * 0.32
        far = max(image_width * 0.08, near * far_scale)
        return float(near), float(far)

    @staticmethod
    def _width_is_plausible(width: float, image_width: int, far: bool) -> bool:
        if far:
            return image_width * 0.035 <= width <= image_width * 0.58
        return image_width * 0.07 <= width <= image_width * 0.78

    def _update_width(self, store: dict[str, float | None], key: str, value: float) -> None:
        store[key] = self._ema(store[key], value, self.width_ema_alpha)

    def _geometry_score(
        self,
        left: float | None,
        center: float | None,
        right: float | None,
        image_width: int,
        far: bool,
    ) -> float:
        """Score lane order and approximate equal-width road geometry."""

        score = 1.0

        if left is not None and center is not None:
            width_left = center - left
            if width_left <= 0 or not self._width_is_plausible(width_left, image_width, far):
                return 0.0

        if center is not None and right is not None:
            width_right = right - center
            if width_right <= 0 or not self._width_is_plausible(width_right, image_width, far):
                return 0.0

        if left is not None and center is not None and right is not None:
            width_left = center - left
            width_right = right - center
            ratio = width_left / max(width_right, 1e-6)
            if ratio < 0.35 or ratio > 2.85:
                score *= 0.25
            elif ratio < 0.55 or ratio > 1.85:
                score *= 0.65

        return score

    def _estimate_target_at_y(
        self,
        y: int,
        curves: dict[str, LaneCurve | None],
        image_width: int,
        far: bool,
        default_width: float,
    ) -> _TargetEstimate | None:
        x_left = safe_curve_x_at(curves["left"], y, image_width)
        x_center = safe_curve_x_at(curves["center"], y, image_width)
        x_right = safe_curve_x_at(curves["right"], y, image_width)

        geometry_score = self._geometry_score(
            left=x_left,
            center=x_center,
            right=x_right,
            image_width=image_width,
            far=far,
        )

        conf_left = curves["left"].confidence if curves["left"] is not None else 0.0
        conf_center = curves["center"].confidence if curves["center"] is not None else 0.0
        conf_right = curves["right"].confidence if curves["right"] is not None else 0.0

        width_store = self.width_far if far else self.width_near
        width_left_est = width_store["left"] or default_width
        width_right_est = width_store["right"] or default_width

        values: list[float] = []
        weights: list[float] = []
        sources: list[str] = []

        direct_width: float | None = None
        current_left_width: float | None = None

        # Primary measurement: center + right.
        if x_center is not None and x_right is not None and x_right > x_center:
            width = x_right - x_center
            if self._width_is_plausible(width, image_width, far):
                direct_width = width
                width_consistency = 1.0
                if width_store["right"] is not None:
                    relative_error = abs(width - width_store["right"]) / max(width_store["right"], 1.0)
                    width_consistency = float(np.exp(-1.8 * relative_error))

                weight = (
                    2.8
                    * max(0.08, conf_center)
                    * max(0.08, conf_right)
                    * max(0.2, geometry_score)
                    * (0.45 + 0.55 * width_consistency)
                )
                values.append((x_center + x_right) * 0.5)
                weights.append(weight)
                sources.append("center+right")

        # Center-only virtual right boundary.
        if x_center is not None:
            values.append(x_center + width_right_est * 0.5)
            weights.append(0.75 * max(0.08, conf_center))
            sources.append("center+width")

        # Right-only virtual center boundary.
        if x_right is not None:
            values.append(x_right - width_right_est * 0.5)
            weights.append(0.70 * max(0.08, conf_right))
            sources.append("right-width")

        # Left boundary predicts the right-lane center through both lane widths.
        if x_left is not None:
            values.append(x_left + width_left_est + width_right_est * 0.5)
            weights.append(0.42 * max(0.08, conf_left))
            sources.append("left+two-widths")

        # Left + center provide a current measurement of the adjacent lane width.
        if x_left is not None and x_center is not None and x_center > x_left:
            current_left_width = x_center - x_left
            if self._width_is_plausible(current_left_width, image_width, far):
                values.append(x_center + width_right_est * 0.5)
                weights.append(
                    0.95
                    * max(0.08, conf_left)
                    * max(0.08, conf_center)
                    * max(0.25, geometry_score)
                )
                sources.append("left+center+width")

        if not values:
            return None

        # Robustly reject a far-away candidate before weighted averaging.
        median = _weighted_median(values, weights)
        rejection_threshold = max(18.0, image_width * (0.11 if far else 0.095))
        keep = [abs(value - median) <= rejection_threshold for value in values]

        kept_values = [value for value, flag in zip(values, keep) if flag]
        kept_weights = [weight for weight, flag in zip(weights, keep) if flag]
        kept_sources = [source for source, flag in zip(sources, keep) if flag]

        if not kept_values:
            kept_values, kept_weights, kept_sources = values, weights, sources

        total_weight = float(np.sum(kept_weights))
        target = float(np.average(kept_values, weights=np.maximum(kept_weights, 1e-5)))

        detected_count = sum(value is not None for value in (x_left, x_center, x_right))
        count_factor = {1: 0.52, 2: 0.78, 3: 1.0}.get(detected_count, 0.0)
        confidence = float(
            np.clip((1.0 - np.exp(-total_weight)) * count_factor * max(0.35, geometry_score), 0.0, 1.0)
        )

        return _TargetEstimate(
            x=target,
            confidence=confidence,
            source="+".join(kept_sources),
            direct_width=direct_width,
            left_width=current_left_width,
        )

    def _filter_target(
        self,
        value: float,
        confidence: float,
        previous: float | None,
        previous_velocity: float,
        image_width: int,
    ) -> tuple[float, float, bool]:
        """Confidence-adaptive EMA plus a small latency prediction."""

        if previous is None:
            return float(value), 0.0, False

        alpha = float(np.clip(self.target_smoothing * (0.55 + 0.75 * confidence), 0.12, 0.82))
        max_jump = image_width * (0.045 + 0.055 * confidence)
        clipped_value = float(np.clip(value, previous - max_jump, previous + max_jump))

        filtered = previous + alpha * (clipped_value - previous)
        velocity = 0.55 * previous_velocity + 0.45 * (filtered - previous)

        # Prediction remains deliberately small; long extrapolation amplifies mask noise.
        prediction = self.prediction_horizon * velocity * confidence
        predicted = filtered + prediction
        used_prediction = abs(prediction) >= 0.5

        return float(predicted), float(velocity), used_prediction

    def update(self, class_map: np.ndarray) -> ThreeLaneFollowingResult:
        h, w = class_map.shape[:2]

        eval_y = int(np.clip(h * self.eval_y_ratio, 0, h - 1))
        lookahead_y = int(np.clip(h * self.lookahead_y_ratio, 0, h - 1))
        if lookahead_y >= eval_y:
            lookahead_y = max(0, eval_y - max(12, int(h * 0.12)))

        camera_center_x = (w * 0.5) + self.camera_center_offset_px
        far_scale = float(np.clip((lookahead_y + 1.0) / max(eval_y + 1.0, 1.0), 0.38, 0.82))
        default_near_width, default_far_width = self._default_widths(w, far_scale)

        curves = {
            "left": fit_lane_curve_guided(
                class_map,
                self.left_class_id,
                "lane_left",
                self.roi_top_ratio,
                self.previous_curves["left"],
                eval_y,
                lookahead_y,
            ),
            "center": fit_lane_curve_guided(
                class_map,
                self.center_class_id,
                "lane_center",
                self.roi_top_ratio,
                self.previous_curves["center"],
                eval_y,
                lookahead_y,
            ),
            "right": fit_lane_curve_guided(
                class_map,
                self.right_class_id,
                "lane_right",
                self.roi_top_ratio,
                self.previous_curves["right"],
                eval_y,
                lookahead_y,
            ),
        }

        detected_lanes = tuple(name for name, curve in curves.items() if curve is not None)

        near = self._estimate_target_at_y(
            y=eval_y,
            curves=curves,
            image_width=w,
            far=False,
            default_width=default_near_width,
        )
        far = self._estimate_target_at_y(
            y=lookahead_y,
            curves=curves,
            image_width=w,
            far=True,
            default_width=default_far_width,
        )

        if near is None:
            return ThreeLaneFollowingResult(
                left_curve=curves["left"],
                center_curve=curves["center"],
                right_curve=curves["right"],
                lane_center_x=None,
                camera_center_x=camera_center_x,
                offset_px=None,
                offset_norm=None,
                steering=None,
                eval_y=eval_y,
                valid=False,
                reason="no usable right-lane target",
                lookahead_y=lookahead_y,
                near_target_x=None,
                far_target_x=None,
                confidence=0.0,
                detected_lanes=detected_lanes,
            )

        if far is None:
            far = _TargetEstimate(
                x=near.x,
                confidence=near.confidence * 0.72,
                source="near reused",
            )

        # Update lane-width memory only from physically plausible direct pairs.
        if near.direct_width is not None and near.confidence >= 0.20:
            self._update_width(self.width_near, "right", near.direct_width)
        if far.direct_width is not None and far.confidence >= 0.18:
            self._update_width(self.width_far, "right", far.direct_width)
        if near.left_width is not None and near.confidence >= 0.20:
            self._update_width(self.width_near, "left", near.left_width)
        if far.left_width is not None and far.confidence >= 0.18:
            self._update_width(self.width_far, "left", far.left_width)

        if near.direct_width is not None or near.left_width is not None:
            self.width_update_count += 1

        filtered_near, self.near_velocity, used_near_prediction = self._filter_target(
            near.x,
            near.confidence,
            self.filtered_near_target,
            self.near_velocity,
            w,
        )
        filtered_far, self.far_velocity, used_far_prediction = self._filter_target(
            far.x,
            far.confidence,
            self.filtered_far_target,
            self.far_velocity,
            w,
        )

        # Store the non-extrapolated state approximately by removing this frame's prediction.
        self.filtered_near_target = filtered_near - (
            self.prediction_horizon * self.near_velocity * near.confidence
        )
        self.filtered_far_target = filtered_far - (
            self.prediction_horizon * self.far_velocity * far.confidence
        )

        confidence = float(np.clip(0.68 * near.confidence + 0.32 * far.confidence, 0.0, 1.0))

        near_error = (filtered_near - camera_center_x) / (w * 0.5)
        far_error = (filtered_far - camera_center_x) / (w * 0.5)
        heading_error = (filtered_far - filtered_near) / max(float(eval_y - lookahead_y), 1.0)
        heading_error = float(np.clip(heading_error, -1.0, 1.0))

        steering = self.steering_kp * (
            (1.0 - self.lookahead_weight) * near_error
            + self.lookahead_weight * far_error
            + self.heading_weight * heading_error
        )

        # Boundary protection: if the camera center gets too close to either
        # right-lane boundary, add a corrective push before actual departure.
        x_center_near = safe_curve_x_at(curves["center"], eval_y, w)
        x_right_near = safe_curve_x_at(curves["right"], eval_y, w)
        departure_risk = False

        if (
            x_center_near is not None
            and x_right_near is not None
            and x_right_near > x_center_near
        ):
            lane_width = x_right_near - x_center_near
            left_margin = (camera_center_x - x_center_near) / max(lane_width, 1.0)
            right_margin = (x_right_near - camera_center_x) / max(lane_width, 1.0)

            if left_margin < self.departure_margin_ratio:
                severity = (self.departure_margin_ratio - left_margin) / self.departure_margin_ratio
                steering += self.departure_gain * float(np.clip(severity, 0.0, 1.8))
                departure_risk = True
            if right_margin < self.departure_margin_ratio:
                severity = (self.departure_margin_ratio - right_margin) / self.departure_margin_ratio
                steering -= self.departure_gain * float(np.clip(severity, 0.0, 1.8))
                departure_risk = True

        steering = float(np.clip(steering, -self.max_abs_steering, self.max_abs_steering))
        offset_px = float(filtered_near - camera_center_x)
        offset_norm = float(offset_px / (w * 0.5))

        # Keep previous curves only when the measurement has some credibility.
        if confidence >= 0.10:
            for name, curve in curves.items():
                if curve is not None:
                    self.previous_curves[name] = curve

        valid = confidence >= self.min_confidence
        reason_parts = [
            f"three-lane right target ({','.join(detected_lanes) or 'none'})",
            f"conf={confidence:.2f}",
            f"near={near.source}",
            f"far={far.source}",
        ]
        if departure_risk:
            reason_parts.append("boundary protection")

        if not valid:
            reason_parts.append("below confidence threshold")

        return ThreeLaneFollowingResult(
            left_curve=curves["left"],
            center_curve=curves["center"],
            right_curve=curves["right"],
            lane_center_x=filtered_near if valid else None,
            camera_center_x=camera_center_x,
            offset_px=offset_px if valid else None,
            offset_norm=offset_norm if valid else None,
            steering=steering if valid else None,
            eval_y=eval_y,
            valid=valid,
            reason=" | ".join(reason_parts),
            lookahead_y=lookahead_y,
            near_target_x=filtered_near,
            far_target_x=filtered_far,
            confidence=confidence,
            detected_lanes=detected_lanes,
            left_lane_width_px=self.width_near["left"],
            right_lane_width_px=self.width_near["right"],
            departure_risk=departure_risk,
            used_prediction=used_near_prediction or used_far_prediction,
        )


def compute_lane_following_right(
    class_map: np.ndarray,
    left_class_id: int,
    center_class_id: int,
    right_class_id: int,
    roi_top_ratio: float = 0.30,
    eval_y_ratio: float = 0.82,
    lookahead_y_ratio: float = 0.58,
    camera_center_offset_px: float = 0.0,
    steering_kp: float = 1.0,
    max_abs_steering: float = 1.0,
    fallback_lane_width_px: float | None = None,
) -> ThreeLaneFollowingResult:
    """Stateless compatibility wrapper.

    For real driving, instantiate :class:`ThreeLaneRightFollower` once and call
    ``update`` each new inference frame so width/curve history is preserved.
    """

    follower = ThreeLaneRightFollower(
        left_class_id=left_class_id,
        center_class_id=center_class_id,
        right_class_id=right_class_id,
        roi_top_ratio=roi_top_ratio,
        eval_y_ratio=eval_y_ratio,
        lookahead_y_ratio=lookahead_y_ratio,
        camera_center_offset_px=camera_center_offset_px,
        steering_kp=steering_kp,
        max_abs_steering=max_abs_steering,
        fallback_lane_width_px=fallback_lane_width_px,
    )
    return follower.update(class_map)


def _draw_curve(
    image: np.ndarray,
    curve: LaneCurve | None,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    if curve is None:
        return

    h, w = image.shape[:2]
    pts = []
    for y in np.linspace(0, h - 1, 90):
        x = curve.x_at(float(y))
        if np.isfinite(x) and 0 <= x < w:
            pts.append([int(x), int(y)])

    if len(pts) >= 2:
        cv2.polylines(
            image,
            [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)],
            isClosed=False,
            color=color,
            thickness=thickness,
        )


def draw_lane_following_right_debug(
    image: np.ndarray,
    result: ThreeLaneFollowingResult,
) -> np.ndarray:
    """Draw all three curves and both steering targets."""

    debug = image.copy()
    h, w = debug.shape[:2]

    _draw_curve(debug, result.left_curve, (255, 80, 40))
    _draw_curve(debug, result.center_curve, (0, 230, 255))
    _draw_curve(debug, result.right_curve, (60, 220, 60))

    camera_x = int(np.clip(result.camera_center_x, 0, w - 1))
    cv2.line(debug, (camera_x, 0), (camera_x, h - 1), (255, 255, 255), 2)
    cv2.line(debug, (0, result.eval_y), (w - 1, result.eval_y), (190, 190, 190), 1)
    cv2.line(debug, (0, result.lookahead_y), (w - 1, result.lookahead_y), (130, 130, 130), 1)

    if result.near_target_x is not None:
        near_x = int(np.clip(result.near_target_x, 0, w - 1))
        cv2.circle(debug, (near_x, result.eval_y), 7, (0, 0, 255), -1)
        cv2.arrowedLine(
            debug,
            (camera_x, result.eval_y),
            (near_x, result.eval_y),
            (0, 0, 255),
            3,
            tipLength=0.18,
        )

    if result.far_target_x is not None:
        far_x = int(np.clip(result.far_target_x, 0, w - 1))
        cv2.circle(debug, (far_x, result.lookahead_y), 6, (255, 0, 255), -1)

        if result.near_target_x is not None:
            near_x = int(np.clip(result.near_target_x, 0, w - 1))
            cv2.line(
                debug,
                (near_x, result.eval_y),
                (far_x, result.lookahead_y),
                (255, 0, 255),
                2,
            )

    if result.valid:
        text = (
            f"off={result.offset_px:+.1f}px steer={result.steering:+.3f} "
            f"conf={result.confidence:.2f} lanes={','.join(result.detected_lanes)}"
        )
        color = (0, 255, 0) if not result.departure_risk else (0, 165, 255)
    else:
        text = f"invalid conf={result.confidence:.2f}: {result.reason}"
        color = (0, 0, 255)

    cv2.putText(
        debug,
        text,
        (12, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )

    return debug
