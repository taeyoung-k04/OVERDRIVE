#!/usr/bin/env python3
"""Fast two-boundary right-lane following.

Only these semantic classes are used for control:
- lane_center (yellow): left boundary of the target/right lane
- lane_right  (green):  right boundary of the target/right lane

The lane_left/blue class is never fitted, validated, or used as a fallback.
The desired path is intentionally biased slightly toward lane_right/green so
mechanical left pull and corner cutting do not move the vehicle toward lane 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from lane_following import LaneCurve, safe_curve_x_at


@dataclass
class TwoLineFollowingResult:
    """Right-lane control result using only center and right boundaries."""

    # left_curve is kept as a compatibility field and is always None.
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

    near_error: float = 0.0
    far_error: float = 0.0
    curvature_error: float = 0.0
    lateral_term: float = 0.0
    curvature_term: float = 0.0
    boundary_term: float = 0.0
    steering_trim: float = 0.0
    perspective_scale: float = 0.0
    neutralized: bool = False
    sign_guarded: bool = False
    target_lane_ratio: float = 0.50
    active_target_lane_ratio: float = 0.50
    curve_strength: float = 0.0
    active_lookahead_weight: float = 0.0
    active_heading_weight: float = 0.0
    lane_position_ratio: float | None = None
    used_held_center: bool = False
    used_held_right: bool = False
    pair_validated: bool = False


@dataclass
class _Candidate:
    curve: LaneCurve
    score: float


@dataclass
class _Target:
    x: float
    confidence: float
    source: str
    direct_width: float | None = None


def _extract_component_points(
    labels: np.ndarray,
    label: int,
    bbox: tuple[int, int, int, int],
    y_offset: int,
    row_step: int = 3,
) -> np.ndarray:
    """Extract median x/y points from one connected component's bounding box."""

    x0, y0, bw, bh = bbox
    sub = labels[y0 : y0 + bh, x0 : x0 + bw]
    ys, xs = np.where(sub == label)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xs = xs + x0
    ys = ys + y0
    bins = ys // max(1, row_step)
    points: list[tuple[float, float]] = []

    for bin_id in np.unique(bins):
        keep = bins == bin_id
        points.append(
            (
                float(np.median(xs[keep])),
                float(y_offset + np.median(ys[keep])),
            )
        )

    return np.asarray(points, dtype=np.float32)


def _robust_polyfit(
    points: np.ndarray,
    degree: int,
    image_width: int,
    max_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit x=f(y) while preserving real corner curvature.

    The previous implementation removed outliers with a straight-line fit before
    fitting a quadratic. On a sharp corner, valid curved points could therefore
    be rejected as noise. This routine fits the intended polynomial first and
    rejects points by polynomial residual using MAD.
    """

    if len(points) < degree + 2:
        return None

    work = points.astype(np.float64, copy=True)
    coeffs: np.ndarray | None = None

    for _ in range(max_iterations + 1):
        xs = work[:, 0]
        ys = work[:, 1]
        y_span = float(np.ptp(ys)) if len(ys) else 0.0
        if y_span > 1e-6:
            y_norm = (ys - np.min(ys)) / y_span
        else:
            y_norm = np.zeros_like(ys)
        weights = 1.0 + 2.4 * y_norm

        try:
            coeffs = np.polyfit(ys, xs, deg=degree, w=weights)
        except np.linalg.LinAlgError:
            return None

        residuals = np.abs(np.polyval(coeffs, ys) - xs)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median))) + 1e-6
        threshold = max(7.0, median + 3.2 * 1.4826 * mad, image_width * 0.012)
        keep = residuals <= threshold

        min_keep = max(degree + 2, 4)
        if int(np.sum(keep)) < min_keep or bool(np.all(keep)):
            break
        work = work[keep]

    if coeffs is None or len(work) < max(degree + 2, 4):
        return None

    final_residual = float(
        np.mean(np.abs(np.polyval(coeffs, work[:, 1]) - work[:, 0]))
    )
    return coeffs, work.astype(np.float32), final_residual


def _fit_points(
    points: np.ndarray,
    class_name: str,
    image_width: int,
    min_points: int = 7,
    relaxed_min_points: int = 4,
) -> LaneCurve | None:
    if len(points) < relaxed_min_points:
        return None

    ys = points[:, 1]
    y_span = float(np.ptp(ys)) if len(ys) else 0.0
    preferred_degree = 2 if len(points) >= min_points and y_span >= 38.0 else 1

    fitted = _robust_polyfit(points, preferred_degree, image_width)
    if fitted is None and preferred_degree == 2:
        fitted = _robust_polyfit(points, 1, image_width)
    if fitted is None:
        return None

    coeffs, inliers, residual = fitted
    inlier_ratio = float(len(inliers) / max(len(points), 1))
    point_score = min(1.0, len(inliers) / 18.0)
    span_score = float(np.clip(y_span / 105.0, 0.12, 1.0))
    residual_score = float(
        np.clip(1.0 - residual / max(7.0, image_width * 0.075), 0.08, 1.0)
    )
    confidence = (
        point_score
        * (0.28 + 0.72 * span_score)
        * residual_score
        * (0.55 + 0.45 * inlier_ratio)
    )

    return LaneCurve(
        coeffs=coeffs,
        points=inliers,
        class_name=class_name,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _curve_distance(
    curve_a: LaneCurve,
    curve_b: LaneCurve,
    sample_ys: Iterable[int],
) -> float:
    distances = [
        abs(curve_a.x_at(float(y)) - curve_b.x_at(float(y)))
        for y in sample_ys
    ]
    return float(np.mean(distances)) if distances else float("inf")


def _lane_candidates(
    class_map: np.ndarray,
    class_id: int,
    class_name: str,
    roi_top_ratio: float,
    previous_curve: LaneCurve | None,
    eval_y: int,
    lookahead_y: int,
    max_candidates: int = 5,
    min_component_area: int = 8,
    min_component_height: int = 9,
) -> list[_Candidate]:
    """Generate several plausible curves instead of committing too early."""

    h, w = class_map.shape[:2]
    start_y = int(np.clip(h * roi_top_ratio, 0, h - 1))
    mask = (class_map[start_y:h, :] == class_id).astype(np.uint8) * 255

    # Curved and dashed masks need both directional and compact closing.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    compact_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, compact_kernel, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates: list[_Candidate] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        if area < min_component_area or bh < min_component_height:
            continue

        points = _extract_component_points(
            labels=labels,
            label=label,
            bbox=(x0, y0, bw, bh),
            y_offset=start_y,
            row_step=3,
        )
        curve = _fit_points(points, class_name, w)
        if curve is None:
            continue

        coverage_score = float(np.clip(bh / max(18.0, h * 0.40), 0.12, 1.0))
        component_bottom = start_y + y0 + bh
        near_reach = float(
            np.clip(1.0 - abs(component_bottom - eval_y) / max(h * 0.35, 1.0), 0.15, 1.0)
        )
        score = curve.confidence * (0.52 + 0.30 * coverage_score + 0.18 * near_reach)

        if previous_curve is not None:
            distance = _curve_distance(
                curve,
                previous_curve,
                (lookahead_y, (lookahead_y + eval_y) // 2, eval_y),
            )
            # Temporal continuity helps, but cannot dominate a clearly better mask.
            temporal_score = float(np.exp(-distance / max(10.0, w * 0.12)))
            score *= 0.72 + 0.28 * temporal_score

        candidates.append(_Candidate(curve=curve, score=float(score)))

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: max(1, int(max_candidates))]


def _merge_aligned_candidate(
    candidates: list[_Candidate],
    class_name: str,
    image_width: int,
    eval_y: int,
    lookahead_y: int,
) -> LaneCurve | None:
    if not candidates:
        return None

    seed = candidates[0].curve
    aligned = [seed.points]
    threshold = max(9.0, image_width * 0.040)
    for item in candidates[1:]:
        distance = _curve_distance(
            seed,
            item.curve,
            (lookahead_y, (lookahead_y + eval_y) // 2, eval_y),
        )
        if distance <= threshold:
            aligned.append(item.curve.points)

    if len(aligned) > 1:
        merged = _fit_points(
            np.concatenate(aligned, axis=0), class_name, image_width
        )
        if merged is not None:
            return merged
    return seed


def _pair_score(
    center_item: _Candidate,
    right_item: _Candidate,
    image_width: int,
    eval_y: int,
    lookahead_y: int,
    learned_near_width: float | None,
    learned_far_width: float | None,
) -> float | None:
    center = center_item.curve
    right = right_item.curve
    xc_near = safe_curve_x_at(center, eval_y, image_width)
    xr_near = safe_curve_x_at(right, eval_y, image_width)
    xc_far = safe_curve_x_at(center, lookahead_y, image_width)
    xr_far = safe_curve_x_at(right, lookahead_y, image_width)
    if None in (xc_near, xr_near, xc_far, xr_far):
        return None

    near_width = float(xr_near - xc_near)
    far_width = float(xr_far - xc_far)
    if near_width <= 0 or far_width <= 0:
        return None
    if not (image_width * 0.07 <= near_width <= image_width * 0.78):
        return None
    if not (image_width * 0.025 <= far_width <= image_width * 0.60):
        return None

    perspective_ratio = far_width / max(near_width, 1.0)
    if not (0.18 <= perspective_ratio <= 1.10):
        return None

    score = float(np.sqrt(max(center_item.score, 1e-6) * max(right_item.score, 1e-6)))
    score *= 0.72 + 0.28 * float(np.exp(-abs(perspective_ratio - 0.58) / 0.38))

    if learned_near_width is not None:
        rel = abs(near_width - learned_near_width) / max(learned_near_width, 1.0)
        score *= 0.65 + 0.35 * float(np.exp(-1.8 * rel))
    if learned_far_width is not None:
        rel = abs(far_width - learned_far_width) / max(learned_far_width, 1.0)
        score *= 0.68 + 0.32 * float(np.exp(-1.6 * rel))
    return score


def _select_pair(
    center_candidates: list[_Candidate],
    right_candidates: list[_Candidate],
    image_width: int,
    eval_y: int,
    lookahead_y: int,
    learned_near_width: float | None,
    learned_far_width: float | None,
) -> tuple[LaneCurve | None, LaneCurve | None, bool]:
    """Choose yellow and green jointly so disconnected noise is not paired."""

    best: tuple[float, LaneCurve, LaneCurve] | None = None
    for center_item in center_candidates:
        for right_item in right_candidates:
            score = _pair_score(
                center_item,
                right_item,
                image_width,
                eval_y,
                lookahead_y,
                learned_near_width,
                learned_far_width,
            )
            if score is not None and (best is None or score > best[0]):
                best = (score, center_item.curve, right_item.curve)

    if best is not None:
        return best[1], best[2], True

    center = _merge_aligned_candidate(
        center_candidates, "lane_center", image_width, eval_y, lookahead_y
    )
    right = _merge_aligned_candidate(
        right_candidates, "lane_right", image_width, eval_y, lookahead_y
    )
    return center, right, False


def _held_curve(curve: LaneCurve | None, miss_count: int) -> LaneCurve | None:
    if curve is None:
        return None
    decay = 0.72 ** max(1, miss_count)
    return LaneCurve(
        coeffs=curve.coeffs.copy(),
        points=curve.points.copy(),
        class_name=curve.class_name,
        confidence=float(curve.confidence * decay),
    )


class TwoLineRightFollower:
    """Stateful right-lane follower using only yellow and green boundaries."""

    def __init__(
        self,
        center_class_id: int,
        right_class_id: int,
        roi_top_ratio: float = 0.30,
        eval_y_ratio: float = 0.82,
        lookahead_y_ratio: float = 0.58,
        camera_center_offset_px: float = 0.0,
        steering_kp: float = 1.0,
        lookahead_weight: float = 0.20,
        heading_weight: float = 0.15,
        max_abs_steering: float = 1.0,
        fallback_lane_width_px: float | None = None,
        target_smoothing: float = 0.70,
        prediction_horizon: float = 0.0,
        instant_target: bool = True,
        width_ema_alpha: float = 0.24,
        allow_single_line_fallback: bool = True,
        min_confidence: float = 0.16,
        departure_margin_ratio: float = 0.16,
        departure_gain: float = 0.55,
        target_lane_ratio: float = 0.58,
        corner_target_lane_ratio: float = 0.64,
        curve_activation_error: float = 0.025,
        curve_full_error: float = 0.10,
        corner_lookahead_weight: float = 0.42,
        corner_heading_weight: float = 0.45,
        left_departure_margin_ratio: float = 0.34,
        right_departure_margin_ratio: float = 0.12,
        left_departure_gain: float = 0.90,
        right_departure_gain: float = 0.45,
        curve_sign_override_error: float = 0.035,
        steering_neutral_error: float = 0.030,
        far_neutral_error: float = 0.045,
        curvature_neutral_error: float = 0.020,
        neutral_exit_multiplier: float = 1.60,
        curvature_term_limit: float = 0.09,
        sign_guard_error: float = 0.020,
        enable_sign_guard: bool = True,
        steering_trim: float = 0.0,
        curve_hold_frames: int = 2,
        lane_max_candidates: int = 5,
        lane_component_min_area: int = 8,
        lane_component_min_height: int = 9,
    ):
        self.center_class_id = int(center_class_id)
        self.right_class_id = int(right_class_id)
        self.roi_top_ratio = float(roi_top_ratio)
        self.eval_y_ratio = float(eval_y_ratio)
        self.lookahead_y_ratio = float(lookahead_y_ratio)
        self.camera_center_offset_px = float(camera_center_offset_px)

        self.steering_kp = float(steering_kp)
        self.lookahead_weight = float(np.clip(lookahead_weight, 0.0, 0.65))
        self.heading_weight = float(max(0.0, heading_weight))
        self.max_abs_steering = float(max(0.01, max_abs_steering))
        self.fallback_lane_width_px = fallback_lane_width_px

        self.target_smoothing = float(np.clip(target_smoothing, 0.0, 1.0))
        self.prediction_horizon = float(max(0.0, prediction_horizon))
        self.instant_target = bool(instant_target)
        self.width_ema_alpha = float(np.clip(width_ema_alpha, 0.01, 1.0))
        self.allow_single_line_fallback = bool(allow_single_line_fallback)
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))

        self.departure_margin_ratio = float(max(0.01, departure_margin_ratio))
        self.departure_gain = float(max(0.0, departure_gain))
        self.target_lane_ratio = float(np.clip(target_lane_ratio, 0.50, 0.80))
        self.corner_target_lane_ratio = float(
            np.clip(corner_target_lane_ratio, self.target_lane_ratio, 0.85)
        )
        self.curve_activation_error = float(max(0.0, curve_activation_error))
        self.curve_full_error = float(
            max(self.curve_activation_error + 1e-4, curve_full_error)
        )
        self.corner_lookahead_weight = float(
            np.clip(corner_lookahead_weight, self.lookahead_weight, 0.80)
        )
        self.corner_heading_weight = float(
            max(self.heading_weight, corner_heading_weight)
        )
        self.left_departure_margin_ratio = float(
            np.clip(left_departure_margin_ratio, 0.05, 0.60)
        )
        self.right_departure_margin_ratio = float(
            np.clip(right_departure_margin_ratio, 0.03, 0.45)
        )
        self.left_departure_gain = float(max(0.0, left_departure_gain))
        self.right_departure_gain = float(max(0.0, right_departure_gain))
        self.curve_sign_override_error = float(max(0.0, curve_sign_override_error))
        self.steering_neutral_error = float(max(0.0, steering_neutral_error))
        self.far_neutral_error = float(max(0.0, far_neutral_error))
        self.curvature_neutral_error = float(max(0.0, curvature_neutral_error))
        self.neutral_exit_multiplier = float(max(1.0, neutral_exit_multiplier))
        self.curvature_term_limit = float(max(0.0, curvature_term_limit))
        self.sign_guard_error = float(max(0.0, sign_guard_error))
        self.enable_sign_guard = bool(enable_sign_guard)
        self.steering_trim = float(steering_trim)
        self.curve_hold_frames = int(max(0, curve_hold_frames))
        self.lane_max_candidates = int(np.clip(lane_max_candidates, 1, 10))
        self.lane_component_min_area = int(max(3, lane_component_min_area))
        self.lane_component_min_height = int(max(4, lane_component_min_height))

        self.previous_center_curve: LaneCurve | None = None
        self.previous_right_curve: LaneCurve | None = None
        self.near_width_px: float | None = None
        self.far_width_px: float | None = None

        self.filtered_near_target: float | None = None
        self.filtered_far_target: float | None = None
        self.near_velocity = 0.0
        self.far_velocity = 0.0
        self.neutral_active = False
        self.center_miss_frames = 0
        self.right_miss_frames = 0

    @staticmethod
    def _ema(previous: float | None, value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return float(previous + alpha * (value - previous))

    def _default_widths(self, image_width: int, far_scale: float) -> tuple[float, float]:
        near = self.fallback_lane_width_px
        if near is None or near <= 0:
            near = image_width * 0.32
        far = max(image_width * 0.07, float(near) * far_scale)
        return float(near), float(far)

    @staticmethod
    def _width_is_plausible(width: float, image_width: int, far: bool) -> bool:
        if far:
            return image_width * 0.035 <= width <= image_width * 0.58
        return image_width * 0.07 <= width <= image_width * 0.78

    def _estimate_target(
        self,
        y: int,
        center_curve: LaneCurve | None,
        right_curve: LaneCurve | None,
        image_width: int,
        fallback_width: float,
        learned_width: float | None,
        far: bool,
        target_ratio: float,
    ) -> _Target | None:
        x_center = safe_curve_x_at(center_curve, y, image_width)
        x_right = safe_curve_x_at(right_curve, y, image_width)
        center_conf = center_curve.confidence if center_curve is not None else 0.0
        right_conf = right_curve.confidence if right_curve is not None else 0.0
        width_ref = learned_width or fallback_width

        if x_center is not None and x_right is not None and x_right > x_center:
            width = x_right - x_center
            if self._width_is_plausible(width, image_width, far):
                consistency = 1.0
                if learned_width is not None:
                    rel = abs(width - learned_width) / max(learned_width, 1.0)
                    consistency = float(np.exp(-1.8 * rel))
                confidence = (
                    np.sqrt(max(0.01, center_conf) * max(0.01, right_conf))
                    * (0.55 + 0.45 * consistency)
                )
                return _Target(
                    x=float(x_center + width * target_ratio),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    source="center+right",
                    direct_width=float(width),
                )

        if not self.allow_single_line_fallback:
            return None

        # If both exist but form an invalid width, trust only the stronger one.
        if x_center is not None and (x_right is None or center_conf >= right_conf):
            return _Target(
                x=float(x_center + width_ref * target_ratio),
                confidence=float(np.clip(center_conf * 0.52, 0.0, 1.0)),
                source="center-only fallback",
            )

        if x_right is not None:
            return _Target(
                x=float(x_right - width_ref * (1.0 - target_ratio)),
                confidence=float(np.clip(right_conf * 0.50, 0.0, 1.0)),
                source="right-only fallback",
            )

        return None

    def _filter_target(
        self,
        value: float,
        confidence: float,
        previous: float | None,
        previous_velocity: float,
        image_width: int,
    ) -> tuple[float, float, bool]:
        if previous is None:
            return float(value), 0.0, False
        if self.instant_target:
            return float(value), float(value - previous), False

        alpha = float(
            np.clip(self.target_smoothing * (0.60 + 0.65 * confidence), 0.14, 0.88)
        )
        max_jump = image_width * (0.05 + 0.06 * confidence)
        clipped = float(np.clip(value, previous - max_jump, previous + max_jump))
        filtered = float(previous + alpha * (clipped - previous))
        velocity = float(0.50 * previous_velocity + 0.50 * (filtered - previous))
        prediction = float(self.prediction_horizon * velocity * confidence)
        return filtered + prediction, velocity, abs(prediction) >= 0.5

    def _invalid_result(
        self,
        center_curve: LaneCurve | None,
        right_curve: LaneCurve | None,
        camera_center_x: float,
        eval_y: int,
        lookahead_y: int,
        detected: tuple[str, ...],
        reason: str,
    ) -> TwoLineFollowingResult:
        return TwoLineFollowingResult(
            left_curve=None,
            center_curve=center_curve,
            right_curve=right_curve,
            lane_center_x=None,
            camera_center_x=camera_center_x,
            offset_px=None,
            offset_norm=None,
            steering=None,
            eval_y=eval_y,
            valid=False,
            reason=reason,
            lookahead_y=lookahead_y,
            near_target_x=None,
            far_target_x=None,
            confidence=0.0,
            detected_lanes=detected,
            right_lane_width_px=self.near_width_px,
        )

    def update(self, class_map: np.ndarray) -> TwoLineFollowingResult:
        h, w = class_map.shape[:2]
        eval_y = int(np.clip(h * self.eval_y_ratio, 0, h - 1))
        lookahead_y = int(np.clip(h * self.lookahead_y_ratio, 0, h - 1))
        if lookahead_y >= eval_y:
            lookahead_y = max(0, eval_y - max(12, int(h * 0.12)))

        camera_center_x = (w * 0.5) + self.camera_center_offset_px
        far_scale = float(
            np.clip((lookahead_y + 1.0) / max(eval_y + 1.0, 1.0), 0.38, 0.82)
        )
        default_near_width, default_far_width = self._default_widths(w, far_scale)

        # Blue/lane_left is intentionally never read here. Generate several
        # yellow and green candidates, then choose the geometrically valid pair.
        center_candidates = _lane_candidates(
            class_map=class_map,
            class_id=self.center_class_id,
            class_name="lane_center",
            roi_top_ratio=self.roi_top_ratio,
            previous_curve=self.previous_center_curve,
            eval_y=eval_y,
            lookahead_y=lookahead_y,
            max_candidates=self.lane_max_candidates,
            min_component_area=self.lane_component_min_area,
            min_component_height=self.lane_component_min_height,
        )
        right_candidates = _lane_candidates(
            class_map=class_map,
            class_id=self.right_class_id,
            class_name="lane_right",
            roi_top_ratio=self.roi_top_ratio,
            previous_curve=self.previous_right_curve,
            eval_y=eval_y,
            lookahead_y=lookahead_y,
            max_candidates=self.lane_max_candidates,
            min_component_area=self.lane_component_min_area,
            min_component_height=self.lane_component_min_height,
        )
        center_curve, right_curve, pair_validated = _select_pair(
            center_candidates=center_candidates,
            right_candidates=right_candidates,
            image_width=w,
            eval_y=eval_y,
            lookahead_y=lookahead_y,
            learned_near_width=self.near_width_px,
            learned_far_width=self.far_width_px,
        )

        used_held_center = False
        used_held_right = False
        if center_curve is None:
            self.center_miss_frames += 1
            if self.center_miss_frames <= self.curve_hold_frames:
                center_curve = _held_curve(
                    self.previous_center_curve, self.center_miss_frames
                )
                used_held_center = center_curve is not None
        else:
            self.center_miss_frames = 0

        if right_curve is None:
            self.right_miss_frames += 1
            if self.right_miss_frames <= self.curve_hold_frames:
                right_curve = _held_curve(
                    self.previous_right_curve, self.right_miss_frames
                )
                used_held_right = right_curve is not None
        else:
            self.right_miss_frames = 0

        detected_names: list[str] = []
        if center_curve is not None:
            detected_names.append("center-held" if used_held_center else "center")
        if right_curve is not None:
            detected_names.append("right-held" if used_held_right else "right")
        detected = tuple(detected_names)

        near = self._estimate_target(
            eval_y,
            center_curve,
            right_curve,
            w,
            default_near_width,
            self.near_width_px,
            far=False,
            target_ratio=self.target_lane_ratio,
        )
        if near is None:
            return self._invalid_result(
                center_curve,
                right_curve,
                camera_center_x,
                eval_y,
                lookahead_y,
                detected,
                "yellow and green boundaries unavailable",
            )

        far = self._estimate_target(
            lookahead_y,
            center_curve,
            right_curve,
            w,
            default_far_width,
            self.far_width_px,
            far=True,
            target_ratio=self.target_lane_ratio,
        )
        if far is None:
            far = _Target(
                x=near.x,
                confidence=near.confidence * 0.65,
                source="near target reused",
            )

        if near.direct_width is not None and near.confidence >= 0.16:
            self.near_width_px = self._ema(
                self.near_width_px, near.direct_width, self.width_ema_alpha
            )
        if far.direct_width is not None and far.confidence >= 0.14:
            self.far_width_px = self._ema(
                self.far_width_px, far.direct_width, self.width_ema_alpha
            )

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
        self.filtered_near_target = float(filtered_near)
        self.filtered_far_target = float(filtered_far)

        confidence = float(np.clip(0.72 * near.confidence + 0.28 * far.confidence, 0.0, 1.0))

        near_width_ref = self.near_width_px or near.direct_width or default_near_width
        far_width_ref = self.far_width_px or far.direct_width or default_far_width
        perspective_scale = float(
            np.clip(far_width_ref / max(near_width_ref, 1.0), 0.25, 0.90)
        )

        # First estimate curvature using the normal right-biased target. A true
        # corner should move the far target away from the position explained by
        # perspective convergence alone.
        expected_far_x_base = camera_center_x + (
            filtered_near - camera_center_x
        ) * perspective_scale
        base_curvature_error = float(
            (filtered_far - expected_far_x_base) / (w * 0.5)
        )
        curve_strength = float(
            np.clip(
                (abs(base_curvature_error) - self.curve_activation_error)
                / max(
                    self.curve_full_error - self.curve_activation_error,
                    1e-6,
                ),
                0.0,
                1.0,
            )
        )

        # In a corner, keep a little more distance from the yellow center line.
        # The shift is proportional to lane width, so it works at both near and
        # look-ahead rows without introducing a fixed-pixel camera dependency.
        active_target_lane_ratio = float(
            self.target_lane_ratio
            + curve_strength
            * (self.corner_target_lane_ratio - self.target_lane_ratio)
        )
        ratio_delta = active_target_lane_ratio - self.target_lane_ratio
        filtered_near = float(filtered_near + near_width_ref * ratio_delta)
        filtered_far = float(filtered_far + far_width_ref * ratio_delta)
        self.filtered_near_target = filtered_near
        self.filtered_far_target = filtered_far

        near_error = float((filtered_near - camera_center_x) / (w * 0.5))
        far_error = float((filtered_far - camera_center_x) / (w * 0.5))
        expected_far_x = camera_center_x + (
            filtered_near - camera_center_x
        ) * perspective_scale
        curvature_error = float((filtered_far - expected_far_x) / (w * 0.5))

        # Binary a/d steering cannot use a larger magnitude, so corner handling
        # must start the correct direction earlier. Increase far-target and
        # curvature authority only when a real corner is detected.
        active_lookahead_weight = float(
            self.lookahead_weight
            + curve_strength
            * (self.corner_lookahead_weight - self.lookahead_weight)
        )
        active_heading_weight = float(
            self.heading_weight
            + curve_strength
            * (self.corner_heading_weight - self.heading_weight)
        )
        lateral_error = float(
            (1.0 - active_lookahead_weight) * near_error
            + active_lookahead_weight * far_error
        )
        lateral_term = float(self.steering_kp * lateral_error)
        raw_curvature_term = float(
            self.steering_kp * active_heading_weight * curvature_error
        )
        dynamic_limit = min(
            self.curvature_term_limit,
            max(0.018, abs(lateral_term) * 0.60 + 0.018),
        )
        curvature_term = float(
            np.clip(raw_curvature_term, -dynamic_limit, dynamic_limit)
        )

        steering_before_boundary = lateral_term + curvature_term
        sign_guarded = False
        if (
            self.enable_sign_guard
            and abs(near_error) >= self.sign_guard_error
            and steering_before_boundary * near_error < 0.0
            and not (
                curve_strength > 0.0
                and abs(curvature_error) >= self.curve_sign_override_error
            )
        ):
            curvature_term = 0.0
            steering_before_boundary = lateral_term
            sign_guarded = True

        x_center_near = safe_curve_x_at(center_curve, eval_y, w)
        x_right_near = safe_curve_x_at(right_curve, eval_y, w)
        departure_risk = False
        boundary_term = 0.0
        lane_position_ratio: float | None = None

        if (
            x_center_near is not None
            and x_right_near is not None
            and x_right_near > x_center_near
        ):
            lane_width = x_right_near - x_center_near
            lane_position_ratio = float(
                (camera_center_x - x_center_near) / max(lane_width, 1.0)
            )
            left_margin = lane_position_ratio
            right_margin = 1.0 - lane_position_ratio

            # Protect the yellow/center boundary earlier and more strongly than
            # the green boundary. Crossing yellow means entering lane 1, which
            # is the failure mode reported during cornering.
            if left_margin < self.left_departure_margin_ratio:
                severity = (
                    self.left_departure_margin_ratio - left_margin
                ) / self.left_departure_margin_ratio
                boundary_term += self.left_departure_gain * float(
                    np.clip(severity, 0.0, 1.8)
                )
                departure_risk = True
            if right_margin < self.right_departure_margin_ratio:
                severity = (
                    self.right_departure_margin_ratio - right_margin
                ) / self.right_departure_margin_ratio
                boundary_term -= self.right_departure_gain * float(
                    np.clip(severity, 0.0, 1.8)
                )
                departure_risk = True

        # Schmitt-trigger neutral zone: enter with normal thresholds, but leave
        # only after error grows meaningfully. This prevents a/d chatter at zero.
        enter_neutral = bool(
            not departure_risk
            and curve_strength < 0.10
            and abs(near_error) <= self.steering_neutral_error
            and abs(far_error) <= self.far_neutral_error
            and abs(curvature_error) <= self.curvature_neutral_error
        )
        stay_neutral = bool(
            self.neutral_active
            and not departure_risk
            and curve_strength < 0.18
            and abs(near_error)
            <= self.steering_neutral_error * self.neutral_exit_multiplier
            and abs(far_error)
            <= self.far_neutral_error * self.neutral_exit_multiplier
            and abs(curvature_error)
            <= self.curvature_neutral_error * self.neutral_exit_multiplier
        )
        self.neutral_active = enter_neutral or stay_neutral

        steering = (
            0.0
            if self.neutral_active
            else steering_before_boundary + boundary_term + self.steering_trim
        )
        steering = float(
            np.clip(steering, -self.max_abs_steering, self.max_abs_steering)
        )

        offset_px = float(filtered_near - camera_center_x)
        offset_norm = float(offset_px / (w * 0.5))

        if confidence >= 0.08:
            if center_curve is not None and not used_held_center:
                self.previous_center_curve = center_curve
            if right_curve is not None and not used_held_right:
                self.previous_right_curve = right_curve

        valid = confidence >= self.min_confidence
        reason_parts = [
            f"two-line right target ({','.join(detected) or 'none'})",
            f"conf={confidence:.2f}",
            f"near={near.source}",
            f"far={far.source}",
        ]
        reason_parts.append("pair-ok" if pair_validated else "pair-fallback")
        if used_held_center or used_held_right:
            reason_parts.append("short curve hold")
        if curve_strength > 0.0:
            reason_parts.append(f"corner={curve_strength:.2f}")
        if departure_risk:
            reason_parts.append("yellow-boundary protection")
        if self.neutral_active:
            reason_parts.append("neutral hold")
        if sign_guarded:
            reason_parts.append("sign guard")
        if not valid:
            reason_parts.append("below confidence threshold")

        return TwoLineFollowingResult(
            left_curve=None,
            center_curve=center_curve,
            right_curve=right_curve,
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
            detected_lanes=detected,
            right_lane_width_px=self.near_width_px,
            departure_risk=departure_risk,
            used_prediction=used_near_prediction or used_far_prediction,
            near_error=near_error,
            far_error=far_error,
            curvature_error=curvature_error,
            lateral_term=lateral_term,
            curvature_term=curvature_term,
            boundary_term=boundary_term,
            steering_trim=self.steering_trim,
            perspective_scale=perspective_scale,
            neutralized=self.neutral_active,
            sign_guarded=sign_guarded,
            target_lane_ratio=self.target_lane_ratio,
            active_target_lane_ratio=active_target_lane_ratio,
            curve_strength=curve_strength,
            active_lookahead_weight=active_lookahead_weight,
            active_heading_weight=active_heading_weight,
            lane_position_ratio=lane_position_ratio,
            used_held_center=used_held_center,
            used_held_right=used_held_right,
            pair_validated=pair_validated,
        )


# Compatibility aliases for older imports.
ThreeLaneRightFollower = TwoLineRightFollower
ThreeLaneFollowingResult = TwoLineFollowingResult


def compute_lane_following_right(
    class_map: np.ndarray,
    center_class_id: int,
    right_class_id: int,
    left_class_id: int | None = None,  # accepted but deliberately ignored
    roi_top_ratio: float = 0.30,
    eval_y_ratio: float = 0.82,
    lookahead_y_ratio: float = 0.58,
    camera_center_offset_px: float = 0.0,
    steering_kp: float = 1.0,
    max_abs_steering: float = 1.0,
    fallback_lane_width_px: float | None = None,
    target_lane_ratio: float = 0.58,
    corner_target_lane_ratio: float = 0.64,
) -> TwoLineFollowingResult:
    follower = TwoLineRightFollower(
        center_class_id=center_class_id,
        right_class_id=right_class_id,
        roi_top_ratio=roi_top_ratio,
        eval_y_ratio=eval_y_ratio,
        lookahead_y_ratio=lookahead_y_ratio,
        camera_center_offset_px=camera_center_offset_px,
        steering_kp=steering_kp,
        max_abs_steering=max_abs_steering,
        fallback_lane_width_px=fallback_lane_width_px,
        target_lane_ratio=target_lane_ratio,
        corner_target_lane_ratio=corner_target_lane_ratio,
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
    pts: list[list[int]] = []
    for y in np.linspace(0, h - 1, 80):
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
    result: TwoLineFollowingResult,
) -> np.ndarray:
    """Draw only yellow/center and green/right control boundaries."""

    debug = image.copy()
    h, w = debug.shape[:2]

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
            f"2-line off={result.offset_px:+.1f}px steer={result.steering:+.3f} "
            f"lat={result.lateral_term:+.3f} curv={result.curvature_term:+.3f} "
            f"bound={result.boundary_term:+.3f} ratio={result.active_target_lane_ratio:.2f}"
        )
        color = (0, 255, 0) if not result.departure_risk else (0, 165, 255)
    else:
        text = f"2-line invalid conf={result.confidence:.2f}: {result.reason}"
        color = (0, 0, 255)

    cv2.putText(
        debug,
        text,
        (12, h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )

    diag = (
        f"yellow+green only | near={result.near_error:+.3f} "
        f"far={result.far_error:+.3f} curve={result.curvature_error:+.3f} "
        f"strength={result.curve_strength:.2f} look={result.active_lookahead_weight:.2f} "
        f"pos={result.lane_position_ratio if result.lane_position_ratio is not None else -1.0:.2f}"
    )
    if result.neutralized:
        diag += " NEUTRAL-HOLD"
    if result.sign_guarded:
        diag += " SIGN-GUARD"
    cv2.putText(
        debug,
        diag,
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )

    return debug
