#!/usr/bin/env python3
"""Lane center and steering calculation from semantic class map.

This file does not require any Arduino code changes.
It only converts semantic masks into:
- left lane curve
- center lane curve
- lane center
- camera-to-lane offset
- steering value

Updated version:
- Better recovery from partially detected lane masks
- More tolerant curve fitting
- Single-lane fallback when only lane_left or lane_center is detected
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneCurve:
    """Lane curve represented as x = f(y)."""

    coeffs: np.ndarray
    points: np.ndarray
    class_name: str
    confidence: float = 1.0

    def x_at(self, y: float) -> float:
        return float(np.polyval(self.coeffs, y))


@dataclass
class LaneFollowingResult:
    left_curve: LaneCurve | None
    center_curve: LaneCurve | None

    lane_center_x: float | None
    camera_center_x: float
    offset_px: float | None
    offset_norm: float | None
    steering: float | None

    eval_y: int
    valid: bool
    reason: str


def clean_mask(
    mask: np.ndarray,
    min_component_area: int = 18,
) -> np.ndarray:
    """Clean a binary lane mask without destroying thin/partial lane lines.

    Original version used OPEN first, which can erase thin lane fragments.
    This version connects broken pieces first, then removes tiny components.
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask = (mask > 0).astype(np.uint8) * 255

    # Connect vertically broken lane fragments.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, square_kernel, iterations=1)

    # Slightly thicken sparse predictions so center-point extraction becomes stable.
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            cleaned[labels == label] = 255

    return cleaned


def extract_center_points(
    mask: np.ndarray,
    roi_top_ratio: float = 0.30,
    row_step: int = 3,
    row_band: int = 5,
    min_pixels_per_band: int = 1,
) -> np.ndarray:
    """Convert a thick lane mask into thin center points.

    Instead of checking only one row at a time, this function checks a small
    band around each y coordinate. This helps when the semantic mask is sparse
    or broken.
    """

    h, _ = mask.shape[:2]

    start_y = int(h * roi_top_ratio)
    start_y = max(0, min(start_y, h - 1))

    half_band = max(1, row_band // 2)

    points = []

    for y in range(start_y, h, row_step):
        y1 = max(start_y, y - half_band)
        y2 = min(h, y + half_band + 1)

        band = mask[y1:y2, :]
        ys_local, xs = np.where(band > 0)

        if len(xs) < min_pixels_per_band:
            continue

        x_center = float(np.median(xs))
        y_center = float(y1 + np.median(ys_local))

        points.append((x_center, y_center))

    if not points:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(points, dtype=np.float32)


def remove_outlier_points(
    points: np.ndarray,
    min_keep: int = 4,
) -> np.ndarray:
    """Remove obvious outlier points using a quick robust line fit."""

    if len(points) < 8:
        return points

    xs = points[:, 0]
    ys = points[:, 1]

    try:
        line_coeffs = np.polyfit(ys, xs, deg=1)
    except np.linalg.LinAlgError:
        return points

    pred_xs = np.polyval(line_coeffs, ys)
    residuals = np.abs(xs - pred_xs)

    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual))) + 1e-6

    # Robust threshold. Keep it not too strict because lane masks are noisy.
    threshold = max(14.0, median_residual + 3.5 * 1.4826 * mad)

    keep = residuals <= threshold

    if int(np.sum(keep)) < min_keep:
        return points

    return points[keep]


def fit_lane_curve(
    class_map: np.ndarray,
    class_id: int,
    class_name: str,
    roi_top_ratio: float = 0.30,
    min_points: int = 6,
    relaxed_min_points: int = 4,
) -> LaneCurve | None:
    """Fit one lane curve from a semantic class mask.

    Polynomial form:
        x = a*y^2 + b*y + c

    If there are enough points, use degree 2.
    If there are only a few points, use degree 1 instead of failing.
    """

    h, w = class_map.shape[:2]

    raw_mask = (class_map == class_id).astype(np.uint8) * 255
    mask = clean_mask(raw_mask)

    points = extract_center_points(
        mask,
        roi_top_ratio=roi_top_ratio,
        row_step=3,
        row_band=5,
        min_pixels_per_band=1,
    )

    if len(points) < relaxed_min_points:
        return None

    points = remove_outlier_points(
        points,
        min_keep=relaxed_min_points,
    )

    if len(points) < relaxed_min_points:
        return None

    xs = points[:, 0]
    ys = points[:, 1]

    y_span = float(np.max(ys) - np.min(ys)) if len(ys) > 0 else 0.0

    # If points are enough and spread vertically, use curve.
    # If points are sparse, use line to avoid unstable quadratic fitting.
    degree = 2 if len(points) >= min_points and y_span >= 45.0 else 1

    # Give lower image area more influence because it matters more for steering.
    if y_span > 1e-6:
        y_norm = (ys - np.min(ys)) / y_span
    else:
        y_norm = np.zeros_like(ys)

    weights = 1.0 + 2.5 * y_norm

    try:
        coeffs = np.polyfit(ys, xs, deg=degree, w=weights)
    except np.linalg.LinAlgError:
        if degree == 2 and len(points) >= relaxed_min_points:
            try:
                coeffs = np.polyfit(ys, xs, deg=1, w=weights)
            except np.linalg.LinAlgError:
                return None
        else:
            return None

    pred_xs = np.polyval(coeffs, ys)
    mean_residual = float(np.mean(np.abs(pred_xs - xs))) if len(xs) else w

    point_score = min(1.0, len(points) / 18.0)
    residual_score = float(np.clip(1.0 - mean_residual / max(1.0, w * 0.12), 0.2, 1.0))
    confidence = point_score * residual_score

    return LaneCurve(
        coeffs=coeffs,
        points=points,
        class_name=class_name,
        confidence=confidence,
    )


def safe_curve_x_at(
    curve: LaneCurve | None,
    y: int,
    image_width: int,
    margin_ratio: float = 0.35,
) -> float | None:
    """Evaluate curve safely and reject extremely unrealistic x values."""

    if curve is None:
        return None

    x = curve.x_at(float(y))

    if not np.isfinite(x):
        return None

    margin = image_width * margin_ratio

    if x < -margin or x > image_width + margin:
        return None

    return float(x)


def make_result(
    left_curve: LaneCurve | None,
    center_curve: LaneCurve | None,
    lane_center_x: float | None,
    camera_center_x: float,
    eval_y: int,
    image_width: int,
    steering_kp: float,
    max_abs_steering: float,
    valid: bool,
    reason: str,
) -> LaneFollowingResult:
    """Build LaneFollowingResult and compute steering if valid."""

    if not valid or lane_center_x is None:
        return LaneFollowingResult(
            left_curve=left_curve,
            center_curve=center_curve,
            lane_center_x=None,
            camera_center_x=camera_center_x,
            offset_px=None,
            offset_norm=None,
            steering=None,
            eval_y=eval_y,
            valid=False,
            reason=reason,
        )

    lane_center_x = float(np.clip(lane_center_x, 0.0, image_width - 1.0))

    offset_px = lane_center_x - camera_center_x
    offset_norm = offset_px / (image_width / 2.0)

    steering = steering_kp * offset_norm
    steering = float(np.clip(steering, -max_abs_steering, max_abs_steering))

    return LaneFollowingResult(
        left_curve=left_curve,
        center_curve=center_curve,
        lane_center_x=lane_center_x,
        camera_center_x=camera_center_x,
        offset_px=offset_px,
        offset_norm=offset_norm,
        steering=steering,
        eval_y=eval_y,
        valid=True,
        reason=reason,
    )


def compute_lane_following(
    class_map: np.ndarray,
    left_class_id: int,
    center_class_id: int,
    roi_top_ratio: float = 0.30,
    eval_y_ratio: float = 0.85,
    camera_center_offset_px: float = 0.0,
    steering_kp: float = 1.0,
    max_abs_steering: float = 1.0,
    allow_single_lane_fallback: bool = True,
    fallback_lane_width_px: float | None = None,
    min_lane_width_px: float | None = None,
    max_lane_width_px: float | None = None,
) -> LaneFollowingResult:
    """Compute lane center and steering error.

    Main case:
        lane_center_x = midpoint between lane_left and lane_center

    Fallback case:
        If only lane_left is detected:
            virtual target = lane_left + fallback_lane_width_px / 2

        If only lane_center is detected:
            virtual target = lane_center - fallback_lane_width_px / 2
    """

    h, w = class_map.shape[:2]

    eval_y = int(h * eval_y_ratio)
    eval_y = max(0, min(eval_y, h - 1))

    camera_center_x = (w / 2.0) + camera_center_offset_px

    if fallback_lane_width_px is None or fallback_lane_width_px <= 0:
        # Approximate distance between lane_left and lane_center.
        # Tune this value for your camera angle if needed.
        fallback_lane_width_px = w * 0.32

    if min_lane_width_px is None or min_lane_width_px <= 0:
        min_lane_width_px = w * 0.08

    if max_lane_width_px is None or max_lane_width_px <= 0:
        max_lane_width_px = w * 0.75

    left_curve = fit_lane_curve(
        class_map=class_map,
        class_id=left_class_id,
        class_name="lane_left",
        roi_top_ratio=roi_top_ratio,
        min_points=6,
        relaxed_min_points=4,
    )

    center_curve = fit_lane_curve(
        class_map=class_map,
        class_id=center_class_id,
        class_name="lane_center",
        roi_top_ratio=roi_top_ratio,
        min_points=6,
        relaxed_min_points=4,
    )

    x_left = safe_curve_x_at(
        curve=left_curve,
        y=eval_y,
        image_width=w,
    )

    x_center = safe_curve_x_at(
        curve=center_curve,
        y=eval_y,
        image_width=w,
    )

    if x_left is None:
        left_curve = None

    if x_center is None:
        center_curve = None

    # Best case: both lane_left and lane_center are detected.
    if left_curve is not None and center_curve is not None and x_left is not None and x_center is not None:
        if x_left >= x_center:
            return make_result(
                left_curve=left_curve,
                center_curve=center_curve,
                lane_center_x=None,
                camera_center_x=camera_center_x,
                eval_y=eval_y,
                image_width=w,
                steering_kp=steering_kp,
                max_abs_steering=max_abs_steering,
                valid=False,
                reason=f"invalid lane order: x_left={x_left:.1f}, x_center={x_center:.1f}",
            )

        lane_width_px = x_center - x_left

        if lane_width_px < min_lane_width_px or lane_width_px > max_lane_width_px:
            return make_result(
                left_curve=left_curve,
                center_curve=center_curve,
                lane_center_x=None,
                camera_center_x=camera_center_x,
                eval_y=eval_y,
                image_width=w,
                steering_kp=steering_kp,
                max_abs_steering=max_abs_steering,
                valid=False,
                reason=f"unrealistic lane width: {lane_width_px:.1f}px",
            )

        lane_center_x = (x_left + x_center) / 2.0

        return make_result(
            left_curve=left_curve,
            center_curve=center_curve,
            lane_center_x=lane_center_x,
            camera_center_x=camera_center_x,
            eval_y=eval_y,
            image_width=w,
            steering_kp=steering_kp,
            max_abs_steering=max_abs_steering,
            valid=True,
            reason="ok",
        )

    # Fallback case 1: only lane_left is detected.
    if allow_single_lane_fallback and left_curve is not None and x_left is not None:
        lane_center_x = x_left + (fallback_lane_width_px / 2.0)

        return make_result(
            left_curve=left_curve,
            center_curve=None,
            lane_center_x=lane_center_x,
            camera_center_x=camera_center_x,
            eval_y=eval_y,
            image_width=w,
            steering_kp=steering_kp,
            max_abs_steering=max_abs_steering,
            valid=True,
            reason="fallback: lane_left only",
        )

    # Fallback case 2: only lane_center is detected.
    if allow_single_lane_fallback and center_curve is not None and x_center is not None:
        lane_center_x = x_center - (fallback_lane_width_px / 2.0)

        return make_result(
            left_curve=None,
            center_curve=center_curve,
            lane_center_x=lane_center_x,
            camera_center_x=camera_center_x,
            eval_y=eval_y,
            image_width=w,
            steering_kp=steering_kp,
            max_abs_steering=max_abs_steering,
            valid=True,
            reason="fallback: lane_center only",
        )

    if left_curve is None and center_curve is None:
        reason = "lane_left and lane_center were not detected"
    elif left_curve is None:
        reason = "lane_left was not detected"
    else:
        reason = "lane_center was not detected"

    return make_result(
        left_curve=left_curve,
        center_curve=center_curve,
        lane_center_x=None,
        camera_center_x=camera_center_x,
        eval_y=eval_y,
        image_width=w,
        steering_kp=steering_kp,
        max_abs_steering=max_abs_steering,
        valid=False,
        reason=reason,
    )


def draw_curve(
    image: np.ndarray,
    curve: LaneCurve | None,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw a fitted lane curve."""

    if curve is None:
        return

    h, w = image.shape[:2]

    points = []

    for y in np.linspace(0, h - 1, 80):
        x = curve.x_at(y)

        if 0 <= x < w:
            points.append([int(x), int(y)])

    if len(points) >= 2:
        points = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)


def draw_lane_following_debug(
    image: np.ndarray,
    result: LaneFollowingResult,
) -> np.ndarray:
    """Draw lane following debug information."""

    debug = image.copy()
    h, w = debug.shape[:2]

    draw_curve(debug, result.left_curve, color=(255, 80, 40))
    draw_curve(debug, result.center_curve, color=(0, 230, 255))

    camera_x = int(result.camera_center_x)

    # Camera center: white vertical line
    cv2.line(debug, (camera_x, 0), (camera_x, h - 1), (255, 255, 255), 2)

    # Evaluation row: gray horizontal line
    cv2.line(debug, (0, result.eval_y), (w - 1, result.eval_y), (180, 180, 180), 1)

    if result.valid and result.lane_center_x is not None:
        lane_x = int(result.lane_center_x)

        # Desired lane center: red vertical line
        cv2.line(debug, (lane_x, 0), (lane_x, h - 1), (0, 0, 255), 2)

        cv2.arrowedLine(
            debug,
            (camera_x, result.eval_y),
            (lane_x, result.eval_y),
            (0, 0, 255),
            3,
            tipLength=0.2,
        )

        text = (
            f"offset={result.offset_px:+.1f}px "
            f"norm={result.offset_norm:+.3f} "
            f"steer={result.steering:+.3f} "
            f"{result.reason}"
        )
        color = (0, 255, 0)

    else:
        text = f"invalid: {result.reason}"
        color = (0, 0, 255)

    cv2.putText(
        debug,
        text,
        (12, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return debug