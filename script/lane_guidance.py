#!/usr/bin/env python3
"""Preview lane-guidance error from the BEV semantic lane map."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import cv2
import numpy as np

LANE_DETECTION_DIR = Path(__file__).resolve().parent / "lane_detection"
if str(LANE_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(LANE_DETECTION_DIR))

from lane_detection.infer_sem_class import (
    DEFAULT_WEIGHTS,
    load_semantic_model,
    make_class_overlay,
    semantic_to_class_map,
)
from lane_detection.realtime_sem_class_camera import (
    LatestFrameReader,
    draw_delay,
    draw_fps,
    keep_lane_marking_classes,
    make_classmap_canvas,
    normalize_frame_size,
    open_camera,
)
from lane_detection.utils.perspective import (
    add_perspective_args,
    apply_perspective,
    make_perspective_config,
)
from lane_detection.utils.postprocess import (
    CLASS_TO_ID,
    add_postprocess_args,
    postprocess_class_map,
)

ROI_TOP_RATIO = 0.65  # ROI box range
ROI_BOTTOM_RATIO = 0.9

FIT_TOP_RATIO = 0.18  # y-range used for lane fitting
FIT_BOTTOM_RATIO = 0.95
MIN_FIT_ROWS = 16  # Minimum valid rows required for lane fitting
MIN_RUN_PIXELS = 2  # Minimum pixels in a row for lane fitting

CONTROL_Y_RATIO = 0.78  # Y position where steering error is sampled

MAX_GUIDED_YELLOW_DELTA_RATIO = 0.08  # Max mean x-delta before using green-guided yellow.
MAX_CURVE_DELTA_RATIO = 0.05  # Max bend delta "
MAX_CURVE_RATIO = 2.0  # Max yellow bend ratio "
MIN_CURVE_SIGN_PX = 8.0  # Minimum bend magnitude before comparing curve direction.

MAX_YELLOW_ROW_SEGMENTS = 1  # Max yellow blobs allowed in one row

GAP_SMA_WINDOW = 7
MIN_VALID_GAP_RATIO = 0.18  # Minimum valid yellow-green gap
MAX_VALID_GAP_RATIO = 0.60  # Maximum valid yellow-green gap


@dataclass(frozen=True)
class LaneError:
    vehicle_x: int
    target_x: int | None
    center_lane_x: int | None
    right_lane_x: int | None
    error_px: int | None
    roi_top: int
    roi_bottom: int
    fit_top: int
    fit_bottom: int
    control_y: int
    center_fit: np.ndarray | None
    right_fit: np.ndarray | None
    guided_center_fit: np.ndarray | None
    target_fit: np.ndarray | None


class GapTracker:
    def __init__(self, window: int = GAP_SMA_WINDOW) -> None:
        self.values = deque(maxlen=window)

    def update(self, observed_gap: float | None, width: int) -> float | None:
        min_gap = width * MIN_VALID_GAP_RATIO
        max_gap = width * MAX_VALID_GAP_RATIO
        if observed_gap is not None and min_gap <= observed_gap <= max_gap:
            self.values.append(float(observed_gap))

        if not self.values:
            return None
        return float(np.mean(self.values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="onnx")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-fps", type=int, default=20)
    parser.add_argument("--no-force-size", action="store_true")
    parser.add_argument("--buffered-camera", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show-fps", action="store_true")
    add_perspective_args(parser)
    add_postprocess_args(parser)
    return parser.parse_args()


def collect_lane_points(mask: np.ndarray, top: int, bottom: int) -> tuple[np.ndarray, np.ndarray]:
    sample_xs = []
    sample_ys = []

    for y in range(top, bottom):
        xs = np.flatnonzero(mask[y])
        if xs.size < MIN_RUN_PIXELS:
            continue
        sample_xs.append(float(np.median(xs)))
        sample_ys.append(float(y))

    return np.array(sample_xs, dtype=np.float32), np.array(sample_ys, dtype=np.float32)


def count_row_segments(xs: np.ndarray) -> int:
    if xs.size == 0:
        return 0
    return int(np.count_nonzero(np.diff(xs) > 1) + 1)


def remove_split_yellow_rows(class_map: np.ndarray) -> np.ndarray:
    cleaned = class_map.copy()
    center_id = CLASS_TO_ID["lane_center"]
    for y in range(cleaned.shape[0]):
        xs = np.flatnonzero(cleaned[y] == center_id)
        if count_row_segments(xs) > MAX_YELLOW_ROW_SEGMENTS:
            cleaned[y, xs] = CLASS_TO_ID["background"]
    return cleaned


def fit_lane_curve(class_map: np.ndarray, class_id: int, top: int, bottom: int) -> np.ndarray | None:
    mask = class_map == class_id
    xs, ys = collect_lane_points(mask, top, bottom)
    if xs.size < MIN_FIT_ROWS:
        return None
    return np.polyfit(ys, xs, 2)


def evaluate_curve_x(curve: np.ndarray | None, y: int, width: int) -> int | None:
    if curve is None:
        return None
    x = float(np.polyval(curve, y))
    return int(np.clip(round(x), 0, width - 1))


def shifted_curve(curve: np.ndarray | None, x_offset: float) -> np.ndarray | None:
    if curve is None:
        return None
    shifted = curve.copy()
    shifted[2] += x_offset
    return shifted


def curve_mean_delta(first: np.ndarray, second: np.ndarray, top: int, bottom: int) -> float:
    ys = np.linspace(top, bottom - 1, 24)
    return float(np.mean(np.abs(np.polyval(first, ys) - np.polyval(second, ys))))


def estimate_yellow_green_gap(class_map: np.ndarray, top: int, bottom: int) -> float | None:
    gaps = []
    center_mask = class_map == CLASS_TO_ID["lane_center"]
    right_mask = class_map == CLASS_TO_ID["lane_right"]

    for y in range(top, bottom):
        center_xs = np.flatnonzero(center_mask[y])
        right_xs = np.flatnonzero(right_mask[y])
        if center_xs.size < MIN_RUN_PIXELS or right_xs.size < MIN_RUN_PIXELS:
            continue

        gap = float(np.median(right_xs) - np.median(center_xs))
        if gap > 0:
            gaps.append(gap)

    if len(gaps) < MIN_FIT_ROWS:
        return None
    return float(np.median(gaps))


def curve_bend_px(curve: np.ndarray, top: int, bottom: int) -> float:
    span = float(bottom - top)
    return float(curve[0] * span * span)


def should_use_guided_center(
    center_fit: np.ndarray,
    guided_center_curve: np.ndarray,
    top: int,
    bottom: int,
    width: int,
) -> bool:
    delta = curve_mean_delta(center_fit, guided_center_curve, top, bottom)
    if delta > width * MAX_GUIDED_YELLOW_DELTA_RATIO:
        return True

    center_bend = curve_bend_px(center_fit, top, bottom)
    guided_bend = curve_bend_px(guided_center_curve, top, bottom)
    if abs(center_bend - guided_bend) > width * MAX_CURVE_DELTA_RATIO:
        return True

    if abs(center_bend) > MIN_CURVE_SIGN_PX and abs(guided_bend) > MIN_CURVE_SIGN_PX:
        if np.sign(center_bend) != np.sign(guided_bend):
            return True

    guided_abs = max(abs(guided_bend), MIN_CURVE_SIGN_PX)
    if abs(center_bend) > guided_abs * MAX_CURVE_RATIO:
        return True

    return False


def select_center_curve(
    center_fit: np.ndarray | None,
    guided_center_curve: np.ndarray | None,
    top: int,
    bottom: int,
    width: int,
) -> np.ndarray | None:
    if center_fit is None:
        return guided_center_curve
    if guided_center_curve is None:
        return center_fit

    if should_use_guided_center(center_fit, guided_center_curve, top, bottom, width):
        return guided_center_curve

    return center_fit


def calculate_lane_error(class_map: np.ndarray, gap_tracker: GapTracker) -> LaneError:
    height, width = class_map.shape[:2]
    roi_top = int(np.clip(round(height * ROI_TOP_RATIO), 0, height - 1))
    roi_bottom = int(np.clip(round(height * ROI_BOTTOM_RATIO), roi_top + 1, height))
    fit_top = int(np.clip(round(height * FIT_TOP_RATIO), 0, height - 1))
    fit_bottom = int(np.clip(round(height * FIT_BOTTOM_RATIO), fit_top + 1, height))
    control_y = int(np.clip(round(height * CONTROL_Y_RATIO), 0, height - 1))
    vehicle_x = width // 2

    raw_center_fit = fit_lane_curve(class_map, CLASS_TO_ID["lane_center"], fit_top, fit_bottom)
    raw_right_fit = fit_lane_curve(class_map, CLASS_TO_ID["lane_right"], fit_top, fit_bottom)
    observed_gap_px = estimate_yellow_green_gap(class_map, fit_top, fit_bottom)
    gap_px = gap_tracker.update(observed_gap_px, width)
    guided_center_fit = shifted_curve(raw_right_fit, -gap_px) if gap_px is not None else None
    center_fit = select_center_curve(raw_center_fit, guided_center_fit, fit_top, fit_bottom, width)
    center_lane_x = evaluate_curve_x(center_fit, control_y, width)
    right_lane_x = evaluate_curve_x(raw_right_fit, control_y, width)

    target_fit = None
    target_x = None
    if center_fit is not None and gap_px is not None:
        target_fit = shifted_curve(center_fit, gap_px * 0.5)
        target_x = evaluate_curve_x(target_fit, control_y, width)

    error_px = None if target_x is None else target_x - vehicle_x
    return LaneError(
        vehicle_x=vehicle_x,
        target_x=target_x,
        center_lane_x=center_lane_x,
        right_lane_x=right_lane_x,
        error_px=error_px,
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        fit_top=fit_top,
        fit_bottom=fit_bottom,
        control_y=control_y,
        center_fit=center_fit,
        right_fit=raw_right_fit,
        guided_center_fit=guided_center_fit,
        target_fit=target_fit,
    )


def draw_vertical_line(frame: np.ndarray, x: int | None, color: tuple[int, int, int], thickness: int) -> None:
    if x is None:
        return
    height, width = frame.shape[:2]
    x = int(np.clip(x, 0, width - 1))
    cv2.line(frame, (x, 0), (x, height - 1), color, thickness, cv2.LINE_AA)


def draw_curve(
    frame: np.ndarray,
    curve: np.ndarray | None,
    top: int,
    bottom: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if curve is None:
        return

    height, width = frame.shape[:2]
    ys = np.linspace(top, bottom - 1, 48)
    points = []
    for y in ys:
        x = float(np.polyval(curve, y))
        points.append((int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))))
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_lane_error(preview: np.ndarray, lane_error: LaneError) -> None:
    width = preview.shape[1]
    cv2.rectangle(
        preview,
        (0, lane_error.roi_top),
        (width - 1, lane_error.roi_bottom),
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    draw_vertical_line(preview, lane_error.vehicle_x, (255, 120, 40), 2)
    draw_curve(preview, lane_error.center_fit, lane_error.fit_top, lane_error.fit_bottom, (0, 230, 255), 2)
    draw_curve(preview, lane_error.right_fit, lane_error.fit_top, lane_error.fit_bottom, (80, 255, 80), 2)
    draw_curve(preview, lane_error.target_fit, lane_error.fit_top, lane_error.fit_bottom, (255, 0, 255), 3)

    if lane_error.target_x is not None:
        cv2.arrowedLine(
            preview,
            (lane_error.vehicle_x, lane_error.control_y),
            (lane_error.target_x, lane_error.control_y),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.08,
        )


def main() -> None:
    args = parse_args()
    model = load_semantic_model(args.weights, args.backend)

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reader = None if args.buffered_camera else LatestFrameReader(capture)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height} @ {actual_fps:.1f} FPS",
        flush=True,
    )

    window_name = "Lane Guidance"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    previous_time = time.perf_counter()
    fps = 0.0
    gap_tracker = GapTracker()

    try:
        while True:
            if reader is not None:
                ok, frame, frame_time = reader.read_with_timestamp()
            else:
                ok, frame = capture.read()
                frame_time = time.perf_counter()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")

            if args.flip:
                frame = cv2.flip(frame, 1)
            frame = normalize_frame_size(frame, args.width, args.height, not args.no_force_size)
            perspective_config = make_perspective_config(args, frame.shape[:2])

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                task="semantic",
                verbose=False,
                stream=True,
            )
            result = next(iter(results))
            class_map = semantic_to_class_map(result.semantic_mask, frame.shape[:2])
            if args.postprocess:
                class_map = postprocess_class_map(class_map)
            class_map = keep_lane_marking_classes(class_map)
            class_map = apply_perspective(class_map, perspective_config, cv2.INTER_NEAREST)
            class_map = remove_split_yellow_rows(class_map)

            lane_error = calculate_lane_error(class_map, gap_tracker)
            classmap_canvas = make_classmap_canvas(class_map, frame.dtype)
            preview = make_class_overlay(classmap_canvas, class_map)
            draw_lane_error(preview, lane_error)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            if args.show_fps:
                draw_fps(preview, fps)
                draw_delay(preview, now - frame_time)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if reader is not None:
            reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
