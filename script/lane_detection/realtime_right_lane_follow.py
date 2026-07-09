#!/usr/bin/env python3
"""Real-time right-lane following for a DC-motor steering Arduino car.

The controller uses the semantic ``lane_right`` class (green lane marking), fits
its curve, creates a virtual path to the left of that boundary, and computes a
continuous steering correction in the range -1.0..+1.0.

Serial protocol used by the matching Arduino sketch:
    C,<steering_milli>,<drive_pwm>\n
Examples:
    C,-350,90     # left correction, forward PWM 90
    C,420,75      # right correction, forward PWM 75
    X             # immediate stop

A background serial thread repeats the newest command independently of inference
FPS, so the Arduino watchdog remains fed even when ONNX inference is slow.
"""

from __future__ import annotations

#   아두이노 업로드 해야됨 : lane_follow_dc_steering.ino
#
#   실행 명령:
#   python .\script\lane_detection\realtime_right_lane_follow.py `
#   --weights .\runs\semantic\yolo_lane_sem_class\train_cpu_640_yolo26n_ade20k\weights\best.onnx `
#   --backend onnx `
#   --camera 1 `
#   --width 640 `
#   --height 360 `
#   --imgsz 640 `
#   --show-fps
import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from infer_sem_class import (
    CLASS_TO_ID,
    DEFAULT_WEIGHTS,
    add_postprocess_args,
    load_postprocess_config,
    load_semantic_model,
    make_class_overlay,
    postprocess_class_map,
    semantic_to_class_map,
)


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model / camera
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-force-size", action="store_true")
    parser.add_argument("--buffered-camera", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--preview", choices=("overlay", "debug"), default="debug")
    add_postprocess_args(parser)

    # Lane extraction
    parser.add_argument(
        "--control-source",
        choices=("raw", "processed"),
        default="raw",
        help="Use raw or postprocessed class map for steering. Raw is often safer when road postprocessing removes a valid right lane.",
    )
    parser.add_argument("--roi-top-ratio", type=float, default=0.38)
    parser.add_argument("--min-component-area", type=int, default=90)
    parser.add_argument("--min-lane-points", type=int, default=18)
    parser.add_argument("--min-lane-confidence", type=float, default=0.18)

    # Virtual path geometry
    parser.add_argument(
        "--vehicle-x-ratio",
        type=float,
        default=0.50,
        help="Vehicle forward-axis x position as a fraction of image width. Calibrate this to the camera mount; use --right-offset-ratio for lateral bias.",
    )
    parser.add_argument("--near-y-ratio", type=float, default=0.84)
    parser.add_argument("--far-y-ratio", type=float, default=0.58)
    parser.add_argument(
        "--right-offset-ratio",
        type=float,
        default=0.22,
        help="Desired distance left of the right lane at near-y, divided by image width.",
    )
    parser.add_argument(
        "--vanishing-y-ratio",
        type=float,
        default=0.31,
        help="Approximate vanishing-point y ratio used to scale the lane offset with perspective.",
    )

    # Steering controller
    parser.add_argument("--kp", type=float, default=1.15)
    parser.add_argument("--kd", type=float, default=0.18)
    parser.add_argument("--near-weight", type=float, default=0.65)
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument(
        "--adaptive-lookahead",
        dest="adaptive_lookahead",
        action="store_true",
        help="Increase far-point influence automatically on curves (default).",
    )
    adaptive_group.add_argument(
        "--no-adaptive-lookahead",
        dest="adaptive_lookahead",
        action="store_false",
        help="Disable automatic far-point weighting on curves.",
    )
    parser.set_defaults(adaptive_lookahead=True)
    parser.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--steering-deadband", type=float, default=0.015)
    parser.add_argument(
        "--new-command-weight",
        type=float,
        default=0.78,
        help="Weight of the newest command. Higher reacts faster; lower is smoother.",
    )
    parser.add_argument("--max-command-change", type=float, default=0.24)
    parser.add_argument("--lost-hold-frames", type=int, default=2)
    parser.add_argument("--lost-stop-frames", type=int, default=8)

    # Drive-speed conversion. Steering remains normalized (-1.0..+1.0)
    # and is converted to -1000..+1000 only at the serial boundary.
    parser.add_argument("--speed-straight", type=int, default=100)
    parser.add_argument("--speed-turn", type=int, default=78)
    parser.add_argument("--speed-min", type=int, default=65)
    parser.add_argument(
        "--constant-speed",
        type=int,
        default=None,
        help="Use a fixed forward PWM (0..255) instead of slowing on turns.",
    )

    # Arduino serial. The matching sketch uses C,<steering>,<speed> and X.
    parser.add_argument(
        "--arduino-port",
        default=None,
        help="For example COM6 or /dev/ttyACM0. Omit for vision-only mode.",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--serial-timeout", type=float, default=0.10)
    parser.add_argument("--arduino-reset-wait", type=float, default=1.8)
    parser.add_argument(
        "--command-rate",
        type=float,
        default=20.0,
        help="Background serial command rate in Hz. Keep above the Arduino watchdog rate.",
    )
    parser.add_argument(
        "--steering-command-scale",
        type=int,
        default=1000,
        help="Convert normalized steering to this integer range for Arduino.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    errors: list[str] = []
    for backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            capture.set(cv2.CAP_PROP_FPS, fps)

        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                return capture
            errors.append(f"backend={backend}: opened but frame read failed")
        else:
            errors.append(f"backend={backend}: open failed")
        capture.release()

    raise RuntimeError(
        f"Could not open camera index {camera_index}. " + "; ".join(errors)
    )


class LatestFrameReader:
    """Continuously drain the camera and expose only the most recent frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self.condition = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.stopped = False
        self.failed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            with self.condition:
                if self.stopped:
                    return

            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.failed = True
                    self.stopped = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.condition.notify_all()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self.condition:
            if self.frame is None and not self.stopped:
                self.condition.wait(timeout=2.0)
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)


def normalize_frame_size(
    frame: np.ndarray,
    width: int,
    height: int,
    force_size: bool,
) -> np.ndarray:
    if not force_size or width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# -----------------------------------------------------------------------------
# Right-lane extraction
# -----------------------------------------------------------------------------


@dataclass
class RightLaneObservation:
    valid: bool
    coefficients: Optional[np.ndarray] = None  # x = a*y_norm^2 + b*y_norm + c
    points: Optional[np.ndarray] = None  # shape (N, 2), columns x/y
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    residual_px: float = math.inf
    reason: str = ""


def _clean_lane_mask(mask: np.ndarray) -> np.ndarray:
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel5)
    return cleaned


def _select_right_lane_component(
    mask: np.ndarray,
    min_component_area: int,
) -> Optional[np.ndarray]:
    height, width = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_label: Optional[int] = None
    best_score = -math.inf

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        component = labels == label
        ys, xs = np.nonzero(component)
        if xs.size == 0:
            continue

        x_median = float(np.median(xs))
        y_min = float(ys.min())
        y_max = float(ys.max())
        vertical_span = max(1.0, y_max - y_min)

        # The right boundary should normally be in the right half and extend
        # toward the bottom of the frame. These are soft scores, not hard cuts.
        rightness = np.clip(x_median / max(width - 1, 1), 0.0, 1.0)
        bottomness = np.clip(y_max / max(height - 1, 1), 0.0, 1.0)
        span_score = np.clip(vertical_span / max(height * 0.45, 1.0), 0.0, 1.0)

        # Penalize components far into the left side, where center/left noise
        # is more likely to be mistaken for the right boundary.
        left_penalty = 0.35 if x_median < width * 0.42 else 1.0

        score = (
            math.log1p(area)
            * (0.40 + 0.60 * rightness)
            * (0.35 + 0.65 * bottomness)
            * (0.35 + 0.65 * span_score)
            * left_penalty
        )

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None
    return (labels == best_label).astype(np.uint8)


def extract_right_lane_curve(
    class_map: np.ndarray,
    lane_right_id: int,
    roi_top_ratio: float,
    min_component_area: int,
    min_lane_points: int,
) -> RightLaneObservation:
    height, width = class_map.shape[:2]
    roi_top = int(np.clip(round(height * roi_top_ratio), 0, height - 1))

    mask = (class_map == lane_right_id).astype(np.uint8)
    mask[:roi_top, :] = 0
    mask = _clean_lane_mask(mask)

    component = _select_right_lane_component(mask, min_component_area)
    if component is None:
        return RightLaneObservation(valid=False, mask=mask, reason="no right-lane component")

    row_points: list[tuple[float, float]] = []
    # Sampling every second row reduces duplicated information without losing
    # the curve shape at typical 360p/540p resolutions.
    for y in range(roi_top, height, 2):
        xs = np.flatnonzero(component[y])
        if xs.size == 0:
            continue
        row_points.append((float(np.median(xs)), float(y)))

    if len(row_points) < min_lane_points:
        return RightLaneObservation(
            valid=False,
            points=np.asarray(row_points, dtype=np.float32) if row_points else None,
            mask=component,
            reason=f"too few points: {len(row_points)}",
        )

    points = np.asarray(row_points, dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    y_norm = ys / max(height - 1, 1)

    try:
        coefficients = np.polyfit(y_norm, xs, 2)
    except (np.linalg.LinAlgError, ValueError):
        return RightLaneObservation(valid=False, points=points, mask=component, reason="polyfit failed")

    # Two robust-refit passes remove isolated segmentation blobs.
    inliers = np.ones(len(points), dtype=bool)
    for _ in range(2):
        predicted = np.polyval(coefficients, y_norm)
        residuals = np.abs(xs - predicted)
        median_residual = float(np.median(residuals[inliers])) if np.any(inliers) else math.inf
        robust_threshold = float(np.clip(2.8 * median_residual + 3.0, 6.0, 24.0))
        inliers = residuals <= robust_threshold
        if int(np.count_nonzero(inliers)) < min_lane_points:
            break
        try:
            coefficients = np.polyfit(y_norm[inliers], xs[inliers], 2)
        except (np.linalg.LinAlgError, ValueError):
            break

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < min_lane_points:
        return RightLaneObservation(
            valid=False,
            coefficients=coefficients,
            points=points,
            mask=component,
            reason=f"too few inliers: {inlier_count}",
        )

    points = points[inliers]
    xs = points[:, 0]
    ys = points[:, 1]
    y_norm = ys / max(height - 1, 1)
    predicted = np.polyval(coefficients, y_norm)
    residual_px = float(np.mean(np.abs(xs - predicted)))

    vertical_coverage = float((ys.max() - ys.min()) / max(height * (1.0 - roi_top_ratio), 1.0))
    point_score = float(np.clip(len(points) / max(height * 0.22, 1.0), 0.0, 1.0))
    residual_score = float(np.clip(1.0 - residual_px / 22.0, 0.0, 1.0))
    confidence = float(
        np.clip(0.45 * vertical_coverage + 0.30 * point_score + 0.25 * residual_score, 0.0, 1.0)
    )

    # Reject obviously impossible curves at the control rows.
    for y_ratio in (0.58, 0.84):
        x = float(np.polyval(coefficients, y_ratio))
        if x < -width * 0.20 or x > width * 1.20:
            return RightLaneObservation(
                valid=False,
                coefficients=coefficients,
                points=points,
                mask=component,
                confidence=confidence,
                residual_px=residual_px,
                reason="curve extrapolated outside frame",
            )

    return RightLaneObservation(
        valid=True,
        coefficients=coefficients,
        points=points,
        mask=component,
        confidence=confidence,
        residual_px=residual_px,
        reason="ok",
    )


def evaluate_curve_x(coefficients: np.ndarray, y_ratio: float) -> float:
    return float(np.polyval(coefficients, y_ratio))


# -----------------------------------------------------------------------------
# Steering controller
# -----------------------------------------------------------------------------


@dataclass
class ControlOutput:
    valid: bool
    steering: float
    steering_command: int
    speed: int
    confidence: float
    lost_frames: int
    near_error: float = 0.0
    far_error: float = 0.0
    combined_error: float = 0.0
    derivative: float = 0.0
    curvature: float = 0.0
    vehicle_x: float = 0.0
    green_near_x: float = 0.0
    green_far_x: float = 0.0
    target_near_x: float = 0.0
    target_far_x: float = 0.0
    near_weight: float = 0.0
    reason: str = ""


class RightLaneFollower:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.previous_error = 0.0
        self.previous_steering = 0.0
        self.lost_frames = 0

    def reset(self) -> None:
        """Reset temporal controller state before a new driving run."""
        self.previous_error = 0.0
        self.previous_steering = 0.0
        self.lost_frames = 0

    def _offset_at_y(self, width: int, y_ratio: float) -> float:
        near_offset = width * self.args.right_offset_ratio
        denominator = self.args.near_y_ratio - self.args.vanishing_y_ratio
        if denominator <= 1e-6:
            return near_offset

        scale = (y_ratio - self.args.vanishing_y_ratio) / denominator
        scale = float(np.clip(scale, 0.08, 1.25))
        return near_offset * scale

    def _steering_to_command(self, steering: float) -> int:
        scale = max(1, int(self.args.steering_command_scale))
        return int(np.clip(round(steering * scale), -scale, scale))

    def _steering_to_speed(self, steering: float, confidence: float) -> int:
        if self.args.constant_speed is not None:
            return max(0, int(self.args.constant_speed))

        turn_amount = float(np.clip(abs(steering), 0.0, 1.0))
        speed = self.args.speed_straight + turn_amount * (
            self.args.speed_turn - self.args.speed_straight
        )

        # Low-confidence frames run more slowly instead of making a full-speed
        # correction based on uncertain segmentation.
        confidence_scale = float(np.clip(0.70 + 0.30 * confidence, 0.55, 1.0))
        speed *= confidence_scale
        return int(max(self.args.speed_min, round(speed)))

    def _lost_output(self, observation: RightLaneObservation) -> ControlOutput:
        self.lost_frames += 1

        if self.lost_frames <= self.args.lost_hold_frames:
            steering = self.previous_steering
            speed = max(self.args.speed_min, self.args.speed_turn)
            reason = "temporary lane loss: holding steering"
        elif self.lost_frames < self.args.lost_stop_frames:
            steering = self.previous_steering * 0.72
            speed = self.args.speed_min
            reason = "lane loss: returning toward center"
        else:
            steering = 0.0
            speed = 0
            reason = "lane lost: safety stop"

        self.previous_steering = steering
        return ControlOutput(
            valid=False,
            steering=steering,
            steering_command=self._steering_to_command(steering),
            speed=speed,
            confidence=observation.confidence,
            lost_frames=self.lost_frames,
            reason=f"{reason} ({observation.reason})",
        )

    def compute(
        self,
        observation: RightLaneObservation,
        frame_shape: tuple[int, int],
    ) -> ControlOutput:
        height, width = frame_shape[:2]

        if (
            not observation.valid
            or observation.coefficients is None
            or observation.confidence < self.args.min_lane_confidence
        ):
            return self._lost_output(observation)

        self.lost_frames = 0
        coefficients = observation.coefficients

        green_near_x = evaluate_curve_x(coefficients, self.args.near_y_ratio)
        green_far_x = evaluate_curve_x(coefficients, self.args.far_y_ratio)

        target_near_x = green_near_x - self._offset_at_y(width, self.args.near_y_ratio)
        target_far_x = green_far_x - self._offset_at_y(width, self.args.far_y_ratio)
        vehicle_x = width * self.args.vehicle_x_ratio

        normalization = max(width * 0.5, 1.0)
        near_error = (target_near_x - vehicle_x) / normalization
        far_error = (target_far_x - vehicle_x) / normalization

        # Approximate curve strength from how much the detected boundary moves
        # horizontally between the far and near rows.
        curvature = abs(green_near_x - green_far_x) / max(width, 1)

        near_weight = float(np.clip(self.args.near_weight, 0.05, 0.95))
        if self.args.adaptive_lookahead:
            # On a curve, shift up to 20 percentage points toward the far point
            # so steering begins earlier.
            far_bonus = float(np.clip(curvature * 1.8, 0.0, 0.20))
            near_weight = float(np.clip(near_weight - far_bonus, 0.42, 0.85))
        far_weight = 1.0 - near_weight

        combined_error = near_weight * near_error + far_weight * far_error
        derivative = combined_error - self.previous_error

        raw_steering = self.args.steering_sign * (
            self.args.kp * combined_error + self.args.kd * derivative
        )

        if abs(raw_steering) < self.args.steering_deadband:
            raw_steering = 0.0
        raw_steering = float(np.clip(raw_steering, -1.0, 1.0))

        new_weight = float(np.clip(self.args.new_command_weight, 0.0, 1.0))
        steering = new_weight * raw_steering + (1.0 - new_weight) * self.previous_steering

        max_change = max(0.0, float(self.args.max_command_change))
        steering = float(
            np.clip(
                steering,
                self.previous_steering - max_change,
                self.previous_steering + max_change,
            )
        )
        steering = float(np.clip(steering, -1.0, 1.0))

        steering_command = self._steering_to_command(steering)
        speed = self._steering_to_speed(steering, observation.confidence)

        self.previous_error = combined_error
        self.previous_steering = steering

        return ControlOutput(
            valid=True,
            steering=steering,
            steering_command=steering_command,
            speed=speed,
            confidence=observation.confidence,
            lost_frames=0,
            near_error=near_error,
            far_error=far_error,
            combined_error=combined_error,
            derivative=derivative,
            curvature=curvature,
            vehicle_x=vehicle_x,
            green_near_x=green_near_x,
            green_far_x=green_far_x,
            target_near_x=target_near_x,
            target_far_x=target_far_x,
            near_weight=near_weight,
            reason="ok",
        )


# -----------------------------------------------------------------------------
# Arduino serial output
# -----------------------------------------------------------------------------


def format_arduino_command(steering_command: int, speed: int) -> bytes:
    """Encode the protocol shared with the matching Arduino sketch."""
    return f"C,{int(steering_command)},{int(speed)}\n".encode("ascii")


class ArduinoSender:
    """Repeat the latest command in a background thread.

    ONNX inference can occasionally take longer than the Arduino watchdog. The
    writer thread therefore transmits at a fixed rate rather than only once per
    camera frame.
    """

    def __init__(
        self,
        port: Optional[str],
        baud: int,
        timeout: float,
        reset_wait: float,
        command_rate: float,
        steering_scale: int,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self.reset_wait = float(reset_wait)
        self.command_rate = max(1.0, float(command_rate))
        self.steering_scale = max(1, int(steering_scale))

        self.serial = None
        self.state_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.latest_command = (0, 0)
        self.driving_active = False
        self.thread_stop = threading.Event()
        self.writer_thread: Optional[threading.Thread] = None

        if not port:
            print(
                "No --arduino-port supplied; SPACE toggles visual simulation only.",
                flush=True,
            )
        else:
            print(
                f"Arduino ready on {port}. Press SPACE to connect and start.",
                flush=True,
            )

    @property
    def configured(self) -> bool:
        return bool(self.port)

    @property
    def enabled(self) -> bool:
        return self.serial is not None

    def connect(self) -> bool:
        if self.serial is not None:
            return True
        if not self.port:
            return False

        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Run: python -m pip install pyserial"
            ) from exc

        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not open Arduino port {self.port}: {exc}") from exc

        if self.reset_wait > 0:
            time.sleep(self.reset_wait)

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.thread_stop.clear()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.emergency_stop()
        print(
            f"Arduino connected: {self.port} @ {self.baud}, "
            f"protocol=C,steering,speed @ {self.command_rate:.1f} Hz",
            flush=True,
        )
        return True

    def _write(self, payload: bytes) -> None:
        if self.serial is None:
            return
        with self.write_lock:
            try:
                self.serial.write(payload)
            except Exception as exc:
                raise RuntimeError(f"Arduino serial write failed: {exc}") from exc

    def _writer_loop(self) -> None:
        period = 1.0 / self.command_rate
        next_send = time.perf_counter()
        while not self.thread_stop.is_set():
            with self.state_lock:
                active = self.driving_active
                steering_command, speed = self.latest_command

            if active and self.serial is not None:
                try:
                    self._write(format_arduino_command(steering_command, speed))
                except RuntimeError as exc:
                    print(f"SERIAL ERROR: {exc}", file=sys.stderr, flush=True)
                    with self.state_lock:
                        self.driving_active = False
                    try:
                        self._write(b"X\n")
                    except Exception:
                        pass

            next_send += period
            delay = next_send - time.perf_counter()
            if delay <= 0:
                next_send = time.perf_counter()
                delay = 0.001
            self.thread_stop.wait(delay)

    def start_driving(self) -> None:
        with self.state_lock:
            self.latest_command = (0, 0)
            self.driving_active = True

    def update_command(self, steering: float, speed: int) -> None:
        scale = self.steering_scale
        steering_command = int(np.clip(round(float(steering) * scale), -scale, scale))
        speed_command = int(np.clip(round(speed), 0, 255))
        with self.state_lock:
            self.latest_command = (steering_command, speed_command)

    def emergency_stop(self) -> None:
        with self.state_lock:
            self.driving_active = False
            self.latest_command = (0, 0)
        if self.serial is not None:
            try:
                self._write(b"X\n")
                with self.write_lock:
                    self.serial.flush()
            except Exception as exc:
                print(f"WARNING: failed to send emergency stop: {exc}", file=sys.stderr)

    def close(self) -> None:
        self.emergency_stop()
        self.thread_stop.set()
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=1.0)
            self.writer_thread = None
        if self.serial is not None:
            self.serial.close()
            self.serial = None


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def _safe_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(round(x), 0, width - 1)),
        int(np.clip(round(y), 0, height - 1)),
    )


def draw_steering_arrow(
    preview: np.ndarray,
    steering: float,
    vehicle_x: int,
    driving_enabled: bool,
) -> None:
    """Draw a center-referenced arrow showing the computed steering direction."""
    height, width = preview.shape[:2]
    steering = float(np.clip(steering, -1.0, 1.0))

    start = (vehicle_x, height - 82)
    arrow_height = max(76, int(height * 0.24))
    horizontal_range = max(70, int(width * 0.25))
    end_x = int(np.clip(vehicle_x + steering * horizontal_range, 8, width - 9))
    end_y = max(8, start[1] - arrow_height)

    if steering > 0.06:
        direction = "RIGHT"
    elif steering < -0.06:
        direction = "LEFT"
    else:
        direction = "STRAIGHT"

    # Green while commands are being sent; orange while showing a preview only.
    arrow_color = (80, 255, 80) if driving_enabled else (0, 190, 255)
    cv2.arrowedLine(
        preview,
        start,
        (end_x, end_y),
        arrow_color,
        7,
        cv2.LINE_AA,
        tipLength=0.22,
    )
    cv2.circle(preview, start, 8, (255, 255, 255), -1, cv2.LINE_AA)

    label = ("STEERING " if driving_enabled else "PLANNED ") + direction
    text_size, _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    )
    label_x = int(np.clip(end_x - text_size[0] // 2, 8, width - text_size[0] - 8))
    label_y = max(28, end_y - 12)
    cv2.putText(
        preview,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        arrow_color,
        2,
        cv2.LINE_AA,
    )


def draw_debug(
    preview: np.ndarray,
    observation: RightLaneObservation,
    control: ControlOutput,
    args: argparse.Namespace,
    driving_enabled: bool,
    arduino_connected: bool,
    arduino_configured: bool,
) -> None:
    height, width = preview.shape[:2]
    roi_y = int(height * args.roi_top_ratio)
    cv2.line(preview, (0, roi_y), (width - 1, roi_y), (90, 90, 90), 1)

    if observation.points is not None:
        for x, y in observation.points[:: max(1, len(observation.points) // 80)]:
            cv2.circle(preview, _safe_point(x, y, width, height), 2, (0, 255, 0), -1)

    if observation.coefficients is not None:
        actual_points: list[tuple[int, int]] = []
        target_points: list[tuple[int, int]] = []
        for y_ratio in np.linspace(args.roi_top_ratio, 0.98, 70):
            y = y_ratio * height
            green_x = evaluate_curve_x(observation.coefficients, float(y_ratio))

            near_offset = width * args.right_offset_ratio
            denominator = args.near_y_ratio - args.vanishing_y_ratio
            if denominator > 1e-6:
                scale = float(
                    np.clip(
                        (y_ratio - args.vanishing_y_ratio) / denominator,
                        0.08,
                        1.25,
                    )
                )
            else:
                scale = 1.0
            target_x = green_x - near_offset * scale

            actual_points.append(_safe_point(green_x, y, width, height))
            target_points.append(_safe_point(target_x, y, width, height))

        if len(actual_points) >= 2:
            cv2.polylines(
                preview,
                [np.asarray(actual_points, dtype=np.int32)],
                False,
                (80, 255, 80),
                4,
                cv2.LINE_AA,
            )
        if len(target_points) >= 2:
            cv2.polylines(
                preview,
                [np.asarray(target_points, dtype=np.int32)],
                False,
                (255, 0, 255),
                3,
                cv2.LINE_AA,
            )

    vehicle_x = int(round(width * args.vehicle_x_ratio))
    cv2.line(
        preview,
        (vehicle_x, height - 1),
        (vehicle_x, int(height * 0.44)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    draw_steering_arrow(
        preview,
        steering=control.steering,
        vehicle_x=vehicle_x,
        driving_enabled=driving_enabled,
    )

    if control.valid:
        near_y = args.near_y_ratio * height
        far_y = args.far_y_ratio * height
        cv2.circle(
            preview,
            _safe_point(control.green_near_x, near_y, width, height),
            7,
            (80, 255, 80),
            -1,
        )
        cv2.circle(
            preview,
            _safe_point(control.green_far_x, far_y, width, height),
            7,
            (80, 255, 80),
            -1,
        )
        cv2.circle(
            preview,
            _safe_point(control.target_near_x, near_y, width, height),
            8,
            (255, 0, 255),
            -1,
        )
        cv2.circle(
            preview,
            _safe_point(control.target_far_x, far_y, width, height),
            8,
            (255, 0, 255),
            -1,
        )

    if driving_enabled:
        run_text = "DRIVING - SPACE: STOP"
        run_color = (80, 255, 80)
    else:
        run_text = "STOPPED - SPACE: START"
        run_color = (0, 190, 255)

    if arduino_configured:
        serial_text = "ARDUINO CONNECTED" if arduino_connected else "ARDUINO DISCONNECTED"
    else:
        serial_text = "VISION-ONLY MODE"

    banner_width = min(width - 24, 390)
    cv2.rectangle(preview, (10, height - 68), (10 + banner_width, height - 10), (20, 20, 20), -1)
    cv2.putText(
        preview,
        run_text,
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        run_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        serial_text,
        (20, height - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    status_color = (80, 255, 80) if control.valid else (0, 0, 255)
    output_speed = control.speed if driving_enabled and arduino_connected else 0
    lines = [
        f"lane={'OK' if control.valid else 'LOST'} conf={control.confidence:.2f}",
        f"steer={control.steering:+.3f} cmd={control.steering_command:+d} planned={control.speed} output={output_speed}",
        f"error={control.combined_error:+.3f} nearW={control.near_weight:.2f} curve={control.curvature:.3f}",
    ]
    if not control.valid:
        lines.append(f"lost={control.lost_frames}: {control.reason}")

    for index, text in enumerate(lines):
        cv2.putText(
            preview,
            text,
            (12, 62 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color if index == 0 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> None:
    ratio_names = (
        "roi_top_ratio",
        "vehicle_x_ratio",
        "near_y_ratio",
        "far_y_ratio",
        "right_offset_ratio",
        "vanishing_y_ratio",
        "near_weight",
        "new_command_weight",
    )
    for name in ratio_names:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1, got {value}")

    if args.far_y_ratio >= args.near_y_ratio:
        raise SystemExit("--far-y-ratio must be smaller than --near-y-ratio")
    if args.lost_stop_frames <= args.lost_hold_frames:
        raise SystemExit("--lost-stop-frames must be larger than --lost-hold-frames")
    if args.command_rate <= 0:
        raise SystemExit("--command-rate must be positive")
    for name in ("speed_straight", "speed_turn", "speed_min"):
        value = int(getattr(args, name))
        if not 0 <= value <= 255:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 255")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 255:
        raise SystemExit("--constant-speed must be between 0 and 255")


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = load_semantic_model(args.weights, args.backend)
    postprocess_config = load_postprocess_config(args.postprocess_config) if args.postprocess else None

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reader = None if args.buffered_camera else LatestFrameReader(capture)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height} @ {actual_fps:.1f} FPS",
        flush=True,
    )
    print(
        "Right controller: "
        f"source={args.control_source}, vehicle_x={args.vehicle_x_ratio:.3f}, "
        f"offset={args.right_offset_ratio:.3f}, near/far={args.near_y_ratio:.2f}/{args.far_y_ratio:.2f}",
        flush=True,
    )

    arduino = ArduinoSender(
        port=args.arduino_port,
        baud=args.baud,
        timeout=args.serial_timeout,
        reset_wait=args.arduino_reset_wait,
        command_rate=args.command_rate,
        steering_scale=args.steering_command_scale,
    )
    follower = RightLaneFollower(args)

    window_name = "Right Lane Follow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0
    lane_right_id = CLASS_TO_ID["lane_right"]
    driving_enabled = False
    print("Controls: SPACE=start/stop, S=emergency stop, Q or ESC=quit", flush=True)

    try:
        while True:
            ok, frame = reader.read() if reader is not None else capture.read()
            if not ok or frame is None:
                raise RuntimeError("Could not read a frame from the camera")

            if args.flip:
                frame = cv2.flip(frame, 1)
            frame = normalize_frame_size(
                frame,
                args.width,
                args.height,
                force_size=not args.no_force_size,
            )

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                task="semantic",
                rect=False,
                verbose=False,
            )
            raw_class_map = semantic_to_class_map(results[0].semantic_mask, frame.shape[:2])

            if postprocess_config is not None:
                processed_class_map = postprocess_class_map(raw_class_map, postprocess_config)
            else:
                processed_class_map = raw_class_map

            control_class_map = (
                raw_class_map if args.control_source == "raw" else processed_class_map
            )
            observation = extract_right_lane_curve(
                control_class_map,
                lane_right_id=lane_right_id,
                roi_top_ratio=args.roi_top_ratio,
                min_component_area=args.min_component_area,
                min_lane_points=args.min_lane_points,
            )
            control = follower.compute(observation, frame.shape[:2])
            if driving_enabled and arduino.enabled:
                arduino.update_command(control.steering, control.speed)

            preview = make_class_overlay(frame, processed_class_map)
            if args.preview == "debug":
                draw_debug(
                    preview,
                    observation,
                    control,
                    args,
                    driving_enabled=driving_enabled,
                    arduino_connected=arduino.enabled,
                    arduino_configured=arduino.configured,
                )

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = 0.88 * fps + 0.12 * instant_fps if fps else instant_fps
            if args.show_fps:
                draw_fps(preview, fps)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if driving_enabled:
                    driving_enabled = False
                    arduino.emergency_stop()
                    follower.reset()
                    print("STOPPED: drive and steering outputs disabled.", flush=True)
                else:
                    if arduino.configured and not arduino.enabled:
                        try:
                            arduino.connect()
                        except RuntimeError as exc:
                            print(f"START FAILED: {exc}", file=sys.stderr, flush=True)
                            continue
                    follower.reset()
                    if arduino.enabled:
                        # Start from zero; the next fresh inference frame updates
                        # the command consumed by the background serial writer.
                        arduino.emergency_stop()
                        arduino.start_driving()
                    driving_enabled = True
                    if arduino.enabled:
                        print("DRIVING: fresh-frame Arduino commands will start now.", flush=True)
                    else:
                        print("SIMULATION RUNNING: no Arduino port configured.", flush=True)
            if key in (ord("s"), ord("S")):
                driving_enabled = False
                arduino.emergency_stop()
                follower.reset()
                print("EMERGENCY STOP: drive and steering outputs disabled.", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        # Stop the car before releasing the camera or closing the serial port.
        arduino.emergency_stop()
        arduino.close()
        if reader is not None:
            reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
