#!/usr/bin/env python3
"""Drive in parking PHASE 0 by following the fitted parking-dot line."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.lane_follower import ControlOutput, RightLaneFollower
from hardware.arduino import ArduinoSender, format_arduino_command
from hardware.camera import LatestFrameReader, normalize_frame_size, open_camera
from perception.lane_detector import LaneBoundary, LaneCurve

from infer_sem_class import (
    CLASS_TO_ID,
    DEFAULT_WEIGHTS,
    load_semantic_model,
    make_overlay,
    semantic_to_class_map,
)
from utils.filter_cars import (
    filter_cars_in_parking_region,
    remove_car_detections,
)
from utils.lane_detect import (
    Line,
    ParkingDotLineDetector,
    ParkingLineDetector,
    ReferenceLineDetector,
    draw_line,
    draw_line_points,
    draw_parking_lines,
)
from utils.phase_control import PhaseController


KEYBOARD_DRIVE_SPEED = 140
KEYBOARD_STEERING_MIN = -1000
KEYBOARD_STEERING_MAX = 1000
KEYBOARD_STEERING_STEP = 75
KEYBOARD_SEND_INTERVAL = 0.05


class ParkingArduinoSender(ArduinoSender):
    """Arduino sender that preserves signed drive PWM for reverse parking."""

    def update_command(
        self,
        steering: float,
        speed: int,
        *,
        immediate: bool = False,
    ) -> None:
        scale = self.steering_scale
        steering_command = int(
            np.clip(round(float(steering) * scale), -scale, scale)
        )
        speed_command = int(np.clip(round(speed), -255, 255))

        with self.state_lock:
            self.latest_command = (steering_command, speed_command)
            active = self.driving_active

        if immediate and active and self.enabled:
            try:
                self._write(
                    format_arduino_command(
                        steering_command,
                        speed_command,
                    )
                )
            except RuntimeError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-force-size", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show-fps", action="store_true")

    # The parking-dot line is treated as the right-side boundary of a virtual
    # driving path, matching the existing single-boundary lane follower.
    parser.add_argument("--vehicle-x-ratio", type=float, default=0.50)
    parser.add_argument("--near-y-ratio", type=float, default=0.84)
    parser.add_argument("--far-y-ratio", type=float, default=0.58)
    parser.add_argument("--vanishing-y-ratio", type=float, default=0.31)

    # Keep the main lane follower's control defaults.
    parser.add_argument("--kp", type=float, default=1.15)
    parser.add_argument("--kd", type=float, default=0.18)
    parser.add_argument("--near-weight", type=float, default=0.65)
    parser.add_argument(
        "--adaptive-lookahead",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--steering-deadband", type=float, default=0.015)
    parser.add_argument("--new-command-weight", type=float, default=0.78)
    parser.add_argument("--max-command-change", type=float, default=0.24)
    parser.add_argument("--large-error-threshold", type=float, default=0.14)
    parser.add_argument("--large-error-full", type=float, default=0.52)
    parser.add_argument("--large-error-gain", type=float, default=1.55)
    parser.add_argument("--large-error-power", type=float, default=1.35)
    parser.add_argument("--recovery-near-weight-bonus", type=float, default=0.18)
    parser.add_argument("--recovery-command-change-bonus", type=float, default=0.34)
    parser.add_argument("--recovery-new-command-weight", type=float, default=0.94)
    parser.add_argument("--recovery-min-confidence", type=float, default=0.28)
    parser.add_argument("--recovery-speed", type=int, default=200)
    parser.add_argument("--lane-loss-grace-seconds", type=float, default=1.20)
    parser.add_argument("--lane-loss-speed", type=int, default=200)
    parser.add_argument("--lane-loss-straighten-delay", type=float, default=0.30)
    parser.add_argument("--lane-loss-min-steering-retain", type=float, default=0.38)
    parser.add_argument("--lane-loss-confidence-decay", type=float, default=0.80)
    parser.add_argument("--lane-curve-new-weight", type=float, default=0.72)
    parser.add_argument("--max-lane-jump-ratio", type=float, default=0.24)
    parser.add_argument("--min-lane-confidence", type=float, default=0.18)
    parser.add_argument("--speed-straight", type=int, default=255)
    parser.add_argument("--speed-turn", type=int, default=255)
    parser.add_argument("--speed-min", type=int, default=100)
    parser.add_argument("--constant-speed", type=int, default=140)

    parser.add_argument("--arduino-port", default="COM7")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--serial-timeout", type=float, default=0.10)
    parser.add_argument("--arduino-reset-wait", type=float, default=1.8)
    parser.add_argument("--command-rate", type=float, default=20.0)
    parser.add_argument("--steering-command-scale", type=int, default=1000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "vehicle_x_ratio",
        "near_y_ratio",
        "far_y_ratio",
        "vanishing_y_ratio",
        "near_weight",
        "new_command_weight",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be between 0 and 1"
            )
    if args.far_y_ratio >= args.near_y_ratio:
        raise SystemExit("--far-y-ratio must be smaller than --near-y-ratio")
    if args.command_rate <= 0:
        raise SystemExit("--command-rate must be positive")
    for name in (
        "speed_straight",
        "speed_turn",
        "speed_min",
        "recovery_speed",
        "lane_loss_speed",
    ):
        if not 0 <= int(getattr(args, name)) <= 255:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 255")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 255:
        raise SystemExit("--constant-speed must be between 0 and 255")


def line_to_lane_curve(line: Line, frame_height: int) -> LaneCurve:
    """Convert a fitted image-space line to the follower's x(y_norm) curve."""
    if (
        not line.valid
        or line.point is None
        or line.direction is None
        or abs(float(line.direction[1])) < 1e-6
    ):
        return LaneCurve(
            boundary=LaneBoundary.RIGHT,
            valid=False,
            confidence=line.confidence,
            reason=line.reason or "parking dot line unavailable",
        )

    point_x, point_y = map(float, line.point)
    direction_x, direction_y = map(float, line.direction)
    dx_per_y = direction_x / direction_y
    y_scale = max(frame_height - 1, 1)
    coefficients = np.asarray(
        [
            dx_per_y * y_scale,
            point_x - dx_per_y * point_y,
        ],
        dtype=np.float64,
    )
    return LaneCurve(
        boundary=LaneBoundary.RIGHT,
        valid=True,
        coefficients=coefficients,
        points=line.points,
        mask=line.mask,
        confidence=line.confidence,
        reason="parking dot line",
    )


def detect_out_follow_line(
    class_map: np.ndarray,
    *,
    min_component_area: int = 20,
) -> Line:
    """Project the largest detected out component vertically through its center."""
    out_mask = (class_map == CLASS_TO_ID["out"]).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        out_mask,
        connectivity=8,
    )
    candidates = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area
    ]
    if not candidates:
        return Line(
            valid=False,
            mask=out_mask,
            reason="out component unavailable",
        )

    label = max(
        candidates,
        key=lambda item: int(stats[item, cv2.CC_STAT_AREA]),
    )
    component = (labels == label).astype(np.uint8)
    center_x = float(centroids[label][0])
    component_area = int(stats[label, cv2.CC_STAT_AREA])
    total_out_area = max(int(np.count_nonzero(out_mask)), 1)
    height, _ = class_map.shape
    component_bottom = (
        int(stats[label, cv2.CC_STAT_TOP])
        + int(stats[label, cv2.CC_STAT_HEIGHT])
        - 1
    )

    return Line(
        valid=True,
        point=np.asarray(
            [center_x, float(centroids[label][1])],
            dtype=np.float64,
        ),
        direction=np.asarray([0.0, 1.0], dtype=np.float64),
        angle_deg=90.0,
        slope=math.inf,
        intercept=center_x,
        confidence=float(component_area / total_out_area),
        mask=component,
        segment=(
            (int(round(center_x)), component_bottom),
            (int(round(center_x)), height - 1),
        ),
        reason="vertical projection from out component",
    )


def midpoint_follow_line(
    first: Line,
    second: Line,
    image_shape: tuple[int, ...],
) -> Line:
    """Build the exact image-space center line between two fitted lines."""
    height, _ = image_shape[:2]
    if not first.valid or not second.valid:
        return Line(
            valid=False,
            reason="reference or parking-dot line unavailable",
        )

    top_y = 0.0
    bottom_y = float(max(height - 1, 0))
    first_top_x = first.x_at(top_y)
    first_bottom_x = first.x_at(bottom_y)
    second_top_x = second.x_at(top_y)
    second_bottom_x = second.x_at(bottom_y)
    if None in (
        first_top_x,
        first_bottom_x,
        second_top_x,
        second_bottom_x,
    ):
        return Line(
            valid=False,
            reason="reference or parking-dot line is horizontal",
        )

    top_x = (float(first_top_x) + float(second_top_x)) * 0.5
    bottom_x = (
        float(first_bottom_x) + float(second_bottom_x)
    ) * 0.5
    direction = np.asarray(
        [bottom_x - top_x, bottom_y - top_y],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return Line(valid=False, reason="midpoint line has no length")
    direction /= norm
    point = np.asarray([top_x, top_y], dtype=np.float64)
    angle_deg = float(
        math.degrees(math.atan2(direction[1], direction[0]))
    )

    return Line(
        valid=True,
        point=point,
        direction=direction,
        angle_deg=angle_deg,
        slope=(
            math.inf
            if abs(float(direction[0])) < 1e-6
            else float(direction[1] / direction[0])
        ),
        intercept=top_x,
        confidence=min(first.confidence, second.confidence),
        reason="midpoint of reference and parking-dot lines",
    )


def stopped_output(reason: str) -> ControlOutput:
    return ControlOutput(
        valid=False,
        steering=0.0,
        steering_command=0,
        speed=0,
        confidence=0.0,
        lost_frames=0,
        reason=reason,
    )


def phase_1_keyboard_wa_output(
    steering_command: int,
    steering_scale: int,
) -> ControlOutput:
    """Return the command produced while drive_keyboard holds W and A."""
    scale = max(1, int(steering_scale))
    steering_command = int(
        np.clip(steering_command, -scale, scale)
    )
    return ControlOutput(
        valid=True,
        steering=steering_command / scale,
        steering_command=steering_command,
        speed=KEYBOARD_DRIVE_SPEED,
        confidence=1.0,
        lost_frames=0,
        reason="phase 1: drive_keyboard W+A",
    )


def phase_2_max_right_output(
    steering_command: int,
    steering_scale: int,
) -> ControlOutput:
    """Steer fully right first, then drive at drive_keyboard speed."""
    scale = max(1, int(steering_scale))
    steering_command = int(
        np.clip(steering_command, -scale, scale)
    )
    at_max_right = steering_command >= min(
        KEYBOARD_STEERING_MAX,
        scale,
    )
    return ControlOutput(
        valid=True,
        steering=steering_command / scale,
        steering_command=steering_command,
        speed=-KEYBOARD_DRIVE_SPEED if at_max_right else 0,
        confidence=1.0,
        lost_frames=0,
        reason=(
            "phase 2: reversing at maximum right steering"
            if at_max_right
            else "phase 2: moving steering to maximum right"
        ),
    )


def phase_3_output(
    phase_started_at: float | None,
    reverse_seconds: float,
) -> ControlOutput:
    """Reverse straight briefly, then remain fully stopped."""
    elapsed = (
        math.inf
        if phase_started_at is None
        else max(0.0, time.perf_counter() - phase_started_at)
    )
    reversing = elapsed < reverse_seconds
    return ControlOutput(
        valid=True,
        steering=0.0,
        steering_command=0,
        speed=-KEYBOARD_DRIVE_SPEED if reversing else 0,
        confidence=1.0,
        lost_frames=0,
        reason=(
            "phase 3: reversing straight for 0.5 seconds"
            if reversing
            else "phase 3: stopped for 4 seconds"
        ),
    )


def phase_4_max_right_output(
    steering_command: int,
    steering_scale: int,
) -> ControlOutput:
    """Steer fully right first, then drive forward."""
    scale = max(1, int(steering_scale))
    steering_command = int(
        np.clip(steering_command, -scale, scale)
    )
    at_max_right = steering_command >= min(
        KEYBOARD_STEERING_MAX,
        scale,
    )
    return ControlOutput(
        valid=True,
        steering=steering_command / scale,
        steering_command=steering_command,
        speed=KEYBOARD_DRIVE_SPEED if at_max_right else 0,
        confidence=1.0,
        lost_frames=0,
        reason=(
            "phase 4: driving forward at maximum right steering"
            if at_max_right
            else "phase 4: moving steering to maximum right"
        ),
    )


def draw_runtime_status(
    frame: np.ndarray,
    phase: int,
    control: ControlOutput,
    driving_enabled: bool,
    fps: float,
    show_fps: bool,
) -> None:
    phase_text = f"PHASE {phase}"
    size, _ = cv2.getTextSize(
        phase_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2,
    )
    cv2.putText(
        frame,
        phase_text,
        (frame.shape[1] - size[0] - 12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if show_fps:
        cv2.putText(
            frame,
            f"FPS: {fps:4.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        (
            f"{'DRIVING' if driving_enabled else 'STOPPED'}  "
            f"steer={control.steering:+.3f} speed={control.speed}"
        ),
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (70, 255, 70) if driving_enabled else (0, 80, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = load_semantic_model(args.weights, args.backend)
    capture = open_camera(
        args.camera,
        args.width,
        args.height,
        args.camera_fps,
    )
    reader = LatestFrameReader(capture)
    arduino = ParkingArduinoSender(
        port=args.arduino_port,
        baud=args.baud,
        timeout=args.serial_timeout,
        reset_wait=args.arduino_reset_wait,
        command_rate=args.command_rate,
        steering_scale=args.steering_command_scale,
    )

    reference_detector = ReferenceLineDetector()
    parking_dot_detector = ParkingDotLineDetector()
    parking_line_detector = ParkingLineDetector()
    phase_controller = PhaseController()
    follower = RightLaneFollower(args)

    window_name = "Parking Auto Drive"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    driving_enabled = False
    previous_phase = phase_controller.phase
    last_phase_0_steering_command = 0
    phase_1_steering_command = 0
    phase_1_last_step_time: float | None = None
    phase_2_steering_command = 0
    phase_2_last_step_time: float | None = None
    phase_4_steering_command = 0
    phase_4_last_step_time: float | None = None
    previous_time = time.perf_counter()
    fps = 0.0
    print(
        "Controls: SPACE=start/stop, S=emergency stop, "
        "R=reset Arduino fault, Q or ESC=quit",
        flush=True,
    )

    try:
        while True:
            ok, frame = reader.read()
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
                verbose=False,
                stream=True,
            )
            result = next(iter(results))
            class_map = semantic_to_class_map(
                result.semantic_mask,
                frame.shape[:2],
            )

            reference_line = reference_detector.detect(class_map)
            parking_dot_line: Line | None = None
            parking_lines = None
            out_follow_line: Line | None = None
            phase_0_follow_line: Line | None = None
            if phase_controller.phase == 0:
                parking_dot_line = parking_dot_detector.detect(class_map)
                phase_0_follow_line = midpoint_follow_line(
                    reference_line,
                    parking_dot_line,
                    frame.shape,
                )
                parking_lines = parking_line_detector.detect(
                    class_map,
                    excluded_points=parking_dot_line.rejected_points,
                )
                class_map = filter_cars_in_parking_region(
                    class_map,
                    parking_dot_line,
                    parking_lines,
                )
                if driving_enabled:
                    phase_controller.update(
                        class_map,
                        parking_lines,
                        parking_dot_line,
                        now=time.perf_counter(),
                    )
            elif (
                driving_enabled
                and phase_controller.phase in (1, 2, 3, 4)
            ):
                phase_controller.update(
                    class_map,
                    reference_line=reference_line,
                    now=time.perf_counter(),
                )

            if phase_controller.phase == 5:
                out_follow_line = detect_out_follow_line(class_map)

            if phase_controller.phase >= 1:
                class_map = remove_car_detections(class_map)

            if phase_controller.phase == 0 and phase_0_follow_line is not None:
                observation = line_to_lane_curve(
                    phase_0_follow_line,
                    frame.shape[0],
                )
                control = follower.compute(
                    observation=observation,
                    frame_shape=frame.shape[:2],
                    offset_ratio=0.0,
                )
                last_phase_0_steering_command = control.steering_command
            elif phase_controller.phase == 1:
                control_now = time.perf_counter()
                if phase_1_last_step_time is None:
                    phase_1_last_step_time = control_now
                elapsed = control_now - phase_1_last_step_time
                step_count = int(elapsed / KEYBOARD_SEND_INTERVAL)
                if step_count > 0:
                    phase_1_steering_command = max(
                        KEYBOARD_STEERING_MIN,
                        phase_1_steering_command
                        - KEYBOARD_STEERING_STEP * step_count,
                    )
                    phase_1_last_step_time += (
                        step_count * KEYBOARD_SEND_INTERVAL
                    )
                control = phase_1_keyboard_wa_output(
                    phase_1_steering_command,
                    args.steering_command_scale,
                )
            elif phase_controller.phase == 2:
                control_now = time.perf_counter()
                if phase_2_last_step_time is None:
                    phase_2_last_step_time = control_now
                elapsed = control_now - phase_2_last_step_time
                step_count = int(elapsed / KEYBOARD_SEND_INTERVAL)
                if step_count > 0:
                    phase_2_steering_command = min(
                        KEYBOARD_STEERING_MAX,
                        phase_2_steering_command
                        + KEYBOARD_STEERING_STEP * step_count,
                    )
                    phase_2_last_step_time += (
                        step_count * KEYBOARD_SEND_INTERVAL
                    )
                control = phase_2_max_right_output(
                    phase_2_steering_command,
                    args.steering_command_scale,
                )
            elif phase_controller.phase == 3:
                control = phase_3_output(
                    phase_controller.phase_started_at,
                    phase_controller.phase_3_reverse_seconds,
                )
            elif phase_controller.phase == 4:
                control_now = time.perf_counter()
                if phase_4_last_step_time is None:
                    phase_4_last_step_time = control_now
                elapsed = control_now - phase_4_last_step_time
                step_count = int(elapsed / KEYBOARD_SEND_INTERVAL)
                if step_count > 0:
                    phase_4_steering_command = min(
                        KEYBOARD_STEERING_MAX,
                        phase_4_steering_command
                        + KEYBOARD_STEERING_STEP * step_count,
                    )
                    phase_4_last_step_time += (
                        step_count * KEYBOARD_SEND_INTERVAL
                    )
                control = phase_4_max_right_output(
                    phase_4_steering_command,
                    args.steering_command_scale,
                )
            elif phase_controller.phase == 5 and out_follow_line is not None:
                observation = line_to_lane_curve(
                    out_follow_line,
                    frame.shape[0],
                )
                control = follower.compute(
                    observation=observation,
                    frame_shape=frame.shape[:2],
                    offset_ratio=0.0,
                )
            else:
                control = stopped_output(
                    f"parking-dot control disabled in phase {phase_controller.phase}"
                )
                follower.reset()

            if phase_controller.phase != previous_phase:
                print(
                    f"PHASE {previous_phase} -> {phase_controller.phase}",
                    flush=True,
                )
                if phase_controller.phase == 1:
                    phase_1_steering_command = int(
                        np.clip(
                            last_phase_0_steering_command,
                            KEYBOARD_STEERING_MIN,
                            args.steering_command_scale,
                        )
                    )
                    phase_1_last_step_time = time.perf_counter()
                    control = phase_1_keyboard_wa_output(
                        phase_1_steering_command,
                        args.steering_command_scale,
                    )
                elif phase_controller.phase == 2:
                    phase_2_steering_command = phase_1_steering_command
                    phase_2_last_step_time = time.perf_counter()
                    control = phase_2_max_right_output(
                        phase_2_steering_command,
                        args.steering_command_scale,
                    )
                elif phase_controller.phase == 4:
                    phase_4_steering_command = 0
                    phase_4_last_step_time = time.perf_counter()
                    control = phase_4_max_right_output(
                        phase_4_steering_command,
                        args.steering_command_scale,
                    )
                elif phase_controller.phase != 0:
                    arduino.update_command(0.0, 0, immediate=True)
                previous_phase = phase_controller.phase

            if driving_enabled and arduino.enabled:
                arduino.update_command(
                    control.steering,
                    control.speed,
                )

            preview = make_overlay(frame, class_map)
            if phase_controller.phase == 0 and parking_lines is not None:
                draw_parking_lines(
                    preview,
                    parking_lines,
                    color=(0, 140, 255),
                    thickness=3,
                )
            draw_line(
                preview,
                reference_line,
                color=(0, 255, 0),
                thickness=3,
            )
            if phase_controller.phase == 5 and out_follow_line is not None:
                draw_line(
                    preview,
                    out_follow_line,
                    color=(255, 180, 0),
                    thickness=3,
                )
            if phase_controller.phase == 0 and phase_0_follow_line is not None:
                draw_line(
                    preview,
                    phase_0_follow_line,
                    color=(255, 0, 255),
                    thickness=3,
                )
            if phase_controller.phase == 0 and parking_dot_line is not None:
                draw_line(
                    preview,
                    parking_dot_line,
                    color=(0, 255, 255),
                    thickness=3,
                )
                draw_line_points(
                    preview,
                    parking_dot_line,
                    color=(0, 200, 255),
                    radius=6,
                )

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = 0.88 * fps + 0.12 * instant_fps if fps else instant_fps
            draw_runtime_status(
                preview,
                phase_controller.phase,
                control,
                driving_enabled,
                fps,
                args.show_fps,
            )

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if driving_enabled:
                    driving_enabled = False
                    arduino.emergency_stop()
                    follower.reset()
                    reference_detector.reset()
                    parking_dot_detector.reset()
                    phase_controller = PhaseController()
                    previous_phase = 0
                    print("STOPPED: drive and steering outputs disabled.", flush=True)
                else:
                    if arduino.configured and not arduino.enabled:
                        try:
                            arduino.connect()
                        except RuntimeError as exc:
                            print(
                                f"START FAILED: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                    follower.reset()
                    if arduino.enabled:
                        arduino.emergency_stop()
                        arduino.start_driving()
                    driving_enabled = True
                    print(
                        "DRIVING: PHASE 0 follows the reference/dot midpoint; "
                        "PHASE 1 holds W+A; PHASE 2 reverses right; "
                        "PHASE 4 drives forward right; "
                        "PHASE 5 follows out vertically.",
                        flush=True,
                    )
            if key in (ord("s"), ord("S")):
                driving_enabled = False
                arduino.emergency_stop()
                follower.reset()
                reference_detector.reset()
                parking_dot_detector.reset()
                phase_controller = PhaseController()
                previous_phase = 0
                print("EMERGENCY STOP: drive and steering outputs disabled.", flush=True)
            if key in (ord("r"), ord("R")):
                driving_enabled = False
                arduino.emergency_stop()
                follower.reset()
                reference_detector.reset()
                parking_dot_detector.reset()
                phase_controller = PhaseController()
                previous_phase = 0
                if arduino.enabled:
                    arduino.reset_fault()
                    print("ARDUINO FAULT RESET requested.", flush=True)
                else:
                    print("Arduino is not connected.", flush=True)
    finally:
        arduino.emergency_stop()
        arduino.close()
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
