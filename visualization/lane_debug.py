#!/usr/bin/env python3
"""Debug visualization for lane following."""

from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np

from control.lane_follower import (
    ControlOutput,
    DrivingLane,
    evaluate_curve_x,
)
from hardware.arduino import ArduinoTelemetry
from perception.lane_detector import LaneCurve

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


def draw_actual_steering_arrow(
    preview: np.ndarray,
    actual_steering: float,
    vehicle_x: int,
) -> None:
    """Draw the potentiometer-measured steering direction as a shorter cyan arrow."""
    height, width = preview.shape[:2]
    actual_steering = float(np.clip(actual_steering, -1.0, 1.0))
    start = (vehicle_x, height - 60)
    horizontal_range = max(58, int(width * 0.19))
    end_x = int(np.clip(vehicle_x + actual_steering * horizontal_range, 8, width - 9))
    end_y = max(8, start[1] - max(48, int(height * 0.15)))
    color = (255, 255, 0)
    cv2.arrowedLine(
        preview,
        start,
        (end_x, end_y),
        color,
        3,
        cv2.LINE_AA,
        tipLength=0.22,
    )
    cv2.putText(
        preview,
        "ACTUAL",
        (int(np.clip(end_x - 34, 8, width - 75)), max(20, end_y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_debug(
    preview: np.ndarray,
    observation: LaneCurve,
    control: ControlOutput,
    args: argparse.Namespace,
    offset_ratio: float,
    reference_label: str,
    current_lane: DrivingLane,
    driving_enabled: bool,
    arduino_connected: bool,
    arduino_configured: bool,
    telemetry: Optional[ArduinoTelemetry],
) -> None:
    height, width = preview.shape[:2]

    cv2.putText(
        preview,
        (
            f"LANE={current_lane.value} | "
            f"REF={reference_label} | "
            f"OFFSET={offset_ratio:.3f}"
        ),
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    roi_y = int(height * args.roi_top_ratio)
    cv2.line(preview, (0, roi_y), (width - 1, roi_y), (90, 90, 90), 1)

    if observation.points is not None:
        for x, y in observation.points[:: max(1, len(observation.points) // 80)]:
            cv2.circle(preview, _safe_point(x, y, width, height), 2, (0, 255, 0), -1)

    curve_coefficients = (
        observation.coefficients
        if observation.coefficients is not None
        else control.display_coefficients
    )
    if curve_coefficients is not None:
        actual_points: list[tuple[int, int]] = []
        target_points: list[tuple[int, int]] = []
        for y_ratio in np.linspace(args.roi_top_ratio, 0.98, 70):
            y = y_ratio * height
            green_x = evaluate_curve_x(curve_coefficients, float(y_ratio))

            near_offset = width * float(offset_ratio)
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
                (0, 190, 255) if control.predicted else (80, 255, 80),
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
    if telemetry is not None:
        actual_steering = telemetry.actual_command / max(1, args.steering_command_scale)
        draw_actual_steering_arrow(
            preview,
            actual_steering=actual_steering,
            vehicle_x=vehicle_x,
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
        if telemetry is not None and telemetry.fault != 0:
            serial_text = f"ARDUINO FAULT {telemetry.fault}"
        else:
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

    if control.valid:
        lane_label = "OK"
        status_color = (80, 255, 80)
    elif control.predicted:
        lane_label = "PREDICT"
        status_color = (0, 190, 255)
    else:
        lane_label = "LOST"
        status_color = (0, 0, 255)

    output_speed = control.speed if driving_enabled and arduino_connected else 0
    lines = [
        f"lane={lane_label} conf={control.confidence:.2f} loss={control.lane_loss_age:.2f}s",
        f"steer={control.steering:+.3f} cmd={control.steering_command:+d} planned={control.speed} output={output_speed}",
        f"error={control.combined_error:+.3f} gain={control.gain_multiplier:.2f} recovery={control.recovery_level:.2f}",
        f"nearW={control.near_weight:.2f} curve={control.curvature:.3f}",
    ]
    if telemetry is not None:
        lines.append(
            f"pot={telemetry.sensor}->{telemetry.target} err={telemetry.error:+d} "
            f"pwm={telemetry.steering_pwm} actual={telemetry.actual_command:+d} fault={telemetry.fault}"
        )
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
