#!/usr/bin/env python3
"""Debug visualization for car detection and ultrasonic obstacle avoidance."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from control.obstacle_avoidance import ObstacleAvoidanceCommand
from hardware.arduino import ArduinoDistances
from perception.infer_car import CarInferenceResult


def _validate_preview(preview: np.ndarray) -> None:
    """Validate the OpenCV preview image."""

    if not isinstance(preview, np.ndarray):
        raise TypeError("preview must be a numpy.ndarray")

    if preview.ndim != 3 or preview.shape[2] != 3:
        raise ValueError(
            f"preview must have BGR shape (H, W, 3), got {preview.shape}"
        )


def _distance_text(distance_cm: Optional[int]) -> str:
    """Format an optional ultrasonic distance."""

    if distance_cm is None:
        return "N/A"

    return f"{int(distance_cm)}cm"


def _distance_color(
    distance_cm: Optional[int],
    *,
    obstacle_distance_cm: int,
    clear_distance_cm: int,
) -> tuple[int, int, int]:
    """Return a BGR color representing ultrasonic distance state."""

    if distance_cm is None:
        return (130, 130, 130)

    if distance_cm <= obstacle_distance_cm:
        return (0, 0, 255)

    if distance_cm < clear_distance_cm:
        return (0, 165, 255)

    return (80, 255, 80)


def _draw_distance_value(
    preview: np.ndarray,
    *,
    label: str,
    distance_cm: Optional[int],
    x: int,
    y: int,
    obstacle_distance_cm: int,
    clear_distance_cm: int,
) -> None:
    """Draw the ultrasonic distance as a large numeric value."""

    color = _distance_color(
        distance_cm,
        obstacle_distance_cm=obstacle_distance_cm,
        clear_distance_cm=clear_distance_cm,
    )

    cv2.putText(
        preview,
        f"{label}: {_distance_text(distance_cm)}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_obstacle_debug(
    preview: np.ndarray,
    car_result: CarInferenceResult,
    distances: Optional[ArduinoDistances],
    control: Optional[ObstacleAvoidanceCommand],
    *,
    corridor_center_ratio: float = 0.50,
    corridor_width_ratio: float = 0.36,
    obstacle_distance_cm: int = 200,
    clear_distance_cm: int = 250,
    display_max_cm: int = 250,
    mock_enabled: bool = False,
) -> None:
    """Draw car-detection and ultrasonic obstacle-avoidance debug information.

    The visualization includes:
    - the visual forward corridor,
    - connected car-region bounding boxes,
    - each car region's area and bottom-y ratio,
    - the center ultrasonic distance,
    - visual ``front_blocked`` state,
    - fused camera/ultrasonic obstacle state,
    - obstacle latch and confirmation counters,
    - the target lane and lane-change request.
    """

    _validate_preview(preview)

    if not 0.0 <= corridor_center_ratio <= 1.0:
        raise ValueError("corridor_center_ratio must be in [0, 1]")

    if not 0.0 < corridor_width_ratio <= 1.0:
        raise ValueError("corridor_width_ratio must be in (0, 1]")

    if obstacle_distance_cm < 1:
        raise ValueError("obstacle_distance_cm must be at least 1")

    if clear_distance_cm <= obstacle_distance_cm:
        raise ValueError(
            "clear_distance_cm must be greater than obstacle_distance_cm"
        )

    if display_max_cm < clear_distance_cm:
        raise ValueError(
            "display_max_cm must be at least clear_distance_cm"
        )

    height, width = preview.shape[:2]

    if car_result.class_map.shape != (height, width):
        raise ValueError(
            "car_result.class_map shape must match preview height and width"
        )

    # -----------------------------------------------------------------
    # Visual forward corridor
    # -----------------------------------------------------------------

    corridor_top = int(round(height * 0.18))

    corridor_color = (
        (0, 0, 255)
        if car_result.front_blocked
        else (255, 180, 0)
    )

    overlay = preview.copy()
    lane_mask = (
        car_result.current_lane_mask
        > 0
    )
    overlay[lane_mask] = corridor_color
    cv2.addWeighted(
        overlay,
        0.09,
        preview,
        0.91,
        0.0,
        preview,
    )

    lane_contours, _ = cv2.findContours(
        car_result.current_lane_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        preview,
        lane_contours,
        -1,
        corridor_color,
        2,
    )

    cv2.putText(
        preview,
        "CURRENT LANE OBSTACLE PATH",
        (
            8,
            min(height - 8, corridor_top + 20),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        corridor_color,
        1,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------
    # Car connected components
    # -----------------------------------------------------------------

    for index, detection in enumerate(car_result.detections, start=1):
        x, y, box_width, box_height = detection.bbox

        x1 = int(np.clip(x, 0, width - 1))
        y1 = int(np.clip(y, 0, height - 1))
        x2 = int(np.clip(x + box_width, 0, width - 1))
        y2 = int(np.clip(y + box_height, 0, height - 1))

        box_color = (
            (0, 0, 255)
            if detection.blocks_path
            else (
                (255, 0, 255)
                if (
                    detection.on_road
                    and detection.in_current_lane
                )
                else (
                    (255, 180, 0)
                    if detection.on_road
                    else (130, 130, 130)
                )
            )
        )

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
        )

        center_x = int(np.clip(detection.center[0], 0, width - 1))
        center_y = int(np.clip(detection.center[1], 0, height - 1))

        cv2.circle(
            preview,
            (center_x, center_y),
            4,
            box_color,
            -1,
            cv2.LINE_AA,
        )

        label = (
            f"CAR{index} "
            f"A={detection.area} "
            f"B={detection.bottom_y_ratio:.2f} "
            f"ROAD={int(detection.on_road)} "
            f"LANE={int(detection.in_current_lane)}"
        )

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1,
        )

        label_x = int(
            np.clip(
                x1,
                2,
                max(2, width - text_size[0] - 3),
            )
        )
        label_y = max(16, y1 - 5)

        cv2.putText(
            preview,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------------------
    # Resolve sensor and controller values
    # -----------------------------------------------------------------

    center_cm = (
        distances.center_cm
        if distances is not None
        else None
    )

    distance_age = (
        distances.age
        if distances is not None
        else None
    )

    front_blocked = bool(
        getattr(
            control,
            "front_blocked",
            car_result.front_blocked,
        )
    )

    ultrasonic_close = bool(
        getattr(
            control,
            "ultrasonic_close",
            (
                center_cm is not None
                and center_cm <= obstacle_distance_cm
            ),
        )
    )

    fused_obstacle = bool(
        getattr(
            control,
            "fused_front_obstacle",
            front_blocked and ultrasonic_close,
        )
    )

    obstacle_latched = bool(
        getattr(
            control,
            "obstacle_latched",
            False,
        )
    )

    lane_change_requested = bool(
        getattr(
            control,
            "lane_change_requested",
            False,
        )
    )

    blocked_frames = int(
        getattr(
            control,
            "blocked_frames",
            0,
        )
    )

    clear_frames = int(
        getattr(
            control,
            "clear_frames",
            0,
        )
    )

    target_lane = getattr(
        control,
        "target_lane",
        None,
    )

    if target_lane is None:
        target_lane_text = "N/A"
    else:
        target_lane_text = str(
            getattr(
                target_lane,
                "name",
                target_lane,
            )
        )

    # -----------------------------------------------------------------
    # Right-side obstacle banner
    # -----------------------------------------------------------------

    banner_margin = 10
    banner_width = min(
        315,
        max(250, width - banner_margin * 2),
    )
    banner_height = min(
        198,
        max(150, height - 155),
    )

    banner_x = max(
        banner_margin,
        width - banner_width - banner_margin,
    )

    # The intersection banner occupies the upper-right corner.
    banner_y = min(
        max(142, int(height * 0.38)),
        max(banner_margin, height - banner_height - banner_margin),
    )

    banner_right = min(
        width - banner_margin,
        banner_x + banner_width,
    )
    banner_bottom = min(
        height - banner_margin,
        banner_y + banner_height,
    )

    if lane_change_requested:
        state_text = "LANE CHANGE REQUEST"
        state_color = (0, 255, 255)
    elif fused_obstacle:
        state_text = "FUSED OBSTACLE"
        state_color = (0, 0, 255)
    elif front_blocked:
        state_text = "VISION BLOCKED"
        state_color = (0, 165, 255)
    elif ultrasonic_close:
        state_text = "ULTRASONIC CLOSE"
        state_color = (0, 165, 255)
    else:
        state_text = "PATH CLEAR"
        state_color = (80, 255, 80)

    banner_overlay = preview.copy()

    cv2.rectangle(
        banner_overlay,
        (banner_x, banner_y),
        (banner_right, banner_bottom),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        banner_overlay,
        0.84,
        preview,
        0.16,
        0.0,
        preview,
    )

    cv2.rectangle(
        preview,
        (banner_x, banner_y),
        (banner_right, banner_bottom),
        state_color,
        2,
    )

    text_x = banner_x + 10

    cv2.putText(
        preview,
        f"OBSTACLE: {state_text}",
        (text_x, banner_y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        state_color,
        2,
        cv2.LINE_AA,
    )

    mode_text = "MOCK" if mock_enabled else "LIVE"
    age_text = (
        f"{distance_age:.2f}s"
        if distance_age is not None
        else "N/A"
    )

    cv2.putText(
        preview,
        (
            f"MODE={mode_text} "
            f"CAR={int(front_blocked)} "
            f"US={int(ultrasonic_close)} "
            f"FUSED={int(fused_obstacle)}"
        ),
        (text_x, banner_y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        (
            f"LATCH={int(obstacle_latched)} "
            f"BLOCK={blocked_frames} "
            f"CLEAR={clear_frames} "
            f"DIST_AGE={age_text}"
        ),
        (text_x, banner_y + 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        (
            f"TARGET={target_lane_text} "
            f"CHANGE={int(lane_change_requested)} "
            f"CARS={len(car_result.detections)}"
        ),
        (text_x, banner_y + 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    _draw_distance_value(
        preview,
        label="CENTER",
        distance_cm=center_cm,
        x=text_x,
        y=banner_y + 122,
        obstacle_distance_cm=obstacle_distance_cm,
        clear_distance_cm=clear_distance_cm,
    )

    # Highlight the exact center-distance threshold used for lane changing.
    if (
        center_cm is not None
        and center_cm <= obstacle_distance_cm
    ):
        warning_text = (
            f"CENTER <= {obstacle_distance_cm}cm"
        )

        cv2.putText(
            preview,
            warning_text,
            (
                text_x,
                min(
                    banner_bottom - 7,
                    banner_y + 192,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


__all__ = [
    "draw_obstacle_debug",
]
