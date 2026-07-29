#!/usr/bin/env python3
"""Debug visualization for intersection control."""

from __future__ import annotations

import cv2
import numpy as np

from control.intersection_controller import (
    IntersectionControlOutput,
    IntersectionState,
)


def draw_intersection_debug(
    preview: np.ndarray,
    control: IntersectionControlOutput,
) -> None:
    """Draw the current intersection state and final drive command.

    The banner shows:
    - current intersection state,
    - final steering and speed sent to the vehicle,
    - whether traffic-light and stop-line perception are active,
    - the controller's state-transition reason.
    """

    if not isinstance(preview, np.ndarray):
        raise TypeError("preview must be a numpy.ndarray")

    if preview.ndim != 3 or preview.shape[2] != 3:
        raise ValueError(
            f"preview must have BGR shape (H, W, 3), got {preview.shape}"
        )

    height, width = preview.shape[:2]

    # -----------------------------------------------------------------
    # State-specific text and color
    # -----------------------------------------------------------------

    if control.state == IntersectionState.SEARCHING_RED:
        state_text = "SEARCHING RED"
        state_color = (80, 255, 80)

    elif control.state == IntersectionState.APPROACHING_STOP_LINE:
        state_text = "RED / STOP LINE ARMED"
        state_color = (0, 165, 255)

    elif control.state == IntersectionState.WAITING_FOR_GREEN:
        state_text = "STOPPED / WAITING GREEN"
        state_color = (0, 0, 255)

    elif control.state == IntersectionState.CLEARING_INTERSECTION:
        state_text = "GREEN / CLEARING"
        state_color = (0, 255, 0)

    else:
        state_text = str(control.state.value).upper()
        state_color = (255, 255, 255)

    # IntersectionControlOutput에 새 필드가 아직 반영되지 않은 상태에서도
    # debug 창이 바로 종료되지 않도록 getattr fallback을 사용한다.
    traffic_light_required = bool(
        getattr(
            control,
            "traffic_light_required",
            False,
        )
    )
    stop_line_required = bool(
        getattr(
            control,
            "stop_line_required",
            False,
        )
    )
    override_active = bool(
        getattr(
            control,
            "override_active",
            False,
        )
    )

    # -----------------------------------------------------------------
    # Banner geometry
    # -----------------------------------------------------------------

    banner_margin = 10
    banner_width = min(
        390,
        max(250, width - banner_margin * 2),
    )
    banner_height = 125

    banner_x = max(
        banner_margin,
        width - banner_width - banner_margin,
    )
    banner_y = banner_margin

    banner_right = min(
        width - banner_margin,
        banner_x + banner_width,
    )
    banner_bottom = min(
        height - banner_margin,
        banner_y + banner_height,
    )

    # Semi-opaque-looking dark banner.
    # OpenCV의 rectangle에는 alpha blending이 없으므로
    # 별도 overlay를 만들어 합성한다.
    overlay = preview.copy()

    cv2.rectangle(
        overlay,
        (banner_x, banner_y),
        (banner_right, banner_bottom),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.82,
        preview,
        0.18,
        0.0,
        preview,
    )

    # State-colored border
    cv2.rectangle(
        preview,
        (banner_x, banner_y),
        (banner_right, banner_bottom),
        state_color,
        2,
    )

    # -----------------------------------------------------------------
    # Text rows
    # -----------------------------------------------------------------

    text_x = banner_x + 10

    cv2.putText(
        preview,
        f"INTERSECTION: {state_text}",
        (text_x, banner_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        state_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        (
            f"FINAL steer={control.steering:+.3f} "
            f"speed={control.speed}"
        ),
        (text_x, banner_y + 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        (
            f"TL={int(traffic_light_required)} "
            f"STOPLINE={int(stop_line_required)} "
            f"OVERRIDE={int(override_active)}"
        ),
        (text_x, banner_y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    reason = str(
        getattr(
            control,
            "reason",
            "",
        )
    )

    cv2.putText(
        preview,
        reason[:58],
        (text_x, banner_y + 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------
    # Extra safety indicator while stopped
    # -----------------------------------------------------------------

    if (
        control.state == IntersectionState.WAITING_FOR_GREEN
        and control.speed == 0
    ):
        stop_text = "VEHICLE STOP COMMAND ACTIVE"

        text_size, _ = cv2.getTextSize(
            stop_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            2,
        )

        stop_text_x = max(
            10,
            (width - text_size[0]) // 2,
        )
        stop_text_y = max(
            25,
            height - 20,
        )

        cv2.putText(
            preview,
            stop_text,
            (stop_text_x, stop_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


__all__ = [
    "draw_intersection_debug",
]