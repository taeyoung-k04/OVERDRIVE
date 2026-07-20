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
    """Draw the current intersection state and final drive command."""

    height, width = preview.shape[:2]

    if control.state == IntersectionState.CRUISE:
        state_text = "CRUISE"
        state_color = (80, 255, 80)

    elif control.state == IntersectionState.WAITING_FOR_GREEN:
        state_text = "WAITING FOR GREEN"
        state_color = (0, 0, 255)

    elif control.state == IntersectionState.CLEARING_STOP_LINE:
        state_text = "CLEARING STOP LINE"
        state_color = (0, 190, 255)

    else:
        state_text = str(control.state.value).upper()
        state_color = (255, 255, 255)

    banner_x = max(10, width - 330)
    banner_y = 10
    banner_width = min(320, width - banner_x - 10)
    banner_height = 90

    cv2.rectangle(
        preview,
        (banner_x, banner_y),
        (
            banner_x + banner_width,
            banner_y + banner_height,
        ),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        preview,
        f"INTERSECTION: {state_text}",
        (banner_x + 10, banner_y + 27),
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
        (banner_x + 10, banner_y + 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        control.reason[:48],
        (banner_x + 10, banner_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )