#!/usr/bin/env python3
"""HSV-based traffic-light detector for OVERDRIVE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class TrafficLightState(str, Enum):
    UNKNOWN = "unknown"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass(frozen=True)
class TrafficLightDetection:
    """Traffic-light detection result for one frame."""

    observed_state: TrafficLightState
    stable_state: TrafficLightState
    confidence: float
    center: Optional[tuple[int, int]] = None
    radius: int = 0


class TrafficLightDetector:
    """Detect red, yellow, and green circular lights from a BGR frame."""

    def __init__(
        self,
        *,
        roi_bottom_ratio: float = 0.70,
        min_area: int = 20,
        min_circularity: float = 0.45,
        stable_frames: int = 3,
    ) -> None:
        self.roi_bottom_ratio = float(roi_bottom_ratio)
        self.min_area = int(min_area)
        self.min_circularity = float(min_circularity)
        self.stable_frames = max(1, int(stable_frames))

        self._history: deque[TrafficLightState] = deque(
            maxlen=self.stable_frames
        )
        self._stable_state = TrafficLightState.UNKNOWN

    def reset(self) -> None:
        self._history.clear()
        self._stable_state = TrafficLightState.UNKNOWN

    @staticmethod
    def _make_masks(hsv: np.ndarray) -> dict[TrafficLightState, np.ndarray]:
        """Create HSV masks for each traffic-light color."""

        # 빨간색은 Hue 범위의 양쪽 끝에 걸쳐 있다.
        red_low = cv2.inRange(
            hsv,
            np.array([0, 100, 80]),
            np.array([10, 255, 255]),
        )
        red_high = cv2.inRange(
            hsv,
            np.array([170, 100, 80]),
            np.array([179, 255, 255]),
        )
        red = cv2.bitwise_or(red_low, red_high)

        yellow = cv2.inRange(
            hsv,
            np.array([18, 100, 80]),
            np.array([40, 255, 255]),
        )

        green = cv2.inRange(
            hsv,
            np.array([40, 80, 70]),
            np.array([90, 255, 255]),
        )

        return {
            TrafficLightState.RED: red,
            TrafficLightState.YELLOW: yellow,
            TrafficLightState.GREEN: green,
        }

    def detect(self, frame: np.ndarray) -> TrafficLightDetection:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have BGR shape (H, W, 3), got {frame.shape}"
            )

        height, width = frame.shape[:2]

        # 도로 바닥에 있는 색상 물체를 신호등으로 오인하지 않도록
        # 영상의 위쪽 영역만 검사한다.
        roi_bottom = int(height * self.roi_bottom_ratio)
        roi = frame[:roi_bottom, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masks = self._make_masks(hsv)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        best_state = TrafficLightState.UNKNOWN
        best_confidence = 0.0
        best_center: Optional[tuple[int, int]] = None
        best_radius = 0

        for state, mask in masks.items():
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue

                perimeter = float(cv2.arcLength(contour, True))
                if perimeter <= 0:
                    continue

                circularity = float(
                    4.0 * np.pi * area / (perimeter * perimeter)
                )

                if circularity < self.min_circularity:
                    continue

                (center_x, center_y), radius = cv2.minEnclosingCircle(
                    contour
                )

                if radius < 3:
                    continue

                circle_area = np.pi * radius * radius
                fill_ratio = min(1.0, area / max(circle_area, 1.0))

                confidence = float(
                    np.clip(
                        0.65 * circularity + 0.35 * fill_ratio,
                        0.0,
                        1.0,
                    )
                )

                if confidence > best_confidence:
                    best_state = state
                    best_confidence = confidence
                    best_center = (
                        int(round(center_x)),
                        int(round(center_y)),
                    )
                    best_radius = int(round(radius))

        self._history.append(best_state)

        # 같은 색이 연속된 프레임에서 나왔을 때만 안정된 결과로 인정한다.
        if (
            len(self._history) == self.stable_frames
            and len(set(self._history)) == 1
        ):
            self._stable_state = self._history[-1]

        return TrafficLightDetection(
            observed_state=best_state,
            stable_state=self._stable_state,
            confidence=best_confidence,
            center=best_center,
            radius=best_radius,
        )

    @staticmethod
    def draw_debug(
        image: np.ndarray,
        detection: TrafficLightDetection,
    ) -> np.ndarray:
        """Draw the current traffic-light result on an image."""

        result = image.copy()

        if detection.center is not None:
            cv2.circle(
                result,
                detection.center,
                detection.radius,
                (255, 255, 255),
                2,
            )

        text = (
            f"Traffic: {detection.stable_state.value.upper()} "
            f"(raw={detection.observed_state.value}, "
            f"conf={detection.confidence:.2f})"
        )

        cv2.putText(
            result,
            text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return result