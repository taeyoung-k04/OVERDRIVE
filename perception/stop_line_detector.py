#!/usr/bin/env python3
"""Semantic stop-line detector for OVERDRIVE.

This module performs perception only:
- receives a 2-D semantic class map,
- extracts geometrically plausible stop-line components,
- checks whether the line is close enough to stop,
- confirms the result over several frames.

Traffic-light interpretation and Arduino commands should remain in the runtime
or in a separate intersection controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from perception.infer_sem_class import CLASS_TO_ID


@dataclass(frozen=True)
class StopLineDetection:
    """Stop-line detection result for one semantic frame."""

    observed: bool
    confirmed: bool
    should_stop: bool
    confidence: float

    bbox: Optional[tuple[int, int, int, int]] = None
    center: Optional[tuple[int, int]] = None

    area: float = 0.0
    width_ratio: float = 0.0
    center_y_ratio: float = 0.0
    bottom_y_ratio: float = 0.0
    aspect_ratio: float = 0.0
    angle_deg: float = 90.0
    corridor_overlap: float = 0.0

    reason: str = ""


@dataclass(frozen=True)
class _StopLineCandidate:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]

    area: float
    width_ratio: float
    center_y_ratio: float
    bottom_y_ratio: float
    aspect_ratio: float
    angle_deg: float
    fill_ratio: float
    corridor_overlap: float

    confidence: float
    selection_score: float


class StopLineDetector:
    """Detect a nearby horizontal stop line from a semantic class map.

    The stop-line class is taken from ``CLASS_TO_ID["stop_line"]``.

    ``observed`` means that a geometrically plausible stop line exists in the
    current frame. ``should_stop`` becomes true only when that line is close
    enough and has been observed repeatedly across recent frames.
    """

    def __init__(
        self,
        *,
        roi_top_ratio: float = 0.45,
        trigger_y_ratio: float = 0.72,
        min_area_ratio: float = 0.00045,
        min_area_px: int = 90,
        min_width_ratio: float = 0.16,
        min_aspect_ratio: float = 2.2,
        max_height_ratio: float = 0.20,
        max_angle_deg: float = 22.0,
        min_fill_ratio: float = 0.18,
        vehicle_x_ratio: float = 0.50,
        vehicle_corridor_ratio: float = 0.30,
        min_corridor_overlap: float = 0.20,
        min_confidence: float = 0.52,
        history_size: int = 5,
        confirm_frames: int = 3,
    ) -> None:
        if not 0.0 <= roi_top_ratio < 1.0:
            raise ValueError("roi_top_ratio must be in [0, 1)")
        if not 0.0 <= trigger_y_ratio <= 1.0:
            raise ValueError("trigger_y_ratio must be in [0, 1]")
        if trigger_y_ratio <= roi_top_ratio:
            raise ValueError("trigger_y_ratio must be below roi_top_ratio")
        if min_area_ratio < 0.0:
            raise ValueError("min_area_ratio must be non-negative")
        if min_area_px < 0:
            raise ValueError("min_area_px must be non-negative")
        if not 0.0 <= min_width_ratio <= 1.0:
            raise ValueError("min_width_ratio must be in [0, 1]")
        if min_aspect_ratio <= 0.0:
            raise ValueError("min_aspect_ratio must be positive")
        if not 0.0 < max_height_ratio <= 1.0:
            raise ValueError("max_height_ratio must be in (0, 1]")
        if not 0.0 < max_angle_deg <= 90.0:
            raise ValueError("max_angle_deg must be in (0, 90]")
        if not 0.0 <= min_fill_ratio <= 1.0:
            raise ValueError("min_fill_ratio must be in [0, 1]")
        if not 0.0 <= vehicle_x_ratio <= 1.0:
            raise ValueError("vehicle_x_ratio must be in [0, 1]")
        if not 0.0 < vehicle_corridor_ratio <= 1.0:
            raise ValueError("vehicle_corridor_ratio must be in (0, 1]")
        if not 0.0 <= min_corridor_overlap <= 1.0:
            raise ValueError("min_corridor_overlap must be in [0, 1]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if not 1 <= confirm_frames <= history_size:
            raise ValueError(
                "confirm_frames must be between 1 and history_size"
            )

        self.roi_top_ratio = float(roi_top_ratio)
        self.trigger_y_ratio = float(trigger_y_ratio)

        self.min_area_ratio = float(min_area_ratio)
        self.min_area_px = int(min_area_px)
        self.min_width_ratio = float(min_width_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_height_ratio = float(max_height_ratio)
        self.max_angle_deg = float(max_angle_deg)
        self.min_fill_ratio = float(min_fill_ratio)

        self.vehicle_x_ratio = float(vehicle_x_ratio)
        self.vehicle_corridor_ratio = float(vehicle_corridor_ratio)
        self.min_corridor_overlap = float(min_corridor_overlap)

        self.min_confidence = float(min_confidence)
        self.confirm_frames = int(confirm_frames)
        self._close_history: deque[bool] = deque(
            maxlen=int(history_size)
        )

    def reset(self) -> None:
        """Clear temporal confirmation history."""
        self._close_history.clear()

    @staticmethod
    def _validate_class_map(class_map: np.ndarray) -> None:
        if not isinstance(class_map, np.ndarray):
            raise TypeError("class_map must be a numpy.ndarray")
        if class_map.ndim != 2:
            raise ValueError(
                f"class_map must have shape (H, W), got {class_map.shape}"
            )
        if class_map.size == 0:
            raise ValueError("class_map must not be empty")

    @staticmethod
    def _odd(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise and connect short horizontal fragments."""
        height, width = mask.shape

        open_size = self._odd(max(3, round(min(height, width) * 0.006)))
        close_width = self._odd(max(9, round(width * 0.025)))
        close_height = self._odd(max(3, round(height * 0.010)))

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_width, close_height),
        )

        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
        )
        return cleaned

    @staticmethod
    def _horizontal_angle_deg(contour: np.ndarray) -> float:
        """Return the contour's absolute angle from the horizontal axis."""
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 2:
            return 90.0

        vx, vy, _, _ = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).reshape(-1)

        angle = abs(math.degrees(math.atan2(float(vy), float(vx))))
        if angle > 90.0:
            angle = 180.0 - angle
        return float(angle)

    def _corridor_overlap(
        self,
        *,
        x: int,
        width_px: int,
        image_width: int,
    ) -> float:
        corridor_width = max(
            1.0,
            image_width * self.vehicle_corridor_ratio,
        )
        corridor_center = image_width * self.vehicle_x_ratio
        corridor_left = corridor_center - corridor_width * 0.5
        corridor_right = corridor_center + corridor_width * 0.5

        component_left = float(x)
        component_right = float(x + width_px)

        overlap = max(
            0.0,
            min(component_right, corridor_right)
            - max(component_left, corridor_left),
        )
        return float(np.clip(overlap / corridor_width, 0.0, 1.0))

    @staticmethod
    def _normalized_score(
        value: float,
        minimum: float,
        full_score_at: float,
    ) -> float:
        if full_score_at <= minimum:
            return float(value >= minimum)

        return float(
            np.clip(
                (value - minimum) / (full_score_at - minimum),
                0.0,
                1.0,
            )
        )

    def _build_candidate(
        self,
        *,
        contour: np.ndarray,
        frame_shape: tuple[int, int],
        minimum_area: float,
    ) -> Optional[_StopLineCandidate]:
        height, width = frame_shape

        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            return None

        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            return None

        width_ratio = box_width / max(width, 1)
        height_ratio = box_height / max(height, 1)
        aspect_ratio = box_width / max(box_height, 1)
        fill_ratio = area / max(float(box_width * box_height), 1.0)
        angle_deg = self._horizontal_angle_deg(contour)

        center_x = x + box_width * 0.5
        center_y = y + box_height * 0.5
        center_y_ratio = center_y / max(height - 1, 1)
        bottom_y_ratio = (y + box_height) / max(height - 1, 1)

        corridor_overlap = self._corridor_overlap(
            x=x,
            width_px=box_width,
            image_width=width,
        )

        if width_ratio < self.min_width_ratio:
            return None
        if aspect_ratio < self.min_aspect_ratio:
            return None
        if height_ratio > self.max_height_ratio:
            return None
        if angle_deg > self.max_angle_deg:
            return None
        if fill_ratio < self.min_fill_ratio:
            return None
        if corridor_overlap < self.min_corridor_overlap:
            return None

        area_score = float(
            np.clip(area / max(minimum_area * 4.0, 1.0), 0.0, 1.0)
        )
        width_score = self._normalized_score(
            width_ratio,
            self.min_width_ratio,
            0.55,
        )
        aspect_score = self._normalized_score(
            aspect_ratio,
            self.min_aspect_ratio,
            7.0,
        )
        angle_score = float(
            np.clip(
                1.0 - angle_deg / max(self.max_angle_deg, 1e-6),
                0.0,
                1.0,
            )
        )
        fill_score = self._normalized_score(
            fill_ratio,
            self.min_fill_ratio,
            0.65,
        )

        confidence = float(
            np.clip(
                0.16 * area_score
                + 0.25 * width_score
                + 0.18 * aspect_score
                + 0.19 * angle_score
                + 0.10 * fill_score
                + 0.12 * corridor_overlap,
                0.0,
                1.0,
            )
        )

        # If several valid horizontal components exist, prefer the one that is
        # both reliable and closest to the vehicle.
        selection_score = float(
            0.78 * confidence + 0.22 * center_y_ratio
        )

        return _StopLineCandidate(
            bbox=(x, y, box_width, box_height),
            center=(
                int(round(center_x)),
                int(round(center_y)),
            ),
            area=area,
            width_ratio=width_ratio,
            center_y_ratio=center_y_ratio,
            bottom_y_ratio=bottom_y_ratio,
            aspect_ratio=aspect_ratio,
            angle_deg=angle_deg,
            fill_ratio=fill_ratio,
            corridor_overlap=corridor_overlap,
            confidence=confidence,
            selection_score=selection_score,
        )

    def detect(self, class_map: np.ndarray) -> StopLineDetection:
        """Detect and temporally confirm a nearby stop line."""
        self._validate_class_map(class_map)

        height, width = class_map.shape
        roi_top = int(
            np.clip(
                round(height * self.roi_top_ratio),
                0,
                height - 1,
            )
        )

        stop_line_id = int(CLASS_TO_ID["stop_line"])
        mask = (class_map == stop_line_id).astype(np.uint8)
        mask[:roi_top, :] = 0
        mask = self._clean_mask(mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        minimum_area = max(
            float(self.min_area_px),
            float(height * width) * self.min_area_ratio,
        )

        candidates: list[_StopLineCandidate] = []
        for contour in contours:
            candidate = self._build_candidate(
                contour=contour,
                frame_shape=(height, width),
                minimum_area=minimum_area,
            )
            if candidate is not None:
                candidates.append(candidate)

        best = (
            max(candidates, key=lambda candidate: candidate.selection_score)
            if candidates
            else None
        )

        observed = (
            best is not None
            and best.confidence >= self.min_confidence
        )
        close_now = (
            observed
            and best is not None
            and best.center_y_ratio >= self.trigger_y_ratio
        )

        self._close_history.append(bool(close_now))

        recent_history = list(
            self._close_history
        )[-self.confirm_frames:]

        confirmed = (
                len(recent_history) == self.confirm_frames
                and all(recent_history)
                )
        
        should_stop = bool(confirmed)

        if best is None:
            return StopLineDetection(
                observed=False,
                confirmed=False,
                should_stop=False,
                confidence=0.0,
                reason="no geometrically valid stop-line component",
            )

        if not observed:
            reason = (
                f"candidate confidence too low: "
                f"{best.confidence:.2f} < {self.min_confidence:.2f}"
            )
        elif best.center_y_ratio < self.trigger_y_ratio:
            reason = (
                f"stop line observed but still far: "
                f"y={best.center_y_ratio:.3f} < "
                f"{self.trigger_y_ratio:.3f}"
            )
        elif not confirmed:
            reason = (
                f"near stop line awaiting temporal confirmation: "
                f"{sum(self._close_history)}/{self.confirm_frames}"
            )
        else:
            reason = "near stop line confirmed"

        return StopLineDetection(
            observed=observed,
            confirmed=confirmed,
            should_stop=should_stop,
            confidence=best.confidence,
            bbox=best.bbox,
            center=best.center,
            area=best.area,
            width_ratio=best.width_ratio,
            center_y_ratio=best.center_y_ratio,
            bottom_y_ratio=best.bottom_y_ratio,
            aspect_ratio=best.aspect_ratio,
            angle_deg=best.angle_deg,
            corridor_overlap=best.corridor_overlap,
            reason=reason,
        )

    def draw_debug(
        self,
        image: np.ndarray,
        detection: StopLineDetection,
    ) -> np.ndarray:
        """Draw stop-line geometry and the calibrated stop threshold."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have BGR shape (H, W, 3), got {image.shape}"
            )

        result = image.copy()
        height, width = result.shape[:2]

        trigger_y = int(
            round((height - 1) * self.trigger_y_ratio)
        )
        cv2.line(
            result,
            (0, trigger_y),
            (width - 1, trigger_y),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        corridor_width = width * self.vehicle_corridor_ratio
        corridor_center = width * self.vehicle_x_ratio
        corridor_left = int(
            round(corridor_center - corridor_width * 0.5)
        )
        corridor_right = int(
            round(corridor_center + corridor_width * 0.5)
        )
        cv2.rectangle(
            result,
            (max(0, corridor_left), 0),
            (min(width - 1, corridor_right), height - 1),
            (120, 120, 120),
            1,
        )

        if detection.bbox is not None:
            x, y, box_width, box_height = detection.bbox
            box_color = (
                (0, 0, 255)
                if detection.should_stop
                else (0, 255, 255)
            )
            cv2.rectangle(
                result,
                (x, y),
                (x + box_width, y + box_height),
                box_color,
                2,
            )

        text = (
            f"StopLine observed={int(detection.observed)} "
            f"confirmed={int(detection.confirmed)} "
            f"y={detection.center_y_ratio:.2f} "
            f"conf={detection.confidence:.2f}"
        )
        cv2.putText(
            result,
            text,
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return result


__all__ = [
    "StopLineDetection",
    "StopLineDetector",
]
