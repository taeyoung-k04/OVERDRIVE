#!/usr/bin/env python3
"""Semantic traffic-light inference for OVERDRIVE.

The semantic model first locates traffic-light pixels.  HSV color detection is
then restricted to those regions to classify the light as red, yellow, green,
or unknown.  This module does not depend on the real-time driving runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception.infer_sem_class import (
    CLASS_TO_ID,
    DEFAULT_WEIGHTS,
    load_semantic_model,
    semantic_to_class_map,
)

from perception.traffic_light_detector import (
    TrafficLightDetection,
    TrafficLightDetector,
)


TRAFFIC_LIGHT_CLASS_ID = CLASS_TO_ID["traffic_light"]


@dataclass(frozen=True)
class TrafficLightRegion:
    """One connected traffic-light region in the semantic output."""

    label: int
    bbox: tuple[int, int, int, int]
    area: int
    center: tuple[int, int]


@dataclass(frozen=True)
class TrafficLightInferenceResult:
    """Semantic region and color-state result for one frame."""

    class_map: np.ndarray
    mask: np.ndarray
    regions: tuple[TrafficLightRegion, ...]
    color: TrafficLightDetection


class TrafficLightInference:
    """Run semantic traffic-light localization followed by HSV color inference."""

    def __init__(
        self,
        weights: Path | str = DEFAULT_WEIGHTS,
        *,
        backend: str = "onnx",
        imgsz: int = 640,
        device: str = "cpu",
        min_region_area: int = 8,
        mask_padding: int = 5,
        stable_frames: int = 3,
        model: Optional[object] = None,
    ) -> None:
        if min_region_area < 1:
            raise ValueError("min_region_area must be at least 1")
        if mask_padding < 0:
            raise ValueError("mask_padding must be non-negative")

        self.imgsz = int(imgsz)
        self.device = str(device)
        self.min_region_area = int(min_region_area)
        self.mask_padding = int(mask_padding)
        self.model = (
            model
            if model is not None
            else load_semantic_model(Path(weights), backend)
        )
        self.color_detector = TrafficLightDetector(
            roi_bottom_ratio=0.70,
            min_area=5,
            min_circularity=0.30,
            stable_frames=stable_frames,
        )

    def reset(self) -> None:
        """Clear temporal color-state history."""

        self.color_detector.reset()

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have BGR shape (H, W, 3), got {frame.shape}"
            )

    def predict(self, frame: np.ndarray) -> TrafficLightInferenceResult:
        """Infer traffic-light regions and color in one BGR frame."""

        self._validate_frame(frame)
        prediction = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            device=self.device,
            task="semantic",
            rect=False,
            verbose=False,
        )[0]
        class_map = semantic_to_class_map(
            prediction.semantic_mask,
            frame.shape[:2],
        )
        return self.from_class_map(frame, class_map)

    def from_class_map(
        self,
        frame: np.ndarray,
        class_map: np.ndarray,
    ) -> TrafficLightInferenceResult:
        """Classify lights using a class map produced elsewhere."""

        self._validate_frame(frame)
        if not isinstance(class_map, np.ndarray) or class_map.ndim != 2:
            raise ValueError("class_map must be a 2-D numpy.ndarray")
        if class_map.shape != frame.shape[:2]:
            raise ValueError(
                "class_map shape must match frame height and width"
            )

        raw_mask = (class_map == TRAFFIC_LIGHT_CLASS_ID).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            raw_mask,
            connectivity=8,
        )

        regions: list[TrafficLightRegion] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_region_area:
                continue

            center_x = float(centroids[label][0])
            center_y = float(centroids[label][1])
            center_y_ratio = center_y / max(frame.shape[0], 1)

            # 바닥이나 차량 주변의 semantic 오인식을 제거한다.
            if center_y_ratio > 0.72:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            
            regions.append(
                TrafficLightRegion(
                    label=label,
                    bbox=(x, y, width, height),
                    area=area,
                    center=(
                        int(round(center_x)),
                        int(round(center_y)),
                    ),
                )
            )

        regions.sort(
            key=lambda item: item.area,
            reverse=True,
        )

        # Preserve every plausible semantic component. The color detector
        # evaluates all contours in this combined mask and chooses the one
        # with the highest color/shape confidence. Keeping only the largest
        # component could discard a smaller real light in favor of a larger
        # segmentation false positive.
        clean_mask = np.zeros_like(raw_mask)
        for region in regions:
            clean_mask[labels == region.label] = 255

        if self.mask_padding > 0 and np.any(clean_mask):
            size = self.mask_padding * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (size, size),
            )
            color_mask = cv2.dilate(clean_mask, kernel)
        else:
            color_mask = clean_mask

        masked_frame = cv2.bitwise_and(
            frame,
            frame,
            mask=color_mask,
        )
        color = self.color_detector.detect(masked_frame)

        return TrafficLightInferenceResult(
            class_map=class_map,
            mask=clean_mask,
            regions=tuple(regions),
            color=color,
        )

    @staticmethod
    def draw_debug(
        frame: np.ndarray,
        result: TrafficLightInferenceResult,
    ) -> np.ndarray:
        """Draw semantic regions and the inferred light color."""

        output = frame.copy()
        for region in result.regions:
            x, y, width, height = region.bbox
            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 165, 255),
                2,
            )
        return TrafficLightDetector.draw_debug(output, result.color)


__all__ = [
    "DEFAULT_WEIGHTS",
    "TRAFFIC_LIGHT_CLASS_ID",
    "TrafficLightInference",
    "TrafficLightInferenceResult",
    "TrafficLightRegion",
]
