#!/usr/bin/env python3
"""Semantic car inference for OVERDRIVE.

This module is independent from the driving runtime.

It extracts the ``car`` class from the trained 8-class semantic output
and converts it into:

- a binary car mask
- connected car regions
- bounding boxes
- a visual ``front_blocked`` decision

The driving runtime combines ``front_blocked`` with the center ultrasonic
distance before requesting a lane change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception.infer_sem_class import (
    load_semantic_model,
    semantic_to_class_map,
)


# Semantic model의 car 클래스 번호
CAR_CLASS_ID = 6

DEFAULT_WEIGHTS = Path(
    "runs/semantic/yolo_lane_sem_class/"
    "train_cpu_640_yolo26n_8class/weights/best.onnx"
)


@dataclass(frozen=True)
class CarDetection:
    """Semantic 출력에서 검출된 하나의 차량 영역."""

    # x, y, width, height
    bbox: tuple[int, int, int, int]

    # 차량으로 분류된 픽셀 개수
    area: int

    # 연결 영역 중심 좌표
    center: tuple[int, int]

    # 차량 bounding box 하단의 이미지 높이 비율
    bottom_y_ratio: float

    # 현재 차량이 전방 경로를 막는다고 판단됐는지
    blocks_path: bool


@dataclass(frozen=True)
class CarInferenceResult:
    """한 프레임의 차량 추론 결과."""

    # 전체 semantic class map
    class_map: np.ndarray

    # car 클래스만 추출한 이진 마스크
    mask: np.ndarray

    # 검출된 차량 영역
    detections: tuple[CarDetection, ...]

    # 영상 기준 전방 경로에 차량이 있는지
    front_blocked: bool


class CarInference:
    """Semantic 모델 출력에서 차량 영역을 추출한다."""

    def __init__(
        self,
        weights: Path | str = DEFAULT_WEIGHTS,
        *,
        backend: str = "onnx",
        imgsz: int = 640,
        device: str = "cpu",
        min_area: int = 120,
        corridor_center_ratio: float = 0.50,
        corridor_width_ratio: float = 0.36,
        blocked_bottom_ratio: float = 0.58,
        model: Optional[object] = None,
    ) -> None:
        """
        Args:
            weights:
                Semantic 모델 가중치 경로.

            backend:
                모델 backend. 예: onnx, pt.

            imgsz:
                모델 입력 영상 크기.

            device:
                추론 장치. 예: cpu, cuda.

            min_area:
                차량으로 인정할 최소 연결 영역 픽셀 수.

            corridor_center_ratio:
                전방 주행 영역 중심의 x축 위치.
                0.5이면 영상 중앙.

            corridor_width_ratio:
                전방 주행 영역의 너비를 전체 영상 너비의
                비율로 표현한 값.

            blocked_bottom_ratio:
                차량 bounding box의 하단 위치가 이 비율 이상이면
                영상 기준으로 충분히 가까운 차량으로 판단한다.

            model:
                이미 로드된 semantic 모델.

                실시간 주행 코드에서는 같은 모델을 재사용하기 위해
                이 인자로 모델을 전달한다.
        """

        if min_area < 1:
            raise ValueError(
                "min_area must be at least 1"
            )

        if not 0.0 <= corridor_center_ratio <= 1.0:
            raise ValueError(
                "corridor_center_ratio must be in [0, 1]"
            )

        if not 0.0 < corridor_width_ratio <= 1.0:
            raise ValueError(
                "corridor_width_ratio must be in (0, 1]"
            )

        if not 0.0 <= blocked_bottom_ratio <= 1.0:
            raise ValueError(
                "blocked_bottom_ratio must be in [0, 1]"
            )

        self.imgsz = int(imgsz)
        self.device = str(device)
        self.min_area = int(min_area)

        self.corridor_center_ratio = float(
            corridor_center_ratio
        )
        self.corridor_width_ratio = float(
            corridor_width_ratio
        )
        self.blocked_bottom_ratio = float(
            blocked_bottom_ratio
        )

        # 실시간 주행 코드에서 이미 로드한 모델이 전달되면
        # 해당 모델을 그대로 사용한다.
        #
        # model이 전달되지 않은 단독 실행 환경에서는
        # weights를 이용해 새 모델을 로드한다.
        self.model = (
            model
            if model is not None
            else load_semantic_model(
                Path(weights),
                backend,
            )
        )

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> None:
        """입력 카메라 프레임 형식을 확인한다."""

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "frame must be a numpy.ndarray"
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must have BGR shape "
                f"(H, W, 3), got {frame.shape}"
            )

        if frame.shape[0] < 1 or frame.shape[1] < 1:
            raise ValueError(
                "frame height and width must be positive"
            )

    @staticmethod
    def _validate_class_map(
        class_map: np.ndarray,
    ) -> None:
        """Semantic class map 형식을 확인한다."""

        if not isinstance(class_map, np.ndarray):
            raise TypeError(
                "class_map must be a numpy.ndarray"
            )

        if class_map.ndim != 2:
            raise ValueError(
                "class_map must be 2-D, "
                f"got shape {class_map.shape}"
            )

        if (
            class_map.shape[0] < 1
            or class_map.shape[1] < 1
        ):
            raise ValueError(
                "class_map height and width must be positive"
            )

    def predict(
        self,
        frame: np.ndarray,
    ) -> CarInferenceResult:
        """
        하나의 BGR 프레임에서 semantic 추론까지 직접 수행한다.

        단독 차량 추론 테스트에서 사용할 수 있다.

        실시간 주행 코드에서는 이미 생성된 raw_class_map을
        from_class_map()에 전달하는 방식을 권장한다.
        """

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

        return self.from_class_map(
            class_map
        )

    def from_class_map(
        self,
        class_map: np.ndarray,
        *,
        corridor_left_ratio: Optional[float] = None,
        corridor_right_ratio: Optional[float] = None,
        blocked_bottom_ratio: Optional[float] = None,
    ) -> CarInferenceResult:
        """
        다른 코드에서 이미 생성한 semantic class map으로부터
        차량 영역을 추출한다.

        실시간 주행 코드에서는 모델 추론을 한 번만 수행한 뒤
        이 메서드를 호출한다.
        """

        self._validate_class_map(
            class_map
        )

        height, width = class_map.shape

        if (
            corridor_left_ratio is None
        ) != (
            corridor_right_ratio is None
        ):
            raise ValueError(
                "corridor_left_ratio and corridor_right_ratio "
                "must be provided together"
            )

        if corridor_left_ratio is None:
            corridor_half = (
                self.corridor_width_ratio
                * 0.5
            )
            resolved_corridor_left_ratio = (
                self.corridor_center_ratio
                - corridor_half
            )
            resolved_corridor_right_ratio = (
                self.corridor_center_ratio
                + corridor_half
            )
        else:
            resolved_corridor_left_ratio = float(
                corridor_left_ratio
            )
            resolved_corridor_right_ratio = float(
                corridor_right_ratio
            )

        if not (
            0.0
            <= resolved_corridor_left_ratio
            < resolved_corridor_right_ratio
            <= 1.0
        ):
            raise ValueError(
                "corridor ratios must satisfy "
                "0 <= left < right <= 1"
            )

        resolved_blocked_bottom_ratio = (
            self.blocked_bottom_ratio
            if blocked_bottom_ratio is None
            else float(blocked_bottom_ratio)
        )

        if not 0.0 <= resolved_blocked_bottom_ratio <= 1.0:
            raise ValueError(
                "blocked_bottom_ratio must be in [0, 1]"
            )

        # car 클래스만 255로 설정한 이진 마스크
        mask = (
            (class_map == CAR_CLASS_ID)
            .astype(np.uint8)
            * 255
        )

        # 서로 연결된 차량 픽셀을 개별 영역으로 분리
        count, _, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        # 주행 경로 영역 계산. 일반 주행은 생성자 기본값을 사용하고,
        # 고정 트랙의 특수 코너에서는 호출자가 비대칭 영역을 지정한다.
        corridor_left = (
            width
            * resolved_corridor_left_ratio
        )

        corridor_right = (
            width
            * resolved_corridor_right_ratio
        )

        detections: list[CarDetection] = []

        # label 0은 배경이므로 1부터 시작
        for label in range(1, count):
            x = int(
                stats[label, cv2.CC_STAT_LEFT]
            )
            y = int(
                stats[label, cv2.CC_STAT_TOP]
            )
            box_width = int(
                stats[label, cv2.CC_STAT_WIDTH]
            )
            box_height = int(
                stats[label, cv2.CC_STAT_HEIGHT]
            )
            area = int(
                stats[label, cv2.CC_STAT_AREA]
            )

            # 너무 작은 영역은 노이즈로 제거
            if area < self.min_area:
                continue

            right = x + box_width
            bottom = y + box_height

            # 차량 bounding box가 전방 corridor와 겹치는지
            overlaps_corridor = (
                right >= corridor_left
                and x <= corridor_right
            )

            # 차량 bounding box 하단의 이미지 높이 비율
            bottom_y_ratio = (
                bottom / height
            )

            # 영상 기준 전방 경로를 막는 차량인지 판단
            blocks_path = (
                overlaps_corridor
                and bottom_y_ratio
                >= resolved_blocked_bottom_ratio
            )

            detections.append(
                CarDetection(
                    bbox=(
                        x,
                        y,
                        box_width,
                        box_height,
                    ),
                    area=area,
                    center=(
                        int(
                            round(
                                centroids[label][0]
                            )
                        ),
                        int(
                            round(
                                centroids[label][1]
                            )
                        ),
                    ),
                    bottom_y_ratio=float(
                        bottom_y_ratio
                    ),
                    blocks_path=blocks_path,
                )
            )

        # 영상 아래쪽에 가까운 차량부터 정렬
        detections.sort(
            key=lambda item: item.bottom_y_ratio,
            reverse=True,
        )

        return CarInferenceResult(
            class_map=class_map,
            mask=mask,
            detections=tuple(
                detections
            ),
            front_blocked=any(
                detection.blocks_path
                for detection in detections
            ),
        )

    @staticmethod
    def draw_debug(
        frame: np.ndarray,
        result: CarInferenceResult,
    ) -> np.ndarray:
        """차량 bounding box와 front_blocked 상태를 표시한다."""

        CarInference._validate_frame(
            frame
        )

        if result.class_map.shape != frame.shape[:2]:
            raise ValueError(
                "result.class_map shape must match "
                "frame height and width"
            )

        output = frame.copy()

        for detection in result.detections:
            x, y, width, height = (
                detection.bbox
            )

            # 주행 경로를 막으면 빨간색,
            # 그 외 차량은 자홍색
            color = (
                (0, 0, 255)
                if detection.blocks_path
                else (255, 0, 255)
            )

            cv2.rectangle(
                output,
                (x, y),
                (
                    x + width,
                    y + height,
                ),
                color,
                2,
            )

        status_text = (
            "CAR: BLOCKED"
            if result.front_blocked
            else "CAR: CLEAR"
        )

        status_color = (
            (0, 0, 255)
            if result.front_blocked
            else (0, 255, 0)
        )

        cv2.putText(
            output,
            status_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        return output


__all__ = [
    "CAR_CLASS_ID",
    "DEFAULT_WEIGHTS",
    "CarDetection",
    "CarInference",
    "CarInferenceResult",
]
