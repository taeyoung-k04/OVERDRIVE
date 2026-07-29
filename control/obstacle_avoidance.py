from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from control.lane_follower import DrivingLane


@dataclass(frozen=True)
class ObstacleAvoidanceCommand:
    """장애물 회피 컨트롤러의 한 프레임 판단 결과."""

    # 차선 추종기가 따라가야 할 목표 차선
    target_lane: DrivingLane

    # 이번 update에서 새롭게 차선 변경이 요청되었는지
    lane_change_requested: bool

    # 현재 장애물을 이미 처리한 상태인지
    obstacle_latched: bool

    # 영상과 초음파 조건이 동시에 만족되었는지
    fused_front_obstacle: bool

    # 영상에서 현재 주행 경로 안에 차량이 인식되었는지
    front_blocked: bool

    # 중앙 초음파가 장애물 거리 범위 안인지
    ultrasonic_close: bool

    # 사용된 중앙 초음파 거리
    center_distance_cm: Optional[int]

    # 장애물 연속 감지 프레임 수
    blocked_frames: int

    # 기존 visualization과의 호환성을 위해 유지
    clear_frames: int


class ObstacleAvoidanceController:
    """
    차량 인식 결과와 중앙 초음파 거리 정보를 결합하여
    차선 변경 여부를 결정한다.

    차선 변경 조건:
        1. 영상에서 현재 주행 경로 안에 차량이 인식됨
        2. 중앙 초음파 거리가 obstacle_distance_cm 이하
        3. 위 조건이 detection_confirm_frames 동안 연속으로 유지됨

    차선 변경 방향:
        LANE_1 -> LANE_2
        LANE_2 -> LANE_1

    같은 장애물에 대해서는 차선 변경 중 한 번만 반응한다.

    실제 1.5초 차선 변경이 끝나면 lane_change_completed()를
    호출한다. 이때 근접 상태와 장애물 래치를 즉시 해제하며,
    다음 프레임부터 새 차선의 장애물을 다시 감지한다.
    """

    def __init__(
        self,
        initial_lane: DrivingLane,
        *,
        obstacle_distance_cm: int = 200,
        clear_distance_cm: int = 250,
        detection_confirm_frames: int = 3,
        clear_confirm_frames: int = 10,
    ) -> None:
        """
        Args:
            initial_lane:
                차량이 처음 주행하는 차선.

            obstacle_distance_cm:
                이 거리 이하에서 장애물이 있다고 판단한다.

            clear_distance_cm:
                초음파 근접 상태의 히스테리시스 해제 거리.

                단, 차선 변경 완료 시에는 이 거리를 기다리지 않고
                lane_change_completed()에서 즉시 근접 상태를 해제한다.

            detection_confirm_frames:
                영상과 초음파 조건이 몇 프레임 연속 참이어야
                차선을 변경할지 결정한다.

            clear_confirm_frames:
                기존 실시간 주행 코드와의 호환성을 위해 유지한다.

                차선 변경 후 재활성화는 거리 확인이 아니라
                lane_change_completed() 호출로 처리한다.
        """

        if obstacle_distance_cm < 1:
            raise ValueError(
                "obstacle_distance_cm must be at least 1"
            )

        if clear_distance_cm <= obstacle_distance_cm:
            raise ValueError(
                "clear_distance_cm must be greater than "
                "obstacle_distance_cm"
            )

        if detection_confirm_frames < 1:
            raise ValueError(
                "detection_confirm_frames must be at least 1"
            )

        if clear_confirm_frames < 1:
            raise ValueError(
                "clear_confirm_frames must be at least 1"
            )

        self._obstacle_distance_cm = int(
            obstacle_distance_cm
        )

        self._clear_distance_cm = int(
            clear_distance_cm
        )

        self._detection_confirm_frames = int(
            detection_confirm_frames
        )

        # 기존 생성 코드와의 호환성을 위해 저장만 한다.
        self._clear_confirm_frames = int(
            clear_confirm_frames
        )

        self.reset(initial_lane)

    @property
    def target_lane(self) -> DrivingLane:
        """현재 차선 추종기가 따라가야 할 목표 차선."""

        return self._target_lane

    @property
    def obstacle_latched(self) -> bool:
        """현재 장애물을 이미 처리했는지 반환한다."""

        return self._obstacle_latched

    @staticmethod
    def _opposite_lane(
        lane: DrivingLane,
    ) -> DrivingLane:
        """현재 차선의 반대 차선을 반환한다."""

        if lane == DrivingLane.LANE_1:
            return DrivingLane.LANE_2

        if lane == DrivingLane.LANE_2:
            return DrivingLane.LANE_1

        raise ValueError(
            f"Unsupported driving lane: {lane!r}"
        )

    @staticmethod
    def _distance_is_valid(
        distance_cm: Optional[int],
    ) -> bool:
        """
        초음파 거리값이 유효한지 확인한다.

        Arduino에서는 echo를 받지 못하면 -1을 전송하므로
        None, 0, 음수는 유효하지 않은 값으로 처리한다.
        """

        return (
            distance_cm is not None
            and distance_cm > 0
        )

    def _update_ultrasonic_state(
        self,
        center_distance_cm: Optional[int],
    ) -> tuple[bool, bool]:
        """
        중앙 초음파의 장애물 근접 상태를 갱신한다.

        Returns:
            ultrasonic_close:
                현재 초음파 기준 장애물이 가까운지.

            distance_valid:
                전달받은 거리값이 정상적인지.
        """

        distance_valid = self._distance_is_valid(
            center_distance_cm
        )

        if not distance_valid:
            # 센서값이 없다고 해서 장애물이 사라졌다고
            # 판단하지 않는다.
            return self._ultrasonic_close, False

        assert center_distance_cm is not None

        if self._ultrasonic_close:
            # 일반 주행 중에는 170cm 이상일 때
            # 초음파 근접 상태를 해제한다.
            if center_distance_cm >= self._clear_distance_cm:
                self._ultrasonic_close = False

        else:
            # 150cm 이하가 되면 근접 상태로 진입한다.
            if center_distance_cm <= self._obstacle_distance_cm:
                self._ultrasonic_close = True

        return self._ultrasonic_close, True

    def update(
        self,
        *,
        front_blocked: bool,
        center_distance_cm: Optional[int],
        allow_vision_only: bool = False,
    ) -> ObstacleAvoidanceCommand:
        """
        영상과 초음파 값을 받아 차선 변경 여부를 판단한다.

        차선 변경이 요청되면 obstacle_latched가 True가 되어
        1.5초 차선 변경 중 추가 차선 변경 요청을 막는다.

        실제 차선 변경이 끝나면 lane_change_completed()를
        호출해 즉시 다시 장애물을 감지할 수 있도록 한다.
        """

        ultrasonic_close, distance_valid = (
            self._update_ultrasonic_state(
                center_distance_cm
            )
        )

        fused_front_obstacle = (
            bool(front_blocked)
            if allow_vision_only
            else (
                bool(front_blocked)
                and distance_valid
                and ultrasonic_close
            )
        )

        lane_change_requested = False

        if fused_front_obstacle:
            self._blocked_frames += 1
            self._clear_frames = 0

            obstacle_confirmed = (
                self._blocked_frames
                >= self._detection_confirm_frames
            )

            # 차선 변경 중에는 obstacle_latched가 True이므로
            # 추가 차선 변경을 요청하지 않는다.
            if (
                obstacle_confirmed
                and not self._obstacle_latched
            ):
                self._target_lane = self._opposite_lane(
                    self._target_lane
                )

                self._obstacle_latched = True
                lane_change_requested = True

        else:
            # 영상과 초음파 결합 조건이 끊어지면
            # 연속 감지 프레임만 초기화한다.
            self._blocked_frames = 0
            self._clear_frames = 0

            # 기존처럼 거리 조건으로 obstacle_latched를
            # 해제하지 않는다.
            #
            # 래치는 실제 1.5초 차선 변경이 끝났을 때
            # lane_change_completed()에서 즉시 해제한다.

        return ObstacleAvoidanceCommand(
            target_lane=self._target_lane,
            lane_change_requested=lane_change_requested,
            obstacle_latched=self._obstacle_latched,
            fused_front_obstacle=fused_front_obstacle,
            front_blocked=bool(front_blocked),
            ultrasonic_close=ultrasonic_close,
            center_distance_cm=center_distance_cm,
            blocked_frames=self._blocked_frames,
            clear_frames=self._clear_frames,
        )

    def lane_change_completed(
        self,
        lane: DrivingLane,
    ) -> None:
        """
        실제 1.5초 차선 변경이 끝난 순간 호출한다.

        이전 장애물의 래치와 초음파 근접 상태를 즉시 해제한다.
        다음 update부터 새 차선의 장애물을 바로 다시 감지한다.

        새 차선에도 장애물이 있다면:

            front_blocked=True
            center_distance_cm <= obstacle_distance_cm

        조건을 detection_confirm_frames 동안 다시 확인한 뒤
        새로운 차선 변경을 요청한다.
        """

        if lane not in (
            DrivingLane.LANE_1,
            DrivingLane.LANE_2,
        ):
            raise ValueError(
                f"Unsupported driving lane: {lane!r}"
            )

        self._target_lane = lane

        # 이전 차선 장애물의 연속 감지 상태 초기화
        self._blocked_frames = 0
        self._clear_frames = 0

        # 1.5초 차선 변경 완료 즉시 재활성화
        self._obstacle_latched = False
        self._ultrasonic_close = False

    def reset(
        self,
        lane: DrivingLane,
    ) -> None:
        """
        지정한 현재 차선을 기준으로 컨트롤러 상태를 초기화한다.
        """

        if lane not in (
            DrivingLane.LANE_1,
            DrivingLane.LANE_2,
        ):
            raise ValueError(
                f"Unsupported driving lane: {lane!r}"
            )

        self._target_lane = lane

        self._blocked_frames = 0
        self._clear_frames = 0

        self._obstacle_latched = False
        self._ultrasonic_close = False
