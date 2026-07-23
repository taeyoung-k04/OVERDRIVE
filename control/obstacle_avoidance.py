from __future__ import annotations

from dataclasses import dataclass

from control.lane_follower import DrivingLane


@dataclass(frozen=True)
class ObstacleAvoidanceCommand:
    """장애물 회피 판단 결과."""

    target_lane: DrivingLane
    lane_change_requested: bool
    obstacle_latched: bool


class ObstacleAvoidanceController:
    """
    새로운 전방 장애물이 감지될 때마다 목표 차선을 전환한다.

    LANE_1 -> LANE_2
    LANE_2 -> LANE_1

    같은 장애물이 계속 감지되는 동안에는 한 번만 차선을 변경한다.
    장애물이 일정 프레임 동안 사라져야 다음 장애물을 받을 수 있다.
    """

    def __init__(
        self,
        initial_lane: DrivingLane,
        detection_confirm_frames: int = 3,
        clear_confirm_frames: int = 10,
    ) -> None:
        if detection_confirm_frames < 1:
            raise ValueError(
                "detection_confirm_frames must be at least 1"
            )

        if clear_confirm_frames < 1:
            raise ValueError(
                "clear_confirm_frames must be at least 1"
            )

        self._detection_confirm_frames = detection_confirm_frames
        self._clear_confirm_frames = clear_confirm_frames

        self.reset(initial_lane)

    @property
    def target_lane(self) -> DrivingLane:
        return self._target_lane

    @staticmethod
    def _opposite_lane(lane: DrivingLane) -> DrivingLane:
        if lane == DrivingLane.LANE_1:
            return DrivingLane.LANE_2

        return DrivingLane.LANE_1

    def update(
        self,
        front_blocked: bool,
    ) -> ObstacleAvoidanceCommand:
        """
        전방 장애물 감지값을 받아 목표 차선을 결정한다.

        Args:
            front_blocked:
                True: 전방 장애물 감지
                False: 전방 장애물 없음
        """

        lane_change_requested = False

        if front_blocked:
            self._blocked_frames += 1
            self._clear_frames = 0

            obstacle_confirmed = (
                self._blocked_frames
                >= self._detection_confirm_frames
            )

            # 현재 장애물을 아직 처리하지 않았을 때만 변경
            if obstacle_confirmed and not self._obstacle_latched:
                self._target_lane = self._opposite_lane(
                    self._target_lane
                )

                self._obstacle_latched = True
                lane_change_requested = True

        else:
            self._blocked_frames = 0

            if self._obstacle_latched:
                self._clear_frames += 1

                obstacle_cleared = (
                    self._clear_frames
                    >= self._clear_confirm_frames
                )

                if obstacle_cleared:
                    # 다음 장애물 감지를 받을 수 있도록 재활성화
                    self._obstacle_latched = False
                    self._clear_frames = 0

            else:
                self._clear_frames = 0

        return ObstacleAvoidanceCommand(
            target_lane=self._target_lane,
            lane_change_requested=lane_change_requested,
            obstacle_latched=self._obstacle_latched,
        )

    def reset(
        self,
        lane: DrivingLane,
    ) -> None:
        """현재 차선을 기준으로 장애물 회피 상태를 초기화한다."""

        self._target_lane = lane
        self._blocked_frames = 0
        self._clear_frames = 0
        self._obstacle_latched = False