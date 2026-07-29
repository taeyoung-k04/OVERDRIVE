#!/usr/bin/env python3
"""Intersection state controller for OVERDRIVE.

Perception modules answer:
- Is a nearby stop line confirmed?
- What traffic-light color is stable?

This controller answers:
- Should the vehicle keep following the lane?
- Should it stop and wait?
- When may it leave the intersection?
- When may stop-line detection be armed again?

The controller does not run OpenCV or semantic inference. It only combines
perception results with the lane follower's planned steering and speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Optional

import numpy as np

from perception.stop_line_detector import StopLineDetection
from perception.traffic_light_detector import (
    TrafficLightDetection,
    TrafficLightState,
)


class IntersectionState(str, Enum):
    """High-level states for one stop-line/traffic-light interaction."""

    SEARCHING_RED = "searching_red"
    APPROACHING_STOP_LINE = "approaching_stop_line"
    WAITING_FOR_GREEN = "waiting_for_green"
    CLEARING_INTERSECTION = "clearing_intersection"
    


@dataclass(frozen=True)
class IntersectionControlOutput:
    """Final command and state produced for one frame."""

    state: IntersectionState
    steering: float
    speed: int

    # True while intersection logic is replacing or constraining lane control.
    override_active: bool

    # The runtime should run TrafficLightDetector only while this is true.
    traffic_light_required: bool

    stop_line_required: bool = False

    # Reset TrafficLightDetector after this frame. This prevents an old stable
    # GREEN value from being reused at the next intersection.
    reset_traffic_light_detector: bool = False
    reset_stop_line_detector: bool = False

    entered_waiting: bool = False
    released_on_green: bool = False
    intersection_cleared: bool = False

    reason: str = ""


class IntersectionController:
    """Control stopping, signal waiting, and intersection release.

    State flow
    ----------
    CRUISE
        Lane follower controls steering/speed. A confirmed nearby stop line
        changes the state to WAITING_FOR_GREEN.

    WAITING_FOR_GREEN
        Speed is forced to zero. UNKNOWN, RED, and YELLOW all keep the vehicle
        stopped. A stable GREEN changes the state to CLEARING_STOP_LINE.

    CLEARING_STOP_LINE
        Lane following resumes, but the same stop line is ignored until it has
        disappeared for several frames. This prevents immediate re-stopping.
    """

    def __init__(
        self,
        *,
        clear_confirm_frames: int = 8,
        max_clearing_seconds: float = 10.0,
        minimum_green_confidence: float = 0.0,
        hold_steering_while_stopped: bool = True,
        departure_speed_cap: Optional[int] = None,
    ) -> None:
        if clear_confirm_frames < 1:
            raise ValueError("clear_confirm_frames must be at least 1")
        if max_clearing_seconds <= 0.0:
            raise ValueError("max_clearing_seconds must be positive")
        if not 0.0 <= minimum_green_confidence <= 1.0:
            raise ValueError(
                "minimum_green_confidence must be between 0 and 1"
            )
        if departure_speed_cap is not None and not 0 <= departure_speed_cap <= 255:
            raise ValueError("departure_speed_cap must be between 0 and 255")

        self.clear_confirm_frames = int(clear_confirm_frames)
        self.max_clearing_seconds = float(max_clearing_seconds)
        self.minimum_green_confidence = float(minimum_green_confidence)
        self.hold_steering_while_stopped = bool(
            hold_steering_while_stopped
        )
        self.departure_speed_cap = (
            None
            if departure_speed_cap is None
            else int(departure_speed_cap)
        )

        self.state = IntersectionState.SEARCHING_RED
        self._red_seen = False
        self._clear_frames = 0
        self._state_started_at = time.perf_counter()
        self._stopped_steering = 0.0

    @property
    def requires_traffic_light(self) -> bool:
        return self.state in {
            IntersectionState.SEARCHING_RED,
            IntersectionState.APPROACHING_STOP_LINE,
            IntersectionState.WAITING_FOR_GREEN,
        }

    @property
    def requires_stop_line(self) -> bool:
        return self.state in {
            IntersectionState.APPROACHING_STOP_LINE,
            IntersectionState.CLEARING_INTERSECTION,
        }

    @property
    def allows_obstacle_avoidance(self) -> bool:
        """Disable lane changes while waiting for or clearing an intersection."""
        return self.state == IntersectionState.SEARCHING_RED

    def reset(self) -> None:
        """Return to normal lane-following state."""
        self.state = IntersectionState.SEARCHING_RED
        self._red_seen = False
        self._clear_frames = 0
        self._state_started_at = time.perf_counter()
        self._stopped_steering = 0.0

    def _transition(
        self,
        new_state: IntersectionState,
        now: float,
    ) -> None:
        self.state = new_state
        self._state_started_at = now

        # if new_state != IntersectionState.CLEARING_STOP_LINE:
        #     self._clear_frames = 0

    @staticmethod
    def _signal_confirmed(
        detection: Optional[TrafficLightDetection],
        expected: TrafficLightState,
        *,
        visible: bool,
        minimum_confidence: float,
    ) -> bool:
        return bool(
            visible
            and detection is not None
            and detection.observed_state == expected
            and detection.stable_state == expected
            and detection.confidence >= minimum_confidence
        )
    
    @staticmethod
    def _stop_line_observed(
        detection: Optional[StopLineDetection],
    ) -> bool:
        if detection is None:
            return False

        observed = getattr(
            detection,
            "observed",
            None,
        )

        if observed is not None:
            return bool(observed)

        return bool(
            getattr(detection, "detected", False)
        )
    def _stopped_command(
        self,
        *,
        reason: str,
        reset_traffic_light_detector: bool = False,
        entered_waiting: bool = False,
    ) -> IntersectionControlOutput:
        steering = (
            self._stopped_steering
            if self.hold_steering_while_stopped
            else 0.0
        )

        return IntersectionControlOutput(
            state=self.state,
            steering=steering,
            speed=0,
            override_active=True,
            traffic_light_required=True,
            stop_line_required=False,
            reset_traffic_light_detector=reset_traffic_light_detector,
            entered_waiting=entered_waiting,
            reason=reason,
        )

    def _departure_speed(self, base_speed: int) -> int:
        speed = int(np.clip(round(base_speed), 0, 255))
        if self.departure_speed_cap is not None:
            speed = min(speed, self.departure_speed_cap)
        return speed

    def update(
        self,
        *,
        stop_line: Optional[StopLineDetection],
        traffic_light: Optional[TrafficLightDetection],
        traffic_light_visible: bool,
        base_steering: float,
        base_speed: int,
        driving_enabled: bool,
        now: Optional[float] = None,
    ) -> IntersectionControlOutput:
        """Combine perception results with the lane follower's planned command."""
        current_time = (
            time.perf_counter()
            if now is None
            else float(now)
        )

        planned_steering = float(
            np.clip(base_steering, -1.0, 1.0)
        )
        planned_speed = int(
            np.clip(round(base_speed), 0, 255)
        )

        if not driving_enabled:
            # Keep the controller deterministic between manual drive sessions.
            self.reset()
            return IntersectionControlOutput(
                state=self.state,
                steering=0.0,
                speed=0,
                override_active=True,
                traffic_light_required=False,
                stop_line_required=False,
                reset_traffic_light_detector=True,
                reason="driving disabled",
            )

        # -------------------------------------------------------------
        # Normal lane following
        # -------------------------------------------------------------
        if self.state == IntersectionState.SEARCHING_RED:
            red_confirmed = self._signal_confirmed(
                traffic_light,
                TrafficLightState.RED,
                visible=traffic_light_visible,
                minimum_confidence=self.minimum_green_confidence,
            )

            if red_confirmed:
                self._red_seen = True
                self._transition(
                    IntersectionState.APPROACHING_STOP_LINE,
                    current_time,
                )

                return IntersectionControlOutput(
                    state=self.state,
                    steering=planned_steering,
                    speed=planned_speed,
                    override_active=False,
                    traffic_light_required=True,
                    stop_line_required=True,
                    reset_stop_line_detector=True,
                    reason="stable red detected: stop-line detection armed",
                )

            return IntersectionControlOutput(
                state=self.state,
                steering=planned_steering,
                speed=planned_speed,
                override_active=False,
                traffic_light_required=True,
                stop_line_required=False,
                reason="searching for initial red traffic light",
            )

        #-------------------------------------------------------------
        # Stop at the line
        #-------------------------------------------------------------
        if self.state == IntersectionState.APPROACHING_STOP_LINE:
            should_stop = bool(
                stop_line is not None
                and stop_line.should_stop
            )

            if should_stop:
                self._stopped_steering = planned_steering
                self._transition(
                    IntersectionState.WAITING_FOR_GREEN,
                    current_time,
                )

                return IntersectionControlOutput(
                    state=self.state,
                    steering=self._stopped_steering,
                    speed=0,
                    override_active=True,
                    traffic_light_required=True,
                    stop_line_required=False,
                    entered_waiting=True,
                    reason="red seen and nearby stop line confirmed",
                )

            return IntersectionControlOutput(
                state=self.state,
                steering=planned_steering,
                speed=planned_speed,
                override_active=False,
                traffic_light_required=True,
                stop_line_required=True,
                reason="red seen: approaching stop line",
            )
        # -------------------------------------------------------------
        # Full stop until a stable green signal
        # -------------------------------------------------------------
        if self.state == IntersectionState.WAITING_FOR_GREEN:
            green_confirmed = (
                self._red_seen
                and self._signal_confirmed(
                    traffic_light,
                    TrafficLightState.GREEN,
                    visible=traffic_light_visible,
                    minimum_confidence=self.minimum_green_confidence,
                )
            )

            if not green_confirmed:
                return IntersectionControlOutput(
                    state=self.state,
                    steering=self._stopped_steering,
                    speed=0,
                    override_active=True,
                    traffic_light_required=True,
                    stop_line_required=False,
                    reason="vehicle stopped: waiting for stable green",
                )

            self._transition(
                IntersectionState.CLEARING_INTERSECTION,
                current_time,
            )
            self._clear_frames = 0

            return IntersectionControlOutput(
                state=self.state,
                steering=planned_steering,
                speed=self._departure_speed(planned_speed),
                override_active=True,
                traffic_light_required=False,
                stop_line_required=True,
                reset_traffic_light_detector=True,
                released_on_green=True,
                reason="stable green confirmed: departing",
            )

        # -------------------------------------------------------------
        # Ignore the same line until it disappears
        # -------------------------------------------------------------
        line_visible = self._stop_line_observed(stop_line)

        if line_visible:
            self._clear_frames = 0
        else:
            self._clear_frames += 1

        clearing_age = max(
            0.0,
            current_time - self._state_started_at,
        )
        cleared_by_frames = (
            self._clear_frames >= self.clear_confirm_frames
        )
        cleared_by_timeout = (
            clearing_age >= self.max_clearing_seconds
        )

        if cleared_by_frames or cleared_by_timeout:
            clear_reason = (
                f"stop line absent for {self._clear_frames} frames"
                if cleared_by_frames
                else (
                    f"clearing timeout after "
                    f"{clearing_age:.2f}s"
                )
            )

            self._red_seen = False
            self._transition(
                IntersectionState.SEARCHING_RED,
                current_time,
            )

            return IntersectionControlOutput(
                state=self.state,
                steering=planned_steering,
                speed=planned_speed,
                override_active=False,
                traffic_light_required=False,
                intersection_cleared=True,
                reason=clear_reason,
            )

        return IntersectionControlOutput(
            state=self.state,
            steering=planned_steering,
            speed=planned_speed,
            override_active=True,
            traffic_light_required=False,
            reason=(
                "clearing previous stop line: "
                f"visible={line_visible}, "
                f"clear={self._clear_frames}/"
                f"{self.clear_confirm_frames}"
            ),
        )


__all__ = [
    "IntersectionControlOutput",
    "IntersectionController",
    "IntersectionState",
]
