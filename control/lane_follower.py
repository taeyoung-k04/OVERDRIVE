#!/usr/bin/env python3
"""Lane-following steering and speed controller."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from perception.lane_detector import (
    LaneBoundary,
    LaneCurve,
    LaneObservation,
)


# -----------------------------------------------------------------------------
# Steering controller
# -----------------------------------------------------------------------------

def evaluate_curve_x(coefficients: np.ndarray, y_ratio: float) -> float:
    return float(np.polyval(coefficients, y_ratio))


@dataclass
class ControlOutput:
    valid: bool
    steering: float
    steering_command: int
    speed: int
    confidence: float
    lost_frames: int
    predicted: bool = False
    recovery_level: float = 0.0
    gain_multiplier: float = 1.0
    lane_loss_age: float = 0.0
    display_coefficients: Optional[np.ndarray] = None
    near_error: float = 0.0
    far_error: float = 0.0
    combined_error: float = 0.0
    derivative: float = 0.0
    curvature: float = 0.0
    vehicle_x: float = 0.0
    green_near_x: float = 0.0
    green_far_x: float = 0.0
    target_near_x: float = 0.0
    target_far_x: float = 0.0
    near_weight: float = 0.0
    reason: str = ""

class DrivingLane(Enum):
    LANE_1 = 1
    LANE_2 = 2

@dataclass(frozen=True)
class LaneReference:
    observation: LaneCurve
    offset_ratio: float
    label: str

def select_lane_reference(
    lanes: LaneObservation,
    current_lane: DrivingLane,
    args: argparse.Namespace,
) -> LaneReference:
    if current_lane == DrivingLane.LANE_1:
        return LaneReference(
            observation=lanes.center,
            offset_ratio=float(args.center_offset_ratio),
            label="lane_center",
        )

    return LaneReference(
        observation=lanes.right,
        offset_ratio=float(args.right_offset_ratio),
        label="lane_right",
    )
    
class RightLaneFollower:
    
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        self.args = args

        # 현재 follower가 추적 중인 semantic 경계
        self.active_boundary: Optional[LaneBoundary] = None

        self._reset_tracking_state()

    def _reset_tracking_state(self) -> None:
        """Reset temporal values belonging to one reference boundary."""
        self.previous_error = 0.0
        self.previous_steering = 0.0
        self.lost_frames = 0

        self.smoothed_coefficients: Optional[np.ndarray] = None
        self.last_valid_coefficients: Optional[np.ndarray] = None
        self.last_valid_confidence = 0.0
        self.last_valid_time: Optional[float] = None
        self.last_valid_curvature = 0.0
        self.last_valid_combined_error = 0.0

    def reset(self) -> None:
        """Reset all state before a new driving run."""
        self.active_boundary = None
        self._reset_tracking_state()

    def _activate_boundary(
        self,
        boundary: LaneBoundary,
    ) -> None:
        """Reset old geometry when the reference boundary changes."""
        if self.active_boundary == boundary:
            return

        previous = self.active_boundary

        self._reset_tracking_state()
        self.active_boundary = boundary

        if previous is not None:
            print(
                f"Lane reference changed: "
                f"{previous.value} -> {boundary.value}",
                flush=True,
            )

    def _offset_at_y(self, width: int, y_ratio: float, offset_ratio: float) -> float:
        near_offset = width * float(offset_ratio)

        denominator = self.args.near_y_ratio - self.args.vanishing_y_ratio
        if denominator <= 1e-6:
            return near_offset

        scale = (y_ratio - self.args.vanishing_y_ratio) / denominator
        scale = float(np.clip(scale, 0.08, 1.25))
        return near_offset * scale

    def _steering_to_command(self, steering: float) -> int:
        scale = max(1, int(self.args.steering_command_scale))
        return int(np.clip(round(steering * scale), -scale, scale))

    def _steering_to_speed(
        self,
        steering: float,
        confidence: float,
        recovery_level: float = 0.0,
    ) -> int:
        if self.args.constant_speed is not None:
            return max(0, int(self.args.constant_speed))

        turn_amount = float(np.clip(abs(steering), 0.0, 1.0))
        speed = self.args.speed_straight + turn_amount * (
            self.args.speed_turn - self.args.speed_straight
        )

        confidence_scale = float(np.clip(0.70 + 0.30 * confidence, 0.55, 1.0))
        speed *= confidence_scale

        # A large cross-track error gets stronger steering, but also a lower
        # forward speed so the car has time to return without overshooting.
        recovery_level = float(np.clip(recovery_level, 0.0, 1.0))
        if recovery_level > 0.0:
            recovery_cap = int(
                round(
                    self.args.speed_turn
                    + recovery_level * (self.args.recovery_speed - self.args.speed_turn)
                )
            )
            speed = min(speed, recovery_cap)

        return int(max(self.args.speed_min, round(speed)))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _recovery_level(self, near_error: float, combined_error: float, confidence: float) -> float:
        if confidence < self.args.recovery_min_confidence:
            return 0.0

        magnitude = max(abs(near_error), abs(combined_error))
        start = max(0.0, float(self.args.large_error_threshold))
        full = max(start + 1e-4, float(self.args.large_error_full))
        normalized = (magnitude - start) / (full - start)
        return self._smoothstep(normalized)

    def _prepare_valid_coefficients(
        self,
        coefficients: np.ndarray,
        width: int,
        confidence: float,
    ) -> Optional[np.ndarray]:
        candidate = np.asarray(coefficients, dtype=np.float64)
        previous = self.smoothed_coefficients
        if previous is None:
            self.smoothed_coefficients = candidate.copy()
            return candidate

        # Reject a one-frame lane jump unless the new detection is very strong.
        near_old = evaluate_curve_x(previous, self.args.near_y_ratio)
        near_new = evaluate_curve_x(candidate, self.args.near_y_ratio)
        far_old = evaluate_curve_x(previous, self.args.far_y_ratio)
        far_new = evaluate_curve_x(candidate, self.args.far_y_ratio)
        jump_ratio = max(abs(near_new - near_old), abs(far_new - far_old)) / max(width, 1)
        if jump_ratio > self.args.max_lane_jump_ratio and confidence < 0.72:
            return None

        new_weight = float(np.clip(self.args.lane_curve_new_weight, 0.0, 1.0))
        smoothed = new_weight * candidate + (1.0 - new_weight) * previous
        self.smoothed_coefficients = smoothed
        return smoothed

    def _compute_from_coefficients(
        self,
        coefficients: np.ndarray,
        confidence: float,
        frame_shape: tuple[int, int],
        predicted: bool,
        offset_ratio: float,
        reference_label: str,
        lane_loss_age: float = 0.0,
    ) -> ControlOutput:
        height, width = frame_shape[:2]

        green_near_x = evaluate_curve_x(coefficients, self.args.near_y_ratio)
        green_far_x = evaluate_curve_x(coefficients, self.args.far_y_ratio)

        target_near_x = green_near_x - self._offset_at_y(width, self.args.near_y_ratio, offset_ratio,)
        target_far_x = green_far_x - self._offset_at_y(width, self.args.far_y_ratio, offset_ratio,)
        vehicle_x = width * self.args.vehicle_x_ratio

        normalization = max(width * 0.5, 1.0)
        near_error = (target_near_x - vehicle_x) / normalization
        far_error = (target_far_x - vehicle_x) / normalization
        curvature = abs(green_near_x - green_far_x) / max(width, 1)

        near_weight = float(np.clip(self.args.near_weight, 0.05, 0.95))
        if self.args.adaptive_lookahead:
            far_bonus = float(np.clip(curvature * 1.8, 0.0, 0.20))
            near_weight = float(np.clip(near_weight - far_bonus, 0.42, 0.85))

        initial_combined = near_weight * near_error + (1.0 - near_weight) * far_error
        recovery_level = self._recovery_level(near_error, initial_combined, confidence)

        # During lane departure, favor the near point because it reflects the
        # vehicle's current lateral displacement more directly than lookahead.
        near_weight = float(
            np.clip(
                near_weight + recovery_level * self.args.recovery_near_weight_bonus,
                0.42,
                0.92,
            )
        )
        far_weight = 1.0 - near_weight
        combined_error = near_weight * near_error + far_weight * far_error
        derivative = combined_error - self.previous_error

        gain_multiplier = 1.0 + self.args.large_error_gain * (
            recovery_level ** max(0.1, self.args.large_error_power)
        )
        raw_steering = self.args.steering_sign * (
            self.args.kp * combined_error * gain_multiplier
            + self.args.kd * derivative * (1.0 + 0.45 * recovery_level)
        )

        if abs(raw_steering) < self.args.steering_deadband:
            raw_steering = 0.0
        raw_steering = float(np.clip(raw_steering, -1.0, 1.0))

        base_new_weight = float(np.clip(self.args.new_command_weight, 0.0, 1.0))
        recovery_new_weight = float(np.clip(self.args.recovery_new_command_weight, 0.0, 1.0))
        new_weight = base_new_weight + recovery_level * (recovery_new_weight - base_new_weight)
        steering = new_weight * raw_steering + (1.0 - new_weight) * self.previous_steering

        dynamic_max_change = max(0.0, float(self.args.max_command_change)) + (
            recovery_level * max(0.0, float(self.args.recovery_command_change_bonus))
        )
        steering = float(
            np.clip(
                steering,
                self.previous_steering - dynamic_max_change,
                self.previous_steering + dynamic_max_change,
            )
        )
        steering = float(np.clip(steering, -1.0, 1.0))

        if predicted:
            # With no current visual observation, retain useful steering briefly
            # but gradually trend toward straight. On a known curve, retain more.
            delay = max(0.0, float(self.args.lane_loss_straighten_delay))
            grace = max(delay + 1e-3, float(self.args.lane_loss_grace_seconds))
            progress = float(np.clip((lane_loss_age - delay) / (grace - delay), 0.0, 1.0))
            min_retain = float(np.clip(self.args.lane_loss_min_steering_retain, 0.0, 1.0))
            curve_retain = float(np.clip(self.last_valid_curvature * 7.0, 0.0, 0.35))
            retain = max(min_retain + curve_retain, 1.0 - progress)
            # Never create a stronger command from stale geometry. Hold the
            # last real command and then decay it gradually toward straight.
            steering = self.previous_steering * float(np.clip(retain, 0.0, 1.0))
            recovery_level *= max(0.0, 1.0 - progress * 0.5)
            speed = int(np.clip(self.args.lane_loss_speed, 0, 255))
        else:
            speed = self._steering_to_speed(steering, confidence, recovery_level)

        steering_command = self._steering_to_command(steering)
        self.previous_error = combined_error
        self.previous_steering = steering

        return ControlOutput(
            valid=not predicted,
            predicted=predicted,
            steering=steering,
            steering_command=steering_command,
            speed=speed,
            confidence=confidence,
            lost_frames=self.lost_frames,
            recovery_level=recovery_level,
            gain_multiplier=gain_multiplier,
            lane_loss_age=lane_loss_age,
            display_coefficients=coefficients.copy(),
            near_error=near_error,
            far_error=far_error,
            combined_error=combined_error,
            derivative=derivative,
            curvature=curvature,
            vehicle_x=vehicle_x,
            green_near_x=green_near_x,
            green_far_x=green_far_x,
            target_near_x=target_near_x,
            target_far_x=target_far_x,
            near_weight=near_weight,
            reason=f"predicted from {reference_label}" if predicted else f"following {reference_label}",
        )

    def _lost_output(
        self,
        observation: LaneCurve,
        frame_shape: tuple[int, int],
        offset_ratio: float,
        reference_label: str,
    ) -> ControlOutput:
        self.lost_frames += 1
        now = time.perf_counter()

        if self.last_valid_time is None or self.last_valid_coefficients is None:
            self.previous_steering = 0.0
            return ControlOutput(
                valid=False,
                predicted=False,
                steering=0.0,
                steering_command=0,
                speed=0,
                confidence=0.0,
                lost_frames=self.lost_frames,
                reason=f"lane unavailable with no history ({observation.reason})",
            )

        lane_loss_age = max(0.0, now - self.last_valid_time)
        grace = max(0.0, float(self.args.lane_loss_grace_seconds))
        if lane_loss_age <= grace:
            decay_seconds = max(1e-3, float(self.args.lane_loss_confidence_decay))
            confidence = self.last_valid_confidence * math.exp(-lane_loss_age / decay_seconds)
            output = self._compute_from_coefficients(
                self.last_valid_coefficients,
                confidence=confidence,
                frame_shape=frame_shape,
                predicted=True,
                offset_ratio=offset_ratio,
                reference_label=reference_label,
                lane_loss_age=lane_loss_age,
            )
            output.reason = f"coasting on last {reference_label} ({observation.reason})"

            return output

        self.previous_steering = 0.0
        return ControlOutput(
            valid=False,
            predicted=False,
            steering=0.0,
            steering_command=0,
            speed=0,
            confidence=0.0,
            lost_frames=self.lost_frames,
            lane_loss_age=lane_loss_age,
            reason=f"{reference_label} lost for {lane_loss_age:.2f}s: safety stop ({observation.reason})",
        )

    def compute(
        self,
        observation: LaneCurve,
        frame_shape: tuple[int, int],
        offset_ratio: float,
    ) -> ControlOutput:
        height, width = frame_shape[:2]

        # center와 right가 바뀌면 이전 경계의 smoothing/history를 제거한다.
        self._activate_boundary(
            observation.boundary
        )

        reference_label = observation.boundary.value

        if (
            not observation.valid
            or observation.coefficients is None
            or observation.confidence
            < self.args.min_lane_confidence
        ):
            return self._lost_output(
                observation=observation,
                frame_shape=frame_shape,
                offset_ratio=offset_ratio,
                reference_label=reference_label,
            )

        coefficients = self._prepare_valid_coefficients(
            observation.coefficients,
            width=width,
            confidence=observation.confidence,
        )

        if coefficients is None:
            rejected = LaneCurve(
                boundary=observation.boundary,
                valid=False,
                confidence=observation.confidence,
                reason="sudden lane jump rejected",
            )

            return self._lost_output(
                observation=rejected,
                frame_shape=frame_shape,
                offset_ratio=offset_ratio,
                reference_label=reference_label,
            )

        self.lost_frames = 0
        now = time.perf_counter()

        output = self._compute_from_coefficients(
            coefficients=coefficients,
            confidence=observation.confidence,
            frame_shape=frame_shape,
            predicted=False,
            offset_ratio=offset_ratio,
            reference_label=reference_label,
        )

        self.last_valid_coefficients = (
            coefficients.copy()
        )
        self.last_valid_confidence = (
            observation.confidence
        )
        self.last_valid_time = now
        self.last_valid_curvature = output.curvature
        self.last_valid_combined_error = (
            output.combined_error
        )

        return output