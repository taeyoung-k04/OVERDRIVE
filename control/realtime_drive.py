#!/usr/bin/env python3
"""Real-time lane-following runtime for OVERDRIVE."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from pathlib import Path

import cv2

from control.lane_follower import (
    DrivingLane,
    RightLaneFollower,
    select_lane_reference,
)
from control.obstacle_avoidance import (
    ObstacleAvoidanceController,
)
from hardware.arduino import ArduinoSender
from hardware.camera import (
    LatestFrameReader,
    normalize_frame_size,
    open_camera,
)
from perception.infer_sem_class import (
    DEFAULT_WEIGHTS,
    add_postprocess_args,
    load_postprocess_config,
    load_semantic_model,
    make_class_overlay,
    postprocess_class_map,
    semantic_to_class_map,
)
from perception.lane_detector import LaneDetector
from visualization.lane_debug import (
    draw_debug,
    draw_fps,
)

from visualization.intersection_debug import (
    draw_intersection_debug,
)

from visualization.obstacle_debug import (
    draw_obstacle_debug,
)

from control.intersection_controller import (
    IntersectionControlOutput,
    IntersectionController,
    IntersectionState,
)

from perception.stop_line_detector import (
    StopLineDetector,
)

from perception.infer_traffic_light import (
    TrafficLightInference,
)
from perception.infer_car import CarInference

# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model / camera
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="onnx")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-force-size", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--preview", choices=("overlay", "debug"), default="debug")
    add_postprocess_args(parser)

    # Intersection stopping. A lower trigger ratio stops farther away because
    # image y increases toward the bottom of the frame.
    parser.add_argument(
        "--stop-line-trigger-y-ratio",
        type=float,
        default=0.72,
        help="Stop-line center y ratio that triggers braking. Lower reacts earlier.",
    )
    parser.add_argument(
        "--stop-line-confirm-frames",
        type=int,
        default=1,
        help="Consecutive near-line frames required before braking.",
    )
    parser.add_argument(
        "--stop-line-history-size",
        type=int,
        default=3,
        help="Recent stop-line observations retained by the detector.",
    )
    parser.add_argument(
        "--no-intersection",
        action="store_true",
        help="Start with intersection detection and control disabled (toggle with I).",
    )
    parser.add_argument(
        "--no-obstacle",
        action="store_true",
        help="Start with obstacle detection and avoidance disabled (toggle with B).",
    )

    # Lane extraction
    parser.add_argument(
        "--control-source",
        choices=("raw", "processed"),
        default="raw",
        help="Use raw or postprocessed class map for steering. Raw is often safer when road postprocessing removes a valid right lane.",
    )
    parser.add_argument("--roi-top-ratio", type=float, default=0.38)
    parser.add_argument("--min-component-area", type=int, default=90)
    parser.add_argument("--min-lane-points", type=int, default=18)
    parser.add_argument("--min-lane-confidence", type=float, default=0.18)

    # Virtual path geometry
    parser.add_argument(
        "--initial-lane",
        type=int,
        choices=(1, 2),
        default=2,
        help="Initial driving lane: 1 follows lane_center, 2 follows lane_right.",
    )

    parser.add_argument(
        "--center-offset-ratio",
        type=float,
        default=0.22,
        help=(
            "Desired distance left of lane_center while following lane 1, "
            "divided by image width."
        ),
    )
    parser.add_argument(
        "--vehicle-x-ratio",
        type=float,
        default=0.50,
        help="Vehicle forward-axis x position as a fraction of image width. Calibrate this to the camera mount; use --right-offset-ratio for lateral bias.",
    )
    parser.add_argument("--near-y-ratio", type=float, default=0.84)
    parser.add_argument("--far-y-ratio", type=float, default=0.58)
    parser.add_argument(
        "--right-offset-ratio",
        type=float,
        default=0.22,
        help="Desired distance left of the right lane at near-y, divided by image width.",
    )
    parser.add_argument(
        "--vanishing-y-ratio",
        type=float,
        default=0.31,
        help="Approximate vanishing-point y ratio used to scale the lane offset with perspective.",
    )

    # Steering controller
    parser.add_argument("--kp", type=float, default=1.15)
    parser.add_argument("--kd", type=float, default=0.18)
    parser.add_argument("--near-weight", type=float, default=0.65)
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument(
        "--adaptive-lookahead",
        dest="adaptive_lookahead",
        action="store_true",
        help="Increase far-point influence automatically on curves (default).",
    )
    adaptive_group.add_argument(
        "--no-adaptive-lookahead",
        dest="adaptive_lookahead",
        action="store_false",
        help="Disable automatic far-point weighting on curves.",
    )
    parser.set_defaults(adaptive_lookahead=True)
    parser.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--steering-deadband", type=float, default=0.015)
    parser.add_argument(
        "--new-command-weight",
        type=float,
        default=0.78,
        help="Weight of the newest command. Higher reacts faster; lower is smoother.",
    )
    parser.add_argument("--max-command-change", type=float, default=0.24)

    # Strong lane-recovery control. Small errors keep the normal gain, while
    # large lateral errors receive progressively stronger steering and faster
    # command updates.
    parser.add_argument("--large-error-threshold", type=float, default=0.14)
    parser.add_argument("--large-error-full", type=float, default=0.52)
    parser.add_argument("--large-error-gain", type=float, default=1.55)
    parser.add_argument("--large-error-power", type=float, default=1.35)
    parser.add_argument("--recovery-near-weight-bonus", type=float, default=0.18)
    parser.add_argument("--recovery-command-change-bonus", type=float, default=0.34)
    parser.add_argument("--recovery-new-command-weight", type=float, default=0.94)
    parser.add_argument("--recovery-min-confidence", type=float, default=0.28)
    parser.add_argument("--recovery-speed", type=int, default=200)

    # Short semantic dropouts are bridged with the most recent fitted lane.
    # Time-based limits are used so behavior is similar at different FPS.
    parser.add_argument("--lane-loss-grace-seconds", type=float, default=1.20)
    parser.add_argument("--lane-loss-speed", type=int, default=200)
    parser.add_argument("--lane-loss-straighten-delay", type=float, default=0.30)
    parser.add_argument("--lane-loss-min-steering-retain", type=float, default=0.38)
    parser.add_argument("--lane-loss-confidence-decay", type=float, default=0.80)
    parser.add_argument("--lane-curve-new-weight", type=float, default=0.72)
    parser.add_argument("--max-lane-jump-ratio", type=float, default=0.24)

    # Kept for command-line compatibility with earlier versions. The new
    # controller uses the time-based lane-loss settings above.
    parser.add_argument("--lost-hold-frames", type=int, default=3)
    parser.add_argument("--lost-stop-frames", type=int, default=24)

    # Drive-speed conversion. Steering remains normalized (-1.0..+1.0)
    # and is converted to -1000..+1000 only at the serial boundary.
    parser.add_argument("--speed-straight", type=int, default=255)
    parser.add_argument("--speed-turn", type=int, default=255)
    parser.add_argument("--speed-min", type=int, default=100)
    parser.add_argument(
        "--constant-speed",
        type=int,
        default=None,
        help="Use a fixed forward PWM (0..255) instead of slowing on turns.",
    )

    # Lane-change maneuver. The target lane reference changes immediately, but
    # steering and speed are temporarily overridden to make the lateral move
    # decisive instead of relying only on the normal lane-following error.
    parser.add_argument(
        "--lane-change-steering",
        type=float,
        default=1.0,
        help="Absolute normalized steering used during the hard lane-change phase (0..1).",
    )
    parser.add_argument(
        "--lane-change-right-steering",
        type=float,
        default=0.5,
        help="Steering magnitude used when changing from LANE_1 to LANE_2.",
    )
    parser.add_argument(
        "--lane-change-speed",
        type=int,
        default=175,
        help="Maximum forward PWM while a lane change is active (0..255).",
    )
    parser.add_argument(
        "--lane-change-full-steer-seconds",
        type=float,
        default=0.55,#0.8
        help="How long to hold configured steering at the start of a lane change.",
    )
    parser.add_argument(
        "--lane-change-total-seconds",
        type=float,
        default=2.0,#1.50
        help="Total lane-change time when moving from LANE_2 to LANE_1.",
    )
    parser.add_argument(
        "--lane-change-right-total-seconds",
        type=float,
        default=2.5,
        help="Total lane-change time when moving from LANE_1 to LANE_2.",
    )
    parser.add_argument(
        "--obstacle-path-half-width-ratio",
        type=float,
        default=0.13,
        help=(
            "Half-width around the perspective-scaled virtual lane path "
            "used to assign a detected car to the current lane."
        ),
    )
    parser.add_argument(
        "--obstacle-path-min-scale",
        type=float,
        default=0.40,
        help=(
            "Minimum perspective scale for the obstacle path width. "
            "Higher values keep the far-field lane search wider."
        ),
    )
    parser.add_argument(
        "--obstacle-road-margin-ratio",
        type=float,
        default=0.025,
        help=(
            "Image-width margin added around the learned main-road mask "
            "to bridge small far-field segmentation gaps."
        ),
    )
    parser.add_argument(
        "--obstacle-min-bottom-ratio",
        type=float,
        default=0.30,
        help=(
            "Minimum car bounding-box bottom y ratio. Lower values detect "
            "vehicles farther ahead."
        ),
    )
    parser.add_argument(
        "--obstacle-car-min-area",
        type=int,
        default=60,
        help="Minimum semantic car component area in pixels.",
    )
    parser.add_argument(
        "--obstacle-lane-memory-seconds",
        type=float,
        default=1.0,
        help=(
            "How long the obstacle detector may reuse the latest valid "
            "lane curve during a short semantic dropout."
        ),
    )
    parser.add_argument(
        "--obstacle-analysis-hz",
        type=float,
        default=8.0,
        help=(
            "Maximum obstacle postprocessing rate. Lane following still "
            "runs every frame; only obstacle analysis is rate-limited."
        ),
    )
    # Arduino serial. The matching sketch uses C,<steering>,<speed> and X.
    parser.add_argument(
        "--arduino-port",
        default="COM8",
        help="For example COM8 or /dev/ttyACM0. Omit for vision-only mode.",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--serial-timeout", type=float, default=0.10)
    parser.add_argument("--arduino-reset-wait", type=float, default=1.8)
    parser.add_argument(
        "--command-rate",
        type=float,
        default=20.0,
        help="Background serial command rate in Hz. Keep above the Arduino watchdog rate.",
    )
    parser.add_argument(
        "--steering-command-scale",
        type=int,
        default=1000,
        help="Convert normalized steering to this integer range for Arduino.",
    )
    parser.add_argument(
        "--telemetry-stale-seconds",
        type=float,
        default=0.8,
        help="Hide Arduino position feedback after this many seconds without telemetry.",
    )
    

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> None:
    ratio_names = (
        "roi_top_ratio",
        "vehicle_x_ratio",
        "near_y_ratio",
        "far_y_ratio",
        "center_offset_ratio",
        "right_offset_ratio",
        "vanishing_y_ratio",
        "near_weight",
        "new_command_weight",
        "stop_line_trigger_y_ratio",
    )
    for name in ratio_names:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1, got {value}")

    if args.far_y_ratio >= args.near_y_ratio:
        raise SystemExit("--far-y-ratio must be smaller than --near-y-ratio")
    if args.lost_stop_frames <= args.lost_hold_frames:
        raise SystemExit("--lost-stop-frames must be larger than --lost-hold-frames")
    if args.large_error_full <= args.large_error_threshold:
        raise SystemExit("--large-error-full must be larger than --large-error-threshold")
    if args.lane_loss_grace_seconds < 0:
        raise SystemExit("--lane-loss-grace-seconds must be non-negative")
    if args.lane_loss_straighten_delay < 0:
        raise SystemExit("--lane-loss-straighten-delay must be non-negative")
    if args.lane_loss_confidence_decay <= 0:
        raise SystemExit("--lane-loss-confidence-decay must be positive")
    if not 0.0 <= args.lane_curve_new_weight <= 1.0:
        raise SystemExit("--lane-curve-new-weight must be between 0 and 1")
    if not 0.0 <= args.lane_loss_min_steering_retain <= 1.0:
        raise SystemExit("--lane-loss-min-steering-retain must be between 0 and 1")
    if args.command_rate <= 0:
        raise SystemExit("--command-rate must be positive")
    if args.stop_line_trigger_y_ratio <= 0.45:
        raise SystemExit(
            "--stop-line-trigger-y-ratio must be larger than the stop-line "
            "ROI top ratio (0.45)"
        )
    if not 1 <= args.stop_line_confirm_frames <= args.stop_line_history_size:
        raise SystemExit(
            "--stop-line-confirm-frames must be between 1 and "
            "--stop-line-history-size"
        )
    if args.telemetry_stale_seconds <= 0:
        raise SystemExit("--telemetry-stale-seconds must be positive")
    if not 0.0 < args.lane_change_steering <= 1.0:
        raise SystemExit("--lane-change-steering must be greater than 0 and at most 1")
    if not 0.0 < args.lane_change_right_steering <= 1.0:
        raise SystemExit(
            "--lane-change-right-steering must be greater than 0 and at most 1"
        )
    if args.lane_change_full_steer_seconds < 0:
        raise SystemExit("--lane-change-full-steer-seconds must be non-negative")
    if args.lane_change_total_seconds <= 0:
        raise SystemExit("--lane-change-total-seconds must be positive")
    if args.lane_change_right_total_seconds <= 0:
        raise SystemExit(
            "--lane-change-right-total-seconds must be positive"
        )
    if args.lane_change_full_steer_seconds > args.lane_change_total_seconds:
        raise SystemExit(
            "--lane-change-full-steer-seconds must not exceed "
            "--lane-change-total-seconds"
        )
    if (
        args.lane_change_full_steer_seconds
        > args.lane_change_right_total_seconds
    ):
        raise SystemExit(
            "--lane-change-full-steer-seconds must not exceed "
            "--lane-change-right-total-seconds"
        )
    if not 0.0 < args.obstacle_path_half_width_ratio <= 0.5:
        raise SystemExit(
            "--obstacle-path-half-width-ratio must be in (0, 0.5]"
        )
    if not 0.0 < args.obstacle_path_min_scale <= 1.0:
        raise SystemExit(
            "--obstacle-path-min-scale must be in (0, 1]"
        )
    if not 0.0 <= args.obstacle_road_margin_ratio <= 0.25:
        raise SystemExit(
            "--obstacle-road-margin-ratio must be in [0, 0.25]"
        )
    if not 0.0 <= args.obstacle_min_bottom_ratio <= 1.0:
        raise SystemExit(
            "--obstacle-min-bottom-ratio must be in [0, 1]"
        )
    if args.obstacle_car_min_area < 1:
        raise SystemExit(
            "--obstacle-car-min-area must be at least 1"
        )
    if args.obstacle_lane_memory_seconds < 0:
        raise SystemExit(
            "--obstacle-lane-memory-seconds must be non-negative"
        )
    if args.obstacle_analysis_hz <= 0:
        raise SystemExit(
            "--obstacle-analysis-hz must be positive"
        )
    for name in (
        "speed_straight",
        "speed_turn",
        "speed_min",
        "recovery_speed",
        "lane_loss_speed",
        "lane_change_speed",
    ):
        value = int(getattr(args, name))
        if not 0 <= value <= 255:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 255")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 255:
        raise SystemExit("--constant-speed must be between 0 and 255")


class LaneChangeManeuver:
    """Temporarily override steering and speed during a lane change."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.max_steering = float(args.lane_change_steering)
        self.right_max_steering = float(
            args.lane_change_right_steering
        )
        self.active_max_steering = self.max_steering
        self.speed_cap = int(args.lane_change_speed)
        self.full_steer_seconds = float(args.lane_change_full_steer_seconds)
        self.total_seconds = float(args.lane_change_total_seconds)
        self.right_total_seconds = float(
            args.lane_change_right_total_seconds
        )
        self.active_total_seconds = self.total_seconds
        self.steering_sign = float(args.steering_sign)

        self.active = False
        self.started_at = 0.0
        self.direction = 0.0
        self.direction_name = "NONE"
        self.target_lane: DrivingLane | None = None

    def start(
        self,
        previous_lane: DrivingLane,
        target_lane: DrivingLane,
        *,
        now: float,
    ) -> None:
        if previous_lane == target_lane:
            return

        # LANE_1 is the left lane and LANE_2 is the right lane.
        # steering_sign keeps this consistent with the normal follower's
        # configured hardware direction.
        moving_left = (
            previous_lane == DrivingLane.LANE_2
            and target_lane == DrivingLane.LANE_1
        )
        self.direction = -1.0 if moving_left else 1.0
        self.direction_name = "LEFT" if moving_left else "RIGHT"
        self.active_max_steering = (
            self.max_steering
            if moving_left
            else self.right_max_steering
        )
        self.active_total_seconds = (
            self.total_seconds
            if moving_left
            else self.right_total_seconds
        )
        self.target_lane = target_lane
        self.started_at = now
        self.active = True

        print(
            f"LANE CHANGE START: {self.direction_name}, "
            f"steering={self.steering_command:+.2f}, "
            f"speed_cap={self.speed_cap}, "
            f"full_steer={self.full_steer_seconds:.2f}s, "
            f"total={self.active_total_seconds:.2f}s",
            flush=True,
        )

    @property
    def steering_command(self) -> float:
        return (
            self.direction
            * self.active_max_steering
            * self.steering_sign
        )

    def cancel(self, reason: str | None = None) -> None:
        if self.active and reason:
            print(f"LANE CHANGE CANCELLED: {reason}", flush=True)
        self.active = False
        self.direction = 0.0
        self.direction_name = "NONE"
        self.target_lane = None

    def apply(
        self,
        *,
        now: float,
        base_steering: float,
        base_speed: int,
    ) -> tuple[float, int]:
        if not self.active:
            return base_steering, base_speed

        elapsed = now - self.started_at
        if elapsed >= self.active_total_seconds:
            print(
                f"LANE CHANGE COMPLETE: target={self.target_lane.name if self.target_lane else 'UNKNOWN'}",
                flush=True,
            )
            self.cancel()
            return base_steering, base_speed

        # Never increase a speed restriction imposed by another controller.
        limited_speed = min(int(base_speed), self.speed_cap)

        # Hold the wheel at the configured maximum only for the initial phase.
        # Afterwards, keep the reduced speed but let the lane follower settle
        # onto the newly selected reference line.
        if elapsed < self.full_steer_seconds:
            return self.steering_command, limited_speed

        return base_steering, limited_speed

    def remaining_seconds(self, now: float) -> float:
        if not self.active:
            return 0.0
        return max(
            0.0,
            self.active_total_seconds
            - (now - self.started_at),
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = load_semantic_model(args.weights, args.backend)
    postprocess_config = load_postprocess_config(args.postprocess_config) if args.postprocess else None

    car_inference = CarInference(
        model=model,
        imgsz=args.imgsz,
        device=args.device,
        min_area=args.obstacle_car_min_area,
        corridor_center_ratio=args.vehicle_x_ratio,
        corridor_width_ratio=0.36,
        blocked_bottom_ratio=0.58,
    )

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reader = LatestFrameReader(capture)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height} @ {actual_fps:.1f} FPS",
        flush=True,
    )
    print(
        "Lane Boundary Follow: "
        f"source={args.control_source}, vehicle_x={args.vehicle_x_ratio:.3f}, "
        f"offset={args.right_offset_ratio:.3f}, near/far={args.near_y_ratio:.2f}/{args.far_y_ratio:.2f}, "
        f"loss_grace={args.lane_loss_grace_seconds:.2f}s, recovery_gain={args.large_error_gain:.2f}",
        flush=True,
    )

    arduino = ArduinoSender(
        port=args.arduino_port,
        baud=args.baud,
        timeout=args.serial_timeout,
        reset_wait=args.arduino_reset_wait,
        command_rate=args.command_rate,
        steering_scale=args.steering_command_scale,
    )

    lane_detector = LaneDetector(
        roi_top_ratio=args.roi_top_ratio,
        min_lane_points=args.min_lane_points,

        right_min_component_area=args.min_component_area,
        right_fragment_min_area=max(
            8,
            args.min_component_area // 6,
        ),

        center_min_component_area=18,
        center_fragment_min_area=8,

        max_lane_jump_ratio=args.max_lane_jump_ratio,

        # 기존 follower에도 곡선 smoothing이 있으므로
        # detector 쪽은 비교적 빠르게 반응하도록 둔다.
        curve_new_weight=0.65,
    )

    follower = RightLaneFollower(args)

    current_lane = DrivingLane(
        args.initial_lane
    )

    #intersection Controller
    stop_line_detector = StopLineDetector(
        roi_top_ratio=0.45,
        trigger_y_ratio=args.stop_line_trigger_y_ratio,
        history_size=args.stop_line_history_size,
        confirm_frames=args.stop_line_confirm_frames,
        vehicle_x_ratio=args.vehicle_x_ratio,
        vehicle_corridor_ratio=0.30,
    )

    traffic_light_inference = TrafficLightInference(
        model=model,
        imgsz=args.imgsz,
        device=args.device,
        min_region_area=6,
        mask_padding=8,
        stable_frames=2,
    )

    intersection_controller = IntersectionController(
        # 정지선이 8프레임 연속 사라지면 교차로 통과 완료
        clear_confirm_frames=6,

        # 정지선 인식이 계속 남는 경우를 위한 안전한 timeout
        max_clearing_seconds=10.0,
        minimum_green_confidence=0.55,
        # 정차 중에는 마지막 조향각 유지
        hold_steering_while_stopped=True,

        # 초록불 출발 시 너무 빠르게 튀어나가는 것을 방지
        departure_speed_cap=80,
        # signal_memory_seconds=1.5,
        # minimum_stop_seconds=2.0,
    )

    #avoid Obstacle
    obstacle_avoidance = ObstacleAvoidanceController(
        initial_lane=current_lane,
        obstacle_distance_cm=200,
        clear_distance_cm=250,
        detection_confirm_frames=2,
        clear_confirm_frames=10,
    )

    lane_change = LaneChangeManeuver(args)

    # 라이다 연결 전 임시 테스트값
    mock_front_blocked = False
    obstacle_enabled = not args.no_obstacle
    intersection_enabled = not args.no_intersection

    window_name = "Lane Boundary Follow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0
    driving_enabled = False
    obstacle_lane_memory = {
        DrivingLane.LANE_1: None,
        DrivingLane.LANE_2: None,
    }
    cached_car_result = None
    last_obstacle_analysis_time = float("-inf")
    last_avoidance_command = None
    obstacle_auto_disabled_for_traffic_light = False
    require_ultrasonic_after_lane_1_change = False
    obstacle_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="obstacle-analysis",
    )
    obstacle_analysis_future: Future | None = None
    print(
        "Controls: SPACE=start/stop, 1/2=select lane, "
        "B=toggle obstacle, I=toggle intersection, "
        "O=toggle mock obstacle, S=emergency stop, "
        "R=reset Arduino fault, Q or ESC=quit",
        flush=True,
    )
    print(
        f"Features: obstacle={'ON' if obstacle_enabled else 'OFF'}, "
        f"intersection={'ON' if intersection_enabled else 'OFF'}",
        flush=True,
    )

    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                raise RuntimeError("Could not read a frame from the camera")

            if args.flip:
                frame = cv2.flip(frame, 1)
            frame = normalize_frame_size(
                frame,
                args.width,
                args.height,
                force_size=not args.no_force_size,
            )

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                task="semantic",
                rect=False,
                verbose=False,
            )
            raw_class_map = semantic_to_class_map(results[0].semantic_mask, frame.shape[:2])
            
            car_result = (
                cached_car_result
                if obstacle_enabled
                else None
            )
            obstacle_sample_ready = False
            if postprocess_config is not None:
                processed_class_map = postprocess_class_map(raw_class_map, postprocess_config)
            else:
                processed_class_map = raw_class_map

            traffic_light_result = None
            traffic_light_detection = None

            if (
                intersection_enabled
                and intersection_controller.requires_traffic_light
            ):
                traffic_light_result = (
                    traffic_light_inference.from_class_map(
                        frame,
                        raw_class_map,
                    )
                )
                traffic_light_detection = traffic_light_result.color

            traffic_light_visible = bool(
                traffic_light_result is not None
                and traffic_light_result.regions
            )

            if (
                traffic_light_visible
                and obstacle_enabled
                and not obstacle_auto_disabled_for_traffic_light
            ):
                obstacle_enabled = False
                obstacle_auto_disabled_for_traffic_light = True
                obstacle_avoidance.reset(
                    current_lane
                )
                lane_change.cancel(
                    "traffic light detected"
                )
                cached_car_result = None
                last_avoidance_command = None
                if obstacle_analysis_future is not None:
                    obstacle_analysis_future.cancel()
                    obstacle_analysis_future = None
                last_obstacle_analysis_time = float(
                    "-inf"
                )
                print(
                    "Obstacle detection/avoidance: "
                    "AUTO OFF after traffic light detection.",
                    flush=True,
                )

            if (
                intersection_enabled
                and intersection_controller.requires_stop_line
            ):
                stop_line_detection = stop_line_detector.detect(
                    raw_class_map,
                )
            else:
                stop_line_detection = None
                stop_line_detector.reset()

            control_class_map = (
                raw_class_map if args.control_source == "raw" else processed_class_map
            )
            
            lanes = lane_detector.detect(
                control_class_map,
            )

            if obstacle_enabled:
                obstacle_lane_now = time.perf_counter()

                if (
                    obstacle_analysis_future is not None
                    and obstacle_analysis_future.done()
                ):
                    cached_car_result = (
                        obstacle_analysis_future.result()
                    )
                    car_result = cached_car_result
                    obstacle_analysis_future = None
                    obstacle_sample_ready = True

                for lane_key, curve in (
                    (
                        DrivingLane.LANE_1,
                        lanes.center,
                    ),
                    (
                        DrivingLane.LANE_2,
                        lanes.right,
                    ),
                ):
                    if curve.valid:
                        obstacle_lane_memory[
                            lane_key
                        ] = (
                            curve,
                            obstacle_lane_now,
                        )

                current_curve = (
                    lanes.center
                    if current_lane == DrivingLane.LANE_1
                    else lanes.right
                )
                lane_reference_for_obstacle = (
                    current_curve
                    if current_curve.valid
                    else None
                )
                remembered_curve = (
                    obstacle_lane_memory[
                        current_lane
                    ]
                )
                if (
                    lane_reference_for_obstacle is None
                    and remembered_curve is not None
                    and (
                        obstacle_lane_now
                        - remembered_curve[1]
                        <= args.obstacle_lane_memory_seconds
                    )
                ):
                    lane_reference_for_obstacle = (
                        remembered_curve[0]
                    )
                lane_offset_for_obstacle = (
                    args.center_offset_ratio
                    if current_lane == DrivingLane.LANE_1
                    else args.right_offset_ratio
                )
                obstacle_period = (
                    1.0
                    / args.obstacle_analysis_hz
                )
                if (
                    obstacle_analysis_future is None
                    and (
                        obstacle_lane_now
                        - last_obstacle_analysis_time
                        >= obstacle_period
                    )
                ):
                    obstacle_analysis_future = (
                        obstacle_executor.submit(
                            car_inference.from_class_map,
                            raw_class_map.copy(),
                            lane_reference=(
                                lane_reference_for_obstacle
                            ),
                            lane_offset_ratio=(
                                lane_offset_for_obstacle
                            ),
                            near_y_ratio=args.near_y_ratio,
                            vanishing_y_ratio=(
                                args.vanishing_y_ratio
                            ),
                            path_half_width_ratio=(
                                args.obstacle_path_half_width_ratio
                            ),
                            path_min_perspective_scale=(
                                args.obstacle_path_min_scale
                            ),
                            road_margin_ratio=(
                                args.obstacle_road_margin_ratio
                            ),
                            blocked_bottom_ratio=(
                                args.obstacle_min_bottom_ratio
                            ),
                        )
                    )
                    last_obstacle_analysis_time = (
                        obstacle_lane_now
                    )

            # ---------------------------------------------------------
            # Obstacle avoidance
            # ---------------------------------------------------------

            avoidance_command = (
                last_avoidance_command
            )

            # Arduino가 전송한 최신 초음파 거리 가져오기
            distance_data = (
                arduino.distance_snapshot(stale_seconds=0.35)
                if obstacle_enabled
                else None
            )

            # 중앙 초음파 거리만 추출
            center_distance_cm = (
                distance_data.center_cm
                if distance_data is not None
                else None
            )

            # 현재 진행 경로에 차량이 있는지
            front_blocked = bool(
                car_result is not None
                and car_result.front_blocked
            )

            # O 키 테스트 모드
            # 실제 차량이나 초음파 입력이 없어도 장애물 상황을 강제로 만든다.
            if obstacle_enabled and mock_front_blocked:
                front_blocked = True
                center_distance_cm = 100
            # 주행 중일 때만 장애물로 차선을 변경
            if (
                driving_enabled
                and obstacle_enabled
                and intersection_controller.allows_obstacle_avoidance
                and (
                    obstacle_sample_ready
                    or mock_front_blocked
                )
                # and not stop_line_detection.should_stop
            ):
                previous_lane = current_lane

                avoidance_command = obstacle_avoidance.update(
                    front_blocked=front_blocked,
                    center_distance_cm=center_distance_cm,
                    # Normal avoidance is vision-only. After completing a
                    # LANE_2 -> LANE_1 maneuver, the next avoidance requires
                    # both the current-lane visual obstacle and a fresh,
                    # close center-ultrasonic reading.
                    allow_vision_only=(
                        not require_ultrasonic_after_lane_1_change
                    ),
                )
                last_avoidance_command = (
                    avoidance_command
                )

                if avoidance_command.lane_change_requested:
                    if (
                        require_ultrasonic_after_lane_1_change
                        and previous_lane == DrivingLane.LANE_1
                        and avoidance_command.target_lane
                        == DrivingLane.LANE_2
                    ):
                        require_ultrasonic_after_lane_1_change = False

                    current_lane = (
                        avoidance_command.target_lane
                    )
                    follower.reset()
                    lane_detector.reset()
                    lane_change.start(
                        previous_lane,
                        current_lane,
                        now=time.perf_counter(),
                    )
                    print(
                        "Obstacle detected: "
                        f"car={avoidance_command.front_blocked}, "
                        f"distance={avoidance_command.center_distance_cm}cm, "
                        f"{previous_lane.name} -> {current_lane.name}",
                        flush=True,
                    )

            reference = select_lane_reference(
                lanes=lanes,
                current_lane=current_lane,
                args=args,
            )

            observation = reference.observation

            control = follower.compute(
                observation=observation,
                frame_shape=frame.shape[:2],
                offset_ratio=reference.offset_ratio,
            )

            intersection_control = intersection_controller.update(
                stop_line=stop_line_detection,
                traffic_light=traffic_light_detection,
                traffic_light_visible=traffic_light_visible,
                # lane follower가 원래 보내려던 명령
                base_steering=control.steering,
                base_speed=control.speed,

                driving_enabled=driving_enabled,
            )
            if not intersection_enabled:
                intersection_control = IntersectionControlOutput(
                    state=IntersectionState.SEARCHING_RED,
                    steering=control.steering,
                    speed=control.speed,
                    override_active=False,
                    traffic_light_required=False,
                    reason="intersection disabled",
                )

            # 교차로 대기 상태에 처음 진입한 프레임에서만 실행된다.
            if intersection_control.entered_waiting:
                obstacle_avoidance.reset(current_lane)
                lane_change.cancel(
                    "waiting at traffic light"
                )

                print(
                    "INTERSECTION: stop line confirmed, waiting for green.",
                    flush=True,
                )

            if intersection_control.reset_traffic_light_detector:
                traffic_light_inference.reset()

            if intersection_control.reset_stop_line_detector:
                stop_line_detector.reset()

            if intersection_control.released_on_green:
                print(
                    "INTERSECTION: green confirmed, departing.",
                    flush=True,
                )

            if intersection_control.intersection_cleared:
                traffic_light_inference.reset()
                stop_line_detector.reset()

                print(
                    "INTERSECTION: cleared, searching for next red light.",
                    flush=True,
                )

            command_now = time.perf_counter()
            final_steering = intersection_control.steering
            final_speed = intersection_control.speed
            # final_steering = control.steering
            # final_speed = control.speed

            # Intersection/stop commands always have priority. A lane-change
            # override is allowed only while normal obstacle avoidance is
            # permitted and the vehicle has a positive drive command.
            lane_change_allowed = (
                driving_enabled
                and intersection_controller.allows_obstacle_avoidance
                and final_speed > 0
            )
            if lane_change.active and not lane_change_allowed:
                lane_change.cancel("intersection or stop command has priority")
            elif lane_change_allowed:
                lane_change_was_active = lane_change.active
                completed_lane_change_direction = (
                    lane_change.direction_name
                    if lane_change.active
                    else "NONE"
                )

                final_steering, final_speed = lane_change.apply(
                    now=command_now,
                    base_steering=final_steering,
                    base_speed=final_speed,
                )

                lane_change_just_completed = (
                    lane_change_was_active
                    and not lane_change.active
                )

                if lane_change_just_completed:
                    obstacle_avoidance.lane_change_completed(
                        current_lane
                    )

                    if (
                        completed_lane_change_direction == "LEFT"
                        and current_lane == DrivingLane.LANE_1
                    ):
                        require_ultrasonic_after_lane_1_change = True
                        print(
                            "LANE_1 obstacle recheck: "
                            "camera + ultrasonic confirmation required.",
                            flush=True,
                        )

                    print(
                        "Lane change completed. "
                        "Obstacle detection re-armed immediately.",
                        flush=True,
                    )

            if driving_enabled and arduino.enabled:
                arduino.update_command(
                    final_steering,
                    final_speed,
                    immediate=intersection_control.entered_waiting,
                )

            # if driving_enabled and arduino.enabled:
            #     arduino.update_command(control.steering, control.speed)
                
            telemetry = arduino.telemetry_snapshot(args.telemetry_stale_seconds)
            preview = make_class_overlay(frame, processed_class_map)
            if args.preview == "debug":
                draw_debug(
                    preview,
                    observation,
                    control,
                    args,
                    offset_ratio=reference.offset_ratio,
                    reference_label=reference.label,
                    current_lane=current_lane,
                    driving_enabled=driving_enabled,
                    arduino_connected=arduino.enabled,
                    arduino_configured=arduino.configured,
                    telemetry=telemetry,
                )
                
                if intersection_enabled:
                    draw_intersection_debug(
                        preview,
                        intersection_control,
                    )

                if (
                    obstacle_enabled
                    and car_result is not None
                ):
                    draw_obstacle_debug(
                        preview,
                        car_result,
                        distance_data,
                        avoidance_command,
                        corridor_center_ratio=args.vehicle_x_ratio,
                        corridor_width_ratio=0.36,
                        obstacle_distance_cm=200,
                        clear_distance_cm=250,
                        display_max_cm=250,
                        mock_enabled=mock_front_blocked,
                    )

                cv2.putText(
                    preview,
                    (
                        f"FEATURES obstacle={'ON' if obstacle_enabled else 'OFF'} "
                        f"intersection={'ON' if intersection_enabled else 'OFF'}"
                    ),
                    (12, frame.shape[0] - 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (90, 230, 90),
                    1,
                    cv2.LINE_AA,
                )

                if lane_change.active:
                    cv2.putText(
                        preview,
                        (
                            f"LANE CHANGE {lane_change.direction_name}: "
                            f"steer={final_steering:+.2f} "
                            f"speed={final_speed} "
                            f"remaining={lane_change.remaining_seconds(command_now):.1f}s"
                        ),
                        (12, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = 0.88 * fps + 0.12 * instant_fps if fps else instant_fps
            if args.show_fps:
                draw_fps(preview, fps)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if driving_enabled:
                    driving_enabled = False
                    arduino.emergency_stop()
                    follower.reset()
                    lane_detector.reset()
                    obstacle_avoidance.reset(current_lane)
                    intersection_controller.reset()
                    traffic_light_inference.reset()
                    stop_line_detector.reset()
                    lane_change.cancel()
                    print("STOPPED: drive and steering outputs disabled.", flush=True)
                else:
                    if arduino.configured and not arduino.enabled:
                        try:
                            arduino.connect()
                        except RuntimeError as exc:
                            print(f"START FAILED: {exc}", file=sys.stderr, flush=True)
                            continue
                    follower.reset()
                    lane_detector.reset()
                    if arduino.enabled:
                        # Start from zero; the next fresh inference frame updates
                        # the command consumed by the background serial writer.
                        arduino.emergency_stop()
                        arduino.start_driving()
                    driving_enabled = True
                    if arduino.enabled:
                        print("DRIVING: fresh-frame Arduino commands will start now.", flush=True)
                    else:
                        print("SIMULATION RUNNING: no Arduino port configured.", flush=True)
            if key in (ord("s"), ord("S")):
                driving_enabled = False
                arduino.emergency_stop()
                follower.reset()
                lane_detector.reset()
                lane_change.cancel()
                print("EMERGENCY STOP: drive and steering outputs disabled.", flush=True)
            if key in (ord("r"), ord("R")):
                driving_enabled = False
                arduino.emergency_stop()
                follower.reset()
                lane_detector.reset()
                obstacle_avoidance.reset(current_lane)
                intersection_controller.reset()
                traffic_light_inference.reset()
                stop_line_detector.reset()
                lane_change.cancel()
                if arduino.enabled:
                    arduino.reset_fault()
                    print("ARDUINO FAULT RESET requested. Keep the vehicle stopped and check telemetry.", flush=True)
                else:
                    print("Arduino is not connected; press SPACE once to connect.", flush=True)
            if key in (ord("1"), ord("2")):
                requested_lane = (
                    DrivingLane.LANE_1
                    if key == ord("1")
                    else DrivingLane.LANE_2
                )

                if requested_lane != current_lane:
                    previous_lane = current_lane
                    current_lane = requested_lane

                    if current_lane == DrivingLane.LANE_2:
                        require_ultrasonic_after_lane_1_change = False

                    # 수동으로 선택한 차선을 장애물 회피 제어기에도 즉시 반영
                    obstacle_avoidance.reset(current_lane)

                    # 기존 차선 기준으로 저장돼 있던 조향 및 차선 검출 이력을 제거
                    follower.reset()
                    lane_detector.reset()

                    if driving_enabled:
                        lane_change.start(
                            previous_lane,
                            current_lane,
                            now=time.perf_counter(),
                        )
                    else:
                        lane_change.cancel()

                    lane_reference_name = (
                        "lane_center"
                        if current_lane == DrivingLane.LANE_1
                        else "lane_right"
                    )
                    drive_state = "DRIVING" if driving_enabled else "STOPPED"

                    print(
                        f"MANUAL LANE CHANGE [{drive_state}]: "
                        f"{previous_lane.name} -> {current_lane.name} "
                        f"(following {lane_reference_name})",
                        flush=True,
                    )
                else:
                    print(
                        f"Already following {current_lane.name}.",
                        flush=True,
                    )
            if key in (ord("o"), ord("O")):
                mock_front_blocked = not mock_front_blocked

                print(
                    f"Mock obstacle: front_blocked={mock_front_blocked}",
                    flush=True,
                )
            if key in (ord("b"), ord("B")):
                obstacle_enabled = not obstacle_enabled
                obstacle_avoidance.reset(current_lane)
                lane_change.cancel("obstacle feature toggled")
                cached_car_result = None
                last_avoidance_command = None
                if obstacle_analysis_future is not None:
                    obstacle_analysis_future.cancel()
                    obstacle_analysis_future = None
                last_obstacle_analysis_time = float(
                    "-inf"
                )
                print(
                    "Obstacle detection/avoidance: "
                    f"{'ON' if obstacle_enabled else 'OFF'}",
                    flush=True,
                )
            if key in (ord("i"), ord("I")):
                intersection_enabled = not intersection_enabled
                intersection_controller.reset()
                traffic_light_inference.reset()
                stop_line_detector.reset()
                print(
                    "Intersection detection/control: "
                    f"{'ON' if intersection_enabled else 'OFF'}",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass
    finally:
        # Stop the car before releasing the camera or closing the serial port.
        obstacle_executor.shutdown(
            wait=False
        )
        arduino.emergency_stop()
        arduino.close()
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
