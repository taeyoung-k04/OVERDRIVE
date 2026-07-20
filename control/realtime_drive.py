#!/usr/bin/env python3
"""Real-time lane-following runtime for OVERDRIVE."""

from __future__ import annotations

import argparse
import sys
import time
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

from control.intersection_controller import (
    IntersectionController,
    IntersectionState,
)

from perception.stop_line_detector import (
    StopLineDetector,
)

from perception.traffic_light_detector import (
    TrafficLightDetection,
    TrafficLightDetector,
)

# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model / camera
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="onnx")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-force-size", action="store_true")
    parser.add_argument("--buffered-camera", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--preview", choices=("overlay", "debug"), default="debug")
    add_postprocess_args(parser)

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
    parser.add_argument("--recovery-speed", type=int, default=65)

    # Short semantic dropouts are bridged with the most recent fitted lane.
    # Time-based limits are used so behavior is similar at different FPS.
    parser.add_argument("--lane-loss-grace-seconds", type=float, default=1.20)
    parser.add_argument("--lane-loss-speed", type=int, default=70)
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
    parser.add_argument("--speed-straight", type=int, default=100)
    parser.add_argument("--speed-turn", type=int, default=78)
    parser.add_argument("--speed-min", type=int, default=65)
    parser.add_argument(
        "--constant-speed",
        type=int,
        default=None,
        help="Use a fixed forward PWM (0..255) instead of slowing on turns.",
    )

    # Arduino serial. The matching sketch uses C,<steering>,<speed> and X.
    parser.add_argument(
        "--arduino-port",
        default="COM6",
        help="For example COM6 or /dev/ttyACM0. Omit for vision-only mode.",
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
    if args.telemetry_stale_seconds <= 0:
        raise SystemExit("--telemetry-stale-seconds must be positive")
    for name in ("speed_straight", "speed_turn", "speed_min", "recovery_speed", "lane_loss_speed"):
        value = int(getattr(args, name))
        if not 0 <= value <= 255:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 255")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 255:
        raise SystemExit("--constant-speed must be between 0 and 255")


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = load_semantic_model(args.weights, args.backend)
    postprocess_config = load_postprocess_config(args.postprocess_config) if args.postprocess else None

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reader = None if args.buffered_camera else LatestFrameReader(capture)

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
        trigger_y_ratio=0.72,
        history_size=5,
        confirm_frames=3,
        vehicle_x_ratio=args.vehicle_x_ratio,
        vehicle_corridor_ratio=0.30,
    )

    traffic_light_detector = TrafficLightDetector(
        roi_bottom_ratio=0.70,
        min_area=20,
        min_circularity=0.45,
        stable_frames=3,
    )

    intersection_controller = IntersectionController(
        # 정지선이 8프레임 연속 사라지면 교차로 통과 완료
        clear_confirm_frames=8,

        # 정지선 인식이 계속 남는 경우를 위한 안전한 timeout
        max_clearing_seconds=3.0,

        # 정차 중에는 마지막 조향각 유지
        hold_steering_while_stopped=True,

        # 초록불 출발 시 너무 빠르게 튀어나가는 것을 방지
        departure_speed_cap=80,
    )

    #avoid Obstacle
    obstacle_avoidance = ObstacleAvoidanceController(
        initial_lane=current_lane,
        detection_confirm_frames=3,
        clear_confirm_frames=10,
    )

    # 라이다 연결 전 임시 테스트값
    mock_front_blocked = False

    window_name = "Lane Boundary Follow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0
    driving_enabled = False
    print("Controls: SPACE=start/stop, 1=select lane 1, 2=select lane 2, O=toggle mock obstacle, S=emergency stop, R=reset Arduino fault, Q or ESC=quit", flush=True)

    try:
        while True:
            ok, frame = reader.read() if reader is not None else capture.read()
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

            if postprocess_config is not None:
                processed_class_map = postprocess_class_map(raw_class_map, postprocess_config)
            else:
                processed_class_map = raw_class_map

            stop_line_detection = stop_line_detector.detect(
                raw_class_map,
            )
            traffic_light_detection: TrafficLightDetection | None = None

            if intersection_controller.requires_traffic_light:
                traffic_light_detection = traffic_light_detector.detect(
                    frame,
                )

            control_class_map = (
                raw_class_map if args.control_source == "raw" else processed_class_map
            )
            
            lanes = lane_detector.detect(
                control_class_map,
            )

            # ---------------------------------------------------------
            # Obstacle avoidance
            # ---------------------------------------------------------

            # 현재는 키보드 테스트값 사용
            front_blocked = mock_front_blocked
            # 라이다 연결 후 예시
            #   front_blocked = lidar_observation.front_blocked

            # 주행 중일 때만 장애물로 차선을 변경
            if (
                driving_enabled
                and intersection_controller.allows_obstacle_avoidance
                and not stop_line_detection.should_stop
            ):
                previous_lane = current_lane

                avoidance_command = obstacle_avoidance.update(
                    front_blocked=front_blocked,
                )

                current_lane = avoidance_command.target_lane

                if avoidance_command.lane_change_requested:
                    print(
                        "Obstacle detected: "
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

                # lane follower가 원래 보내려던 명령
                base_steering=control.steering,
                base_speed=control.speed,

                driving_enabled=driving_enabled,
            )

            # 교차로 대기 상태에 처음 진입한 프레임에서만 실행된다.
            if intersection_control.entered_waiting:
                obstacle_avoidance.reset(current_lane)

                print(
                    "INTERSECTION: waiting for green.",
                    flush=True,
                )

            if intersection_control.reset_traffic_light_detector:
                traffic_light_detector.reset()
            if intersection_control.released_on_green:
                print(
                    "INTERSECTION: green confirmed, departing.",
                    flush=True,
                )

            if intersection_control.intersection_cleared:
                print(
                    "INTERSECTION: cleared, normal driving resumed.",
                    flush=True,
                )

            if driving_enabled and arduino.enabled:
                arduino.update_command(intersection_control.steering, intersection_control.speed)

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
                
                draw_intersection_debug(
                    preview,
                    intersection_control,
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
                print("EMERGENCY STOP: drive and steering outputs disabled.", flush=True)
            if key in (ord("r"), ord("R")):
                driving_enabled = False
                arduino.emergency_stop()
                follower.reset()
                lane_detector.reset()
                if arduino.enabled:
                    arduino.reset_fault()
                    print("ARDUINO FAULT RESET requested. Keep the vehicle stopped and check telemetry.", flush=True)
                else:
                    print("Arduino is not connected; press SPACE once to connect.", flush=True)
            if key == ord("1"):
                if driving_enabled:
                    print(
                        "Stop the vehicle before manually selecting lane 1.",
                        flush=True,
                    )
                else:
                    current_lane = DrivingLane.LANE_1
                    obstacle_avoidance.reset(current_lane)
                    follower.reset()
                    lane_detector.reset()

                    print(
                        "Selected LANE 1: following lane_center",
                        flush=True,
                    )
            if key == ord("2"):
                if driving_enabled:
                    print(
                        "Stop the vehicle before manually selecting lane 2.",
                        flush=True,
                    )
                else:
                    current_lane = DrivingLane.LANE_2
                    obstacle_avoidance.reset(current_lane)
                    follower.reset()
                    lane_detector.reset()

                    print(
                        "Selected LANE 2: following lane_right",
                        flush=True,
                    )
            if key in (ord("o"), ord("O")):
                mock_front_blocked = not mock_front_blocked

                print(
                    f"Mock obstacle: front_blocked={mock_front_blocked}",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass
    finally:
        # Stop the car before releasing the camera or closing the serial port.
        arduino.emergency_stop()
        arduino.close()
        if reader is not None:
            reader.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
