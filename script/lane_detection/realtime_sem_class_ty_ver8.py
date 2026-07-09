#!/usr/bin/env python3
"""ONNX yellow+green lane following for the DC steering Arduino protocol.

Runtime behavior:
- Program start: camera, segmentation, and lane calculation only.
- First SPACE: connect to Arduino asynchronously, then start driving.
- Next SPACE: send ``X`` and remain in standby while vision continues.
- Further SPACE presses: toggle between driving and standby.
- Q or ESC: stop, close Arduino, and exit.

Arduino protocol (115200 baud):
- ``C,<steering>,<drive_pwm>
``
  - steering: -1000 .. +1000 (negative=left, positive=right)
  - drive_pwm: 0 .. 255 (forward only)
- ``X
``: immediate drive and steering stop

The lane controller still computes normalized steering in -1.0 .. +1.0. This
script converts it to the Arduino's proportional -1000 .. +1000 command, so
small errors produce short steering pulses and large errors produce stronger
pulses. Changed commands are sent immediately; unchanged commands are refreshed
periodically to satisfy the Arduino watchdog.
"""

from __future__ import annotations

#   아두이노 업로드 해야됨 : lane_follow_dc_steering.ino
# 실행 명령:
# python .\script\lane_detection\realtime_sem_class_ty_ver8.py `
#   --weights .\runs\semantic\yolo_lane_sem_class\train_cpu_640_yolo26n_ade20k\weights\best.onnx `
#   --camera 1 `
#   --lane-follow `
#   --lane-follow-right `
#   --show-fps `
#   --arduino-port COM6
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import infer_sem_class as _sem

CLASS_TO_ID = _sem.CLASS_TO_ID
make_class_overlay = _sem.make_class_overlay
semantic_to_class_map = _sem.semantic_to_class_map

# Newer team version of infer_sem_class.py provides these helpers.
# The fallbacks keep this script usable with the older project layout too.
DEFAULT_WEIGHTS = getattr(
    _sem,
    "DEFAULT_WEIGHTS",
    Path("runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
)
load_semantic_model = getattr(_sem, "load_semantic_model", None)
add_postprocess_args = getattr(_sem, "add_postprocess_args", None)
load_postprocess_config = getattr(_sem, "load_postprocess_config", None)
postprocess_class_map = getattr(_sem, "postprocess_class_map", None)
from lane_following import (
    compute_lane_following,
    draw_lane_following_debug,
)
from lane_following_right_ver8 import (
    TwoLineRightFollower,
    draw_lane_following_right_debug,
)


class LatestFrameCamera:
    """Read camera continuously and keep only the latest frame."""

    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.running = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> "LatestFrameCamera":
        self.running = True
        self.thread.start()
        return self

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.capture.read()

            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame

            if not ok:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return self.ok, self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


def steering_to_dc_command(
    steering: float | None,
    gain: float = 1.0,
    exponent: float = 1.0,
    quantum: int = 10,
    invert: bool = False,
) -> int:
    """Map normalized steering (-1..1) to the Arduino DC command (-1000..1000).

    ``exponent`` controls sensitivity:
    - 1.0: linear
    - >1.0: softer near center
    - <1.0: stronger near center

    ``quantum`` suppresses meaningless one-count changes while retaining fine
    proportional control. The Arduino applies its own final deadband.
    """

    if steering is None:
        return 0

    value = float(np.clip(float(steering) * float(gain), -1.0, 1.0))
    if invert:
        value = -value

    magnitude = abs(value) ** float(exponent)
    command = int(round(np.sign(value) * magnitude * 1000.0))

    quantum = max(1, int(quantum))
    command = int(round(command / quantum) * quantum)
    return int(np.clip(command, -1000, 1000))


class ArduinoCarController:
    """Serial controller for ``C,<steering>,<drive_pwm>`` / ``X`` protocol."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        drive_pwm: int = 100,
        command_interval: float = 0.05,
        invert_steering: bool = False,
        steering_output_gain: float = 1.0,
        steering_output_exponent: float = 1.0,
        steering_command_quantum: int = 10,
        reset_delay: float = 2.0,
    ):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Install it with: pip install pyserial"
            ) from exc

        self.serial = serial.Serial(
            port,
            baudrate,
            timeout=0.0,
            write_timeout=0.05,
        )
        if reset_delay > 0:
            time.sleep(reset_delay)

        # Discard READY / protocol banner lines printed during Arduino reset.
        try:
            self.serial.reset_input_buffer()
        except Exception:
            pass

        self.drive_pwm = int(np.clip(drive_pwm, 0, 255))
        self.command_interval = max(0.001, float(command_interval))
        self.invert_steering = bool(invert_steering)
        self.steering_output_gain = max(0.0, float(steering_output_gain))
        self.steering_output_exponent = max(1e-6, float(steering_output_exponent))
        self.steering_command_quantum = max(1, int(steering_command_quantum))

        self.last_send_time = 0.0
        self.last_steering_command: int | None = None
        self.last_drive_pwm: int | None = None
        self.full_stop_active = True
        self.lock = threading.Lock()

    def _map_steering(self, steering: float | None) -> int:
        return steering_to_dc_command(
            steering=steering,
            gain=self.steering_output_gain,
            exponent=self.steering_output_exponent,
            quantum=self.steering_command_quantum,
            invert=self.invert_steering,
        )

    def _send_line(self, line: str, flush: bool = False) -> None:
        payload = (line.rstrip("\r\n") + "\n").encode("ascii")
        self.serial.write(payload)
        if flush:
            self.serial.flush()

    def _send_control_locked(
        self,
        steering_command: int,
        drive_pwm: int,
        flush: bool,
    ) -> None:
        steering_command = int(np.clip(steering_command, -1000, 1000))
        drive_pwm = int(np.clip(drive_pwm, 0, 255))
        self._send_line(
            f"C,{steering_command},{drive_pwm}",
            flush=flush,
        )
        self.last_steering_command = steering_command
        self.last_drive_pwm = drive_pwm
        self.last_send_time = time.perf_counter()
        self.full_stop_active = False

    def start_drive(self) -> None:
        """Start forward motion with neutral steering; next frame updates it."""
        with self.lock:
            self._send_control_locked(0, self.drive_pwm, flush=True)

    def update(
        self,
        steering: float | None,
        drive: bool = True,
        urgent: bool = False,
    ) -> None:
        """Send proportional steering immediately when it changes.

        Unchanged commands are refreshed at ``command_interval`` so the
        Arduino's 400 ms watchdog never expires. ``urgent`` is intentionally
        not used to resend an unchanged value every inference frame; command
        changes already bypass the heartbeat interval.
        """

        now = time.perf_counter()
        with self.lock:
            if not drive:
                self._full_stop_locked(force=False)
                return

            steering_command = self._map_steering(steering)
            drive_pwm = self.drive_pwm
            changed = (
                steering_command != self.last_steering_command
                or drive_pwm != self.last_drive_pwm
                or self.full_stop_active
            )
            refresh_due = (now - self.last_send_time) >= self.command_interval

            if changed:
                self._send_control_locked(
                    steering_command,
                    drive_pwm,
                    flush=True,
                )
            elif refresh_due:
                self._send_control_locked(
                    steering_command,
                    drive_pwm,
                    flush=False,
                )

    def _full_stop_locked(self, force: bool) -> None:
        if self.full_stop_active and not force:
            return
        self._send_line("X", flush=True)
        self.full_stop_active = True
        self.last_steering_command = None
        self.last_drive_pwm = None
        self.last_send_time = time.perf_counter()

    def stop(self, force: bool = False) -> None:
        with self.lock:
            self._full_stop_locked(force=force)

    def close(self) -> None:
        with self.lock:
            try:
                self._full_stop_locked(force=True)
            finally:
                self.serial.close()


class ArduinoRuntime:
    """Asynchronous Arduino connection and SPACE-key drive/standby state."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    STANDBY = "STANDBY"
    DRIVING = "DRIVING"
    ERROR = "ERROR"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.controller: ArduinoCarController | None = None
        self.state = self.DISCONNECTED
        self.error_message: str | None = None
        self.start_when_connected = False
        self.connection_thread: threading.Thread | None = None
        self.closing = False

    def _make_controller(self) -> ArduinoCarController:
        return ArduinoCarController(
            port=self.args.arduino_port,
            baudrate=self.args.arduino_baudrate,
            drive_pwm=self.args.drive_pwm,
            command_interval=self.args.command_interval,
            invert_steering=self.args.invert_steering,
            steering_output_gain=self.args.steering_output_gain,
            steering_output_exponent=self.args.steering_output_exponent,
            steering_command_quantum=self.args.steering_command_quantum,
            reset_delay=self.args.arduino_reset_delay,
        )

    def _connect_worker(self) -> None:
        try:
            controller = self._make_controller()
        except Exception as exc:  # serial errors vary by platform
            with self.lock:
                self.controller = None
                self.state = self.ERROR
                self.error_message = str(exc)
            print(f"\n[Arduino] connection failed: {exc}", flush=True)
            return

        with self.lock:
            if self.closing:
                should_close = True
                should_drive = False
            else:
                should_close = False
                self.controller = controller
                should_drive = self.start_when_connected
                self.state = self.DRIVING if should_drive else self.STANDBY
                self.error_message = None

        if should_close:
            controller.close()
            return

        if should_drive:
            controller.start_drive()
            print("\n[Arduino] connected; DRIVE ENABLED", flush=True)
        else:
            controller.stop(force=True)
            print("\n[Arduino] connected; STANDBY", flush=True)

    def toggle(self) -> str:
        """Handle one SPACE press and return the resulting state."""
        if not self.args.arduino_port:
            print("\n[Arduino] --arduino-port is missing; cannot connect.", flush=True)
            return self.DISCONNECTED

        controller_to_start: ArduinoCarController | None = None
        controller_to_stop: ArduinoCarController | None = None

        with self.lock:
            if self.state in (self.DISCONNECTED, self.ERROR):
                self.state = self.CONNECTING
                self.error_message = None
                self.start_when_connected = True
                self.connection_thread = threading.Thread(
                    target=self._connect_worker,
                    daemon=True,
                )
                self.connection_thread.start()
                print(
                    f"\n[Arduino] connecting to {self.args.arduino_port}; "
                    "vision continues...",
                    flush=True,
                )
                return self.state

            if self.state == self.CONNECTING:
                self.start_when_connected = not self.start_when_connected
                mode = "DRIVE after connection" if self.start_when_connected else "STANDBY after connection"
                print(f"\n[Arduino] {mode}", flush=True)
                return self.state

            if self.state == self.STANDBY:
                self.state = self.DRIVING
                controller_to_start = self.controller
            elif self.state == self.DRIVING:
                self.state = self.STANDBY
                controller_to_stop = self.controller

            result_state = self.state

        if controller_to_start is not None:
            controller_to_start.start_drive()
            print("\n[Arduino] DRIVE ENABLED", flush=True)
        if controller_to_stop is not None:
            controller_to_stop.stop(force=True)
            print("\n[Arduino] STANDBY", flush=True)

        return result_state

    def update(self, steering: float | None, drive_allowed: bool, urgent: bool) -> None:
        with self.lock:
            controller = self.controller
            driving = self.state == self.DRIVING

        if controller is None or not driving:
            return

        if drive_allowed:
            controller.update(steering=steering, drive=True, urgent=urgent)
        else:
            # Keep the user-selected DRIVE state armed, but stop physically until
            # valid lane control returns. The next valid frame resumes automatically.
            controller.stop(force=False)

    def status(self) -> tuple[str, str]:
        with self.lock:
            state = self.state
            error = self.error_message

        if state == self.DISCONNECTED:
            return state, "SPACE: connect + start"
        if state == self.CONNECTING:
            suffix = "will start" if self.start_when_connected else "will standby"
            return state, suffix
        if state == self.STANDBY:
            return state, "SPACE: start"
        if state == self.DRIVING:
            return state, "SPACE: stop / standby"
        return state, error or "SPACE: retry"

    def close(self) -> None:
        with self.lock:
            self.closing = True
            self.start_when_connected = False
            controller = self.controller
            self.controller = None
            self.state = self.DISCONNECTED

        if controller is not None:
            controller.close()


def smooth_steering(
    previous: float | None,
    target: float,
    alpha: float,
    max_step: float,
) -> float:
    """Apply exponential smoothing and a simple step limit to steering values."""

    if previous is None:
        return float(target)

    smoothed = previous + alpha * (float(target) - previous)

    if max_step is not None and max_step > 0.0:
        delta = smoothed - previous
        if abs(delta) > max_step:
            smoothed = previous + (max_step if delta > 0 else -max_step)

    return float(smoothed)


def is_urgent_steering(
    previous: float | None,
    target: float,
    lane_result,
    large_delta: float,
    reversal_epsilon: float,
) -> tuple[bool, str]:
    """Identify control changes that should bypass conservative smoothing."""
    departure_risk = bool(getattr(lane_result, "departure_risk", False))

    if previous is None:
        return True, "first control"

    reversal = (
        abs(previous) >= reversal_epsilon
        and abs(target) >= reversal_epsilon
        and np.sign(previous) != np.sign(target)
    )
    if reversal:
        return True, "direction reversal"

    if abs(target - previous) >= large_delta:
        return True, "large steering change"

    if departure_risk:
        return True, "departure protection"

    return False, "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=(
            "Path to .pt or .onnx semantic weights. With --backend onnx, "
            "a .pt suffix is automatically replaced with .onnx."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "pt", "onnx"),
        default="onnx",
        help="Model backend. auto selects from the resolved weight suffix.",
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, 0, cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV CPU threads. For small 640x360 masks, 1 is often faster and steadier.",
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--no-force-size", action="store_true")

    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--half", action="store_true")

    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run YOLO every N frames. Higher values improve speed but reduce responsiveness.",
    )

    parser.add_argument(
        "--roi-top-ratio",
        type=float,
        default=0.30,
        help="Ignore top part of image. 0.30 means use bottom 70 percent.",
    )

    parser.add_argument(
        "--lane-follow",
        action="store_true",
        help="Compute lane center and steering value.",
    )
    parser.add_argument(
        "--lane-follow-right",
        action="store_true",
        help="Use the fast yellow+green right-lane controller; blue/lane_left is ignored.",
    )
    parser.add_argument(
        "--balanced-realtime",
        action="store_true",
        help="Use a slightly lighter inference path that preserves lane accuracy while improving responsiveness.",
    )
    parser.add_argument(
        "--recognition-priority",
        action="store_true",
        help=(
            "Prefer lane-mask recall: force fresh inference, at least imgsz 416, "
            "and include slightly more distant road area."
        ),
    )

    parser.add_argument(
        "--eval-y-ratio",
        type=float,
        default=0.82,
        help="Near Y position used for immediate lateral control.",
    )

    parser.add_argument(
        "--lookahead-y-ratio",
        type=float,
        default=0.58,
        help="Far Y position used to anticipate upcoming curvature.",
    )

    parser.add_argument(
        "--lookahead-weight",
        type=float,
        default=0.20,
        help="Weight of the far target in steering calculation.",
    )

    parser.add_argument(
        "--heading-weight",
        type=float,
        default=0.15,
        help=(
            "Weight of perspective-compensated curvature. The old raw "
            "far-minus-near heading caused a persistent left bias."
        ),
    )

    parser.add_argument(
        "--lane-target-smoothing",
        type=float,
        default=0.70,
        help="Temporal smoothing for detected lane targets. Higher reacts faster.",
    )

    parser.add_argument(
        "--target-prediction",
        type=float,
        default=0.30,
        help="Small prediction horizon used to compensate inference latency.",
    )

    parser.add_argument(
        "--min-lane-confidence",
        type=float,
        default=0.16,
        help="Stop/hold when yellow+green target confidence is below this value.",
    )

    parser.add_argument(
        "--curve-hold-frames",
        type=int,
        default=2,
        help="Reuse the last fitted yellow/green curve for this many brief misses.",
    )
    parser.add_argument(
        "--lane-max-candidates",
        type=int,
        default=5,
        help="Maximum connected-component candidates considered per lane color.",
    )
    parser.add_argument(
        "--lane-component-min-area",
        type=int,
        default=8,
        help="Minimum component area retained by lane post-processing.",
    )
    parser.add_argument(
        "--lane-component-min-height",
        type=int,
        default=9,
        help="Minimum component vertical extent retained by lane post-processing.",
    )

    parser.add_argument(
        "--departure-margin-ratio",
        type=float,
        default=0.16,
        help="Start boundary-protection steering inside this fraction of lane width.",
    )

    parser.add_argument(
        "--target-lane-ratio",
        type=float,
        default=0.58,
        help=(
            "Desired position between yellow=0.0 and green=1.0. "
            "0.58 keeps the vehicle slightly right of geometric center."
        ),
    )
    parser.add_argument(
        "--corner-target-lane-ratio",
        type=float,
        default=0.64,
        help="Target ratio used progressively as corner strength increases.",
    )
    parser.add_argument(
        "--curve-activation-error",
        type=float,
        default=0.025,
        help="Perspective-compensated curve error that starts adaptive corner mode.",
    )
    parser.add_argument(
        "--curve-full-error",
        type=float,
        default=0.10,
        help="Curve error at which adaptive corner mode reaches full strength.",
    )
    parser.add_argument(
        "--corner-lookahead-weight",
        type=float,
        default=0.42,
        help="Far-target weight at full corner strength.",
    )
    parser.add_argument(
        "--corner-heading-weight",
        type=float,
        default=0.45,
        help="Perspective-compensated curvature weight at full corner strength.",
    )
    parser.add_argument(
        "--left-boundary-margin-ratio",
        type=float,
        default=0.34,
        help="Start protecting the yellow boundary this early within lane width.",
    )
    parser.add_argument(
        "--right-boundary-margin-ratio",
        type=float,
        default=0.12,
        help="Start protecting the green boundary inside this fraction of lane width.",
    )
    parser.add_argument(
        "--left-boundary-gain",
        type=float,
        default=0.90,
        help="Rightward correction gain when approaching yellow/lane-1 boundary.",
    )
    parser.add_argument(
        "--right-boundary-gain",
        type=float,
        default=0.45,
        help="Leftward correction gain when approaching green boundary.",
    )
    parser.add_argument(
        "--curve-sign-override-error",
        type=float,
        default=0.035,
        help=(
            "Allow a confident corner signal to override the near-point sign guard "
            "once curvature exceeds this value."
        ),
    )

    parser.add_argument(
        "--steering-neutral-error",
        type=float,
        default=0.030,
        help="Near normalized error treated as exactly centered.",
    )
    parser.add_argument(
        "--far-neutral-error",
        type=float,
        default=0.040,
        help="Far normalized error allowed inside the centered deadband.",
    )
    parser.add_argument(
        "--curvature-neutral-error",
        type=float,
        default=0.020,
        help="Perspective-compensated curvature error treated as straight.",
    )
    parser.add_argument(
        "--neutral-exit-multiplier",
        type=float,
        default=1.60,
        help=(
            "Keep zero steering until error exceeds the neutral thresholds by "
            "this multiplier; reduces a/d chatter near lane center."
        ),
    )
    parser.add_argument(
        "--curvature-term-limit",
        type=float,
        default=0.12,
        help="Maximum absolute contribution from look-ahead curvature.",
    )
    parser.add_argument(
        "--sign-guard-error",
        type=float,
        default=0.025,
        help="Trust the near target when another term tries to reverse this clear error.",
    )
    parser.add_argument(
        "--disable-sign-guard",
        action="store_true",
        help="Allow look-ahead curvature to reverse the near-target correction.",
    )
    parser.add_argument(
        "--steering-trim",
        type=float,
        default=0.0,
        help=(
            "Small constant steering correction after geometry. Positive is image-right; "
            "use only for verified mechanical pull, not camera misalignment."
        ),
    )

    parser.add_argument(
        "--steering-kp",
        type=float,
        default=1.0,
        help="Steering gain.",
    )

    parser.add_argument(
        "--camera-center-offset-px",
        type=float,
        default=0.0,
        help="Camera center correction in pixels.",
    )

    parser.add_argument(
        "--fallback-lane-width-px",
        type=float,
        default=0.0,
        help=(
            "Approximate pixel distance between lane_center and lane_right. "
            "0 means auto estimate from image width."
        ),
    )

    parser.add_argument(
        "--disable-single-lane-fallback",
        action="store_true",
        help="Disable fallback when only one lane line is detected.",
    )

    parser.add_argument(
        "--lost-frame-tolerance",
        type=int,
        default=3,
        help=(
            "Number of consecutive invalid lane frames allowed before full stop. "
            "During this period, the last valid steering is reused."
        ),
    )

    parser.add_argument(
        "--lost-steering-decay",
        type=float,
        default=0.65,
        help="Decay ratio applied to stale steering while lane is temporarily lost.",
    )

    parser.add_argument(
        "--instant-control",
        dest="instant_control",
        action="store_true",
        default=True,
        help=(
            "Apply each fresh raw steering result immediately, bypassing lane-target "
            "and steering smoothing. Enabled by default."
        ),
    )
    parser.add_argument(
        "--smoothed-control",
        dest="instant_control",
        action="store_false",
        help="Use the older two-stage smoothing path instead of direct control.",
    )

    parser.add_argument(
        "--steering-smoothing",
        type=float,
        default=0.70,
        help="Low-pass smoothing factor for steering values. Higher reacts faster.",
    )

    parser.add_argument(
        "--max-steering-step",
        type=float,
        default=0.50,
        help="Maximum normal steering change per fresh lane measurement.",
    )

    parser.add_argument(
        "--urgent-steering-smoothing",
        type=float,
        default=0.92,
        help="Smoothing used for reversal, large changes, or departure risk.",
    )
    parser.add_argument(
        "--urgent-max-steering-step",
        type=float,
        default=0.80,
        help="Maximum steering change for urgent corrections.",
    )
    parser.add_argument(
        "--urgent-steering-delta",
        type=float,
        default=0.25,
        help="Raw steering change that triggers urgent response.",
    )
    parser.add_argument(
        "--direction-reversal-epsilon",
        type=float,
        default=0.03,
        help="Minimum magnitude required to classify a direction reversal.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print lane status every N frames to reduce console overhead.",
    )

    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-display", action="store_true")

    parser.add_argument(
        "--arduino-port",
        default=None,
        help="Arduino serial port. Example: COM3, /dev/ttyUSB0, /dev/ttyACM0",
    )
    parser.add_argument(
        "--arduino-baudrate",
        type=int,
        default=115200,
        help="Baud rate for the DC steering protocol. The provided sketch uses 115200.",
    )
    parser.add_argument(
        "--drive-pwm",
        type=int,
        default=100,
        help="Forward drive PWM sent in C,<steering>,<drive_pwm> (0-255).",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=0.05,
        help=(
            "Heartbeat interval in seconds. Changed steering is sent immediately; "
            "0.05 matches the Arduino sketch's expected 20 Hz update rate."
        ),
    )
    parser.add_argument(
        "--steering-output-gain",
        type=float,
        default=1.0,
        help="Gain applied before converting normalized steering to -1000..1000.",
    )
    parser.add_argument(
        "--steering-output-exponent",
        type=float,
        default=1.0,
        help=(
            "Steering response curve: 1.0 linear, >1 softer near center, "
            "<1 more aggressive near center."
        ),
    )
    parser.add_argument(
        "--steering-command-quantum",
        type=int,
        default=10,
        help="Quantize -1000..1000 commands to this step to reduce serial jitter.",
    )
    parser.add_argument(
        "--arduino-reset-delay",
        type=float,
        default=2.0,
        help="Seconds to wait for Arduino reset after opening the serial port.",
    )
    parser.add_argument(
        "--invert-steering",
        action="store_true",
        help=(
            "Invert Python steering sign. Do not also set INVERT_STEERING=true "
            "in the Arduino sketch."
        ),
    )

    if add_postprocess_args is not None:
        add_postprocess_args(parser)
    else:
        # Compatibility options when using the older infer_sem_class.py.
        parser.add_argument(
            "--postprocess",
            action="store_true",
            help="Apply class-map post-processing when supported by infer_sem_class.py.",
        )
        parser.add_argument(
            "--postprocess-config",
            type=Path,
            default=None,
            help="Optional post-processing config path.",
        )

    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    return capture


def normalize_frame_size(
    frame: np.ndarray,
    width: int,
    height: int,
    force_size: bool,
) -> np.ndarray:
    if not force_size or width <= 0 or height <= 0:
        return frame

    if frame.shape[1] == width and frame.shape[0] == height:
        return frame

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def is_cuda_device(device: str) -> bool:
    device = str(device).lower()
    return device.startswith("cuda") or device.isdigit()


def resolve_model_path(weights: Path, backend: str) -> tuple[Path, str]:
    """Resolve the actual model file and effective backend.

    The team loader supports passing a .pt path together with --backend onnx.
    Resolving it here as well makes existence checks and console output explicit.
    """
    path = Path(weights)
    requested = str(backend).lower()

    if requested == "onnx" and path.suffix.lower() != ".onnx":
        path = path.with_suffix(".onnx")
    elif requested == "pt" and path.suffix.lower() != ".pt":
        path = path.with_suffix(".pt")

    if requested == "auto":
        suffix = path.suffix.lower()
        effective = "onnx" if suffix == ".onnx" else "pt"
    else:
        effective = requested

    return path, effective


def load_model_for_backend(weights: Path, backend: str):
    """Load with the team's backend-aware helper, then fall back to Ultralytics."""
    if load_semantic_model is not None:
        return load_semantic_model(weights, backend), "infer_sem_class.load_semantic_model"

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Neither infer_sem_class.load_semantic_model nor ultralytics.YOLO is available."
        ) from exc

    return YOLO(str(weights), task="semantic"), "ultralytics.YOLO fallback"


def run_semantic_predict(
    model,
    source: np.ndarray,
    args: argparse.Namespace,
    use_half: bool,
):
    """Call a PT/ONNX-compatible predict API without passing invalid options."""
    kwargs = dict(
        source=source,
        imgsz=args.imgsz,
        device=args.device,
        task="semantic",
        verbose=False,
    )

    # ONNX precision is fixed at export time. Passing half to an ONNX runtime
    # wrapper may be rejected or silently ignored, so only PT receives it.
    if args.effective_backend == "pt" and use_half:
        kwargs["half"] = True

    try:
        return model.predict(**kwargs)
    except TypeError:
        # Some lightweight wrappers do not expose the task keyword.
        kwargs.pop("task", None)
        return model.predict(**kwargs)


def maybe_postprocess_class_map(
    class_map: np.ndarray,
    postprocess_config,
) -> np.ndarray:
    if postprocess_config is None:
        return class_map
    if postprocess_class_map is None:
        raise RuntimeError(
            "--postprocess was requested, but this infer_sem_class.py does not "
            "provide postprocess_class_map()."
        )
    return postprocess_class_map(class_map, postprocess_config)


def predict_class_map(
    model,
    frame: np.ndarray,
    args: argparse.Namespace,
    use_half: bool,
    postprocess_config=None,
) -> np.ndarray:
    """Run PT/ONNX semantic inference on the ROI and return a full-frame map."""

    h, w = frame.shape[:2]

    roi_top_ratio = max(0.0, min(float(args.roi_top_ratio), 0.95))
    roi_y = int(h * roi_top_ratio)

    if roi_y <= 0:
        results = run_semantic_predict(model, frame, args, use_half)
        class_map = semantic_to_class_map(
            results[0].semantic_mask,
            frame.shape[:2],
        )
        return maybe_postprocess_class_map(class_map, postprocess_config)

    roi = frame[roi_y:h, :]
    results = run_semantic_predict(model, roi, args, use_half)
    roi_class_map = semantic_to_class_map(
        results[0].semantic_mask,
        roi.shape[:2],
    )

    class_map = np.zeros((h, w), dtype=np.uint8)
    class_map[roi_y:h, :] = roi_class_map
    return maybe_postprocess_class_map(class_map, postprocess_config)


def main() -> None:
    args = parse_args()

    cv2.setNumThreads(max(1, args.opencv_threads))

    if args.infer_every < 1:
        raise SystemExit("--infer-every must be >= 1")

    if not 0.0 <= args.roi_top_ratio < 1.0:
        raise SystemExit("--roi-top-ratio must be in [0.0, 1.0)")

    if args.lost_frame_tolerance < 0:
        raise SystemExit("--lost-frame-tolerance must be >= 0")

    if not 0.0 <= args.lost_steering_decay <= 1.0:
        raise SystemExit("--lost-steering-decay must be in [0.0, 1.0]")

    if not 0.0 <= args.steering_smoothing <= 1.0:
        raise SystemExit("--steering-smoothing must be in [0.0, 1.0]")

    if not 0.0 <= args.urgent_steering_smoothing <= 1.0:
        raise SystemExit("--urgent-steering-smoothing must be in [0.0, 1.0]")

    if not 0.0 < args.command_interval < 0.35:
        raise SystemExit(
            "--command-interval must be > 0 and < 0.35 seconds "
            "for the Arduino's 400 ms watchdog"
        )

    if not 0 <= args.drive_pwm <= 255:
        raise SystemExit("--drive-pwm must be in [0, 255]")

    if args.steering_output_gain < 0.0:
        raise SystemExit("--steering-output-gain must be >= 0")

    if args.steering_output_exponent <= 0.0:
        raise SystemExit("--steering-output-exponent must be > 0")

    if not 1 <= args.steering_command_quantum <= 1000:
        raise SystemExit("--steering-command-quantum must be in [1, 1000]")

    if args.arduino_reset_delay < 0.0:
        raise SystemExit("--arduino-reset-delay must be >= 0")

    if not 0.0 <= args.lookahead_y_ratio < args.eval_y_ratio <= 1.0:
        raise SystemExit(
            "Require 0 <= --lookahead-y-ratio < --eval-y-ratio <= 1"
        )

    if not 0.0 <= args.lookahead_weight <= 1.0:
        raise SystemExit("--lookahead-weight must be in [0.0, 1.0]")

    if not 0.0 <= args.lane_target_smoothing <= 1.0:
        raise SystemExit("--lane-target-smoothing must be in [0.0, 1.0]")

    for name in (
        "steering_neutral_error",
        "far_neutral_error",
        "curvature_neutral_error",
        "curvature_term_limit",
        "sign_guard_error",
    ):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")


    if args.print_every < 1:
        raise SystemExit("--print-every must be >= 1")

    resolved_weights, effective_backend = resolve_model_path(
        args.weights,
        args.backend,
    )
    args.weights = resolved_weights
    args.effective_backend = effective_backend

    if not args.weights.exists():
        raise SystemExit(
            f"Weights file does not exist: {args.weights} "
            f"(backend={args.effective_backend})"
        )

    use_half = (
        args.half
        and args.effective_backend == "pt"
        and is_cuda_device(args.device)
    )
    if args.balanced_realtime:
        # Keep inference fresh first; reduce input size before skipping many frames.
        args.infer_every = min(max(1, args.infer_every), 2)
        args.imgsz = min(args.imgsz, 416)
        args.roi_top_ratio = float(np.clip(args.roi_top_ratio, 0.28, 0.38))

    if args.recognition_priority:
        args.infer_every = 1
        args.imgsz = max(args.imgsz, 416)
        args.roi_top_ratio = min(args.roi_top_ratio, 0.25)

    if args.instant_control and args.infer_every != 1:
        print(
            f"[WARN] instant control requires fresh inference; "
            f"forcing --infer-every 1 (was {args.infer_every}).",
            flush=True,
        )
        args.infer_every = 1

    if args.half and not use_half:
        if args.effective_backend == "onnx":
            reason = "ONNX precision is fixed when the model is exported"
        else:
            reason = "the selected device is not CUDA"
        print(f"[WARN] --half is ignored because {reason}.", flush=True)

    postprocess_config = None
    if getattr(args, "postprocess", False):
        if load_postprocess_config is None:
            raise SystemExit(
                "--postprocess requires the team's updated infer_sem_class.py."
            )
        postprocess_path = getattr(args, "postprocess_config", None)
        postprocess_config = load_postprocess_config(postprocess_path)

    print(
        f"Loading model: {args.weights} "
        f"(backend={args.effective_backend})",
        flush=True,
    )
    model, model_loader_name = load_model_for_backend(
        args.weights,
        args.effective_backend,
    )
    print(f"Model loader: {model_loader_name}", flush=True)

    capture = open_camera(args.camera, args.width, args.height)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height}",
        flush=True,
    )

    print(
        f"Inference: backend={args.effective_backend}, device={args.device}, "
        f"imgsz={args.imgsz}, half={use_half}, "
        f"infer_every={args.infer_every}, roi_top_ratio={args.roi_top_ratio}, "
        f"postprocess={postprocess_config is not None}",
        flush=True,
    )

    print(
        "Control mode: "
        + (
            "INSTANT (raw target -> raw steering -> immediate serial change)"
            if args.instant_control
            else "SMOOTHED"
        ),
        flush=True,
    )

    print(
        f"Right controller: yellow+green only, single_line_fallback={not args.disable_single_lane_fallback}, "
        f"fallback_lane_width_px={args.fallback_lane_width_px}, "
        f"lost_frame_tolerance={args.lost_frame_tolerance}",
        flush=True,
    )
    print(
        f"Lane fitting: pair_selection=True, max_candidates={args.lane_max_candidates}, "
        f"curve_hold_frames={args.curve_hold_frames}, "
        f"min_component={args.lane_component_min_area}px/{args.lane_component_min_height}px",
        flush=True,
    )
    print(
        f"Arduino protocol: DC-V1 @ {args.arduino_baudrate} baud | "
        f"drive_pwm={args.drive_pwm} | heartbeat={args.command_interval:.3f}s | "
        f"steering_gain={args.steering_output_gain:.2f} | "
        f"exponent={args.steering_output_exponent:.2f} | "
        f"quantum={args.steering_command_quantum}",
        flush=True,
    )

    arduino = ArduinoRuntime(args)
    if args.arduino_port:
        print(
            f"Arduino is initially disconnected ({args.arduino_port}). "
            "Press SPACE to connect and start.",
            flush=True,
        )
    else:
        print("No --arduino-port supplied; vision-only mode.", flush=True)

    right_lane_follower = None
    if args.lane_follow and args.lane_follow_right:
        right_lane_follower = TwoLineRightFollower(
            center_class_id=CLASS_TO_ID["lane_center"],
            right_class_id=CLASS_TO_ID["lane_right"],
            roi_top_ratio=args.roi_top_ratio,
            eval_y_ratio=args.eval_y_ratio,
            lookahead_y_ratio=args.lookahead_y_ratio,
            camera_center_offset_px=args.camera_center_offset_px,
            steering_kp=args.steering_kp,
            lookahead_weight=args.lookahead_weight,
            heading_weight=args.heading_weight,
            fallback_lane_width_px=(
                None if args.fallback_lane_width_px <= 0 else args.fallback_lane_width_px
            ),
            target_smoothing=args.lane_target_smoothing,
            prediction_horizon=(0.0 if args.instant_control else args.target_prediction),
            instant_target=args.instant_control,
            allow_single_line_fallback=not args.disable_single_lane_fallback,
            min_confidence=args.min_lane_confidence,
            departure_margin_ratio=args.departure_margin_ratio,
            target_lane_ratio=args.target_lane_ratio,
            corner_target_lane_ratio=args.corner_target_lane_ratio,
            curve_activation_error=args.curve_activation_error,
            curve_full_error=args.curve_full_error,
            corner_lookahead_weight=args.corner_lookahead_weight,
            corner_heading_weight=args.corner_heading_weight,
            left_departure_margin_ratio=args.left_boundary_margin_ratio,
            right_departure_margin_ratio=args.right_boundary_margin_ratio,
            left_departure_gain=args.left_boundary_gain,
            right_departure_gain=args.right_boundary_gain,
            curve_sign_override_error=args.curve_sign_override_error,
            steering_neutral_error=args.steering_neutral_error,
            far_neutral_error=args.far_neutral_error,
            curvature_neutral_error=args.curvature_neutral_error,
            neutral_exit_multiplier=args.neutral_exit_multiplier,
            curvature_term_limit=args.curvature_term_limit,
            sign_guard_error=args.sign_guard_error,
            enable_sign_guard=not args.disable_sign_guard,
            steering_trim=args.steering_trim,
            curve_hold_frames=args.curve_hold_frames,
            lane_max_candidates=args.lane_max_candidates,
            lane_component_min_area=args.lane_component_min_area,
            lane_component_min_height=args.lane_component_min_height,
        )

    camera = LatestFrameCamera(capture).start()

    display = not args.no_display
    window_name = "Two-Line Right Lane: Yellow + Green"

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    previous_time = time.perf_counter()
    fps = 0.0
    frame_count = 0
    last_class_map = None
    last_lane_result = None

    lost_lane_frames = 0
    last_valid_steering = None
    last_control_steering = None
    urgent_control = False
    urgent_reason = "normal"

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame = camera.read()

            if not ok or frame is None:
                time.sleep(0.002)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            frame = normalize_frame_size(
                frame,
                args.width,
                args.height,
                force_size=not args.no_force_size,
            )

            frame_count += 1

            should_infer = (
                last_class_map is None
                or frame_count % args.infer_every == 0
                or last_class_map.shape != frame.shape[:2]
            )

            inference_start = time.perf_counter()

            if should_infer:
                last_class_map = predict_class_map(
                    model=model,
                    frame=frame,
                    args=args,
                    use_half=use_half,
                    postprocess_config=postprocess_config,
                )

            inference_end = time.perf_counter()

            class_map = last_class_map

            lane_compute_start = time.perf_counter()

            lane_result = None
            control_steering = None
            control_drive = False
            control_reason = "lane_follow disabled"
            urgent_control = False
            urgent_reason = "normal"

            if args.lane_follow:
                # Only feed temporal lane tracking when a NEW semantic mask exists.
                # Re-processing the same reused mask would falsely count stale data
                # as multiple fresh observations.
                new_lane_measurement = should_infer or last_lane_result is None
                if new_lane_measurement:
                    if args.lane_follow_right:
                        if right_lane_follower is None:
                            raise RuntimeError("Right-lane follower was not initialized")
                        lane_result = right_lane_follower.update(class_map)
                    else:
                        lane_result = compute_lane_following(
                            class_map=class_map,
                                            center_class_id=CLASS_TO_ID["lane_center"],
                            roi_top_ratio=args.roi_top_ratio,
                            eval_y_ratio=args.eval_y_ratio,
                            camera_center_offset_px=args.camera_center_offset_px,
                            steering_kp=args.steering_kp,
                            allow_single_lane_fallback=not args.disable_single_lane_fallback,
                            fallback_lane_width_px=(
                                None if args.fallback_lane_width_px <= 0 else args.fallback_lane_width_px
                            ),
                        )
                    last_lane_result = lane_result
                else:
                    lane_result = last_lane_result

                if lane_result.valid:
                    if new_lane_measurement:
                        lost_lane_frames = 0

                    raw_steering = float(lane_result.steering or 0.0)
                    if args.instant_control:
                        # Direct path: the newest lane result becomes the command
                        # without a second EMA or per-frame step limiter.
                        control_steering = raw_steering
                        urgent_control = bool(new_lane_measurement)
                        urgent_reason = (
                            "instant fresh result" if new_lane_measurement else "instant hold"
                        )
                    else:
                        urgent_control, urgent_reason = is_urgent_steering(
                            previous=last_control_steering,
                            target=raw_steering,
                            lane_result=lane_result,
                            large_delta=args.urgent_steering_delta,
                            reversal_epsilon=args.direction_reversal_epsilon,
                        )
                        control_steering = smooth_steering(
                            previous=last_control_steering,
                            target=raw_steering,
                            alpha=(
                                args.urgent_steering_smoothing
                                if urgent_control
                                else args.steering_smoothing
                            ),
                            max_step=(
                                args.urgent_max_steering_step
                                if urgent_control
                                else args.max_steering_step
                            ),
                        )
                    last_control_steering = control_steering
                    last_valid_steering = control_steering
                    control_drive = True
                    control_reason = lane_result.reason

                    if frame_count % args.print_every == 0:
                        print(
                            f"offset_px={lane_result.offset_px:+.1f} | "
                            f"offset_norm={lane_result.offset_norm:+.3f} | "
                            f"steering={control_steering:+.3f} | "
                            f"dc={steering_to_dc_command(control_steering, args.steering_output_gain, args.steering_output_exponent, args.steering_command_quantum, args.invert_steering):+d} | "
                            f"lat={getattr(lane_result, 'lateral_term', 0.0):+.3f} | "
                            f"curv={getattr(lane_result, 'curvature_term', 0.0):+.3f} | "
                            f"bound={getattr(lane_result, 'boundary_term', 0.0):+.3f} | "
                            f"ratio={getattr(lane_result, 'active_target_lane_ratio', 0.5):.2f} | "
                            f"curve={getattr(lane_result, 'curve_strength', 0.0):.2f} | "
                            f"maskY={int(np.count_nonzero(class_map == CLASS_TO_ID['lane_center']))} | "
                            f"maskG={int(np.count_nonzero(class_map == CLASS_TO_ID['lane_right']))} | "
                            f"{lane_result.reason}",
                            end="\r",
                            flush=True,
                        )

                else:
                    if new_lane_measurement:
                        lost_lane_frames += 1

                    if (
                        last_valid_steering is not None
                        and lost_lane_frames <= args.lost_frame_tolerance
                    ):
                        decay = args.lost_steering_decay ** lost_lane_frames
                        raw_steering = float(last_valid_steering * decay)
                        if args.instant_control:
                            control_steering = raw_steering
                            urgent_control = True
                            urgent_reason = "instant lost-lane fallback"
                        else:
                            control_steering = smooth_steering(
                                previous=last_control_steering,
                                target=raw_steering,
                                alpha=max(0.05, args.steering_smoothing * 0.7),
                                max_step=max(0.05, args.max_steering_step * 0.5),
                            )
                        last_control_steering = control_steering
                        control_drive = True
                        control_reason = (
                            f"temporary lane lost "
                            f"{lost_lane_frames}/{args.lost_frame_tolerance}: "
                            f"{lane_result.reason}"
                        )

                        if frame_count % args.print_every == 0:
                            print(
                                f"{control_reason} | "
                                f"holding steering={control_steering:+.3f}",
                                end="\r",
                                flush=True,
                            )

                    else:
                        control_steering = None
                        control_drive = False
                        control_reason = f"stop: {lane_result.reason}"

                        if frame_count % args.print_every == 0:
                            print(
                                f"lane invalid: {lane_result.reason}",
                                end="\r",
                                flush=True,
                            )

            lane_compute_end = time.perf_counter()
            serial_start = time.perf_counter()
            arduino.update(
                steering=control_steering,
                drive_allowed=bool(args.lane_follow and control_drive),
                urgent=urgent_control,
            )
            serial_end = time.perf_counter()

            post_start = time.perf_counter()

            if display:
                overlay = make_class_overlay(frame, class_map)

                if args.lane_follow and lane_result is not None:
                    if args.lane_follow_right:
                        overlay = draw_lane_following_right_debug(overlay, lane_result)
                    else:
                        overlay = draw_lane_following_debug(overlay, lane_result)

                    # Show control state separately from lane fitting state.
                    cv2.putText(
                        overlay,
                        f"control: {control_reason}",
                        (12, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    overlay,
                    "mode: INSTANT" if args.instant_control else "mode: SMOOTHED",
                    (12, 124),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0) if args.instant_control else (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

                yellow_pixels = int(
                    np.count_nonzero(class_map == CLASS_TO_ID["lane_center"])
                )
                green_pixels = int(
                    np.count_nonzero(class_map == CLASS_TO_ID["lane_right"])
                )
                cv2.putText(
                    overlay,
                    f"raw mask pixels: Y={yellow_pixels} G={green_pixels}",
                    (12, 148),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"backend: {args.effective_backend.upper()} | imgsz={args.imgsz}",
                    (12, 172),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                dc_command = steering_to_dc_command(
                    control_steering,
                    args.steering_output_gain,
                    args.steering_output_exponent,
                    args.steering_command_quantum,
                    args.invert_steering,
                )
                cv2.putText(
                    overlay,
                    f"DC command: steer={dc_command:+d}/1000 drive={args.drive_pwm}",
                    (12, 196),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                arduino_state, arduino_hint = arduino.status()
                state_color = (
                    (0, 255, 0)
                    if arduino_state == ArduinoRuntime.DRIVING
                    else (0, 220, 255)
                    if arduino_state in (ArduinoRuntime.CONNECTING, ArduinoRuntime.STANDBY)
                    else (0, 0, 255)
                    if arduino_state == ArduinoRuntime.ERROR
                    else (220, 220, 220)
                )
                cv2.putText(
                    overlay,
                    f"Arduino: {arduino_state} | {arduino_hint}",
                    (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    state_color,
                    2,
                    cv2.LINE_AA,
                )
                if urgent_control:
                    cv2.putText(
                        overlay,
                        f"urgent steering: {urgent_reason}",
                        (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now

                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps

                if args.show_fps:
                    draw_fps(overlay, fps)

                post_end = time.perf_counter()

                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF

                display_end = time.perf_counter()

                if key == ord(" "):
                    new_state = arduino.toggle()
                    # Do not carry stale steering through a manual stop/start cycle.
                    last_control_steering = None
                    last_valid_steering = None
                    lost_lane_frames = 0
                    print(f"[Control] Arduino state: {new_state}", flush=True)
                elif key in (ord("q"), 27):
                    break

            else:
                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now

                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps

                post_end = time.perf_counter()
                display_end = post_end

            if args.profile and frame_count % 30 == 0:
                total_ms = (display_end - loop_start) * 1000
                infer_ms = (inference_end - inference_start) * 1000
                lane_ms = (lane_compute_end - lane_compute_start) * 1000
                serial_ms = (serial_end - serial_start) * 1000
                post_ms = (post_end - post_start) * 1000
                display_ms = (display_end - post_end) * 1000

                print(
                    f"\n[profile] fps={fps:5.1f} | "
                    f"total={total_ms:6.1f} ms | "
                    f"infer={infer_ms:6.1f} ms | "
                    f"lane={lane_ms:5.1f} ms | "
                    f"serial={serial_ms:5.2f} ms | "
                    f"post={post_ms:5.1f} ms | "
                    f"display={display_ms:5.1f} ms | "
                    f"infer={'yes' if should_infer else 'reuse'}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)

    finally:
        arduino.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()