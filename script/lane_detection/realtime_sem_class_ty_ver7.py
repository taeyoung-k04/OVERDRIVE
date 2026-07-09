#!/usr/bin/env python3
"""Low-latency yellow+green right-lane following with keyboard arming.

Runtime behavior:
- Program start: camera, segmentation, and lane calculation only.
- First SPACE: connect to Arduino asynchronously, then start driving.
- Next SPACE: send a full stop and remain in standby while vision continues.
- Further SPACE presses: toggle between driving and standby.
- Q or ESC: stop, close Arduino, and exit.

Arduino protocol:
- w: drive forward
- x: drive backward
- a: steer left
- d: steer right
- s: full stop
- c: steering-only stop (recommended; can be disabled from CLI)

Latency improvements:
- lane_left/blue is excluded from control; only lane_center/yellow and lane_right/green are fitted.
- The target path is biased toward green/right and gains extra look-ahead authority in corners.
- Instant-control mode is enabled by default: current raw steering bypasses both target EMA and steering EMA.
- Steering direction changes bypass the serial refresh interval automatically.
- Fresh inference results are sent to Arduino before overlay/display work.
- Steering-only stop on neutral/reversal when the Arduino supports command "c".
- Arduino connection happens in a background thread so camera inference does not freeze.
"""

from __future__ import annotations

# 가상환경 활성화:
# .\.venv\Scripts\Activate.ps1
#
# 실행 명령:
# python .\script\lane_detection\realtime_sem_class_ty_ver7.py `
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

from infer_sem_class import CLASS_TO_ID, make_class_overlay, semantic_to_class_map
from lane_following import (
    compute_lane_following,
    draw_lane_following_debug,
)
from lane_following_right_ver7 import (
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


class ArduinoCarController:
    """Low-latency serial controller for the Arduino vehicle.

    ``steering_stop_command`` should normally be ``"c"``. The Arduino sketch
    must implement it as a steering-motor-only stop. Set it to ``None`` when
    using the old sketch; direction reversals will still be sent immediately,
    but neutral steering must then rely on the Arduino steering timeout.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        steering_deadzone: float = 0.02,
        steering_release_deadzone: float = 0.010,
        steering_reversal_deadzone: float = 0.060,
        command_interval: float = 0.02,
        invert_steering: bool = False,
        steering_stop_command: str | None = "c",
        reset_delay: float = 2.0,
    ):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Install it with: pip install pyserial"
            ) from exc

        self.serial = serial.Serial(port, baudrate, timeout=0.01, write_timeout=0.05)
        if reset_delay > 0:
            time.sleep(reset_delay)

        self.steering_deadzone = max(0.0, float(steering_deadzone))
        self.steering_release_deadzone = min(
            self.steering_deadzone,
            max(0.0, float(steering_release_deadzone)),
        )
        self.steering_reversal_deadzone = max(
            self.steering_deadzone,
            float(steering_reversal_deadzone),
        )
        self.command_interval = max(0.0, float(command_interval))
        self.invert_steering = bool(invert_steering)
        self.steering_stop_command = steering_stop_command or None

        self.last_send_time = 0.0
        self.active_steering_command: str | None = None
        self.full_stop_active = True
        self.lock = threading.Lock()

    def send_char(self, cmd: str, flush: bool = False) -> None:
        if len(cmd) != 1:
            raise ValueError(f"Arduino command must be one character: {cmd!r}")
        self.serial.write(cmd.encode("ascii"))
        # For a changed steering command, wait until the byte has left the host
        # serial buffer. This prevents old repeated commands from sitting ahead
        # of the newest direction change.
        if flush:
            self.serial.flush()

    def _desired_steering_command(self, steering: float) -> str | None:
        if self.invert_steering:
            steering = -steering

        # Use a wider threshold only for an a<->d reversal. This removes rapid
        # left/right chatter around zero without delaying the first correction.
        if self.active_steering_command == "d":
            if steering < -self.steering_reversal_deadzone:
                return "a"
            if steering > self.steering_release_deadzone:
                return "d"
            return None

        if self.active_steering_command == "a":
            if steering > self.steering_reversal_deadzone:
                return "d"
            if steering < -self.steering_release_deadzone:
                return "a"
            return None

        if steering > self.steering_deadzone:
            return "d"
        if steering < -self.steering_deadzone:
            return "a"
        return None

    def _stop_steering_locked(self) -> None:
        if self.active_steering_command is None:
            return
        if self.steering_stop_command is not None:
            self.send_char(self.steering_stop_command, flush=True)
        self.active_steering_command = None

    def start_drive(self) -> None:
        """Leave full-stop state and send an immediate forward command."""
        with self.lock:
            self.full_stop_active = False
            self.last_send_time = time.perf_counter()
            self.send_char("w")

    def update(
        self,
        steering: float | None,
        drive: bool = True,
        urgent: bool = False,
    ) -> None:
        """Apply changed steering immediately and refresh unchanged commands periodically.

        A key latency rule is that the command interval is *not* checked before
        deciding whether steering direction changed. A new a/d/c decision is
        therefore transmitted immediately, even when the previous refresh was
        only a few milliseconds ago.
        """
        now = time.perf_counter()

        with self.lock:
            if not drive:
                self._full_stop_locked(force=False)
                return

            self.full_stop_active = False

            desired_command = None
            if steering is not None:
                desired_command = self._desired_steering_command(float(steering))

            steering_changed = desired_command != self.active_steering_command
            refresh_due = (now - self.last_send_time) >= self.command_interval

            # New steering decisions always win over the heartbeat interval.
            if steering_changed:
                self._stop_steering_locked()
                if desired_command is not None:
                    self.send_char(desired_command, flush=True)
                    self.active_steering_command = desired_command
                self.last_send_time = now
                # Keep forward drive alive as part of the same immediate update.
                self.send_char("w")
                return

            # When direction is unchanged, only refresh for Arduino timeouts.
            if not urgent and not refresh_due:
                return

            self.last_send_time = now
            self.send_char("w")
            if desired_command is not None:
                self.send_char(desired_command)

    def _full_stop_locked(self, force: bool) -> None:
        if self.full_stop_active and not force:
            return
        self._stop_steering_locked()
        self.send_char("s", flush=True)
        self.full_stop_active = True
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
        stop_command = None
        if not self.args.disable_steering_stop_command:
            stop_command = self.args.steering_stop_command.strip() or None

        return ArduinoCarController(
            port=self.args.arduino_port,
            baudrate=self.args.arduino_baudrate,
            steering_deadzone=self.args.steering_deadzone,
            steering_release_deadzone=self.args.steering_release_deadzone,
            steering_reversal_deadzone=self.args.steering_reversal_deadzone,
            command_interval=self.args.command_interval,
            invert_steering=self.args.invert_steering,
            steering_stop_command=stop_command,
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
        default=Path("runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
        help="Path to trained YOLO semantic weights.",
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
    parser.add_argument("--arduino-baudrate", type=int, default=9600)
    parser.add_argument(
        "--steering-deadzone",
        type=float,
        default=0.030,
        help="Start steering outside this command deadzone.",
    )
    parser.add_argument(
        "--steering-release-deadzone",
        type=float,
        default=0.010,
        help="Smaller release threshold used for steering hysteresis.",
    )
    parser.add_argument(
        "--steering-reversal-deadzone",
        type=float,
        default=0.060,
        help="Require this magnitude before changing directly from a to d or d to a.",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=0.01,
        help="Heartbeat interval for unchanged Arduino commands; steering changes bypass it.",
    )
    parser.add_argument(
        "--arduino-reset-delay",
        type=float,
        default=2.0,
        help="Seconds to wait for Arduino reset after opening the serial port.",
    )
    parser.add_argument(
        "--steering-stop-command",
        default="c",
        help="One-character steering-only stop command implemented by Arduino.",
    )
    parser.add_argument(
        "--disable-steering-stop-command",
        action="store_true",
        help="Do not send steering-only stop; rely on Arduino steering timeout.",
    )
    parser.add_argument(
        "--invert-steering",
        action="store_true",
        help="Swap left/right steering direction.",
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


def predict_class_map(
    model,
    frame: np.ndarray,
    args: argparse.Namespace,
    use_half: bool,
) -> np.ndarray:
    """Run YOLO on ROI and return full-frame class map."""

    h, w = frame.shape[:2]

    roi_top_ratio = max(0.0, min(float(args.roi_top_ratio), 0.95))
    roi_y = int(h * roi_top_ratio)

    if roi_y <= 0:
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            device=args.device,
            task="semantic",
            half=use_half,
            verbose=False,
        )

        return semantic_to_class_map(results[0].semantic_mask, frame.shape[:2])

    roi = frame[roi_y:h, :]

    results = model.predict(
        source=roi,
        imgsz=args.imgsz,
        device=args.device,
        task="semantic",
        half=use_half,
        verbose=False,
    )

    roi_class_map = semantic_to_class_map(results[0].semantic_mask, roi.shape[:2])

    class_map = np.zeros((h, w), dtype=np.uint8)
    class_map[roi_y:h, :] = roi_class_map

    return class_map


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

    if args.command_interval < 0.0:
        raise SystemExit("--command-interval must be >= 0")

    if args.arduino_reset_delay < 0.0:
        raise SystemExit("--arduino-reset-delay must be >= 0")

    if (
        not args.disable_steering_stop_command
        and len(args.steering_stop_command.strip()) != 1
    ):
        raise SystemExit("--steering-stop-command must be exactly one character")

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
        "steering_reversal_deadzone",
    ):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")

    if args.steering_release_deadzone > args.steering_deadzone:
        raise SystemExit("--steering-release-deadzone must be <= --steering-deadzone")
    if args.steering_reversal_deadzone < args.steering_deadzone:
        raise SystemExit("--steering-reversal-deadzone must be >= --steering-deadzone")

    if args.print_every < 1:
        raise SystemExit("--print-every must be >= 1")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed.") from exc

    if not args.weights.exists():
        raise SystemExit(f"Weights file does not exist: {args.weights}")

    use_half = args.half and is_cuda_device(args.device)
    if args.balanced_realtime:
        # Keep inference fresh first; reduce input size before skipping many frames.
        args.infer_every = min(max(1, args.infer_every), 2)
        args.imgsz = min(args.imgsz, 416)
        args.roi_top_ratio = float(np.clip(args.roi_top_ratio, 0.28, 0.38))

    if args.instant_control and args.infer_every != 1:
        print(
            f"[WARN] instant control requires fresh inference; "
            f"forcing --infer-every 1 (was {args.infer_every}).",
            flush=True,
        )
        args.infer_every = 1

    if args.half and not use_half:
        print("[WARN] --half is ignored because device is not CUDA.", flush=True)

    print(f"Loading model: {args.weights}", flush=True)
    model = YOLO(str(args.weights), task="semantic")

    capture = open_camera(args.camera, args.width, args.height)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Camera {args.camera}: requested {args.width}x{args.height}, "
        f"opened {actual_width}x{actual_height}",
        flush=True,
    )

    print(
        f"Inference: device={args.device}, imgsz={args.imgsz}, "
        f"half={use_half}, infer_every={args.infer_every}, "
        f"roi_top_ratio={args.roi_top_ratio}",
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
                            f"lat={getattr(lane_result, 'lateral_term', 0.0):+.3f} | "
                            f"curv={getattr(lane_result, 'curvature_term', 0.0):+.3f} | "
                            f"bound={getattr(lane_result, 'boundary_term', 0.0):+.3f} | "
                            f"ratio={getattr(lane_result, 'active_target_lane_ratio', 0.5):.2f} | "
                            f"curve={getattr(lane_result, 'curve_strength', 0.0):.2f} | "
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