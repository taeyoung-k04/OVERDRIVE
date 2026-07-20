#!/usr/bin/env python3
"""Arduino serial communication for OVERDRIVE."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# -----------------------------------------------------------------------------
# Arduino serial output
# -----------------------------------------------------------------------------


def format_arduino_command(steering_command: int, speed: int) -> bytes:
    """Encode the protocol shared with the matching Arduino sketch."""
    return f"C,{int(steering_command)},{int(speed)}\n".encode("ascii")


@dataclass(frozen=True)
class ArduinoTelemetry:
    received_time: float
    sensor: int
    target: int
    error: int
    steering_pwm: int
    drive_pwm: int
    actual_command: int
    fault: int

    @property
    def age(self) -> float:
        return max(0.0, time.perf_counter() - self.received_time)


def parse_arduino_telemetry(line: str) -> Optional[ArduinoTelemetry]:
    """Parse either supported Arduino telemetry format.

    Original paired sketch:
        T,sensor,target,error,steer_pwm,drive_pwm,actual_cmd,fault

    Closed-loop sketch:
        POS,raw=...,filtered=...,current=...,target=...,target_raw=...,
        error=...,pwm=...,fault=...
    """
    stripped = line.strip()

    fields = stripped.split(",")
    if len(fields) == 8 and fields[0] == "T":
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError:
            return None
        return ArduinoTelemetry(
            received_time=time.perf_counter(),
            sensor=values[0],
            target=values[1],
            error=values[2],
            steering_pwm=values[3],
            drive_pwm=values[4],
            actual_command=values[5],
            fault=values[6],
        )

    if not stripped.startswith("POS,"):
        return None

    values_by_name = {}
    for item in fields[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values_by_name[key.strip()] = value.strip()

    required = ("raw", "current", "target", "target_raw", "error", "pwm", "fault")
    if any(key not in values_by_name for key in required):
        return None

    try:
        sensor = int(round(float(values_by_name["raw"])))
        target_raw = int(round(float(values_by_name["target_raw"])))
        current_command = int(round(float(values_by_name["current"])))
        target_command = int(round(float(values_by_name["target"])))
        error = int(round(float(values_by_name["error"])))
        steering_pwm = int(round(float(values_by_name["pwm"])))
        fault = int(round(float(values_by_name["fault"])))
    except ValueError:
        return None

    # The POS format does not include drive PWM. The field is retained as zero
    # because the preview only uses the received steering feedback.
    return ArduinoTelemetry(
        received_time=time.perf_counter(),
        sensor=sensor,
        target=target_raw,
        error=error,
        steering_pwm=steering_pwm,
        drive_pwm=0,
        actual_command=current_command,
        fault=fault,
    )


class ArduinoSender:
    """Repeat the latest command in a background thread.

    ONNX inference can occasionally take longer than the Arduino watchdog. The
    writer thread therefore transmits at a fixed rate rather than only once per
    camera frame.
    """

    def __init__(
        self,
        port: Optional[str],
        baud: int,
        timeout: float,
        reset_wait: float,
        command_rate: float,
        steering_scale: int,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self.reset_wait = float(reset_wait)
        self.command_rate = max(1.0, float(command_rate))
        self.steering_scale = max(1, int(steering_scale))

        self.serial = None
        self.state_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.latest_command = (0, 0)
        self.driving_active = False
        self.thread_stop = threading.Event()
        self.writer_thread: Optional[threading.Thread] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.telemetry_lock = threading.Lock()
        self.latest_telemetry: Optional[ArduinoTelemetry] = None

        if not port:
            print(
                "No --arduino-port supplied; SPACE toggles visual simulation only.",
                flush=True,
            )
        else:
            print(
                f"Arduino ready on {port}. Press SPACE to connect and start.",
                flush=True,
            )

    @property
    def configured(self) -> bool:
        return bool(self.port)

    @property
    def enabled(self) -> bool:
        return self.serial is not None

    def connect(self) -> bool:
        if self.serial is not None:
            return True
        if not self.port:
            return False

        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Run: python -m pip install pyserial"
            ) from exc

        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not open Arduino port {self.port}: {exc}") from exc

        if self.reset_wait > 0:
            time.sleep(self.reset_wait)

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.thread_stop.clear()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.reader_thread.start()
        self.writer_thread.start()

        # The uploaded closed-loop sketch sends periodic POS telemetry only
        # after DBG,1. Older paired sketches simply ignore this command.
        try:
            self._write(b"DBG,1\n")
            self._write(b"Q\n")
        except Exception as exc:
            print(f"WARNING: could not enable Arduino telemetry: {exc}", file=sys.stderr)

        self.emergency_stop()
        print(
            f"Arduino connected: {self.port} @ {self.baud}, "
            f"protocol=C,target_steering,speed + potentiometer telemetry @ {self.command_rate:.1f} Hz",
            flush=True,
        )
        return True

    def _write(self, payload: bytes) -> None:
        if self.serial is None:
            return
        with self.write_lock:
            try:
                self.serial.write(payload)
            except Exception as exc:
                raise RuntimeError(f"Arduino serial write failed: {exc}") from exc

    def _reader_loop(self) -> None:
        while not self.thread_stop.is_set():
            serial_port = self.serial
            if serial_port is None:
                self.thread_stop.wait(0.05)
                continue
            try:
                raw = serial_port.readline()
            except Exception as exc:
                if not self.thread_stop.is_set():
                    print(f"SERIAL READ ERROR: {exc}", file=sys.stderr, flush=True)
                self.thread_stop.wait(0.05)
                continue

            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            telemetry = parse_arduino_telemetry(line)
            if telemetry is not None:
                with self.telemetry_lock:
                    self.latest_telemetry = telemetry
            elif line.startswith(("ERR", "FAULT", "WATCHDOG")):
                print(f"ARDUINO: {line}", file=sys.stderr, flush=True)
            elif line.startswith(("READY", "CALIBRATION", "ACK")):
                print(f"ARDUINO: {line}", flush=True)

    def telemetry_snapshot(self, stale_seconds: float) -> Optional[ArduinoTelemetry]:
        with self.telemetry_lock:
            telemetry = self.latest_telemetry
        if telemetry is None:
            return None
        if telemetry.age > max(0.0, float(stale_seconds)):
            return None
        return telemetry

    def _writer_loop(self) -> None:
        period = 1.0 / self.command_rate
        next_send = time.perf_counter()
        while not self.thread_stop.is_set():
            with self.state_lock:
                active = self.driving_active
                steering_command, speed = self.latest_command

            if active and self.serial is not None:
                try:
                    self._write(format_arduino_command(steering_command, speed))
                except RuntimeError as exc:
                    print(f"SERIAL ERROR: {exc}", file=sys.stderr, flush=True)
                    with self.state_lock:
                        self.driving_active = False
                    try:
                        self._write(b"X\n")
                    except Exception:
                        pass

            next_send += period
            delay = next_send - time.perf_counter()
            if delay <= 0:
                next_send = time.perf_counter()
                delay = 0.001
            self.thread_stop.wait(delay)

    def start_driving(self) -> None:
        with self.state_lock:
            self.latest_command = (0, 0)
            self.driving_active = True

    def update_command(self, steering: float, speed: int) -> None:
        scale = self.steering_scale
        steering_command = int(np.clip(round(float(steering) * scale), -scale, scale))
        speed_command = int(np.clip(round(speed), 0, 255))
        with self.state_lock:
            self.latest_command = (steering_command, speed_command)

    def emergency_stop(self) -> None:
        with self.state_lock:
            self.driving_active = False
            self.latest_command = (0, 0)
        if self.serial is not None:
            try:
                self._write(b"X\n")
                with self.write_lock:
                    self.serial.flush()
            except Exception as exc:
                print(f"WARNING: failed to send emergency stop: {exc}", file=sys.stderr)

    def reset_fault(self) -> None:
        """Clear a recoverable fault while keeping drive output disabled.

        The uploaded closed-loop sketch clears ``steeringFault`` in carStop(),
        so X is its reset command. Older paired sketches also safely accept X.
        """
        with self.state_lock:
            self.driving_active = False
            self.latest_command = (0, 0)
        if self.serial is not None:
            self._write(b"X\n")
            self._write(b"Q\n")

    def close(self) -> None:
        self.emergency_stop()
        self.thread_stop.set()
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=1.0)
            self.writer_thread = None
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None
        if self.serial is not None:
            try:
                self._write(b"DBG,0\n")
            except Exception:
                pass
            self.serial.close()
            self.serial = None
        with self.telemetry_lock:
            self.latest_telemetry = None
