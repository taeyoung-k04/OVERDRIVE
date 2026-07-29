#!/usr/bin/env python3
"""Arduino serial communication for OVERDRIVE.

This module keeps drive output disabled whenever the serial link is unhealthy.
A write/read failure latches a serial fault. The fault can only be cleared by a
successful reset command or by reopening the serial port.
"""

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


# -----------------------------------------------------------------------------
# Ultrasonic distance telemetry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ArduinoDistances:
    """Latest center ultrasonic distance reported by the Arduino."""

    received_time: float
    center_cm: Optional[int]

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


def parse_arduino_distances(line: str) -> Optional[ArduinoDistances]:
    """Parse DIST,<center_cm>.

    The matching Arduino sketch reports -1 when no valid echo was received.
    Invalid or non-positive distance values are exposed as ``None``.
    """
    fields = line.strip().split(",")

    if len(fields) != 2 or fields[0] != "DIST":
        return None

    try:
        center_cm = int(fields[1])
    except ValueError:
        return None

    normalized_center = (
        center_cm
        if center_cm > 0
        else None
    )

    return ArduinoDistances(
        received_time=time.perf_counter(),
        center_cm=normalized_center,
    )


class ArduinoSender:
    """Repeat the latest command in a background thread.

    ONNX inference can occasionally take longer than the Arduino watchdog. The
    writer thread therefore transmits at a fixed rate rather than only once per
    camera frame.

    Safety behavior:
    - Any serial read/write error latches a serial fault.
    - A latched fault immediately disables drive output.
    - Commands are not sent again until reset_fault() successfully verifies the
      link or reconnects the port.
    - Failed emergency-stop writes are not repeated on the already-broken port.
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

        # Keep reads responsive while giving Windows enough time to enqueue a
        # short serial write. Reusing a very small read timeout as write_timeout
        # can create false write-timeout faults.
        self.timeout = max(0.01, float(timeout))
        self.write_timeout = max(1.0, float(timeout))

        self.reset_wait = max(0.0, float(reset_wait))
        self.command_rate = max(1.0, float(command_rate))
        self.steering_scale = max(1, int(steering_scale))

        self.serial = None

        self.connection_lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.telemetry_lock = threading.Lock()
        self.distance_lock = threading.Lock()
        self.message_lock = threading.Lock()

        self.latest_command = (0, 0)
        self.driving_active = False

        self.thread_stop = threading.Event()
        self.writer_thread: Optional[threading.Thread] = None
        self.reader_thread: Optional[threading.Thread] = None

        self.latest_telemetry: Optional[ArduinoTelemetry] = None
        self.latest_distances: Optional[ArduinoDistances] = None

        self.serial_fault = threading.Event()
        self.serial_fault_reason: Optional[str] = None
        self.stop_warning_emitted = False
        self.drive_block_warning_emitted = False

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
    def faulted(self) -> bool:
        return self.serial_fault.is_set()

    def _serial_is_open(self) -> bool:
        serial_port = self.serial
        if serial_port is None:
            return False

        try:
            return bool(serial_port.is_open)
        except Exception:
            return False

    @property
    def enabled(self) -> bool:
        """Return True only when the serial port is open and healthy."""
        return self._serial_is_open() and not self.faulted

    def _disable_drive_output(self) -> None:
        with self.state_lock:
            self.driving_active = False
            self.latest_command = (0, 0)

    def _mark_serial_fault(self, source: str, exc: BaseException) -> None:
        """Latch a serial fault and disable drive output.

        Only the first failure is printed. This prevents every writer iteration
        and every emergency-stop call from producing the same warning.
        """
        self._disable_drive_output()

        reason = f"{source}: {exc}"
        first_fault = not self.serial_fault.is_set()
        self.serial_fault_reason = reason
        self.serial_fault.set()

        if first_fault:
            print(
                f"SERIAL ERROR: {reason}",
                file=sys.stderr,
                flush=True,
            )
            print(
                "STOPPED: drive and steering commands are locked until "
                "the Arduino link is reset or reconnected.",
                file=sys.stderr,
                flush=True,
            )

    def _clear_serial_fault(self) -> None:
        self.serial_fault_reason = None
        self.serial_fault.clear()
        with self.message_lock:
            self.stop_warning_emitted = False
            self.drive_block_warning_emitted = False

    def _start_io_threads(self) -> None:
        self.thread_stop.clear()

        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name="arduino-reader",
            daemon=True,
        )
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="arduino-writer",
            daemon=True,
        )

        self.reader_thread.start()
        self.writer_thread.start()

    def _stop_io_threads(self) -> None:
        self.thread_stop.set()
        current_thread = threading.current_thread()

        writer = self.writer_thread
        if writer is not None and writer is not current_thread:
            writer.join(timeout=max(1.0, self.timeout * 4.0))
        self.writer_thread = None

        reader = self.reader_thread
        if reader is not None and reader is not current_thread:
            reader.join(timeout=max(1.0, self.timeout * 4.0))
        self.reader_thread = None

    def _close_serial_port(self) -> None:
        """Close the current port without attempting any additional writes."""
        with self.connection_lock:
            serial_port = self.serial
            self.serial = None

        if serial_port is None:
            return

        # Wait for an in-progress write before closing the handle.
        with self.write_lock:
            try:
                serial_port.close()
            except Exception:
                pass

    def connect(self) -> bool:
        """Open and verify the Arduino serial connection.

        A connection is reported as successful only after the initial stop and
        telemetry commands have been written successfully.
        """
        if self.enabled:
            return True
        if not self.port:
            return False

        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Run: python -m pip install pyserial"
            ) from exc

        # Remove a stale or faulted handle before opening a new one.
        if self.serial is not None or self.reader_thread is not None or self.writer_thread is not None:
            self._disable_drive_output()
            self._stop_io_threads()
            self._close_serial_port()

        try:
            serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
        except Exception as exc:
            self.serial_fault_reason = f"open failed: {exc}"
            self.serial_fault.set()
            raise RuntimeError(
                f"Could not open Arduino port {self.port}: {exc}"
            ) from exc

        with self.connection_lock:
            self.serial = serial_port

        try:
            if self.reset_wait > 0:
                time.sleep(self.reset_wait)

            serial_port.reset_input_buffer()
            serial_port.reset_output_buffer()

            # Clear a previous software fault so the verification writes are
            # permitted. A failed verification write immediately latches it again.
            self._clear_serial_fault()

            # Keep output disabled while verifying the new connection.
            self._disable_drive_output()
            self._write(b"X\n")
            self._write(b"DBG,1\n")
            self._write(b"Q\n")

        except Exception as exc:
            self._mark_serial_fault("connection verification failed", exc)
            self._close_serial_port()
            raise RuntimeError(
                f"Arduino opened on {self.port}, but communication verification failed: {exc}"
            ) from exc

        self._start_io_threads()

        print(
            f"Arduino connected: {self.port} @ {self.baud}, "
            f"protocol=C,target_steering,speed + potentiometer telemetry "
            f"@ {self.command_rate:.1f} Hz",
            flush=True,
        )
        return True

    def reconnect(self, attempts: int = 2) -> bool:
        """Close and reopen the serial link while keeping drive disabled."""
        if not self.port:
            return False

        self._disable_drive_output()
        self._stop_io_threads()
        self._close_serial_port()

        attempt_count = max(1, int(attempts))
        last_error: Optional[BaseException] = None

        for attempt in range(1, attempt_count + 1):
            try:
                print(
                    f"Arduino reconnect attempt {attempt}/{attempt_count}: {self.port}",
                    flush=True,
                )
                if self.connect():
                    print(
                        "Arduino connection restored. Drive remains stopped; "
                        "start driving again explicitly.",
                        flush=True,
                    )
                    return True
            except Exception as exc:
                last_error = exc
                if attempt < attempt_count:
                    time.sleep(0.5)

        reason = last_error or RuntimeError("unknown reconnect failure")
        self._mark_serial_fault("reconnect failed", reason)
        return False

    def _write(self, payload: bytes) -> None:
        """Write one complete protocol message or latch a serial fault."""
        if not isinstance(payload, bytes):
            raise TypeError("Arduino payload must be bytes")
        if not payload:
            return
        if self.faulted:
            raise RuntimeError(
                self.serial_fault_reason or "Arduino serial connection is faulted"
            )

        with self.write_lock:
            serial_port = self.serial

            if serial_port is None:
                exc = RuntimeError("Arduino serial port is not open")
                self._mark_serial_fault("write failed", exc)
                raise exc

            try:
                if not serial_port.is_open:
                    raise RuntimeError("Arduino serial port is closed")

                written = serial_port.write(payload)
                if written != len(payload):
                    raise RuntimeError(
                        f"partial serial write: {written}/{len(payload)} bytes"
                    )

            except Exception as exc:
                self._mark_serial_fault("Arduino serial write failed", exc)
                raise RuntimeError(
                    f"Arduino serial write failed: {exc}"
                ) from exc

    def _reader_loop(self) -> None:
        while not self.thread_stop.is_set():
            serial_port = self.serial

            if serial_port is None or self.faulted:
                self.thread_stop.wait(0.05)
                continue

            try:
                raw = serial_port.readline()
            except Exception as exc:
                if not self.thread_stop.is_set():
                    self._mark_serial_fault("Arduino serial read failed", exc)
                self.thread_stop.wait(0.05)
                continue

            if not raw:
                continue

            line = raw.decode(
                "ascii",
                errors="replace",
            ).strip()

            distances = parse_arduino_distances(line)

            if distances is not None:
                with self.distance_lock:
                    self.latest_distances = distances
                continue

            telemetry = parse_arduino_telemetry(line)

            if telemetry is not None:
                with self.telemetry_lock:
                    self.latest_telemetry = telemetry
            elif line.startswith(("ERR", "FAULT", "WATCHDOG")):
                print(
                    f"ARDUINO: {line}",
                    file=sys.stderr,
                    flush=True,
                )
            elif line.startswith(("READY", "CALIBRATION", "ACK")):
                print(
                    f"ARDUINO: {line}",
                    flush=True,
                )

    def telemetry_snapshot(
        self,
        stale_seconds: float,
    ) -> Optional[ArduinoTelemetry]:
        with self.telemetry_lock:
            telemetry = self.latest_telemetry

        if telemetry is None:
            return None

        if telemetry.age > max(0.0, float(stale_seconds)):
            return None

        return telemetry

    def distance_snapshot(
        self,
        stale_seconds: float,
    ) -> Optional[ArduinoDistances]:
        """Return the latest non-stale ultrasonic distance report."""
        with self.distance_lock:
            distances = self.latest_distances

        if distances is None:
            return None

        if distances.age > max(0.0, float(stale_seconds)):
            return None

        return distances

    def _writer_loop(self) -> None:
        period = 1.0 / self.command_rate
        next_send = time.perf_counter()

        while not self.thread_stop.is_set():
            with self.state_lock:
                active = self.driving_active
                steering_command, speed = self.latest_command

            if active and self.enabled:
                try:
                    self._write(
                        format_arduino_command(
                            steering_command,
                            speed,
                        )
                    )
                except RuntimeError:
                    # _write() already latched and reported the fault. Do not
                    # send X again through the same failed serial path.
                    pass

            next_send += period
            delay = next_send - time.perf_counter()

            if delay <= 0:
                next_send = time.perf_counter()
                delay = 0.001

            self.thread_stop.wait(delay)

    def start_driving(self) -> bool:
        """Enable periodic drive commands only when the link is healthy.

        With no Arduino port configured, visual simulation remains available.
        """
        if self.configured and not self.enabled:
            with self.message_lock:
                should_print = not self.drive_block_warning_emitted
                self.drive_block_warning_emitted = True

            if should_print:
                reason = self.serial_fault_reason or "serial port is not connected"
                print(
                    f"DRIVING BLOCKED: {reason}. Run reset_fault() or reconnect first.",
                    file=sys.stderr,
                    flush=True,
                )
            self._disable_drive_output()
            return False

        with self.state_lock:
            self.latest_command = (0, 0)
            self.driving_active = True

        return True

    def update_command(
        self,
        steering: float,
        speed: int,
        *,
        immediate: bool = False,
    ) -> None:
        scale = self.steering_scale

        steering_command = int(
            np.clip(
                round(float(steering) * scale),
                -scale,
                scale,
            )
        )

        speed_command = int(
            np.clip(
                round(speed),
                0,
                255,
            )
        )

        with self.state_lock:
            self.latest_command = (
                steering_command,
                speed_command,
            )
            active = self.driving_active

        # Safety-critical transitions should not wait for the next periodic
        # writer tick. The regular writer continues repeating this command.
        if immediate and active and self.enabled:
            try:
                self._write(
                    format_arduino_command(
                        steering_command,
                        speed_command,
                    )
                )
            except RuntimeError:
                # _write() already latched and reported the serial fault.
                pass

    def emergency_stop(self) -> bool:
        """Disable local drive output and send one stop command if possible."""
        self._disable_drive_output()

        if not self.configured:
            return True

        if not self.enabled:
            with self.message_lock:
                should_print = not self.stop_warning_emitted
                self.stop_warning_emitted = True

            if should_print:
                reason = self.serial_fault_reason or "serial port is not connected"
                print(
                    "WARNING: emergency stop could not be transmitted because "
                    f"the Arduino link is unavailable ({reason}).",
                    file=sys.stderr,
                    flush=True,
                )
            return False

        try:
            # Do not call flush() here. On a broken Windows serial handle flush
            # can block while waiting for queued output. write() is bounded by
            # write_timeout and is sufficient for this short command.
            self._write(b"X\n")
            return True
        except RuntimeError:
            with self.message_lock:
                should_print = not self.stop_warning_emitted
                self.stop_warning_emitted = True

            if should_print:
                print(
                    "WARNING: emergency stop transmission failed. The software "
                    "output is disabled; the Arduino watchdog must stop hardware.",
                    file=sys.stderr,
                    flush=True,
                )
            return False

    def reset_fault(self) -> bool:
        """Clear a recoverable fault while keeping drive output disabled.

        A healthy link receives X and Q directly. If the port is faulted or the
        reset write fails, the method closes and reopens the port. It does not
        propagate a write-timeout exception to the main driving loop.
        """
        self._disable_drive_output()

        if not self.port:
            return False

        if not self.enabled:
            return self.reconnect()

        try:
            self._write(b"X\n")
            self._write(b"Q\n")
            print(
                "Arduino fault reset confirmed. Drive remains stopped.",
                flush=True,
            )
            return True
        except RuntimeError:
            return self.reconnect()

    def close(self) -> None:
        """Stop output, terminate worker threads, and close the port."""
        self._disable_drive_output()

        if self.enabled:
            try:
                self._write(b"X\n")
                self._write(b"DBG,0\n")
            except RuntimeError:
                # The fault has already been latched. Continue closing without
                # sending any more data through the failed handle.
                pass

        self._stop_io_threads()
        self._close_serial_port()

        with self.telemetry_lock:
            self.latest_telemetry = None

        with self.distance_lock:
            self.latest_distances = None

        self._disable_drive_output()
