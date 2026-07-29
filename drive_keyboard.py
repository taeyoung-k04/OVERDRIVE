from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import keyboard
import serial
from serial import SerialException


# Connection settings
PORT = "COM8"
BAUD = 115200

SEND_INTERVAL = 0.05  # 50ms


# Driving and steering settings
DRIVE_SPEED = 140

STEERING_MIN = -1000
STEERING_MAX = 1000
STEERING_STEP = 75  # Amount by which the steering target changes every 50 ms


@dataclass
class ControllerState:
    # steering_target
    steering_command: int = 0

    fault_latched: bool = False
    fault_message: str = ""


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def send_command(ser: serial.Serial, command: str) -> None:
    ser.write((command + "\n").encode("ascii"))
    ser.flush()


def stop_arduino(ser: serial.Serial) -> None:
    send_command(ser, "X")


def read_arduino_messages(
    ser: serial.Serial,
    state: ControllerState,
) -> None:
    while ser.in_waiting > 0:
        message = ser.readline().decode("utf-8", errors="replace").strip()

        if not message:
            continue

        print(f"[Arduino] {message}")

        if message.startswith("FAULT,"):
            state.fault_latched = True
            state.fault_message = message

            try:
                stop_arduino(ser)
            except SerialException:
                pass

            print(f"Steering fault—motors stopped: {message}.")
            print("Resolve the issue, then press R to clear the fault.")

        elif message == "WATCHDOG,STOP":
            print("The Arduino watchdog stopped the motors.")


def keyboard_pressed_once(
    key: str,
    previous_states: Dict[str, bool],
) -> bool:
    """Return True only when the key is initially pressed."""

    current = keyboard.is_pressed(key)
    previous = previous_states.get(key, False)

    previous_states[key] = current

    return current and not previous


def print_controls() -> None:
    print("----------------------------------------")
    print("w         : 전진")
    print("a         : 왼쪽 조향")
    print("d         : 오른쪽 조향")
    print("c         : 조향 초기화")
    print("space     : 정지")
    print("esc       : 정지 후 종료")
    print("----------------------------------------")
    print("q         : Arduino 상태 출력")
    print("----------------------------------------")


def main() -> None:
    state = ControllerState()
    previous_key_states: Dict[str, bool] = {}

    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            timeout=0.02,
            write_timeout=0.2,
        )

    except SerialException as error:
        print(f"Unable to open {PORT}: {error}")
        return

    try:
        # Arduino may reset when the serial connection is opened.
        time.sleep(2.0)

        stop_arduino(ser)
        time.sleep(0.1)

        print_controls()

        send_command(ser, "Q")

        last_send_time = time.monotonic()

        while True:
            read_arduino_messages(ser, state)

            if keyboard.is_pressed("esc"):
                stop_arduino(ser)
                break

            if keyboard.is_pressed("space"):
                stop_arduino(ser)

                time.sleep(SEND_INTERVAL)
                continue

            if keyboard_pressed_once("q", previous_key_states):
                send_command(ser, "Q")

            restart_pressed = keyboard_pressed_once("r", previous_key_states)
            if state.fault_latched and restart_pressed:
                stop_arduino(ser)
                time.sleep(0.1)

                state.fault_latched = False
                state.fault_message = ""

                send_command(ser, "Q")
                print("The fault has been cleared.")

            if state.fault_latched:
                time.sleep(0.02)  # Prevent excessive CPU usage.
                continue

            if keyboard_pressed_once("c", previous_key_states):
                state.steering_command = 0


            now = time.monotonic()
            if now - last_send_time >= SEND_INTERVAL:
                forward_pressed = keyboard.is_pressed("w")
                left_pressed = keyboard.is_pressed("a")
                right_pressed = keyboard.is_pressed("d")

                if forward_pressed:
                    drive_pwm = DRIVE_SPEED
                else:
                    drive_pwm = 0

                if left_pressed and not right_pressed:
                    state.steering_command -= STEERING_STEP

                elif right_pressed and not left_pressed:
                    state.steering_command += STEERING_STEP

                state.steering_command = clamp(
                    state.steering_command,
                    STEERING_MIN,
                    STEERING_MAX,
                )

                command = f"C,{state.steering_command},{drive_pwm}"
                send_command(ser, command)

                last_send_time = now

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass

    except SerialException as error:
        print(f"Serial communication error: {error}")

    finally:
        try:
            if ser.is_open:
                stop_arduino(ser)
                time.sleep(0.05)
                ser.close()

        except SerialException:
            pass


if __name__ == "__main__":
    main()
