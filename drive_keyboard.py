from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import keyboard
import serial
from serial import SerialException


# ---------------------------------------------------------------------
# 연결 설정
# ---------------------------------------------------------------------

PORT = "COM6"
BAUD = 115200

# Arduino watchdog 400ms보다 빠르게 명령 전송
SEND_INTERVAL = 0.05

# ---------------------------------------------------------------------
# 주행 및 조향 설정
# ---------------------------------------------------------------------

DRIVE_PWM = 140

STEERING_MIN = -1000
STEERING_MAX = 1000

# 0.05초마다 조향 목표값이 변경되는 양
# 25면 중앙에서 끝까지 약 2초
STEERING_STEP = 75


@dataclass
class ControllerState:
    # 현재 유지할 조향 목표값
    steering_command: int = 0

    fault_latched: bool = False
    fault_message: str = ""


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def send_line(ser: serial.Serial, command: str) -> None:
    """Arduino에 줄바꿈을 포함한 명령 전송."""
    ser.write((command + "\n").encode("ascii"))
    ser.flush()


def stop_arduino(ser: serial.Serial) -> None:
    """주행과 조향 모터를 즉시 정지."""
    send_line(ser, "X")


def read_arduino_messages(
    ser: serial.Serial,
    state: ControllerState,
) -> None:
    """Arduino 상태 메시지 확인."""

    while ser.in_waiting > 0:
        message = ser.readline().decode(
            "utf-8",
            errors="replace",
        ).strip()

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

            print()
            print("========================================")
            print("조향 오류가 발생해 모든 모터를 정지했습니다.")
            print(f"오류: {message}")
            print("문제를 확인한 뒤 R을 눌러 다시 시작하세요.")
            print("========================================")
            print()

        elif message == "WATCHDOG,STOP":
            print("Arduino watchdog으로 모터가 정지했습니다.")


def key_pressed_once(
    key: str,
    previous_states: Dict[str, bool],
) -> bool:
    """키가 처음 눌린 순간에만 True 반환."""

    current = keyboard.is_pressed(key)
    previous = previous_states.get(key, False)

    previous_states[key] = current

    return current and not previous


def print_controls() -> None:
    print()
    print("Keyboard control start")
    print("----------------------------------------")
    print("w         : 전진")
    print("a         : 누르는 동안 계속 왼쪽 조향")
    print("d         : 누르는 동안 계속 오른쪽 조향")
    print("w + a     : 전진하면서 왼쪽 조향")
    print("w + d     : 전진하면서 오른쪽 조향")
    print("a/d 해제  : 현재 조향각 유지")
    print("w 해제    : 전진 정지, 조향각 유지")
    print("c         : 조향 중앙 복귀")
    print("space / x : 즉시 정지")
    print("r         : fault 해제")
    print("q         : Arduino 상태 출력")
    print("esc       : 정지 후 종료")
    print("----------------------------------------")
    print()


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
        print(f"{PORT}를 열 수 없습니다: {error}")
        print("Arduino IDE 시리얼 모니터를 닫고 COM 포트를 확인하세요.")
        return

    try:
        # Uno/Nano는 시리얼 연결 시 재부팅될 수 있음
        time.sleep(2.0)

        stop_arduino(ser)
        time.sleep(0.1)

        print_controls()

        print(
            f"PORT={PORT}, "
            f"DRIVE_PWM={DRIVE_PWM}, "
            f"STEERING_STEP={STEERING_STEP}"
        )

        send_line(ser, "Q")

        last_send_time = time.monotonic()

        while True:
            read_arduino_messages(ser, state)

            # ---------------------------------------------------------
            # 종료
            # ---------------------------------------------------------

            if keyboard.is_pressed("esc"):
                stop_arduino(ser)
                print("정지 후 종료합니다.")
                break

            # ---------------------------------------------------------
            # 즉시 정지
            # ---------------------------------------------------------

            if (
                keyboard.is_pressed("space")
                or keyboard.is_pressed("x")
            ):
                stop_arduino(ser)

                # Python의 steering_command는 변경하지 않음
                # 키를 놓으면 기존 조향 목표값을 다시 유지
                time.sleep(SEND_INTERVAL)
                continue

            # ---------------------------------------------------------
            # 상태 출력
            # ---------------------------------------------------------

            if key_pressed_once("q", previous_key_states):
                send_line(ser, "Q")

            # ---------------------------------------------------------
            # Fault 해제
            # ---------------------------------------------------------

            if key_pressed_once("r", previous_key_states):
                stop_arduino(ser)
                time.sleep(0.1)

                state.fault_latched = False
                state.fault_message = ""

                send_line(ser, "Q")
                print("Fault를 해제했습니다.")

            if state.fault_latched:
                time.sleep(0.02)
                continue

            # ---------------------------------------------------------
            # 중앙 복귀
            # ---------------------------------------------------------

            if key_pressed_once("c", previous_key_states):
                state.steering_command = 0
                print("조향 중앙 복귀")

            # ---------------------------------------------------------
            # 20Hz 주기로 조향과 주행을 동시에 처리
            # ---------------------------------------------------------

            now = time.monotonic()

            if now - last_send_time >= SEND_INTERVAL:
                left_pressed = keyboard.is_pressed("a")
                right_pressed = keyboard.is_pressed("d")
                forward_pressed = keyboard.is_pressed("w")

                # A를 누르는 동안 목표값을 계속 왼쪽으로 이동
                if left_pressed and not right_pressed:
                    state.steering_command -= STEERING_STEP

                # D를 누르는 동안 목표값을 계속 오른쪽으로 이동
                elif right_pressed and not left_pressed:
                    state.steering_command += STEERING_STEP

                # 아무 키도 누르지 않거나 A/D를 동시에 누르면
                # steering_command를 변경하지 않아 현재 각도를 유지
                state.steering_command = clamp(
                    state.steering_command,
                    STEERING_MIN,
                    STEERING_MAX,
                )

                # W가 눌려 있으면 전진
                if forward_pressed:
                    drive_pwm = DRIVE_PWM
                else:
                    drive_pwm = 0

                # 하나의 명령에 조향값과 전진 속도를 함께 전송
                command = (
                    f"C,{state.steering_command},{drive_pwm}"
                )

                send_line(ser, command)
                last_send_time = now

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("Ctrl+C로 정지했습니다.")

    except SerialException as error:
        print(f"시리얼 통신 오류: {error}")

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