import time

import serial
from serial import SerialException, SerialTimeoutException


PORT = "COM6"       # 실제 포트로 변경
BAUDRATE = 115200   # Arduino의 Serial.begin 값과 동일하게 변경


def main() -> None:
    ser = None

    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            timeout=0.2,
            write_timeout=1.0,

            # 일반 USB Arduino에서는 모두 False 권장
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

        # 포트를 열 때 Arduino가 리셋될 수 있으므로 대기
        time.sleep(2.0)

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        commands = [
            b"X\n",
            b"S\n",
        ]

        for command in commands:
            print(f"Sending: {command!r}")

            written = ser.write(command)

            if written != len(command):
                raise RuntimeError(
                    f"Partial write: {written}/{len(command)} bytes"
                )

            ser.flush()
            print(f"Success: {written} bytes")

            time.sleep(0.5)

    except SerialTimeoutException as exc:
        print(f"WRITE TIMEOUT: {exc}")

    except SerialException as exc:
        print(f"SERIAL ERROR: {exc}")

    finally:
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()