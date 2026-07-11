import os
from dotenv import load_dotenv

import time
import serial
import keyboard

load_dotenv()

PORT = os.getenv("ARDUINO_PORT")
BAUD = 9600

SEND_INTERVAL = 0.05

ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(2)

print("Keyboard control start")
print("w:     forward")
print("a:     steering left")
print("s:     backward")
print("d:     steering right")
print("space: stop")
print("esc:   quit")

try:
    while True:
        if keyboard.is_pressed("w"):
            ser.write(b"w\n")

        if keyboard.is_pressed("a"):
            ser.write(b"a\n")

        if keyboard.is_pressed("s"):
            ser.write(b"s\n")

        if keyboard.is_pressed("d"):
            ser.write(b"d\n")

        if keyboard.is_pressed("space"):
            ser.write(b"stop\n")

        if keyboard.is_pressed("esc"):
            ser.write(b"stop\n")
            break

        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    ser.write(b"stop\n")

finally:
    ser.close()
