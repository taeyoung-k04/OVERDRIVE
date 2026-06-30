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
print("w: forward")
print("x: backward")
print("a: steering A")
print("d: steering D")
print("space: stop")
print("esc: quit")

try:
    while True:
        if keyboard.is_pressed("esc"):
            ser.write(b"s")
            print("Quit")
            break

        if keyboard.is_pressed("w"):
            ser.write(b"w")

        if keyboard.is_pressed("x"):
            ser.write(b"x")

        if keyboard.is_pressed("a"):
            ser.write(b"a")

        if keyboard.is_pressed("d"):
            ser.write(b"d")

        if keyboard.is_pressed("space"):
            ser.write(b"s")

        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    ser.write(b"s")
    print("Stopped by Ctrl+C")

finally:
    ser.close()
