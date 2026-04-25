# =============================================================================
# sensors.py — Arduino serial interface and dummy sensor fallback
# =============================================================================

import threading
import time
from typing import List

from config import ARDUINO_ENABLED
from structures import SensorReading


def _dummy_sensor() -> SensorReading:
    """
    Returns a safe-default SensorReading used when ARDUINO_ENABLED=0.
    """
    return SensorReading(
        tof_distances=[999.0] * 5,
        tof_ok=False,
        gyro_pitch=0.0,
        gyro_roll=0.0,
        gyro_yaw=0.0,
        ldr_value=1.0,
        ultra_front=999.0,
        ultra_rear=999.0,
        speed=0.0,
        timestamp=time.time(),
    )


class ArduinoInterface:
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self.serial_conn = None
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()
        self.latest = _dummy_sensor()

    def connect(self):
        if not ARDUINO_ENABLED:
            print("[Arduino] DISABLED — using dummy sensor readings")
            return

        import serial, os

        if not os.path.exists(self.port):
            print(f"[WARN] Arduino port {self.port} not found")
            print(f"Available ports: {[f for f in os.listdir('/dev') if 'tty' in f]}")
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2)  # Arduino reset delay
            print(f"[Arduino] Connected on {self.port}")
        except serial.SerialException as e:
            print(f"[WARN] Arduino connection failed ({e})")
            self.serial_conn = None

    def update(self):
        if not ARDUINO_ENABLED or self.serial_conn is None or not self.serial_conn.is_open:
            return

        try:
            line = self.serial_conn.readline().decode("utf-8").strip()
            if not line:
                return

            p = line.split(",")
            if len(p) != 13:
                return

            prev = self.latest

            # ── TOF ───────────────────────────────────────────
            tof_raw = [float(p[i]) for i in range(5)]
            tof = [
                v if 0.0 < v < 8.0 else prev.tof_distances[i]
                for i, v in enumerate(tof_raw)
            ]
            tof_valid = any(v < 8.0 for v in tof)

            # ── IMU ───────────────────────────────────────────
            raw_pitch = float(p[5])
            raw_roll  = float(p[6])
            raw_yaw   = float(p[7])

            gyro_pitch = raw_pitch if abs(raw_pitch) <= 180.0 else prev.gyro_pitch
            gyro_roll  = raw_roll  if abs(raw_roll)  <= 180.0 else prev.gyro_roll
            gyro_yaw   = raw_yaw   if abs(raw_yaw)   <= 500.0 else prev.gyro_yaw

            # ── LDR ───────────────────────────────────────────
            ldr = max(0.0, min(1.0, float(p[8])))

            # ── ULTRASONIC ────────────────────────────────────
            ultra_front_raw = float(p[9])
            ultra_rear_raw  = float(p[10])

            ultra_front = ultra_front_raw if 0.0 < ultra_front_raw < 8.0 else prev.ultra_front
            ultra_rear  = ultra_rear_raw  if 0.0 < ultra_rear_raw  < 8.0 else prev.ultra_rear

            # ── SPEED ─────────────────────────────────────────
            speed = max(0.0, float(p[11]))

            # ── BUILD READING ─────────────────────────────────
            reading = SensorReading(
                tof_distances=tof,
                tof_ok=int(p[12]) == 1 and tof_valid,
                gyro_pitch=gyro_pitch,
                gyro_roll=gyro_roll,
                gyro_yaw=gyro_yaw,
                ldr_value=ldr,
                ultra_front=ultra_front,
                ultra_rear=ultra_rear,
                speed=speed,
                timestamp=time.time(),
            )

            with self.lock:
                self.latest = reading

        except (ValueError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"[WARN] Arduino read error ({e})")
            self.serial_conn = None

    def send_command(self, action: str):
        if not ARDUINO_ENABLED or self.serial_conn is None or not self.serial_conn.is_open:
            return

        char_map = {
            "FORWARD":    b"F",
            "SLOW":       b"S",
            "TURN_LEFT":  b"L",
            "TURN_RIGHT": b"R",
            "STOP":       b"X",
        }

        char = char_map.get(action, b"X")

        try:
            self.serial_conn.write(char)
        except Exception as e:
            print(f"[WARN] Failed to send motor command ({e})")
            self.serial_conn = None

    def get_latest(self) -> SensorReading:
        with self.lock:
            return self.latest