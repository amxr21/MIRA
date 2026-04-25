"""
test_sensors.py
===============
Tests all sensors by reading from the Arduino Mega over USB serial.
All sensors (TOF, IMU, LDR, ultrasonic, encoder) are wired to the Arduino —
this script just reads and displays what the Arduino is already sending.

Same serial protocol as the main pipeline:
  TOF0,TOF1,TOF2,TOF3,TOF4,pitch,roll,yaw,ldr,ultra_front,speed,tof_ok

── WIRING REMINDER ─────────────────────────────────────────────────────────────
  All sensors → Arduino Mega
  Arduino Mega → Pi via USB cable (one cable, handles both power and serial)

── SENSOR INDEX MAP ─────────────────────────────────────────────────────────────
  TOF0 = Center-front
  TOF1 = Left-front  (~30°)
  TOF2 = Right-front (~30°)
  TOF3 = Left-side   (90°)
  TOF4 = Right-side  (90°)

── HOW TO RUN ───────────────────────────────────────────────────────────────────
  1. Flash arduino_mira.ino to the Arduino Mega
  2. Connect Arduino to Pi via USB
  3. Run: python3 tests/test_sensors.py
  4. Press Ctrl+C to stop

── INSTALL ──────────────────────────────────────────────────────────────────────
  pip install pyserial --break-system-packages
"""

import time
import os
import serial

# ── Config ───────────────────────────────────────────────────
PORT      = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if ttyUSB0 not found
BAUD      = 115200
TIMEOUT   = 2.0              # seconds to wait for a line before warning

TOF_NAMES = ["C-FRONT", "L-FRONT", "R-FRONT", "L-SIDE ", "R-SIDE "]


# ── Port detection ───────────────────────────────────────────
def find_arduino_port() -> str:
    """Try to auto-detect the Arduino serial port."""
    candidates = [f"/dev/{p}" for p in os.listdir("/dev")
                  if p.startswith("ttyUSB") or p.startswith("ttyACM")]
    candidates.sort()
    if not candidates:
        print("[ERROR] No USB serial ports found in /dev/")
        print("        Is the Arduino plugged in?")
        raise SystemExit(1)
    if PORT in candidates:
        return PORT
    print(f"[WARN] {PORT} not found. Using {candidates[0]} instead.")
    print(f"       All available: {candidates}")
    return candidates[0]


# ── Connect ──────────────────────────────────────────────────
port = find_arduino_port()
print(f"[Init] Connecting to Arduino on {port} @ {BAUD} baud...")
try:
    ser = serial.Serial(port, BAUD, timeout=TIMEOUT)
    time.sleep(2)   # wait for Arduino boot after serial open
    print(f"[Init] Connected — waiting for sensor data...\n")
except serial.SerialException as e:
    print(f"[ERROR] Could not open {port}: {e}")
    raise SystemExit(1)


# ── Parse one CSV line ───────────────────────────────────────
def parse_line(line: str):
    """
    Parse the 12-field CSV from Arduino.
    Returns None if the line is malformed.
    """
    p = line.strip().split(",")
    if len(p) != 12:
        return None
    try:
        tof      = [float(p[i]) for i in range(5)]
        pitch    = float(p[5])
        roll     = float(p[6])
        yaw      = float(p[7])
        ldr      = float(p[8])
        ultra    = float(p[9])
        speed    = float(p[10])
        tof_ok   = int(p[11]) == 1
        return tof, pitch, roll, yaw, ldr, ultra, speed, tof_ok
    except (ValueError, IndexError):
        return None


# ── Display helpers ──────────────────────────────────────────
def tof_bar(m: float) -> str:
    """Visual proximity bar for quick reading."""
    if m >= 8.0:
        return "  --  "
    filled = max(0, min(10, int((8.0 - m) / 8.0 * 10)))
    return "[" + "█" * filled + "░" * (10 - filled) + "]"

def tof_flag(m: float) -> str:
    if m < 0.5:
        return " ⚠ STOP"
    if m < 1.5:
        return " ! SLOW"
    if m >= 8.0:
        return " -- (timeout)"
    return ""


# ── Main loop ────────────────────────────────────────────────
print("=" * 60)
print("  MIRA Sensor Test — reading from Arduino serial")
print("  Press Ctrl+C to stop")
print("=" * 60)

line_errors  = 0
frames_read  = 0
t_start      = time.time()

try:
    while True:
        raw = ser.readline().decode("utf-8", errors="replace").strip()

        if not raw:
            print("[WARN] No data received — check Arduino is flashed and running")
            continue

        parsed = parse_line(raw)
        if parsed is None:
            line_errors += 1
            if line_errors <= 5:
                print(f"[WARN] Malformed line ({line_errors}): {raw!r}")
            continue

        tof, pitch, roll, yaw, ldr, ultra, speed, tof_ok = parsed
        frames_read += 1
        fps = frames_read / (time.time() - t_start)

        # Clear and redraw
        print("\033[2J\033[H", end="")   # clear terminal

        print(f"{'MIRA Sensor Test':^60}")
        print(f"{'fps=' + f'{fps:.1f}':>60}")
        print("=" * 60)

        # TOF sensors
        bumper = "TOF" if tof_ok else "ULTRA (TOF all failed)"
        print(f"\n  TOF Sensors  [active bumper: {bumper}]\n")
        for i, m in enumerate(tof):
            m_val = m if m < 8.0 else None
            dist_str = f"{m:.2f}m" if m_val is not None else "timeout"
            print(f"  {TOF_NAMES[i]}  {tof_bar(m)}  {dist_str:>8}{tof_flag(m)}")

        # IMU
        print(f"\n  IMU (MPU-6050)\n")
        print(f"  Pitch  : {pitch:+7.2f}°  {'⚠ STOP' if abs(pitch) > 20 else ('! SLOW' if abs(pitch) > 10 else 'ok')}")
        print(f"  Roll   : {roll:+7.2f}°  {'⚠ STOP' if abs(roll) > 25 else ('! SLOW' if abs(roll) > 15 else 'ok')}")
        print(f"  Yaw    : {yaw:+7.2f} °/s  (raw gyro Z)")

        # Environment
        print(f"\n  Environment\n")
        ldr_label = "bright" if ldr >= 0.6 else ("dim" if ldr >= 0.25 else "DARK — TOF only")
        print(f"  LDR    : {ldr:.2f}  ({ldr_label})")
        print(f"  Ultra  : {ultra:.2f}m  {'(bumper active — TOF failed)' if not tof_ok else '(standby)'}")
        print(f"  Speed  : {speed:.2f} m/s")

        print("\n" + "-" * 60)
        print(f"  Raw line: {raw}")

except KeyboardInterrupt:
    print("\n\n[Done] Stopped by user")
    ser.close()
except serial.SerialException as e:
    print(f"\n[ERROR] Serial connection lost: {e}")
    print("        Arduino may have been disconnected")