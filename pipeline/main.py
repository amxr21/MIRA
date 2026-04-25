"""
main.py — MIRA Pipeline Entry Point
====================================
RoverPipeline owns all shared state, creates threads, and manages
startup and shutdown. All logic lives in the other modules — this
file only orchestrates them.

Run:
  python3 main.py

Adjust these flags in config.py before running:
  ARDUINO_ENABLED = 0 or 1
  RECORD_ENABLED  = 0 or 1
  COCO_ENABLED    = 0 or 1
"""

import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import (
    ARDUINO_ENABLED, RECORD_ENABLED, COCO_ENABLED, MOTORS_ENABLED,
    HAILO_AVAILABLE, CONFIG
)
from structures import FusionResult, NavigationCommand
from sensors import ArduinoInterface
from models import (
    YOLODetectorHailo, YOLODetectorONNX,
    DepthEstimatorHailo, DepthEstimatorONNX,
    YOLODetectorCOCO, tag_unknown_objects
)
from fusion import fuse
from decision import estimate_motion, decide
from recorder import VideoRecorder


class RoverPipeline:
    """
    Top-level orchestrator. Owns:
      - Arduino interface
      - AI model instances (YOLO, Depth, COCO)
      - Shared state (latest_fusion, latest_command, latest_depth)
      - Thread locks for all shared state
      - History buffer for motion tracking
      - VideoRecorder
      - Camera capture

    Thread structure:
      Fast loop (~30Hz) — Arduino read + decide()       → daemon thread
      Slow loop (8–13 FPS Hailo / 1–2 FPS CPU)         → daemon thread
      Main loop (50ms tick) — NAV print + shutdown      → main thread
    """

    def __init__(self):
        self.arduino = ArduinoInterface()
        self.coco    = None   # set inside _load_models
        self.yolo, self.depth, self._yolo_mode, self._depth_mode = self._load_models()
        self.recorder: Optional[VideoRecorder] = None

        # Shared state — each variable has its own lock
        self.latest_fusion:  Optional[FusionResult]      = None
        self.latest_command: Optional[NavigationCommand]  = None
        self.latest_depth:   Optional[np.ndarray]         = None
        self.fusion_lock  = threading.Lock()
        self.command_lock = threading.Lock()
        self.depth_lock   = threading.Lock()

        # History buffer for motion tracking
        # RULE: always append BEFORE calling estimate_motion()
        # so history[-1] = current frame, history[-2] = previous frame
        self.history: deque = deque(maxlen=CONFIG["history_len"])

        self.cap     = None
        self.running = False

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_models(self):
        yolo_mode  = "ONNX"
        depth_mode = "ONNX"
        yolo  = None
        depth = None

        # YOLO — Hailo primary, ONNX CPU fallback
        if HAILO_AVAILABLE and os.path.exists(CONFIG["yolo_path_hef"]):
            try:
                yolo = YOLODetectorHailo(CONFIG["yolo_path_hef"])
                yolo_mode = "Hailo"
            except Exception as e:
                print(f"[WARN] YOLO Hailo failed ({e}) — falling back to ONNX CPU")
        if yolo is None:
            yolo = YOLODetectorONNX(CONFIG["yolo_path_onnx"])

        # Depth — Hailo primary, ONNX CPU fallback (independent of YOLO)
        # CPU depth = 200–350ms → worst case scenario — Hailo strongly preferred
        if HAILO_AVAILABLE and os.path.exists(CONFIG["depth_path_hef"]):
            try:
                depth = DepthEstimatorHailo(CONFIG["depth_path_hef"])
                depth_mode = "Hailo"
            except Exception as e:
                print(f"[WARN] Depth Hailo failed ({e}) — falling back to ONNX CPU "
                      f"(slow loop will degrade to ~2 FPS)")
        if depth is None:
            depth = DepthEstimatorONNX(CONFIG["depth_path_onnx"])

        # Scenario label for boot info
        if yolo_mode == "Hailo" and depth_mode == "Hailo":
            self._scenario = "1 — Both on Hailo NPU         (~13-20 FPS)"
        elif yolo_mode == "Hailo" or depth_mode == "Hailo":
            faster = "YOLO" if yolo_mode == "Hailo" else "Depth"
            self._scenario = f"2 — {faster} on Hailo, other on CPU  (~5-8 FPS)"
        else:
            self._scenario = "3 — Both on CPU ONNX (worst case) (~1-2 FPS)"

        # COCO — only if enabled AND YOLO is on Hailo (separate silicon)
        if COCO_ENABLED and yolo_mode == "Hailo":
            try:
                self.coco = YOLODetectorCOCO(CONFIG["coco_path_onnx"])
                print("[COCO] Loaded on CPU ONNX — unknown_object tagging active")
            except Exception as e:
                print(f"[WARN] COCO model failed ({e}) — unknown_object tagging disabled")
                self.coco = None
        elif COCO_ENABLED and yolo_mode != "Hailo":
            print("[COCO] Disabled — YOLO is on CPU, COCO would conflict. "
                  "Set YOLO to Hailo to enable.")
        else:
            print("[COCO] Disabled by COCO_ENABLED=0")

        return yolo, depth, yolo_mode, depth_mode

    # ─────────────────────────────────────────────────────────────────────────
    # Boot info
    # ─────────────────────────────────────────────────────────────────────────

    def _print_boot_info(self):
        try:
            import subprocess
            temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode().strip()
        except Exception:
            temp = "N/A"
        coco_status = (
            "ON" if self.coco is not None
            else ("OFF (YOLO on CPU)" if COCO_ENABLED and self._yolo_mode != "Hailo"
                  else "OFF")
        )
        print("=" * 56)
        print(f"  Scenario   : {self._scenario}")
        print(f"  YOLO       : {self._yolo_mode}")
        print(f"  Depth      : {self._depth_mode}")
        print(f"  COCO       : {coco_status}")
        print(f"  CPU Temp   : {temp}")
        print(f"  Arduino    : {'ON' if ARDUINO_ENABLED else 'OFF (dummy)'}")
        print(f"  Motors     : {'AUTO (pipeline control)' if MOTORS_ENABLED else 'MANUAL (keyboard W/A/S/D)'}")
        print(f"  Recording  : {'ON → ' + CONFIG['record_output'] if RECORD_ENABLED else 'OFF'}")
        print("=" * 56)

    # ─────────────────────────────────────────────────────────────────────────
    # Startup
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        self.arduino.connect()

        self.cap = cv2.VideoCapture(CONFIG["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["capture_w"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["capture_h"])
        self.cap.set(cv2.CAP_PROP_FPS,          CONFIG["capture_fps"])
        if not self.cap.isOpened():
            print("[ERROR] Camera failed to open — check camera_index in config.py and USB connection")
            print("[ERROR] Pipeline cannot run without camera. Exiting.")
            raise SystemExit(1)

        if RECORD_ENABLED:
            try:
                self.recorder = VideoRecorder(
                    CONFIG["record_output"],
                    CONFIG["record_fps"],
                    CONFIG["capture_w"],
                    CONFIG["capture_h"],
                )
            except Exception as e:
                print(f"[WARN] VideoRecorder failed to initialize ({e}) — recording disabled")
                self.recorder = None

        self._print_boot_info()
        self.running = True
        threading.Thread(target=self._fast_loop, daemon=True).start()
        threading.Thread(target=self._slow_loop, daemon=True).start()
        self._main_loop()

    # ─────────────────────────────────────────────────────────────────────────
    # Fast loop — ~30Hz
    # ─────────────────────────────────────────────────────────────────────────

    def _fast_loop(self):
        """Reads Arduino (or dummy) and runs decide() at ~30Hz."""
        while self.running:
            self.arduino.update()
            sensor = self.arduino.get_latest()

            with self.fusion_lock:
                fusion = self.latest_fusion

            cmd = decide(sensor, fusion)

            with self.command_lock:
                self.latest_command = cmd

            time.sleep(0.033)

    # ─────────────────────────────────────────────────────────────────────────
    # Slow loop — 8–13 FPS (Hailo) / 1–2 FPS (CPU)
    # ─────────────────────────────────────────────────────────────────────────

    def _slow_loop(self):
        """Vision loop: YOLO + Depth + COCO + fuse + motion + display/record."""
        frame_count = 0
        t_start     = time.time()

        while self.running:
            t0 = time.time()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                detections = self.yolo.detect(frame)
            except Exception as e:
                print(f"[WARN] YOLO inference failed on this frame ({e}) — skipping")
                time.sleep(0.1)
                continue

            try:
                depth_map = self.depth.estimate(frame)
            except Exception as e:
                print(f"[WARN] Depth inference failed on this frame ({e}) — skipping")
                time.sleep(0.1)
                continue

            sensor = self.arduino.get_latest()

            # COCO on CPU runs alongside Mars on Hailo — no resource conflict
            if self.coco is not None:
                try:
                    coco_dets  = self.coco.detect(frame)
                    detections = tag_unknown_objects(detections, coco_dets)
                except Exception as e:
                    print(f"[WARN] COCO inference failed on this frame ({e}) — skipping unknown tagging")

            try:
                result = fuse(detections, depth_map, sensor)
            except Exception as e:
                print(f"[WARN] Fusion failed on this frame ({e}) — skipping")
                time.sleep(0.1)
                continue

            # Append BEFORE estimate_motion so history[-1] = current frame
            self.history.append(result)
            result.detections = estimate_motion(result.detections, self.history)

            with self.fusion_lock:
                self.latest_fusion = result
            with self.depth_lock:
                self.latest_depth = depth_map
            with self.command_lock:
                cmd = self.latest_command

            # Display + record
            if self.recorder and cmd:
                try:
                    canvas = self.recorder.write(
                        frame, cmd, sensor, result, depth_map,
                        self._yolo_mode, self._depth_mode
                    )
                except Exception as e:
                    print(f"[WARN] Recorder write failed ({e}) — frame skipped")
                    canvas = frame
                try:
                    cv2.imshow("MIRA", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False
                        break
                except Exception:
                    pass  # headless environment — no display, skip imshow

            # ── Terminal output ───────────────────────────────────────────────
            frame_count += 1
            fps = frame_count / (time.time() - t_start)
            ms  = (time.time() - t0) * 1000

            det_strs = []
            for d in result.detections:
                tag    = " ?" if d.class_name == "unknown_object" else (" !" if d.class_name == "big_rock" else "")
                motion = f" v={d.velocity:+.2f}" if d.is_moving else ""
                det_strs.append(f"{d.class_name}{tag}({d.confidence:.0%}) prox={d.depth_mean:.2f}{motion}")

            s       = sensor
            tof_str = (f"C={s.tof_distances[0]:.2f} LF={s.tof_distances[1]:.2f} "
                       f"RF={s.tof_distances[2]:.2f} LS={s.tof_distances[3]:.2f} "
                       f"RS={s.tof_distances[4]:.2f}")
            imu_str = f"pitch={s.gyro_pitch:+.1f} roll={s.gyro_roll:+.1f} yaw={s.gyro_yaw:+.1f}"

            print(
                f"\n[Slow] {ms:.0f}ms  fps={fps:.1f}  terrain={result.terrain_class}"
                f"  slope={'YES' if result.slope_detected else 'no'}"
                f"  depth_conf={result.depth_confidence:.2f}  spd={s.speed:.2f}m/s"
                f"\n  TOF({'TOF' if s.tof_ok else 'ULTRA'}): {tof_str}"
                f"\n  IMU: {imu_str}  LDR={s.ldr_value:.2f}"
                f"\n  Detections: {'  |  '.join(det_strs) if det_strs else 'none'}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop — 50ms tick
    # ─────────────────────────────────────────────────────────────────────────

    def _main_loop(self):
        """
        Two modes controlled by MOTORS_ENABLED in config.py:
          MOTORS_ENABLED = 1 → autonomous: pipeline decisions drive motors
          MOTORS_ENABLED = 0 → manual: keyboard controls motors directly

        Keyboard controls (manual mode):
          W = FORWARD    S = STOP/BACK
          A = TURN_LEFT  D = TURN_RIGHT
          Q = SLOW       Space = STOP
          Ctrl+C = quit
        """
        if MOTORS_ENABLED:
            self._auto_loop()
        else:
            self._manual_loop()

    def _auto_loop(self):
        """Autonomous mode — pipeline sends commands to Arduino."""
        try:
            while self.running:
                with self.command_lock:
                    cmd = self.latest_command
                if cmd:
                    print(f"[NAV] {cmd.action:<12} {cmd.reason:<40} conf={cmd.confidence:.2f}")
                    self.arduino.send_command(cmd.action)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Pipeline] Shutting down...")
        finally:
            self._shutdown()

    def _manual_loop(self):
        """
        Manual keyboard control mode.
        Uses pynput for non-blocking keypress detection so the pipeline
        keeps running (camera, fusion, display) while you drive manually.

        Install: pip install pynput --break-system-packages
        """
        try:
            from pynput import keyboard as kb
        except ImportError:
            print("[ERROR] pynput not installed — run: pip install pynput --break-system-packages")
            self._shutdown()
            return

        KEY_MAP = {
            'w': 'FORWARD',
            'a': 'TURN_LEFT',
            'd': 'TURN_RIGHT',
            'q': 'SLOW',
            's': 'STOP',
            ' ': 'STOP',
        }

        current_key_action = 'STOP'

        def on_press(key):
            nonlocal current_key_action
            try:
                char = key.char.lower() if hasattr(key, 'char') and key.char else None
            except Exception:
                char = None

            # Arrow key support
            if key == kb.Key.up:    char = 'w'
            if key == kb.Key.left:  char = 'a'
            if key == kb.Key.right: char = 'd'
            if key == kb.Key.down:  char = 's'
            if key == kb.Key.space: char = ' '

            if char in KEY_MAP:
                current_key_action = KEY_MAP[char]

        def on_release(key):
            nonlocal current_key_action
            # Stop when key released (except explicit stop keys)
            if key == kb.Key.esc:
                return False  # stop listener
            current_key_action = 'STOP'

        print("\n[Manual Mode] Keyboard control active")
        print("  W / ↑       = FORWARD")
        print("  A / ←       = TURN LEFT")
        print("  D / →       = TURN RIGHT")
        print("  Q           = SLOW")
        print("  S / ↓ / Space = STOP")
        print("  Ctrl+C      = Quit\n")

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        try:
            while self.running:
                self.arduino.send_command(current_key_action)
                print(f"[MANUAL] {current_key_action:<12}", end='\r')
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Pipeline] Shutting down...")
        finally:
            listener.stop()
            self.arduino.send_command('STOP')
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown — release all resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.recorder:
            self.recorder.release()
        cv2.destroyAllWindows()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        RoverPipeline().start()
    except KeyboardInterrupt:
        pass