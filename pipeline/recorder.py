# =============================================================================
# recorder.py — Annotated video output
#
# Three-region frame layout:
# ┌────────────────────────────────────┬──────────────┐
# │  Live camera + boxes + arrows      │  Depth map   │
# │         1280 × 720                 │  320 × 180   │
# ├────────────────────────────────────┴──────────────┤
# │  TOF strip (Row 1) + IMU/LDR/Speed/NAV (Row 2)   │
# │                 1280 × 80                         │
# └───────────────────────────────────────────────────┘
# Output: 1280 × 800 XVID AVI
# =============================================================================

import os
import numpy as np
import cv2
from typing import Optional

from structures import NavigationCommand, SensorReading, FusionResult


class VideoRecorder:
    """
    Writes annotated frames to AVI during the slow loop.

    record_fps MUST match actual slow-loop throughput:
      Hailo (both): ~10 FPS   → record_fps = 10
      CPU only:     ~2 FPS   → record_fps = 2
    Setting record_fps too high makes the video play faster than real time.

    Bounding box color coding:
      Red    (0, 60, 220)   — big_rock       (danger)
      Orange (0, 165, 255)  — unknown_object (caution)
      Green  (30, 200, 30)  — terrain types  (informational)
    """

    SENSOR_STRIP_H = 80
    DEPTH_W        = 320
    DEPTH_H        = 180

    def __init__(self, output_path: str, fps: int, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.out_h   = frame_h + self.SENSOR_STRIP_H
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, self.out_h))
        print(f"[Recorder] {output_path} @ {fps} FPS")

    def write(
        self,
        frame: np.ndarray,
        cmd: NavigationCommand,
        sensor: SensorReading,
        fusion: Optional[FusionResult],
        depth_map: Optional[np.ndarray],
        yolo_mode: str,
        depth_mode: str,
    ) -> np.ndarray:
        """
        Composes the full annotated canvas and writes one frame to the AVI.
        Also returns the canvas so the slow loop can show it via cv2.imshow().
        """
        canvas = np.zeros((self.out_h, self.frame_w, 3), dtype=np.uint8)
        camera = frame.copy()

        # ── Bounding boxes + labels + depth + velocity arrows ────────────────
        if fusion:
            for det in fusion.detections:
                x1, y1, x2, y2 = det.bbox
                if det.class_name == "big_rock":
                    col = (0, 60, 220)       # red
                elif det.class_name == "unknown_object":
                    col = (0, 165, 255)      # orange
                else:
                    col = (30, 200, 30)      # green

                cv2.rectangle(camera, (x1, y1), (x2, y2), col, 2)
                cv2.putText(camera, f"{det.class_name} {det.confidence:.0%}",
                            (x1, max(y1 - 5, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
                cv2.putText(camera, f"prox={det.depth_mean:.2f}",
                            (x1, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 100), 1, cv2.LINE_AA)

                if det.is_moving:
                    cx_d = (x1 + x2) // 2
                    cy_d = (y1 + y2) // 2
                    dy   = -35 if det.velocity < 0 else 35   # up = approaching
                    cv2.arrowedLine(camera, (cx_d, cy_d), (cx_d, cy_d + dy),
                                    (0, 200, 255), 2, tipLength=0.35)

        # ── Depth map inset (top-right corner) ───────────────────────────────
        if depth_map is not None:
            dm_vis   = (depth_map * 255).astype(np.uint8)
            dm_color = cv2.applyColorMap(dm_vis, cv2.COLORMAP_INFERNO)
            dm_small = cv2.resize(dm_color, (self.DEPTH_W, self.DEPTH_H))
            x_off    = self.frame_w - self.DEPTH_W - 4
            camera[4:4 + self.DEPTH_H, x_off:x_off + self.DEPTH_W] = dm_small
            cv2.rectangle(camera, (x_off - 1, 3),
                          (x_off + self.DEPTH_W, 4 + self.DEPTH_H), (200, 200, 200), 1)
            cv2.putText(camera, "DEPTH", (x_off + 4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Mode badge (top-left) ─────────────────────────────────────────────
        cv2.putText(camera, f"Y:{yolo_mode}  D:{depth_mode}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        canvas[:self.frame_h, :] = camera

        # ── Sensor strip ─────────────────────────────────────────────────────
        canvas[self.frame_h:, :] = (22, 22, 22)
        tof        = sensor.tof_distances
        bumper_src = "TOF" if sensor.tof_ok else "ULTRA"

        # Row 1 — TOF values + active bumper source
        tof_txt = (f"TOF C={tof[0]:.2f}m  LF={tof[1]:.2f}m  RF={tof[2]:.2f}m  "
                   f"LS={tof[3]:.2f}m  RS={tof[4]:.2f}m  [{bumper_src}]")
        cv2.putText(canvas, tof_txt, (8, self.frame_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

        # Row 2 — IMU + LDR + speed + NAV command
        imu_txt = (f"pitch={sensor.gyro_pitch:.1f}°  roll={sensor.gyro_roll:.1f}°  "
                   f"yaw={sensor.gyro_yaw:.1f}°  LDR={sensor.ldr_value:.2f}  "
                   f"spd={sensor.speed:.2f}m/s")
        cv2.putText(canvas, imu_txt, (8, self.frame_h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)

        if cmd:
            action_col = {
                "FORWARD":    (0, 200, 0),
                "SLOW":       (0, 180, 255),
                "STOP":       (0, 0, 230),
                "TURN_LEFT":  (200, 100, 0),
                "TURN_RIGHT": (200, 100, 0),
            }.get(cmd.action, (180, 180, 180))
            nav_txt = f"► {cmd.action}  {cmd.reason}  conf={cmd.confidence:.2f}"
            cv2.putText(canvas, nav_txt, (700, self.frame_h + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, action_col, 2, cv2.LINE_AA)

        self.writer.write(canvas)
        return canvas

    def release(self):
        self.writer.release()
        print("[Recorder] File saved.")