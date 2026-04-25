# =============================================================================
# structures.py — Shared data containers
# All dataclasses used across the pipeline. Every other module imports from here.
# No logic, no side effects — pure data definitions.
# =============================================================================

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """
    One object detected by YOLO in a single frame.

    Lifecycle:
      Born in  : _decode_yolo() / YOLODetectorCOCO.detect()
      Enriched : map_depth_to_detections() → adds depth_mean, depth_min
      Enriched : estimate_motion()         → adds is_moving, velocity
      Consumed : decide(), VideoRecorder.write()
    """
    # From YOLO
    class_id:   int
    class_name: str                          # "soil"|"bedrock"|"sand"|"big_rock"|"unknown_object"
    confidence: float
    bbox:       Tuple[int, int, int, int]    # (x1, y1, x2, y2) in original frame pixels

    # Enriched by fusion.map_depth_to_detections()
    mask:       Optional[np.ndarray] = None  # segmentation mask — unused, reserved
    depth_mean: float = 0.0                  # avg proximity across bbox (0=far, 1=close)
    depth_min:  float = 0.0                  # closest point inside bbox

    # Enriched by decision.estimate_motion()
    is_moving:  bool  = False
    velocity:   float = 0.0                  # proximity units/s — negative = approaching


@dataclass
class SensorReading:
    """
    One complete snapshot of all Arduino sensor values (~33ms intervals).

    Born in  : sensors.ArduinoInterface.update() or sensors._dummy_sensor()
    Consumed : fusion.fuse(), decision.decide(), recorder.VideoRecorder.write()
    """
    # 5 TOF sensors: [center-front, left-front, right-front, left-side, right-side]
    tof_distances:  List[float]   # meters — 999.0 = timeout, replaced by last valid per sensor
    tof_ok:         bool          # True if at least one TOF is returning valid readings

    # IMU — front MPU-6050
    gyro_pitch:     float         # forward/backward tilt in degrees
    gyro_roll:      float         # sideways tilt in degrees
    gyro_yaw:       float         # raw gyro Z in °/s (not integrated — informational only)

    # Rear IMU slot — uncomment when second MPU-6050 is installed (AD0=HIGH → addr 0x69)
    # gyro_pitch_rear: float = 0.0
    # gyro_roll_rear:  float = 0.0

    # Ambient light
    ldr_value:      float         # 0.0=dark, 1.0=bright — controls depth model trust weight

    # Ultrasonic — bumper fallback ONLY when tof_ok=False
    ultra_front:    float         # meters
    ultra_rear:    float         # meters

    # Speed from encoder
    speed:          float         # m/s — floored at 0.0

    timestamp:      float = 0.0


@dataclass
class FusionResult:
    """
    Complete output of one slow-loop frame after all models + fusion have run.

    Born in  : fusion.fuse()
    Enriched : decision.estimate_motion() → detections get velocity
    Consumed : decision.decide(), recorder.VideoRecorder.write(), RoverPipeline._slow_loop()
    """
    detections:       List[Detection]
    depth_map:        np.ndarray    # TOF-corrected proximity map H×W float32 (0=far, 1=close)
    terrain_class:    str           # dominant terrain class by bbox area
    slope_detected:   bool
    depth_confidence: float         # 0.0–1.0 — how well depth model agreed with TOF
    timestamp:        float = 0.0


@dataclass
class NavigationCommand:
    """
    Final output of the pipeline — one high-level movement instruction.

    Born in  : decision.decide()
    Consumed : RoverPipeline._main_loop(), recorder.VideoRecorder.write()
    In production: send via serial to motor controller Arduino.
    """
    action:     str     # FORWARD | SLOW | TURN_LEFT | TURN_RIGHT | STOP
    reason:     str     # human-readable explanation for logging and display
    confidence: float   # 0.5 (no fusion data) → 1.0 (certain, e.g. bumper contact)
    timestamp:  float = 0.0