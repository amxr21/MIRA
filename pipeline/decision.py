# =============================================================================
# decision.py — Motion tracking and navigation decision logic
#
# Functions:
#   estimate_motion() — compares detections across frames, fills velocity
#   decide()          — priority waterfall → NavigationCommand
# =============================================================================

import time
from collections import deque
from typing import List, Optional

from config import CONFIG
from structures import Detection, FusionResult, NavigationCommand, SensorReading


def estimate_motion(
    detections: List[Detection],
    history: deque,
) -> List[Detection]:
    """
    Compares current detections to the previous frame using nearest-neighbor
    bounding box center matching. Fills is_moving and velocity on each detection.

    History buffer contract (enforced by RoverPipeline._slow_loop):
      history.append(result) is called BEFORE estimate_motion()
      → history[-1] = current frame FusionResult
      → history[-2] = previous frame FusionResult
      dt = time between those two frames

    Velocity: proximity units/s — negative = approaching (depth_mean increasing).
    Matching threshold: 100px center distance — prevents cross-matching distant objects.
    """
    if len(history) < 2:
        return detections

    prev = history[-2]
    dt   = history[-1].timestamp - prev.timestamp
    if dt <= 0:
        return detections

    for det in detections:
        cx = (det.bbox[0] + det.bbox[2]) / 2
        cy = (det.bbox[1] + det.bbox[3]) / 2
        best, best_dist = None, float("inf")

        for pd in prev.detections:
            pcx = (pd.bbox[0] + pd.bbox[2]) / 2
            pcy = (pd.bbox[1] + pd.bbox[3]) / 2
            d   = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if d < best_dist and d < 100:
                best_dist = d
                best = pd

        if best is not None:
            det.velocity  = (det.depth_mean - best.depth_mean) / dt
            det.is_moving = abs(det.velocity) > 0.1

    return detections


def decide(
    sensor: SensorReading,
    fusion: Optional[FusionResult],
) -> NavigationCommand:
    """
    Priority waterfall — returns at the first condition that fires.
    Lower priorities are never evaluated once a higher one triggers.
    This ordering is safety-critical.

    Priority table:
      1. Ultrasonic bumper   (only when tof_ok=False)     → STOP
      2. Dangerous tilt      (pitch/roll > hard limits)   → STOP
      3. Side TOF threat                                  → TURN
      4. Front obstacle      (TOF/depth/rock/unknown)     → STOP
      5. Approaching object                               → TURN
      6. Slope detected                                   → SLOW/STOP
      7. Moderate tilt                                    → SLOW
      8. Low light                                        → SLOW
      9. Medium range obstacle                            → SLOW
      10. All clear                                       → FORWARD
    """
    now = time.time()
    C   = CONFIG

    # ── Priority 1: Bumper ──────────────────────────────────────────────────
    # Use TOF center if healthy; switch to ultrasonic only when all TOF failed
    front_stop_dist = sensor.tof_distances[0] if sensor.tof_ok else sensor.ultra_front
    if not sensor.tof_ok:
        if sensor.ultra_front < C["ultra_stop"]:
            return NavigationCommand(
                "STOP", f"Front ultrasonic {sensor.ultra_front:.2f}m (TOF failed)", 1.0, now
            )
        if sensor.ultra_rear < C["ultra_stop"]:
            return NavigationCommand(
                "STOP", f"Rear ultrasonic {sensor.ultra_rear:.2f}m (TOF failed)", 1.0, now
            )

    # ── Priority 2: Dangerous tilt ─────────────────────────────────────────
    if abs(sensor.gyro_pitch) > C["pitch_stop"]:
        return NavigationCommand("STOP", f"Dangerous pitch {sensor.gyro_pitch:.1f}°", 1.0, now)
    if abs(sensor.gyro_roll) > C["roll_stop"]:
        return NavigationCommand("STOP", f"Dangerous roll {sensor.gyro_roll:.1f}°", 1.0, now)

    # ── Priority 3: Side TOF danger ────────────────────────────────────────
    if sensor.tof_ok:
        if sensor.tof_distances[3] < C["side_danger"]:
            return NavigationCommand("TURN_RIGHT", f"Left side {sensor.tof_distances[3]:.2f}m", 0.9, now)
        if sensor.tof_distances[4] < C["side_danger"]:
            return NavigationCommand("TURN_LEFT",  f"Right side {sensor.tof_distances[4]:.2f}m", 0.9, now)

    if fusion is not None:

        # ── Priority 4: Front obstacle ──────────────────────────────────────
        if front_stop_dist < C["obs_stop"]:
            src = "TOF" if sensor.tof_ok else "ULTRA"
            return NavigationCommand("STOP", f"[{src}] Obstacle {front_stop_dist:.2f}m", 0.95, now)

        for det in fusion.detections:
            if det.class_name in ("big_rock", "unknown_object") and det.depth_min < C["obs_stop"]:
                label = "Rock" if det.class_name == "big_rock" else "Unknown obj"
                return NavigationCommand("STOP", f"{label} {det.depth_min:.2f}m",
                                        fusion.depth_confidence, now)

        # ── Priority 5: Approaching object ──────────────────────────────────
        for det in fusion.detections:
            if (det.is_moving
                    and det.velocity < C["approach_thresh"]
                    and det.depth_mean < C["obs_slow"]):
                obj_cx = (det.bbox[0] + det.bbox[2]) / 2
                turn   = "TURN_RIGHT" if obj_cx < C["capture_w"] / 2 else "TURN_LEFT"
                return NavigationCommand(turn, f"Approaching obj {det.velocity:.2f}m/s",
                                         fusion.depth_confidence, now)

        # ── Priority 6: Slope ───────────────────────────────────────────────
        if fusion.slope_detected:
            if front_stop_dist < C["obs_slow"]:
                return NavigationCommand("STOP", "Steep slope + TOF confirm",
                                         fusion.depth_confidence, now)
            return NavigationCommand("SLOW", "Slope ahead", fusion.depth_confidence, now)

        # ── Priority 7: Moderate tilt ───────────────────────────────────────
        if abs(sensor.gyro_pitch) > C["pitch_slow"]:
            return NavigationCommand("SLOW", f"Moderate pitch {sensor.gyro_pitch:.1f}°", 0.85, now)
        if abs(sensor.gyro_roll) > C["roll_slow"]:
            return NavigationCommand("SLOW", f"Moderate roll {sensor.gyro_roll:.1f}°",  0.85, now)

        # ── Priority 8: Low light ───────────────────────────────────────────
        if sensor.ldr_value < C["ldr_low"]:
            return NavigationCommand("SLOW", "Extreme low light", 0.6, now)

        # ── Priority 9: Medium range obstacle ───────────────────────────────
        if front_stop_dist < C["obs_slow"]:
            return NavigationCommand("SLOW", f"Obstacle {front_stop_dist:.2f}m", 0.8, now)

    # ── Priority 10: All clear ──────────────────────────────────────────────
    return NavigationCommand("FORWARD", "Path clear", 0.9 if fusion else 0.5, now)