# =============================================================================
# fusion.py — Pure fusion logic functions
# No classes, no state. All functions take inputs and return outputs.
#
# Functions:
#   compute_light_weights()    — LDR → depth/TOF blend weights
#   validate_depth_with_tof()  — correct depth map using TOF readings
#   map_depth_to_detections()  — enrich each Detection with depth stats
#   detect_slope()             — check depth map for downslope ahead
#   get_dominant_terrain()     — find terrain class by largest bbox area
#   fuse()                     — main coordinator, returns FusionResult
# =============================================================================

import time
import numpy as np
from typing import List, Tuple

from config import CONFIG
from structures import Detection, FusionResult, SensorReading


def compute_light_weights(ldr: float) -> Tuple[float, float]:
    """
    Returns (depth_model_weight, tof_weight) based on ambient light.
    In darkness the camera is unreliable → trust TOF fully.
    In bright conditions → trust depth model more, TOF as correction only.
    Weights always sum to 1.0.
    """
    if ldr >= CONFIG["ldr_good"]:
        return (0.7, 0.3)
    elif ldr >= CONFIG["ldr_low"]:
        return (0.3, 0.7)
    return (0.0, 1.0)   # extreme darkness — ignore depth model entirely


def validate_depth_with_tof(
    depth_map: np.ndarray,
    tof_distances: List[float],
    depth_w: float,
    tof_w: float,
) -> Tuple[np.ndarray, float]:
    """
    Corrects the depth map using TOF readings at overlapping camera zones.
    Only sensors 0, 1, 2 (front-facing) have camera FOV overlap.
    Sensors 3 and 4 are side-facing → CONFIG["tof_zones"][3/4] = None → skipped.

    Three correction cases per zone:
      diff ≤ 0.3m  → strong agreement  → blend using LDR weights       → confidence 0.95
      0.3–1.0m     → moderate          → scale entire zone to TOF       → confidence 0.60
      > 1.0m       → large disagreement → stamp zone flat at TOF value  → confidence 0.30

    Returns (corrected_depth_map, overall_confidence).
    """
    corrected   = depth_map.copy()
    confidences = []
    h, w = depth_map.shape[:2]

    for sid in [0, 1, 2]:
        zone = CONFIG["tof_zones"][sid]
        if zone is None:
            continue
        tof_dist = tof_distances[sid]
        cx, cy, zw, zh = zone
        x1 = max(0, int((cx - zw / 2) * w))
        x2 = min(w, int((cx + zw / 2) * w))
        y1 = max(0, int((cy - zh / 2) * h))
        y2 = min(h, int((cy + zh / 2) * h))
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            continue

        model_mean = float(np.mean(region))
        diff = abs(model_mean - tof_dist)

        if diff <= CONFIG["depth_agree"]:
            blended = model_mean * depth_w + tof_dist * tof_w
            corrected[y1:y2, x1:x2] *= blended / (model_mean + 1e-6)
            confidences.append(0.95)
        elif diff <= CONFIG["depth_warn"]:
            corrected[y1:y2, x1:x2] *= tof_dist / (model_mean + 1e-6)
            confidences.append(0.6)
        else:
            corrected[y1:y2, x1:x2] = tof_dist
            confidences.append(0.3)

    confidence = float(np.mean(confidences)) if confidences else 0.5
    return corrected, confidence


def map_depth_to_detections(
    detections: List[Detection],
    depth_map: np.ndarray,
) -> List[Detection]:
    """
    For each Detection, crops the corresponding region from the corrected depth map
    and fills depth_mean (average proximity) and depth_min (closest point).
    Must be called AFTER validate_depth_with_tof so we use corrected values.
    """
    h, w = depth_map.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            continue
        det.depth_mean = float(np.mean(region))
        det.depth_min  = float(np.min(region))
    return detections


def detect_slope(depth_map: np.ndarray) -> bool:
    """
    Checks for a sudden proximity increase in the bottom 30% of the frame.
    If the ground gets rapidly "closer" toward the bottom edge → downslope or cliff.
    Known limitation: large rocks in the lower frame can trigger false positives.
    np.median would be more robust — using np.mean for speed.
    """
    h     = depth_map.shape[0]
    lower = depth_map[int(h * 0.7):, :]
    if lower.size == 0:
        return False
    strip   = max(1, lower.shape[0] // 3)
    top_avg = np.mean(lower[:strip, :])
    bot_avg = np.mean(lower[-strip:, :])
    return (bot_avg - top_avg) > CONFIG["slope_drop"]


def get_dominant_terrain(detections: List[Detection]) -> str:
    """
    Returns the terrain class with the largest total bounding box area.
    Area-based rather than count-based — one large sand region dominates
    over three small bedrock detections.
    Excludes unknown_object from terrain classification.
    """
    areas = {}
    for d in detections:
        if d.class_name == "unknown_object":
            continue
        x1, y1, x2, y2 = d.bbox
        areas[d.class_name] = areas.get(d.class_name, 0) + (x2 - x1) * (y2 - y1)
    return max(areas, key=areas.get) if areas else "unknown"


def fuse(
    detections: List[Detection],
    depth_map: np.ndarray,
    sensor: SensorReading,
) -> FusionResult:
    """
    Main fusion coordinator. Combines YOLO + DepthAnything V2 + TOF + LDR.
    Calls all sub-functions in the correct dependency order.

    Order matters:
      1. compute_light_weights  → get blend weights from LDR
      2. validate_depth_with_tof → correct depth map (uses weights from step 1)
      3. map_depth_to_detections → enrich detections from CORRECTED depth map
      4. get_dominant_terrain + detect_slope → scene-level summaries
    """
    depth_w, tof_w = compute_light_weights(sensor.ldr_value)
    corrected, confidence = validate_depth_with_tof(
        depth_map, sensor.tof_distances, depth_w, tof_w
    )
    detections = map_depth_to_detections(detections, corrected)

    return FusionResult(
        detections=detections,
        depth_map=corrected,
        terrain_class=get_dominant_terrain(detections),
        slope_detected=detect_slope(corrected),
        depth_confidence=confidence,
        timestamp=time.time(),
    )