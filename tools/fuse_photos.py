"""
fuse_photos.py — Run the full fusion pipeline on still images.
Usage (from mira/ root):
    python tools/fuse_photos.py
Outputs go to samples/output/
"""

import os, sys, time
import cv2
import numpy as np

# Resolve project root (mira/) regardless of where this script is called from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import pipeline as p

INPUT_DIR  = os.path.join(ROOT, "samples", "input")
OUTPUT_DIR = os.path.join(ROOT, "samples", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load models once ──────────────────────────────────────────────────────────
print("[fuse_photos] Loading models...")

import onnxruntime as ort

yolo_sess  = ort.InferenceSession(p.CONFIG["yolo_path_onnx"],  providers=["CPUExecutionProvider"])
coco_sess  = ort.InferenceSession(p.CONFIG["coco_path_onnx"],  providers=["CPUExecutionProvider"])
depth_sess = ort.InferenceSession(p.CONFIG["depth_path_onnx"], providers=["CPUExecutionProvider"])

yolo_inp  = yolo_sess.get_inputs()[0].name
coco_inp  = coco_sess.get_inputs()[0].name
depth_inp = depth_sess.get_inputs()[0].name

yolo_det  = p.YOLODetectorONNX(p.CONFIG["yolo_path_onnx"])
coco_det  = p.YOLODetectorCOCO(p.CONFIG["coco_path_onnx"])
depth_est = p.DepthEstimatorONNX(p.CONFIG["depth_path_onnx"])

print("[fuse_photos] Models ready.\n")

# ── dummy sensor (all-clear) ──────────────────────────────────────────────────
sensor = p._dummy_sensor()

# ── process each image ────────────────────────────────────────────────────────
photos = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".jpg"))

for fname in photos:
    t0 = time.time()
    path = os.path.join(INPUT_DIR, fname)
    frame = cv2.imread(path)
    if frame is None:
        print(f"[WARN] Could not read {fname}")
        continue

    # Rotate portrait to landscape
    if frame.shape[0] > frame.shape[1]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    orig_h, orig_w = frame.shape[:2]

    # YOLO MARS
    mars_dets = yolo_det.detect(frame)

    # COCO unknown objects
    coco_dets = coco_det.detect(frame)
    detections = p.tag_unknown_objects(mars_dets, coco_dets)

    # Depth
    depth_map = depth_est.estimate(frame) if not depth_est.broken else None

    # Fusion
    result = p.fuse(detections, depth_map, sensor)

    # ── draw overlays ─────────────────────────────────────────────────────────
    canvas = frame.copy()

    # Depth map thumbnail (top-right corner)
    if depth_map is not None:
        p.draw_depth_zones(canvas, depth_map)
        dv = (depth_map * 255).astype(np.uint8)
        dc = cv2.applyColorMap(dv, cv2.COLORMAP_INFERNO)
        thumb_w, thumb_h = 320, 180
        ds = cv2.resize(dc, (thumb_w, thumb_h))
        xo = orig_w - thumb_w - 4
        canvas[4:4+thumb_h, xo:xo+thumb_w] = ds
        cv2.rectangle(canvas, (xo-1, 3), (xo+thumb_w, 4+thumb_h), (200,200,200), 1)
        cv2.putText(canvas, "DEPTH", (xo+4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "NO DEPTH", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,200), 2, cv2.LINE_AA)

    # Detection boxes
    p._draw_detections(canvas, result.detections)

    # Info bar at top-left
    terrain = result.terrain_class
    slope   = "SLOPE" if result.slope_detected else "flat"
    n_det   = len(result.detections)
    ms      = (time.time() - t0) * 1000
    cv2.putText(canvas, f"terrain={terrain}  slope={slope}  dets={n_det}  {ms:.0f}ms",
                (8, orig_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,255), 1, cv2.LINE_AA)

    # Sensor strip at bottom
    s = sensor
    strip_h = 60
    strip = np.full((strip_h, orig_w, 3), (22,22,22), dtype=np.uint8)
    cv2.putText(strip,
        f"TOF C={s.tof_distances[0]:.2f}  LF={s.tof_distances[1]:.2f}  "
        f"RF={s.tof_distances[2]:.2f}  LS={s.tof_distances[3]:.2f}  RS={s.tof_distances[4]:.2f}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,220,255), 1, cv2.LINE_AA)
    cv2.putText(strip,
        f"pitch={s.gyro_pitch:+.1f}  roll={s.gyro_roll:+.1f}  yaw={s.gyro_yaw:+.1f}  "
        f"LDR={s.ldr_value:.2f}  spd={s.speed:.2f}m/s  depth={'on' if depth_map is not None else 'OFF'}",
        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)

    out_frame = np.vstack([canvas, strip])

    # Save
    stem = os.path.splitext(fname)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_fused.jpg")
    cv2.imwrite(out_path, out_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    det_summary = "  |  ".join(
        f"{'unknown' if d.class_name.startswith('[') else d.class_name} {d.confidence:.0%}"
        for d in result.detections
    ) or "none"
    print(f"[{fname}]  {ms:.0f}ms  terrain={terrain}  dets={n_det}  -> {out_path}")
    print(f"  detections: {det_summary}")

print(f"\n[fuse_photos] Done. Outputs in {OUTPUT_DIR}/")
