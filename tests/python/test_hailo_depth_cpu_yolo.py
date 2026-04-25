"""
test_hailo_depth_cpu_yolo.py
=============================
DepthAnything V2 ViT-S on Hailo-8L NPU | YOLO26n on CPU via ONNX Runtime.
Matches the structure of test_hailo_both.py — swap YOLO Hailo block for ONNX.

Use this when:
  - yolo26n_mars.hef is unavailable
  - comparing YOLO CPU vs NPU latency
  - Scenario 2 verification before full pipeline run

Press Q to quit.
"""

import numpy as np
import cv2
import time
import onnxruntime as ort
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)

# ── Config ───────────────────────────────────────────────────
YOLO_ONNX      = "../models/yolo26/yolo26n_mars.onnx"
DEPTH_HEF      = "../models/depthAnything/depth_anything_v2_vits.hef"
CAMERA_INDEX   = 0
CLASSES        = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
CONF_THRESHOLD = 0.4
DEPTH_W, DEPTH_H = 320, 180

# ── Load YOLO on CPU ─────────────────────────────────────────
print("[Test] Loading YOLO on CPU ONNX...")
yolo_sess  = ort.InferenceSession(YOLO_ONNX, providers=["CPUExecutionProvider"])
yolo_iname = yolo_sess.get_inputs()[0].name

# ── Load Depth on Hailo ──────────────────────────────────────
# DA V2 ViT-S HEF: Input UINT8 NHWC 224×224×3 | Output UINT16 NHWC 224×224×1
print("[Test] Loading Depth on Hailo...")
d_hef = HEF(DEPTH_HEF)
d_target = VDevice()
d_cfg = ConfigureParams.create_from_hef(hef=d_hef, interface=HailoStreamInterface.PCIe)
d_ng  = d_target.configure(d_hef, d_cfg)[0]
d_ngp = d_ng.create_params()
d_inp = InputVStreamParams.make(d_ng, format_type=FormatType.UINT8)
d_out = OutputVStreamParams.make(d_ng, format_type=FormatType.UINT16)
d_name = d_hef.get_input_vstream_infos()[0].name
print("[Test] YOLO:CPU  Depth:Hailo — ready (press Q to quit)")

# ── State ────────────────────────────────────────────────────
smooth      = None
alpha       = 0.7
frame_count = 0
t_start     = time.time()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    t0 = time.time()

    # ── YOLO on CPU ONNX ─────────────────────────────────────
    t1  = time.time()
    img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)  # NCHW float32
    raw = yolo_sess.run(None, {yolo_iname: img})[0]
    if raw.ndim == 3:
        raw = raw[0]
    t_yolo = (time.time() - t1) * 1000

    for row in raw:
        scores = row[4:4 + len(CLASSES)]
        cid    = int(np.argmax(scores))
        conf   = float(scores[cid])
        if conf < CONF_THRESHOLD:
            continue
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = max(0, int((cx - w / 2) * orig_w / 640))
        y1 = max(0, int((cy - h / 2) * orig_h / 640))
        x2 = min(orig_w, int((cx + w / 2) * orig_w / 640))
        y2 = min(orig_h, int((cy + h / 2) * orig_h / 640))
        col = (0, 60, 220) if cid == 3 else (30, 200, 30)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{CLASSES[cid]} {conf:.0%}", (x1, max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    # ── Depth on Hailo ───────────────────────────────────────
    t2   = time.time()
    s    = min(orig_h, orig_w)
    crop = frame[(orig_h - s) // 2:(orig_h + s) // 2, (orig_w - s) // 2:(orig_w + s) // 2]
    d_in = np.expand_dims(cv2.resize(crop, (224, 224)), 0)  # NHWC uint8

    with InferVStreams(d_ng, d_inp, d_out) as pipe:
        with d_ng.activate(d_ngp):
            raw_d = pipe.infer({d_name: d_in})
    depth = list(raw_d.values())[0][0].squeeze().astype(np.float32)  # UINT16 → float32

    t_depth = (time.time() - t2) * 1000

    # ── Postprocess (same as test_hailo_both) ────────────────
    smooth = alpha * depth + (1 - alpha) * smooth if smooth is not None else depth
    norm   = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-6)
    dm_vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    dm_small = cv2.resize(dm_vis, (DEPTH_W, DEPTH_H))
    x_off = orig_w - DEPTH_W - 4
    frame[4:4 + DEPTH_H, x_off:x_off + DEPTH_W] = dm_small
    cv2.rectangle(frame, (x_off - 1, 3), (x_off + DEPTH_W, 4 + DEPTH_H), (200, 200, 200), 1)

    # ── Timing overlay ───────────────────────────────────────
    frame_count += 1
    fps = frame_count / (time.time() - t_start)
    cv2.putText(frame, f"YOLO:CPU {t_yolo:.0f}ms  Depth:Hailo {t_depth:.0f}ms  {fps:.1f}fps",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("MIRA — YOLO:CPU + Depth:Hailo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()