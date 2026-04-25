"""
test_hailo_both.py
==================
Tests YOLO + DepthAnything V2 both running on Hailo-8L NPU.
Displays live camera with bounding boxes and depth map inset.
Press Q to quit.
"""

import numpy as np
import cv2
import time
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)

# ── Config ──────────────────────────────────────────────────
YOLO_HEF       = "../models/yolo26/yolo26n_mars.hef"
DEPTH_HEF      = "../models/depthAnything_v2_vits.hef"
CAMERA_INDEX   = 0
CLASSES        = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
CONF_THRESHOLD = 0.4
DEPTH_W, DEPTH_H = 320, 180


# ── Load models ─────────────────────────────────────────────
def load_hailo(hef_path, fmt_out):
    hef = HEF(hef_path)
    target = VDevice()
    cfg = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    ng  = target.configure(hef, cfg)[0]
    ngp = ng.create_params()
    inp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    out = OutputVStreamParams.make(ng, format_type=fmt_out)
    return hef, ng, ngp, inp, out


print("[Test] Loading YOLO on Hailo...")
y_hef, y_ng, y_ngp, y_inp, y_out = load_hailo(YOLO_HEF, FormatType.FLOAT32)
print("[Test] Loading Depth on Hailo...")
# DA V2 ViT-S HEF outputs UINT16 — confirmed via hailortcli parse-hef
d_hef, d_ng, d_ngp, d_inp, d_out = load_hailo(DEPTH_HEF, FormatType.UINT16)
print("[Test] Both models loaded on Hailo-8L")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

smooth = None
alpha  = 0.7
frame_count = 0
t_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    t0 = time.time()

    # ── YOLO inference ──────────────────────────────────────
    yolo_in = np.expand_dims(cv2.resize(frame, (640, 640)), 0)
    y_name  = y_hef.get_input_vstream_infos()[0].name
    with InferVStreams(y_ng, y_inp, y_out) as pipe:
        with y_ng.activate(y_ngp):
            raw_y = pipe.infer({y_name: yolo_in})
    raw = list(raw_y.values())[0][0]
    if raw.ndim == 3:
        raw = raw[0]

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

    # ── Depth inference ─────────────────────────────────────
    s  = min(orig_h, orig_w)
    crop = frame[(orig_h - s) // 2:(orig_h + s) // 2, (orig_w - s) // 2:(orig_w + s) // 2]
    d_in = np.expand_dims(cv2.resize(crop, (224, 224)), 0)
    d_name = d_hef.get_input_vstream_infos()[0].name
    with InferVStreams(d_ng, d_inp, d_out) as pipe:
        with d_ng.activate(d_ngp):
            raw_d = pipe.infer({d_name: d_in})
    depth = list(raw_d.values())[0][0].squeeze().astype(np.float32)  # UINT16 → float32

    smooth = alpha * depth + (1 - alpha) * smooth if smooth is not None else depth
    norm   = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-6)
    dm_vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    dm_small = cv2.resize(dm_vis, (DEPTH_W, DEPTH_H))
    x_off = orig_w - DEPTH_W - 4
    frame[4:4 + DEPTH_H, x_off:x_off + DEPTH_W] = dm_small
    cv2.rectangle(frame, (x_off - 1, 3), (x_off + DEPTH_W, 4 + DEPTH_H), (200, 200, 200), 1)

    # ── FPS overlay ─────────────────────────────────────────
    frame_count += 1
    fps = frame_count / (time.time() - t_start)
    ms  = (time.time() - t0) * 1000
    cv2.putText(frame, f"HAILO+HAILO  {fps:.1f}fps  {ms:.0f}ms",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("MIRA — Both on Hailo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()