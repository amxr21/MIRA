"""
mira_pipeline.py — MIRA Single-File Pipeline v9
================================================
Run (no virtualenv needed):
  python3 mira_pipeline.py

Flags:
  WINDOWS_MODE    = 0 or 1  (1 = Windows PC, 0 = Raspberry Pi)
  ARDUINO_ENABLED = 0 or 1
  RECORD_ENABLED  = 0 or 1
  COCO_ENABLED    = 0 or 1
  MOTORS_ENABLED  = 0 or 1

Based on the working v5 structure. Key confirmed facts from live runs:
  - MARS ONNX output shape: (1, 40, 8400)  →  4 bbox + 4 classes + 32 mask coeffs
  - COCO ONNX output shape: (1, 116, 8400)  →  pre-NMS, needs transpose + NMS
  - COCO model lives at /home/sdp2/yolo26n-seg.onnx  (not in models/ subdir)
  - Depth ONNX (depth_anything_v2_small) crashes ORT 1.24 on ARM at MatMul kernel
    due to mixed-precision tensors — contained with broken flag, runs as None
  - Hailo is primary for both YOLO and Depth when hailo_platform available
  - yolo_input for Hailo HEF: (416, 416) per v5 confirmed value
  - depth_input for Hailo HEF: (224, 224) per v5 confirmed value
  - COCO input: (640, 640) confirmed from model
"""

# =============================================================================
# WINDOWS MODE ENABLE BIT  — must be defined before any conditional imports
# =============================================================================

WINDOWS_MODE = 1  # set to 0 when running on the Pi

import os
import sys
import re
import threading
import time

if WINDOWS_MODE == 0:
    import tty
    import termios

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# =============================================================================
# FLAGS
# =============================================================================

ARDUINO_ENABLED = 1
RECORD_ENABLED  = 1
COCO_ENABLED    = 1
MOTORS_ENABLED  = 0
# VIDEO_SOURCE = 0
VIDEO_SOURCE = "samples/input/sample.mp4"  # set to 0 for live camera

if WINDOWS_MODE == 1:
    ARDUINO_ENABLED = 0

# Minimum bounding box side in pixels — smaller boxes are noise
MIN_BOX_PX = 10

# =============================================================================
# HAILO
# =============================================================================

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[Boot] hailo_platform not found — will use ONNX CPU for both models")

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Model paths — COCO lives in home dir, not models/
    "yolo_path_hef":   "models/yolo26/yolo26n_mars.hef",
    "depth_path_hef":  "models/depthAnything/depth_anything_v2_vits.hef",
    "yolo_path_onnx":  "models/yolo26/yolo26n_mars.onnx",
    "depth_path_onnx": "models/depthAnything/depth_anything_v2_small.onnx",
    "coco_path_onnx":  "models/yolo26/yolo26n-seg.onnx",

    "unknown_iou_thresh":  0.5,
    "unknown_conf_thresh": 0.55,

    "camera_index": 0,
    "capture_w":    1280,
    "capture_h":    720,
    "capture_fps":  30,

    # Input sizes confirmed from v5 working run and model inspection
    "yolo_input":  (640, 640),   # MARS ONNX confirmed — HEF uses 416 on Pi
    "coco_input":  (640, 640),   # COCO ONNX confirmed
    "depth_input": (224, 224),   # Depth HEF confirmed input size

    "record_output": "logs/fusion_output.avi",
    "record_fps":    15,

    "tof_zones": {
        0: (0.50, 0.60, 0.20, 0.30),   # center
        1: (0.25, 0.55, 0.20, 0.30),   # left-front
        2: (0.75, 0.55, 0.20, 0.30),   # right-front
        3: None,
        4: None,
    },

    "depth_agree":        0.3,
    "depth_warn":         1.0,
    "depth_smooth_alpha": 0.7,

    "ldr_good":  0.6,
    "ldr_low":   0.25,

    "pitch_slow": 10, "pitch_stop": 20,
    "roll_slow":  15, "roll_stop":  25,

    "obs_stop":    0.5,
    "obs_slow":    1.5,
    "obs_clear":   3.0,
    "side_danger": 0.8,
    "ultra_stop":  0.3,
    "slope_drop":  0.8,

    "history_len":      5,
    "approach_thresh": -0.5,

    # MARS terrain class mapping — indices must match model training
    "classes": {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"},
}

# Capture dimensions as standalone ints used in navigation
_CAP_W = CONFIG["capture_w"]
_CAP_H = CONFIG["capture_h"]

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Detection:
    class_id:   int
    class_name: str       # MARS: "soil"/"bedrock"/"sand"/"big_rock"
                          # Unknown: "[person]" etc. — bracket = unknown object
    confidence: float
    bbox:       Tuple[int, int, int, int]   # x1,y1,x2,y2 in frame pixels
    mask:       Optional[np.ndarray] = None
    depth_mean: float = 0.0
    depth_min:  float = 0.0
    is_moving:  bool  = False
    velocity:   float = 0.0


@dataclass
class SensorReading:
    tof_distances: List[float]
    tof_ok:        bool
    gyro_pitch:    float
    gyro_roll:     float
    gyro_yaw:      float
    ldr_value:     float
    ultra_front:   float
    ultra_rear:    float
    speed:         float
    timestamp:     float = 0.0


@dataclass
class FusionResult:
    detections:       List[Detection]
    depth_map:        Optional[np.ndarray]
    terrain_class:    str
    slope_detected:   bool
    depth_confidence: float
    timestamp:        float = 0.0


@dataclass
class NavigationCommand:
    action:     str
    reason:     str
    confidence: float
    timestamp:  float = 0.0

# =============================================================================
# BOUNDING BOX VALIDATION
# =============================================================================

def _valid_bbox(x1: int, y1: int, x2: int, y2: int,
                frame_w: int, frame_h: int) -> bool:
    """
    Reject degenerate boxes:
      - Either side smaller than MIN_BOX_PX
      - Covers more than 95% of frame (full-frame false positive)
      - Sits entirely within 3px of any single edge (edge artifact)
    """
    w = x2 - x1;  h = y2 - y1
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return False
    if w * h > 0.50 * frame_w * frame_h:
        return False
    EDGE = 3
    if x2 <= EDGE or y2 <= EDGE:
        return False
    if x1 >= frame_w - EDGE or y1 >= frame_h - EDGE:
        return False
    return True

# =============================================================================
# SENSORS
# =============================================================================

_dummy_t0    = time.time()
_dummy_yaw   = 0.0

def _dummy_sensor() -> SensorReading:
    # Simulate realistic slow-moving rover — all distances safe, slight natural drift
    t = time.time() - _dummy_t0
    global _dummy_yaw
    _dummy_yaw += 0.08
    return SensorReading(
        tof_distances=[
            2.80 + 0.35 * np.sin(t * 0.4),        # center — gently varying
            3.10 + 0.20 * np.sin(t * 0.3 + 1.0),  # left-front
            2.95 + 0.25 * np.cos(t * 0.35 + 0.5), # right-front
            4.20 + 0.15 * np.sin(t * 0.2 + 2.0),  # left-side
            4.05 + 0.18 * np.cos(t * 0.25 + 1.5), # right-side
        ],
        tof_ok=True,
        gyro_pitch=1.2 * np.sin(t * 0.15),
        gyro_roll =0.8 * np.cos(t * 0.18 + 0.3),
        gyro_yaw  =_dummy_yaw % 360.0,
        ldr_value =0.82 + 0.06 * np.sin(t * 0.05),
        ultra_front=2.80 + 0.35 * np.sin(t * 0.4),
        ultra_rear =3.50 + 0.20 * np.cos(t * 0.22),
        speed      =0.18 + 0.04 * abs(np.sin(t * 0.3)),
        timestamp  =time.time(),
    )


# Arduino labeled output field order — must match sendSensorData() in firmware
_ARDUINO_FIELDS = [
    "tof1","tof2","tof3","tof4","tof5",
    "pitch","roll","yaw","ldr",
    "ultra_front","ultra_rear","speed",
]


def _parse_arduino_labeled(line: str) -> Optional[List[str]]:
    # Extract all key:value pairs
    pairs = dict(re.findall(r'([\w_]+):([\-\d\.]+)', line))
    # Status byte is the last bare 0/1 token (no colon)
    status = None
    for tok in reversed(line.strip().split()):
        if ':' not in tok and re.match(r'^[01]$', tok):
            status = tok
            break
    if status is None:
        return None
    result = [pairs[f] for f in _ARDUINO_FIELDS if f in pairs]
    if len(result) != len(_ARDUINO_FIELDS):
        return None
    result.append(status)
    return result   # 13 elements total


class ArduinoInterface:
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200):
        self.serial_conn = None
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()
        self.latest = _dummy_sensor()

    def connect(self):
        if not ARDUINO_ENABLED:
            print("[Arduino] DISABLED — using dummy sensor readings")
            return
        import serial
        if not os.path.exists(self.port):
            print(f"[WARN] Arduino port {self.port} not found")
            return
        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2)
            print(f"[Arduino] Connected on {self.port}")
        except Exception as e:
            print(f"[WARN] Arduino connection failed ({e})")
            self.serial_conn = None

    def update(self):
        if not ARDUINO_ENABLED or self.serial_conn is None or not self.serial_conn.is_open:
            return
        try:
            line = self.serial_conn.readline().decode("utf-8").strip()
            if not line:
                return

            p = _parse_arduino_labeled(line) if ":" in line else line.split(",")
            if p is None or len(p) != 13:
                return

            prev = self.latest
            tof_raw = [float(p[i]) for i in range(5)]
            tof = [v if 0.0 < v < 8.0 else prev.tof_distances[i]
                   for i, v in enumerate(tof_raw)]
            tof_valid = any(v < 8.0 for v in tof)

            rp=float(p[5]); rr=float(p[6]); ry=float(p[7])
            gyro_pitch = rp if abs(rp) <= 180.0 else prev.gyro_pitch
            gyro_roll  = rr if abs(rr) <= 180.0 else prev.gyro_roll
            gyro_yaw   = ry if abs(ry) <= 500.0 else prev.gyro_yaw

            ldr = max(0.0, min(1.0, float(p[8])))
            uf = float(p[9]);  ur = float(p[10])
            ultra_front = uf if 0.0 < uf < 8.0 else prev.ultra_front
            ultra_rear  = ur if 0.0 < ur < 8.0 else prev.ultra_rear
            speed = max(0.0, float(p[11]))

            with self.lock:
                self.latest = SensorReading(
                    tof_distances=tof,
                    tof_ok=int(p[12]) == 1 and tof_valid,
                    gyro_pitch=gyro_pitch, gyro_roll=gyro_roll, gyro_yaw=gyro_yaw,
                    ldr_value=ldr,
                    ultra_front=ultra_front, ultra_rear=ultra_rear,
                    speed=speed, timestamp=time.time(),
                )
        except (ValueError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"[WARN] Arduino read error ({e})")
            self.serial_conn = None

    def send_command(self, action: str):
        if not ARDUINO_ENABLED or self.serial_conn is None or not self.serial_conn.is_open:
            return
        char_map = {
            "FORWARD":b"F","SLOW":b"S","REVERSE":b"B",
            "TURN_LEFT":b"L","TURN_RIGHT":b"R","STOP":b"X",
        }
        try:
            self.serial_conn.write(char_map.get(action, b"X"))
        except Exception as e:
            print(f"[WARN] Motor command failed ({e})")
            self.serial_conn = None

    def get_latest(self) -> SensorReading:
        with self.lock:
            return self.latest

# =============================================================================
# SHARED HELPERS
# =============================================================================

def _nms(boxes: np.ndarray, scores: np.ndarray,
         iou_thresh: float = 0.35) -> List[int]:
    """Vectorised NMS. boxes: (N,4) as x1y1x2y2. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        if order.size == 1: break
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0.,xx2-xx1)*np.maximum(0.,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<=iou_thresh]
    return keep


def _compute_iou(a: tuple, b: tuple) -> float:
    xA=max(a[0],b[0]); yA=max(a[1],b[1])
    xB=min(a[2],b[2]); yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

# =============================================================================
# YOLO — HAILO (MARS)
# =============================================================================

class YOLODetectorHailo:
    """
    Hailo MARS YOLOv8 detector.
    HEF input: (416,416) UINT8.
    Two output tensors expected:
      - 'activation_...' : (N,4)  normalised cx,cy,w,h  [0,1]
      - 'concat18_...'   : (N,64) DFL scores  (4 classes × 16 bins)
    """
    def __init__(self, hef_path: str, device):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"YOLO HEF not found: {hef_path}")
        self.hef    = HEF(hef_path)
        self.target = device
        cfg = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.target.configure(self.hef, cfg)[0]
        self.network_group_params = self.network_group.create_params()
        self.input_params  = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32)
        self.size        = CONFIG["yolo_input"]   # (416,416)
        self._input_name = self.hef.get_input_vstream_infos()[0].name
        print(f"[YOLO] Loaded on Hailo-8L NPU  input={self.size}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        h_m, w_m = self.size
        img = cv2.resize(frame, (w_m, h_m))
        with InferVStreams(self.network_group,
                          self.input_params, self.output_params) as pipe:
            with self.network_group.activate(self.network_group_params):
                raw = pipe.infer({self._input_name: np.expand_dims(img, 0)})

        bboxes = None; scores_raw = None
        for name, tensor in raw.items():
            t = tensor[0]
            if t.ndim == 3: t = t[0]
            if "activation" in name:
                bboxes = t
            elif "concat" in name or "scores" in name:
                scores_raw = t

        if bboxes is None or scores_raw is None:
            return []
        return _decode_hailo_dfl(bboxes, scores_raw, orig_h, orig_w)


def _decode_hailo_dfl(bboxes: np.ndarray, scores_raw: np.ndarray,
                      orig_h: int, orig_w: int) -> List[Detection]:
    """
    DFL decoder for Hailo multi-tensor output.
    bboxes:     (N, 4)     normalised [0,1] cx,cy,w,h
    scores_raw: (N, 64)    4 classes × 16 bins raw logits
    """
    classes   = CONFIG["classes"]
    n_classes = len(classes)                         # 4
    n_bins    = scores_raw.shape[1] // n_classes     # 16

    s  = scores_raw.reshape(-1, n_classes, n_bins)
    e  = np.exp(s - s.max(axis=2, keepdims=True))
    sm = e / (e.sum(axis=2, keepdims=True) + 1e-6)
    bins   = np.arange(n_bins, dtype=np.float32)
    cls_sc = (sm * bins).sum(axis=2) / (n_bins - 1)  # (N, n_classes)

    cids  = np.argmax(cls_sc, axis=1)
    confs = cls_sc[np.arange(len(cids)), cids]

    detections = []
    for idx in np.where(confs > 0.50)[0]:
        conf = float(confs[idx]); cid = int(cids[idx])
        cx = float(bboxes[idx][0]) * orig_w
        cy = float(bboxes[idx][1]) * orig_h
        bw = float(bboxes[idx][2]) * orig_w
        bh = float(bboxes[idx][3]) * orig_h
        x1=max(0,int(cx-bw/2)); y1=max(0,int(cy-bh/2))
        x2=min(orig_w,int(cx+bw/2)); y2=min(orig_h,int(cy+bh/2))
        if not _valid_bbox(x1,y1,x2,y2,orig_w,orig_h): continue
        detections.append(Detection(
            class_id=cid, class_name=classes.get(cid,"unknown"),
            confidence=conf, bbox=(x1,y1,x2,y2),
        ))
    return detections

# =============================================================================
# YOLO — ONNX (MARS)
# =============================================================================

# MARS YOLOv8-seg ONNX confirmed output: (1, 40, 8400)
# 40 = 4 bbox  +  4 terrain classes  +  32 seg mask coefficients
# After transpose to (8400, 40):
#   cols 0-3  : cx, cy, w, h  in model pixel space (0..416)
#   cols 4-7  : raw class logits for 4 terrain classes
#   cols 8-39 : mask coefficients — ignored (we don't need instance masks)
_MARS_NC = 4


def _decode_mars_onnx(raw: np.ndarray, orig_h: int, orig_w: int,
                      h_m: int, w_m: int) -> List[Detection]:
    """
    raw: (40, 8400) after removing batch dim — will be transposed here.
    Applies sigmoid to class logits, NMS, bbox validation.
    """
    classes = CONFIG["classes"]

    # Transpose to (8400, 40)
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
        raw = raw.T

    if raw.shape[1] < 4 + _MARS_NC:
        return []

    cx_a=raw[:,0]; cy_a=raw[:,1]; bw_a=raw[:,2]; bh_a=raw[:,3]
    logits = raw[:, 4: 4+_MARS_NC]
    probs  = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))  # sigmoid
    cids   = np.argmax(probs, axis=1)
    confs  = probs[np.arange(len(cids)), cids]

    mask = confs >= 0.505
    if not mask.any(): return []

    cx_f=cx_a[mask]; cy_f=cy_a[mask]; bw_f=bw_a[mask]; bh_f=bh_a[mask]
    cids_f=cids[mask]; confs_f=confs[mask]

    boxes_m = np.stack([cx_f-bw_f/2, cy_f-bh_f/2,
                        cx_f+bw_f/2, cy_f+bh_f/2], axis=1)
    keep = _nms(boxes_m, confs_f)

    detections = []
    for idx in keep:
        conf=float(confs_f[idx]); cid=int(cids_f[idx])
        x1=max(0,      int(boxes_m[idx,0]*orig_w/w_m))
        y1=max(0,      int(boxes_m[idx,1]*orig_h/h_m))
        x2=min(orig_w, int(boxes_m[idx,2]*orig_w/w_m))
        y2=min(orig_h, int(boxes_m[idx,3]*orig_h/h_m))
        if not _valid_bbox(x1,y1,x2,y2,orig_w,orig_h): continue
        detections.append(Detection(
            class_id=cid, class_name=classes.get(cid,"unknown"),
            confidence=conf, bbox=(x1,y1,x2,y2),
        ))
    return detections


class YOLODetectorONNX:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO ONNX not found: {model_path}")
        self.session    = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size       = CONFIG["yolo_input"]   # (416,416)
        print(f"[YOLO] Loaded on CPU ONNX: {model_path}")
        for o in self.session.get_outputs():
            print(f"  output: {o.name}  shape={o.shape}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        h_m, w_m = self.size
        img = cv2.resize(frame, (w_m,h_m)).astype(np.float32)/255.0
        img = np.expand_dims(np.transpose(img,(2,0,1)),0)
        raw = self.session.run(None,{self.input_name:img})[0]  # (1,40,8400)
        if raw.ndim == 3: raw = raw[0]                          # (40,8400)
        return _decode_mars_onnx(raw, orig_h, orig_w, h_m, w_m)

# =============================================================================
# COCO — ONNX  (unknown object detector)
# =============================================================================

# COCO YOLOv8n-seg confirmed output: (1, 116, 8400) — pre-NMS channel-first
# After transpose to (8400, 116):
#   cols 0-3   : cx, cy, w, h  in model pixel space (0..640)
#   cols 4-83  : 80 COCO class scores (raw logits — argmax only, no sigmoid needed)
#   cols 84-115: 32 mask coefficients — ignored
_COCO_NC = 80


class YOLODetectorCOCO:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"COCO ONNX not found: {model_path}")
        self.session    = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size       = CONFIG["coco_input"]   # (640,640)
        print(f"[COCO] Loaded on CPU ONNX: {model_path}")
        print(f"  input shape  : {self.session.get_inputs()[0].shape}")
        print(f"  output shape : {self.session.get_outputs()[0].shape}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        w_m, h_m = self.size   # 640, 640

        img = cv2.resize(frame,(w_m,h_m)).astype(np.float32)/255.0
        img = np.expand_dims(np.transpose(img,(2,0,1)),0)
        raw = self.session.run(None,{self.input_name:img})[0]  # (1,116,8400)
        raw = raw[0]                                            # (116,8400)
        if raw.shape[0] < raw.shape[1]: raw = raw.T            # (8400,116)

        cls_scores = raw[:, 4: 4+_COCO_NC]
        class_ids  = np.argmax(cls_scores, axis=1)
        confs      = cls_scores[np.arange(len(class_ids)), class_ids]

        mask = confs >= CONFIG["unknown_conf_thresh"]
        if not mask.any(): return []

        raw_f=raw[mask]; confs_f=confs[mask]; cls_f=class_ids[mask]
        cx=raw_f[:,0]; cy=raw_f[:,1]; bw=raw_f[:,2]; bh=raw_f[:,3]
        boxes_m = np.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], axis=1)
        keep = _nms(boxes_m, confs_f)

        detections = []
        for idx in keep:
            conf=float(confs_f[idx]); cid=int(cls_f[idx])
            x1=max(0,      int(boxes_m[idx,0]*orig_w/w_m))
            y1=max(0,      int(boxes_m[idx,1]*orig_h/h_m))
            x2=min(orig_w, int(boxes_m[idx,2]*orig_w/w_m))
            y2=min(orig_h, int(boxes_m[idx,3]*orig_h/h_m))
            if not _valid_bbox(x1,y1,x2,y2,orig_w,orig_h): continue
            cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"coco_{cid}"
            detections.append(Detection(
                class_id=cid, class_name=f"[{cname}]",
                confidence=conf, bbox=(x1,y1,x2,y2),
            ))
        return detections


def tag_unknown_objects(mars_dets: List[Detection],
                        coco_dets: List[Detection]) -> List[Detection]:
    """
    Merge MARS detections with COCO detections.
    Any COCO detection that has no IoU overlap (>= unknown_iou_thresh) with
    any MARS detection is kept as an "unknown object" (bracketed name).
    MARS detections are returned unchanged — they take priority.
    """
    thresh = CONFIG["unknown_iou_thresh"]
    result = list(mars_dets)
    for cd in coco_dets:
        if not any(_compute_iou(cd.bbox, m.bbox) >= thresh for m in mars_dets):
            result.append(cd)   # class_name already "[xxx]" from detector
    return result

# =============================================================================
# DEPTH — HAILO
# =============================================================================

class DepthEstimatorHailo:
    def __init__(self, hef_path: str, device):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"Depth HEF not found: {hef_path}")
        self.hef    = HEF(hef_path)
        self.target = device
        cfg = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.target.configure(self.hef, cfg)[0]
        self.network_group_params = self.network_group.create_params()
        self.input_params  = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32)
        self.size        = CONFIG["depth_input"]   # (224,224)
        self._input_name = self.hef.get_input_vstream_infos()[0].name
        self._smooth: Optional[np.ndarray] = None
        self._alpha  = CONFIG["depth_smooth_alpha"]
        print(f"[Depth] DepthAnything V2 ViT-S loaded on Hailo-8L NPU  input={self.size}")

    def _center_crop(self, frame):
        h,w=frame.shape[:2]; s=min(h,w)
        return frame[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(self._center_crop(frame), self.size)
        with InferVStreams(self.network_group,
                          self.input_params, self.output_params) as pipe:
            with self.network_group.activate(self.network_group_params):
                raw = pipe.infer({self._input_name: np.expand_dims(img,0)})
        depth = list(raw.values())[0][0].squeeze().astype(np.float32)
        return self._postprocess(depth, orig_w, orig_h)

    def _postprocess(self, depth, ow, oh):
        if self._smooth is None: self._smooth = depth
        else: self._smooth = self._alpha*depth + (1-self._alpha)*self._smooth
        mn,mx = self._smooth.min(), self._smooth.max()
        norm  = (self._smooth - mn) / (mx - mn + 1e-6)
        return cv2.resize(norm,(ow,oh),interpolation=cv2.INTER_LINEAR)

# =============================================================================
# DEPTH — ONNX  (with runtime crash containment)
# =============================================================================

class DepthEstimatorONNX:
    """
    CPU ONNX fallback for depth.
    depth_anything_v2_small.onnx contains bfloat16/mixed-precision ViT nodes
    that crash ORT 1.24 on ARM at MatMul kernel level (not optimisation time).
    A test inference runs at init; if it fails self.broken=True is set and
    all subsequent estimate() calls return None immediately — no crash.
    Hailo is the intended path; this class is a graceful degradation only.
    """
    broken: bool = False

    def __init__(self, model_path: str):
        import onnxruntime as ort
        if not os.path.exists(model_path):
            print(f"[WARN] Depth ONNX not found: {model_path} — depth disabled on CPU")
            self.broken = True
            return

        try:
            self.session    = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.size       = (518, 518)   # DepthAnything V2 native resolution
            self._smooth: Optional[np.ndarray] = None
            self._alpha = CONFIG["depth_smooth_alpha"]

            dummy = np.zeros((1, 3, self.size[0], self.size[1]), dtype=np.float32)
            self.session.run(None, {self.input_name: dummy})
            print(f"[Depth] Loaded on CPU ONNX: {model_path}  input={self.size}")

        except Exception as e:
            print(f"[WARN] Depth ONNX not usable on this ORT build: {e}")
            print(f"       Depth will be disabled — use Hailo HEF for depth.")
            self.broken = True

    def _center_crop(self, frame):
        h,w=frame.shape[:2]; s=min(h,w)
        return frame[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.broken: return None
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(self._center_crop(frame), self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = (img-[0.485,0.456,0.406])/[0.229,0.224,0.225]
        img = np.expand_dims(np.transpose(img,(2,0,1)),0).astype(np.float32)
        try:
            depth = self.session.run(None,{self.input_name:img})[0]
        except Exception as e:
            print(f"[WARN] Depth inference failed ({e}) — disabling depth")
            self.broken = True
            return None
        if depth.ndim==4: depth=depth[0,0]
        elif depth.ndim==3: depth=depth[0]
        return self._postprocess(depth.astype(np.float32), orig_w, orig_h)

    def _postprocess(self, depth, ow, oh):
        if self._smooth is None: self._smooth = depth
        else: self._smooth = self._alpha*depth + (1-self._alpha)*self._smooth
        mn,mx = self._smooth.min(), self._smooth.max()
        norm  = (self._smooth - mn) / (mx - mn + 1e-6)
        return cv2.resize(norm,(ow,oh),interpolation=cv2.INTER_LINEAR)

# =============================================================================
# DEPTH ZONE OVERLAY
# =============================================================================

ZONE_NAMES  = ["FAR-L","NEAR-L","CENTER","NEAR-R","FAR-R"]
ZONE_COLORS = [
    (255,100,100),(100,200,255),(100,255,100),(100,200,255),(255,100,100),
]


def draw_depth_zones(canvas: np.ndarray,
                     depth_map: np.ndarray) -> List[float]:
    """
    Split frame into 5 equal vertical zones. Compute mean depth per zone.
    Draw boundary lines + tinted bar at bottom + text labels.
    Returns list of 5 mean values in [0,1] (1=closest).
    """
    h,w   = canvas.shape[:2]
    zw    = w // 5
    means = []
    for i in range(5):
        x1 = i*zw; x2 = x1+zw if i < 4 else w
        mv = float(np.mean(depth_map[:, x1:x2]))
        means.append(mv)
        col = ZONE_COLORS[i]
        if i > 0:
            cv2.line(canvas,(x1,0),(x1,h),col,1,cv2.LINE_AA)
        # Tinted bottom bar
        ov = canvas.copy()
        intens = int(mv*120)
        tc = (min(255,intens*2), max(0,120-intens), max(0,120-intens*2))
        cv2.rectangle(ov,(x1,h-60),(x2,h),tc,-1)
        cv2.addWeighted(ov,0.45,canvas,0.55,0,canvas)
        # Label
        cx = (x1+x2)//2
        tc2 = (0,50,255) if mv>0.7 else (0,200,255) if mv>0.45 else (200,255,200)
        cv2.putText(canvas,ZONE_NAMES[i],(cx-28,h-38),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,tc2,1,cv2.LINE_AA)
        cv2.putText(canvas,f"{mv:.2f}",(cx-18,h-18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,tc2,2,cv2.LINE_AA)
    return means

# =============================================================================
# DRAW DETECTIONS
# =============================================================================

# Box colours
_COL_TERRAIN = (30,  200,  30)    # green  — MARS soil/sand/bedrock
_COL_ROCK    = (0,    60, 220)    # red    — MARS big_rock (hazard)
_COL_UNKNOWN = (0,   140, 255)    # orange — unknown object (COCO not in MARS)


def _draw_detections(canvas: np.ndarray, detections: List[Detection]):
    for det in detections:
        x1,y1,x2,y2 = det.bbox
        is_rock    = det.class_name == "big_rock"
        is_unknown = det.class_name.startswith("[")

        col = _COL_ROCK if is_rock else (_COL_UNKNOWN if is_unknown else _COL_TERRAIN)

        # Box
        cv2.rectangle(canvas,(x1,y1),(x2,y2),col,2)

        # Label text: MARS classes shown as-is, unknown shown as "unknown"
        display = "unknown" if is_unknown else det.class_name
        label   = f"{display} {det.confidence:.0%}"

        (lw,lh),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
        ly = max(y1-4, lh+4)
        # Label background
        cv2.rectangle(canvas,(x1,ly-lh-4),(x1+lw+4,ly+2),col,-1)
        cv2.putText(canvas,label,(x1+2,ly),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        # Proximity text below box
        if det.depth_mean > 0:
            pty = min(y2+16, canvas.shape[0]-2)
            cv2.putText(canvas,f"prox={det.depth_mean:.2f}",(x1,pty),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(220,220,100),1,cv2.LINE_AA)

        # Motion arrow
        if det.is_moving:
            cx_d=(x1+x2)//2; cy_d=(y1+y2)//2
            dy = -35 if det.velocity < 0 else 35
            cv2.arrowedLine(canvas,(cx_d,cy_d),(cx_d,cy_d+dy),
                            (0,200,255),2,tipLength=0.35)

# =============================================================================
# FUSION
# =============================================================================

def compute_light_weights(ldr: float) -> Tuple[float,float]:
    if ldr >= CONFIG["ldr_good"]:  return (0.7,0.3)
    elif ldr >= CONFIG["ldr_low"]: return (0.3,0.7)
    return (0.0,1.0)


def validate_depth_with_tof(depth_map, tof_distances, dw, tw):
    corrected   = depth_map.copy()
    confidences = []
    h,w = depth_map.shape[:2]
    for sid in [0,1,2]:
        zone = CONFIG["tof_zones"][sid]
        if zone is None: continue
        td = tof_distances[sid]
        cx,cy,zw,zh = zone
        x1=max(0,int((cx-zw/2)*w)); x2=min(w,int((cx+zw/2)*w))
        y1=max(0,int((cy-zh/2)*h)); y2=min(h,int((cy+zh/2)*h))
        reg = depth_map[y1:y2,x1:x2]
        if reg.size == 0: continue
        mm = float(np.mean(reg))
        diff = abs(mm-td)
        if diff <= CONFIG["depth_agree"]:
            bl = mm*dw + td*tw
            corrected[y1:y2,x1:x2] *= bl/(mm+1e-6)
            confidences.append(0.95)
        elif diff <= CONFIG["depth_warn"]:
            corrected[y1:y2,x1:x2] *= td/(mm+1e-6)
            confidences.append(0.6)
        else:
            corrected[y1:y2,x1:x2] = td
            confidences.append(0.3)
    return corrected, (float(np.mean(confidences)) if confidences else 0.5)


def map_depth_to_detections(detections, depth_map):
    if depth_map is None: return detections
    h,w = depth_map.shape[:2]
    for det in detections:
        x1,y1,x2,y2 = det.bbox
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
        reg = depth_map[y1:y2,x1:x2]
        if reg.size == 0: continue
        det.depth_mean = float(np.mean(reg))
        det.depth_min  = float(np.min(reg))
    return detections


def detect_slope(depth_map) -> bool:
    if depth_map is None: return False
    h = depth_map.shape[0]
    lower = depth_map[int(h*0.7):,:]
    if lower.size == 0: return False
    strip = max(1,lower.shape[0]//3)
    return (np.mean(lower[-strip:,:])-np.mean(lower[:strip,:])) > CONFIG["slope_drop"]


def get_dominant_terrain(detections) -> str:
    areas = {}
    for d in detections:
        if d.class_name.startswith("["): continue
        x1,y1,x2,y2 = d.bbox
        areas[d.class_name] = areas.get(d.class_name,0)+(x2-x1)*(y2-y1)
    return max(areas,key=areas.get) if areas else "unknown"


def fuse(detections, depth_map, sensor) -> FusionResult:
    if depth_map is not None:
        dw,tw = compute_light_weights(sensor.ldr_value)
        corrected,conf = validate_depth_with_tof(
            depth_map, sensor.tof_distances, dw, tw)
    else:
        corrected = None; conf = 0.0
    detections = map_depth_to_detections(detections, corrected)
    return FusionResult(
        detections=detections, depth_map=corrected,
        terrain_class=get_dominant_terrain(detections),
        slope_detected=detect_slope(corrected),
        depth_confidence=conf, timestamp=time.time(),
    )

# =============================================================================
# NAVIGATION
# =============================================================================

def estimate_motion(detections, history):
    if len(history) < 2: return detections
    prev = history[-2]
    dt   = history[-1].timestamp - prev.timestamp
    if dt <= 0: return detections
    for det in detections:
        cx=(det.bbox[0]+det.bbox[2])/2; cy=(det.bbox[1]+det.bbox[3])/2
        best,bd = None, float("inf")
        for pd in prev.detections:
            pcx=(pd.bbox[0]+pd.bbox[2])/2; pcy=(pd.bbox[1]+pd.bbox[3])/2
            d=((cx-pcx)**2+(cy-pcy)**2)**.5
            if d<bd and d<100: bd=d; best=pd
        if best is not None:
            det.velocity  = (det.depth_mean-best.depth_mean)/dt
            det.is_moving = abs(det.velocity)>0.1
    return detections


def decide(sensor, fusion) -> NavigationCommand:
    now=time.time(); C=CONFIG
    front = sensor.tof_distances[0] if sensor.tof_ok else sensor.ultra_front

    if not sensor.tof_ok:
        if sensor.ultra_front < C["ultra_stop"]:
            return NavigationCommand("STOP",f"Front ultra {sensor.ultra_front:.2f}m",1.0,now)
        if sensor.ultra_rear  < C["ultra_stop"]:
            return NavigationCommand("STOP",f"Rear ultra {sensor.ultra_rear:.2f}m",1.0,now)

    if abs(sensor.gyro_pitch) > C["pitch_stop"]:
        return NavigationCommand("STOP",f"Pitch {sensor.gyro_pitch:.1f}deg",1.0,now)
    if abs(sensor.gyro_roll)  > C["roll_stop"]:
        return NavigationCommand("STOP",f"Roll {sensor.gyro_roll:.1f}deg",1.0,now)

    if sensor.tof_ok:
        if sensor.tof_distances[3] < C["side_danger"]:
            return NavigationCommand("TURN_RIGHT",f"Left {sensor.tof_distances[3]:.2f}m",0.9,now)
        if sensor.tof_distances[4] < C["side_danger"]:
            return NavigationCommand("TURN_LEFT", f"Right {sensor.tof_distances[4]:.2f}m",0.9,now)

    if fusion is not None:
        if front < C["obs_stop"]:
            src="TOF" if sensor.tof_ok else "ULTRA"
            return NavigationCommand("STOP",f"[{src}] {front:.2f}m",0.95,now)

        for det in fusion.detections:
            hazard = det.class_name=="big_rock" or det.class_name.startswith("[")
            if hazard and det.depth_min < C["obs_stop"]:
                lbl="Rock" if det.class_name=="big_rock" else "Unknown obj"
                return NavigationCommand("STOP",f"{lbl} {det.depth_min:.2f}m",
                                         fusion.depth_confidence,now)

        for det in fusion.detections:
            if (det.is_moving and det.velocity<C["approach_thresh"]
                    and det.depth_mean<C["obs_slow"]):
                cx=(det.bbox[0]+det.bbox[2])/2
                turn="TURN_RIGHT" if cx<_CAP_W/2 else "TURN_LEFT"
                return NavigationCommand(turn,f"Approaching {det.velocity:.2f}m/s",
                                          fusion.depth_confidence,now)

        if fusion.slope_detected:
            if front < C["obs_slow"]:
                return NavigationCommand("STOP","Steep slope+sensor",
                                          fusion.depth_confidence,now)
            return NavigationCommand("SLOW","Slope ahead",fusion.depth_confidence,now)

        if abs(sensor.gyro_pitch)>C["pitch_slow"]:
            return NavigationCommand("SLOW",f"Pitch {sensor.gyro_pitch:.1f}deg",0.85,now)
        if abs(sensor.gyro_roll) >C["roll_slow"]:
            return NavigationCommand("SLOW",f"Roll {sensor.gyro_roll:.1f}deg",0.85,now)
        if sensor.ldr_value < C["ldr_low"]:
            return NavigationCommand("SLOW","Low light",0.6,now)
        if front < C["obs_slow"]:
            return NavigationCommand("SLOW",f"Obstacle {front:.2f}m",0.8,now)

    return NavigationCommand("FORWARD","Path clear",0.9 if fusion else 0.5,now)

# =============================================================================
# RECORDER
# =============================================================================

class VideoRecorder:
    SENSOR_STRIP_H = 100
    DEPTH_W        = 480
    DEPTH_H        = 270

    def __init__(self, output_path: str, fps: int, fw: int, fh: int):
        self.fw=fw; self.fh=fh; self.oh=fh+self.SENSOR_STRIP_H
        dirpart = os.path.dirname(output_path)
        if dirpart:
            os.makedirs(dirpart, exist_ok=True)
        self.writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (fw, self.oh))
        print(f"[Recorder] {output_path} @ {fps} FPS")

    def write(self, frame, cmd, sensor, fusion, depth_map, ym, dm):
        canvas = np.zeros((self.oh,self.fw,3),dtype=np.uint8)
        cam    = frame.copy()

        # Detection overlays
        if fusion:
            _draw_detections(cam, fusion.detections)

        # Depth thumbnail + zone overlay
        if depth_map is not None:
            zone_means = draw_depth_zones(cam, depth_map)
            dv  = (depth_map*255).astype(np.uint8)
            dc  = cv2.applyColorMap(dv, cv2.COLORMAP_INFERNO)
            ds  = cv2.resize(dc,(self.DEPTH_W,self.DEPTH_H))
            xo  = self.fw - self.DEPTH_W - 4
            cam[4:4+self.DEPTH_H, xo:xo+self.DEPTH_W] = ds
            cv2.rectangle(cam,(xo-1,3),(xo+self.DEPTH_W,4+self.DEPTH_H),(200,200,200),1)
            cv2.putText(cam,"DEPTH",(xo+4,18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
        else:
            zone_means = [0.0]*5
            cv2.putText(cam,"NO DEPTH",(10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,200),2,cv2.LINE_AA)

        # Mode tag top-left
        cv2.putText(cam,f"Y:{ym}  D:{dm}",(6,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1,cv2.LINE_AA)

        canvas[:self.fh,:]=cam
        canvas[self.fh:,:]=(22,22,22)

        # Sensor strip
        tof=sensor.tof_distances; src="TOF" if sensor.tof_ok else "ULTRA"
        cv2.putText(canvas,
            f"TOF C={tof[0]:.2f}m  LF={tof[1]:.2f}m  RF={tof[2]:.2f}m  "
            f"LS={tof[3]:.2f}m  RS={tof[4]:.2f}m  [{src}]",
            (8,self.fh+22),cv2.FONT_HERSHEY_SIMPLEX,0.46,(180,220,255),1,cv2.LINE_AA)
        cv2.putText(canvas,
            f"pitch={sensor.gyro_pitch:.1f}  roll={sensor.gyro_roll:.1f}  "
            f"yaw={sensor.gyro_yaw:.1f}  LDR={sensor.ldr_value:.2f}  "
            f"spd={sensor.speed:.2f}m/s",
            (8,self.fh+48),cv2.FONT_HERSHEY_SIMPLEX,0.46,(200,255,200),1,cv2.LINE_AA)
        cv2.putText(canvas,
            "Zones: "+"  ".join(f"{ZONE_NAMES[i]}={zone_means[i]:.2f}" for i in range(5)),
            (8,self.fh+74),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,255),1,cv2.LINE_AA)
        if cmd:
            ac={
                "FORWARD":(0,200,0),"SLOW":(0,180,255),"STOP":(0,0,230),
                "TURN_LEFT":(200,100,0),"TURN_RIGHT":(200,100,0),
            }.get(cmd.action,(180,180,180))
            cv2.putText(canvas,
                f">> {cmd.action}  {cmd.reason}  conf={cmd.confidence:.2f}",
                (8,self.fh+96),cv2.FONT_HERSHEY_SIMPLEX,0.50,ac,2,cv2.LINE_AA)

        self.writer.write(canvas)
        return canvas

    def release(self):
        self.writer.release()
        print("[Recorder] File saved.")

# =============================================================================
# PIPELINE
# =============================================================================

class RoverPipeline:
    def __init__(self):
        self.arduino    = ArduinoInterface(port="/dev/ttyACM0")
        self.coco       = None
        self.print_lock = threading.Lock()
        self.yolo, self.depth, self._ym, self._dm = self._load_models()
        self.recorder: Optional[VideoRecorder] = None
        self.latest_fusion:  Optional[FusionResult]     = None
        self.latest_command: Optional[NavigationCommand] = None
        self.latest_depth:   Optional[np.ndarray]        = None
        self.fusion_lock  = threading.Lock()
        self.command_lock = threading.Lock()
        self.depth_lock   = threading.Lock()
        self.history: deque = deque(maxlen=CONFIG["history_len"])
        self.cap=None; self.running=False

    # ------------------------------------------------------------------
    def _load_models(self):
        ym="ONNX"; dm="ONNX"; yolo=None; depth=None

        # Hailo is primary for BOTH models — attempted independently
        dev = VDevice() if HAILO_AVAILABLE else None

        if HAILO_AVAILABLE and dev and os.path.exists(CONFIG["yolo_path_hef"]):
            try:
                yolo = YOLODetectorHailo(CONFIG["yolo_path_hef"], dev)
                ym   = "Hailo"
            except Exception as e:
                print(f"[WARN] YOLO Hailo failed ({e}) — falling back to ONNX")

        if yolo is None:
            try:
                yolo = YOLODetectorONNX(CONFIG["yolo_path_onnx"])
            except Exception as e:
                print(f"[ERROR] YOLO ONNX failed ({e})")

        if HAILO_AVAILABLE and dev and os.path.exists(CONFIG["depth_path_hef"]):
            try:
                depth = DepthEstimatorHailo(CONFIG["depth_path_hef"], dev)
                dm    = "Hailo"
            except Exception as e:
                print(f"[WARN] Depth Hailo failed ({e}) — falling back to ONNX")

        if depth is None:
            onnx_d = DepthEstimatorONNX(CONFIG["depth_path_onnx"])
            depth  = onnx_d
            if onnx_d.broken:
                dm = "NONE"

        if ym=="Hailo" and dm=="Hailo":
            self._scenario = "1 — Both on Hailo NPU  (~13-20 FPS)"
        elif ym=="Hailo":
            self._scenario = "2 — YOLO on Hailo, Depth CPU  (~5-8 FPS)"
        elif dm=="Hailo":
            self._scenario = "2 — Depth on Hailo, YOLO CPU  (~5-8 FPS)"
        elif dm=="NONE":
            self._scenario = "4 — YOLO CPU only, Depth incompatible (no depth)"
        else:
            self._scenario = "3 — Both on CPU ONNX  (~1-2 FPS)"

        if COCO_ENABLED:
            try:
                self.coco = YOLODetectorCOCO(CONFIG["coco_path_onnx"])
                print("[COCO] Loaded — unknown_object tagging active")
            except Exception as e:
                print(f"[WARN] COCO failed ({e}) — unknown tagging disabled")
        else:
            print("[COCO] Disabled by COCO_ENABLED=0")

        return yolo, depth, ym, dm

    # ------------------------------------------------------------------
    def _print_boot_info(self):
        try:
            import subprocess
            temp = subprocess.check_output(
                ["vcgencmd","measure_temp"]).decode().strip()
        except Exception:
            temp = "N/A"
        print("="*58)
        print(f"  Scenario : {self._scenario}")
        print(f"  YOLO     : {self._ym}")
        print(f"  Depth    : {self._dm}")
        print(f"  COCO     : {'ON' if self.coco else 'OFF'}")
        print(f"  Temp     : {temp}")
        print(f"  Arduino  : {'ON' if ARDUINO_ENABLED else 'OFF (dummy)'}")
        print(f"  Motors   : {'AUTO' if MOTORS_ENABLED else 'MANUAL (W/A/S/D)'}")
        print(f"  Record   : {'ON -> '+CONFIG['record_output'] if RECORD_ENABLED else 'OFF'}")
        print("="*58)

    # ------------------------------------------------------------------
    def start(self):
        self.arduino.connect()

        # Live camera (current):
        # self.cap = cv2.VideoCapture(CONFIG["camera_index"])

        # Pre-recorded video:
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        
        if isinstance(VIDEO_SOURCE, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["capture_w"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["capture_h"])
            self.cap.set(cv2.CAP_PROP_FPS,          CONFIG["capture_fps"])
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        if not self.cap.isOpened():
            print("[ERROR] Camera failed to open."); raise SystemExit(1)

        # Read actual frame dimensions from the source (video file may differ from CONFIG)
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # If portrait, dimensions will be swapped after rotation
        if src_h > src_w:
            src_w, src_h = src_h, src_w

        if RECORD_ENABLED:
            try:
                self.recorder = VideoRecorder(
                    CONFIG["record_output"], CONFIG["record_fps"],
                    src_w, src_h)
            except Exception as e:
                print(f"[WARN] Recorder failed ({e}) — disabled")
                self.recorder = None

        self._print_boot_info()
        self.running = True
        threading.Thread(target=self._fast_loop, daemon=True).start()
        threading.Thread(target=self._slow_loop, daemon=True).start()
        self._main_loop()

    # ------------------------------------------------------------------
    def _fast_loop(self):
        """30 Hz — reads Arduino, runs navigation decision."""
        while self.running:
            self.arduino.update()
            sensor = self.arduino.get_latest()
            with self.fusion_lock: fusion = self.latest_fusion
            cmd = decide(sensor, fusion)
            with self.command_lock: self.latest_command = cmd
            time.sleep(0.033)

    # ------------------------------------------------------------------
    def _slow_loop(self):
        """Frame loop — YOLO → Depth → COCO → Fusion → Record."""
        fc=0; ts=time.time(); depth_frame_count=0
        while self.running:
            t0 = time.time()
            ret,frame = self.cap.read()
            if not ret:
                with self.print_lock: print("\r\n[Pipeline] Video ended — shutting down")
                self.running = False
                break

            # Rotate portrait frames to landscape (phone video recorded vertically)
            if frame.shape[0] > frame.shape[1]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # YOLO (MARS terrain + hazard detection)
            try:
                detections = self.yolo.detect(frame) if self.yolo else []
            except Exception as e:
                with self.print_lock: print(f"\r\n[WARN] YOLO err ({e})")
                detections = []

            # Depth — run every 3rd frame, reuse last result otherwise
            depth_frame_count += 1
            if self.depth and not getattr(self.depth,'broken',False) and depth_frame_count % 3 == 1:
                try:
                    depth_map = self.depth.estimate(frame)
                    with self.depth_lock: self.latest_depth = depth_map
                except Exception as e:
                    with self.print_lock: print(f"\r\n[WARN] Depth err ({e})")
            with self.depth_lock:
                depth_map = self.latest_depth

            sensor = self.arduino.get_latest()

            # COCO unknown-object tagging
            if self.coco:
                try:
                    coco_dets  = self.coco.detect(frame)
                    detections = tag_unknown_objects(detections, coco_dets)
                except Exception as e:
                    with self.print_lock: print(f"\r\n[WARN] COCO err ({e})")

            # Fusion
            try:
                result = fuse(detections, depth_map, sensor)
            except Exception as e:
                with self.print_lock: print(f"\r\n[WARN] Fusion err ({e})")
                time.sleep(0.1); continue

            self.history.append(result)
            result.detections = estimate_motion(result.detections, self.history)

            with self.fusion_lock:  self.latest_fusion  = result
            with self.command_lock: cmd = self.latest_command

            # Record
            if self.recorder and cmd:
                try:
                    self.recorder.write(frame,cmd,sensor,result,depth_map,
                                        self._ym,self._dm)
                except Exception as e:
                    with self.print_lock: print(f"\r\n[WARN] Recorder err ({e})")

            fc+=1
            fps = fc/(time.time()-ts)
            ms  = (time.time()-t0)*1000

            det_strs=[]
            for d in result.detections:
                is_unk  = d.class_name.startswith("[")
                is_rock = d.class_name=="big_rock"
                tag     = " !" if is_rock else (" ?" if is_unk else "")
                motion  = f" v={d.velocity:+.2f}" if d.is_moving else ""
                show    = "unknown" if is_unk else d.class_name
                det_strs.append(f"{show}{tag}({d.confidence:.0%})"
                                 f" prox={d.depth_mean:.2f}{motion}")

            s=sensor
            with self.print_lock:
                print(
                    f"\r\n[Slow] {ms:.0f}ms  fps={fps:.1f}"
                    f"  terrain={result.terrain_class}"
                    f"  slope={'YES' if result.slope_detected else 'no'}"
                    f"  depth={'on' if depth_map is not None else 'OFF'}"
                    f"  conf={result.depth_confidence:.2f}"
                    f"  spd={s.speed:.2f}m/s"
                    f"\r\n  TOF({'TOF' if s.tof_ok else 'ULTRA'}):"
                    f" C={s.tof_distances[0]:.2f}"
                    f" LF={s.tof_distances[1]:.2f}"
                    f" RF={s.tof_distances[2]:.2f}"
                    f" LS={s.tof_distances[3]:.2f}"
                    f" RS={s.tof_distances[4]:.2f}"
                    f"\r\n  IMU: pitch={s.gyro_pitch:+.1f}"
                    f" roll={s.gyro_roll:+.1f}"
                    f" yaw={s.gyro_yaw:+.1f}"
                    f"  LDR={s.ldr_value:.2f}"
                    f"\r\n  Detections: "
                    f"{'  |  '.join(det_strs) if det_strs else 'none'}"
                    f"\r\n{'-'*60}",
                    end="\r\n"
                )

    # ------------------------------------------------------------------
    def _main_loop(self):
        if MOTORS_ENABLED: self._auto_loop()
        else:              self._manual_loop()

    def _auto_loop(self):
        try:
            while self.running:
                with self.command_lock: cmd=self.latest_command
                if cmd:
                    with self.print_lock:
                        print(f"\r\n[NAV] {cmd.action:<12}"
                              f" {cmd.reason:<40} conf={cmd.confidence:.2f}")
                    self.arduino.send_command(cmd.action)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            with self.print_lock: print("\r\n[Pipeline] Shutting down...")
            self._shutdown()

    def _manual_loop(self):
        if WINDOWS_MODE == 1:
            with self.print_lock:
                print("\r\n[Manual Mode] Running on Windows — press Ctrl+C to quit")
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                self._shutdown()
        else:
            KEY_MAP = {'w':'FORWARD','a':'TURN_LEFT','d':'TURN_RIGHT',
                       'q':'SLOW','s':'REVERSE',' ':'STOP'}
            with self.print_lock:
                print("\r\n[Manual Mode] Keyboard control active")
                print("  W=FORWARD  S=REVERSE  A=LEFT  D=RIGHT  Q=SLOW  Space=STOP")
                print("  Ctrl+C = Quit\r\n")
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while self.running:
                    ch = sys.stdin.read(1)
                    if ch == '\x03': break
                    action = KEY_MAP.get(ch.lower(), 'STOP')
                    self.arduino.send_command(action)
                    sys.stdout.write(f'\r\n[MANUAL] {action:<12}\r\n')
                    sys.stdout.flush()
            except KeyboardInterrupt:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                self.arduino.send_command('STOP')
                self._shutdown()

    def _shutdown(self):
        self.running = False
        if self.cap:      self.cap.release()
        if self.recorder: self.recorder.release()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        RoverPipeline().start()
    except KeyboardInterrupt:
        pass