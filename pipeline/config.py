# =============================================================================
# config.py — Top-level flags, CONFIG dict, COCO class list
# Change ARDUINO_ENABLED / RECORD_ENABLED / COCO_ENABLED here only.
# All other files import from this module — nothing else needs changing.
# =============================================================================

# ── Runtime flags ─────────────────────────────────────────────────────────────
ARDUINO_ENABLED = 0   # 0 = dummy sensor readings  | 1 = real Arduino serial
RECORD_ENABLED  = 1   # 0 = display only            | 1 = write AVI to logs/
COCO_ENABLED    = 1   # 0 = Mars model only         | 1 = also run COCO on CPU
MOTORS_ENABLED  = 0   # 0 = manual keyboard control (W/A/S/D + Space)
                      # 1 = autonomous — pipeline drives motors from navigation decisions
                      # Auto-disabled if YOLO falls back to CPU (would double CPU load)

# ── Hailo availability ────────────────────────────────────────────────────────
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[Boot] hailo_platform not found — will use ONNX CPU for both models")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Model paths — Hailo HEF (primary)
    "yolo_path_hef":  "models/yolo26/yolo26n_mars.hef",
    "depth_path_hef": "models/depthAnything/depth_anything_v2_vits.hef",

    # Model paths — ONNX CPU fallback
    "yolo_path_onnx":  "models/yolo26/yolo26n_mars.onnx",
    "depth_path_onnx": "models/depthAnything/depth_anything_v2_small.onnx",

    # COCO model — always CPU, alongside Mars on Hailo (separate silicon)
    "coco_path_onnx":      "models/yolo26/yolo26n-seg.onnx",
    "unknown_iou_thresh":  0.3,    # COCO box IoU < this vs all Mars boxes → unknown_object
    "unknown_conf_thresh": 0.35,   # min COCO confidence to consider a detection

    # Camera
    "camera_index": 0,
    "capture_w":    1280,
    "capture_h":    720,
    "capture_fps":  30,

    # Model input sizes
    "yolo_input":  (640, 640),
    "depth_input": (224, 224),   # DA V2 ViT-S HEF fixed size

    # Recording
    "record_output": "logs/fusion_output.avi",
    "record_fps":    8,          # match actual slow-loop throughput (Hailo:8–13, CPU:2)

    # TOF zone mapping — normalized camera coords (cx, cy, w, h)
    # Sensors 3 and 4 are side-facing — no camera FOV overlap → None
    "tof_zones": {
        0: (0.50, 0.60, 0.20, 0.30),   # Center-front
        1: (0.25, 0.55, 0.20, 0.30),   # Left-front ~30°
        2: (0.75, 0.55, 0.20, 0.30),   # Right-front ~30°
        3: None,
        4: None,
    },

    # Depth validation thresholds (meters)
    "depth_agree": 0.3,   # strong agreement  → blend
    "depth_warn":  1.0,   # moderate          → scale; above = discard

    # Depth temporal smoothing (EMA)
    "depth_smooth_alpha": 0.7,   # 70% current + 30% previous

    # Light thresholds (LDR)
    "ldr_good": 0.6,    # above → trust depth model 70%
    "ldr_low":  0.25,   # below → trust depth model 0%, TOF only

    # Gyroscope limits (degrees)
    "pitch_slow": 10, "pitch_stop": 20,
    "roll_slow":  15, "roll_stop":  25,

    # Obstacle distances (meters)
    "obs_stop":    0.5,
    "obs_slow":    1.5,
    "obs_clear":   3.0,
    "side_danger": 0.8,

    # Ultrasonic bumper fallback — only when tof_ok=False
    "ultra_stop": 0.3,

    # Slope detection
    "slope_drop": 0.8,   # proximity increase in lower 30% of frame

    # Motion tracking
    "history_len":      5,
    "approach_thresh": -0.5,   # m/s — negative = approaching

    # Mars terrain classes
    "classes": {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"},
}

# ── COCO class list (80 classes) ──────────────────────────────────────────────
# Used only for unknown object tagging — the specific class name is discarded.
# Any unmatched COCO detection becomes "unknown_object" in the pipeline.
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
    "toothbrush"
]