# =============================================================================
# models.py — YOLO and Depth model classes + COCO unknown object tagging
#
# Classes:
#   YOLODetectorHailo   — Mars YOLO on Hailo-8L NPU (primary)
#   YOLODetectorONNX    — Mars YOLO on CPU via ONNX Runtime (fallback)
#   YOLODetectorCOCO    — COCO YOLO on CPU, always alongside Mars (unknown tagging)
#   DepthEstimatorHailo — DepthAnything V2 ViT-S on Hailo-8L NPU (primary)
#   DepthEstimatorONNX  — DepthAnything V2 Small on CPU via ONNX Runtime (fallback)
#
# Functions:
#   _decode_yolo()        — shared Mars decoder used by Hailo and ONNX YOLO
#   _compute_iou()        — IoU between two bounding boxes
#   tag_unknown_objects() — cross-checks COCO vs Mars detections, flags unmatched ones
# =============================================================================

import numpy as np
import cv2
from typing import List, Optional

from config import CONFIG, COCO_CLASSES, HAILO_AVAILABLE
from structures import Detection

if HAILO_AVAILABLE:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    )


# =============================================================================
# SHARED MARS DECODER
# =============================================================================

def _decode_yolo(raw: np.ndarray, orig_h: int, orig_w: int,
                 h_m: int, w_m: int) -> List[Detection]:
    """
    Decodes raw YOLO output (Mars model) into Detection objects.
    Shared between YOLODetectorHailo and YOLODetectorONNX — same output format.

    Raw row format: [cx, cy, w, h, score_soil, score_bedrock, score_sand, score_big_rock, ...]
    Coordinates are in model input space (0–640) and rescaled to original frame pixels.
    """
    classes    = CONFIG["classes"]
    detections = []
    for row in raw:
        scores = row[4: 4 + len(classes)]
        cid    = int(np.argmax(scores))
        conf   = float(scores[cid])
        if conf < 0.4:
            continue
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = max(0, int((cx - w / 2) * orig_w / w_m))
        y1 = max(0, int((cy - h / 2) * orig_h / h_m))
        x2 = min(orig_w, int((cx + w / 2) * orig_w / w_m))
        y2 = min(orig_h, int((cy + h / 2) * orig_h / h_m))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(
            class_id=cid,
            class_name=classes.get(cid, "unknown"),
            confidence=conf,
            bbox=(x1, y1, x2, y2),
        ))
    return detections


# =============================================================================
# YOLO — HAILO NPU (primary)
# =============================================================================

class YOLODetectorHailo:
    """
    YOLO26n-seg on Hailo-8L NPU via HEF.
    Input:  NHWC uint8 640×640 — no normalization (baked into HEF)
    Output: decoded Detection list in original frame pixel coordinates
    """

    def __init__(self, hef_path: str):
        import os
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"[WARN] YOLO HEF not found: {hef_path}")
        self.hef = HEF(hef_path)
        self.target = VDevice()
        cfg = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.target.configure(self.hef, cfg)[0]
        self.network_group_params = self.network_group.create_params()
        self.input_vstream_params  = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.size        = CONFIG["yolo_input"]
        self._input_name = self.hef.get_input_vstream_infos()[0].name
        print("[YOLO] Loaded on Hailo-8L NPU")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        h_m, w_m = self.size
        img = cv2.resize(frame, self.size)   # NHWC uint8, no normalization
        with InferVStreams(self.network_group, self.input_vstream_params,
                          self.output_vstream_params) as pipeline:
            with self.network_group.activate(self.network_group_params):
                raw_output = pipeline.infer({self._input_name: np.expand_dims(img, 0)})
        raw = list(raw_output.values())[0][0]
        if raw.ndim == 3:
            raw = raw[0]
        return _decode_yolo(raw, orig_h, orig_w, h_m, w_m)


# =============================================================================
# YOLO — ONNX CPU (fallback)
# =============================================================================

class YOLODetectorONNX:
    """
    YOLO26n-seg on CPU via ONNX Runtime.
    Input:  NCHW float32 640×640 normalized 0–1
    Output: decoded Detection list in original frame pixel coordinates
    """

    def __init__(self, model_path: str):
        import os, onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] YOLO ONNX model not found: {model_path}\n"
                f"        Place yolo26n_mars.onnx in models/yolo26/ and retry."
            )
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size       = CONFIG["yolo_input"]
        print(f"[YOLO] Loaded on CPU ONNX: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        h_m, w_m = self.size
        img = cv2.resize(frame, self.size).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)   # NCHW
        raw = self.session.run(None, {self.input_name: img})[0]
        if raw.ndim == 3:
            raw = raw[0]
        return _decode_yolo(raw, orig_h, orig_w, h_m, w_m)


# =============================================================================
# YOLO — COCO CPU (unknown object detection, optional)
# =============================================================================

class YOLODetectorCOCO:
    """
    yolo26n-seg.onnx (COCO 80-class) on CPU via ONNX Runtime.
    Always runs on CPU even when Mars YOLO is on Hailo — separate silicon, no conflict.
    Only loaded when COCO_ENABLED=1 AND YOLO is confirmed on Hailo.

    Output format: (1, 300, 38) — built-in NMS, xyxy coordinates
    Row: [x1, y1, x2, y2, confidence, class_id, mask×32]
    Coordinates are in model input space (0–640).
    """

    def __init__(self, model_path: str):
        import os, onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[WARN] COCO ONNX model not found: {model_path}\n"
                f"       Unknown object tagging disabled. Place yolo26n-seg.onnx in models/yolo26/ to enable."
            )
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size       = CONFIG["yolo_input"]
        print(f"[COCO] Loaded on CPU ONNX: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        h_m, w_m = self.size
        img = cv2.resize(frame, self.size).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)   # NCHW float32

        raw = self.session.run(None, {self.input_name: img})[0]
        if raw.ndim == 3:
            raw = raw[0]   # (300, 38)

        detections = []
        for row in raw:
            conf     = float(row[4])
            class_id = int(row[5])
            if conf < CONFIG["unknown_conf_thresh"]:
                continue
            x1 = max(0, min(orig_w, int(row[0] * orig_w / w_m)))
            y1 = max(0, min(orig_h, int(row[1] * orig_h / h_m)))
            x2 = max(0, min(orig_w, int(row[2] * orig_w / w_m)))
            y2 = max(0, min(orig_h, int(row[3] * orig_h / h_m)))
            if x2 <= x1 or y2 <= y1:
                continue
            coco_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"coco_{class_id}"
            detections.append(Detection(
                class_id=class_id,
                class_name=coco_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))
        return detections


# =============================================================================
# UNKNOWN OBJECT TAGGING
# =============================================================================

def _compute_iou(boxA: tuple, boxB: tuple) -> float:
    """Intersection over Union between two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def tag_unknown_objects(
    mars_detections: List[Detection],
    coco_detections: List[Detection],
) -> List[Detection]:
    """
    Cross-checks COCO detections against Mars detections using IoU.
    Any COCO box with IoU < unknown_iou_thresh against ALL Mars boxes
    → relabeled as "unknown_object" and appended to the detection list.

    Mars detections are never removed — unknowns are purely additive.
    decide() and VideoRecorder treat unknown_object like big_rock for navigation.
    """
    thresh   = CONFIG["unknown_iou_thresh"]
    unknowns = []
    for coco_det in coco_detections:
        matched = any(
            _compute_iou(coco_det.bbox, mars_det.bbox) >= thresh
            for mars_det in mars_detections
        )
        if not matched:
            unknowns.append(Detection(
                class_id=-1,
                class_name="unknown_object",
                confidence=coco_det.confidence,
                bbox=coco_det.bbox,
            ))
    return mars_detections + unknowns


# =============================================================================
# DEPTH — HAILO NPU (primary)
# =============================================================================

class DepthEstimatorHailo:
    """
    DepthAnything V2 ViT-S on Hailo-8L NPU.
    Input:  UINT8 NHWC 224×224×3 — center-cropped, no normalization (baked into HEF)
    Output: normalized 0–1 proximity map (1=closest) at original frame resolution

    Confirmed via hailortcli parse-hef:
      Input  scdepthv3/input_layer1  UINT8  NHWC(224×224×3)
      Output conv31                  UINT16 NHWC(224×224×1)
    """

    def __init__(self, hef_path: str):
        import os
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"[WARN] Depth HEF not found: {hef_path}")
        self.hef = HEF(hef_path)
        self.target = VDevice()
        cfg = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.target.configure(self.hef, cfg)[0]
        self.network_group_params = self.network_group.create_params()
        self.input_vstream_params  = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.size        = CONFIG["depth_input"]
        self._input_name = self.hef.get_input_vstream_infos()[0].name
        self._smooth: Optional[np.ndarray] = None
        self._alpha  = CONFIG["depth_smooth_alpha"]
        print("[Depth] DepthAnything V2 ViT-S loaded on Hailo-8L NPU")

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        """Crops center square to avoid distortion when resizing 1280×720 → 224×224."""
        h, w = frame.shape[:2]
        s = min(h, w)
        return frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(self._center_crop(frame), self.size)   # NHWC uint8
        with InferVStreams(self.network_group, self.input_vstream_params,
                          self.output_vstream_params) as pipeline:
            with self.network_group.activate(self.network_group_params):
                raw_output = pipeline.infer({self._input_name: np.expand_dims(img, 0)})
        depth = list(raw_output.values())[0][0].squeeze().astype(np.float32)
        return self._postprocess(depth, orig_w, orig_h)

    def _postprocess(self, depth: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        # EMA temporal smoothing — suppresses per-frame normalization flicker
        if self._smooth is None:
            self._smooth = depth
        else:
            self._smooth = self._alpha * depth + (1.0 - self._alpha) * self._smooth
        d_min, d_max = self._smooth.min(), self._smooth.max()
        norm = (self._smooth - d_min) / (d_max - d_min + 1e-6)
        return cv2.resize(norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# DEPTH — ONNX CPU (fallback)
# =============================================================================

class DepthEstimatorONNX:
    """
    DepthAnything V2 Small on CPU via ONNX Runtime.
    Input:  NCHW float32 224×224 — ImageNet-normalized, RGB
    Output: normalized 0–1 proximity map at original frame resolution
    """

    def __init__(self, model_path: str):
        import os, onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] Depth ONNX model not found: {model_path}\n"
                f"        Place depth_anything_v2_small.onnx in models/depthAnything/ and retry."
            )
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size       = CONFIG["depth_input"]
        self._smooth: Optional[np.ndarray] = None
        self._alpha  = CONFIG["depth_smooth_alpha"]
        print(f"[Depth] Loaded on CPU ONNX: {model_path}")

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        s = min(h, w)
        return frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(self._center_crop(frame), self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]   # ImageNet norm
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)   # NCHW
        depth = self.session.run(None, {self.input_name: img})[0]
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]
        return self._postprocess(depth.astype(np.float32), orig_w, orig_h)

    def _postprocess(self, depth: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        if self._smooth is None:
            self._smooth = depth
        else:
            self._smooth = self._alpha * depth + (1.0 - self._alpha) * self._smooth
        d_min, d_max = self._smooth.min(), self._smooth.max()
        norm = (self._smooth - d_min) / (d_max - d_min + 1e-6)
        return cv2.resize(norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)