"""
streaming/dashboard.py
======================
Streams live MIRA data to a web dashboard via Flask + SocketIO.

Streams:
  - Annotated camera frame (JPEG, ~5 FPS to save bandwidth)
  - All sensor readings (TOF, IMU, LDR, speed) as JSON
  - Current navigation command + confidence

Usage on Pi:
  pip install flask flask-socketio --break-system-packages
  python3 dashboard.py

Open in browser:  http://<pi-ip>:5000

This file imports from the pipeline and reads its shared state via the
same threading primitives — it does NOT re-run inference.
Run alongside fusion_pipeline_v3.py or standalone in display-only mode.
"""

import sys
import os
import time
import threading
import numpy as np
import cv2
from flask import Flask, render_template_string, Response
from flask_socketio import SocketIO

# Add pipeline folder to path so we can import shared state
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

# ── Flask + SocketIO setup ───────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "mira-sdp2"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Shared state (filled by pipeline thread, read by dashboard) ─
_latest_frame:   bytes     = b""
_latest_payload: dict      = {}
_state_lock = threading.Lock()


def update_dashboard_state(canvas: np.ndarray, sensor, cmd, fusion):
    """
    Called from the pipeline's slow loop after each frame.
    Encodes frame to JPEG and builds the sensor JSON payload.
    """
    global _latest_frame, _latest_payload

    # Encode frame to JPEG
    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _state_lock:
        _latest_frame = buf.tobytes()

    # Build sensor payload
    tof = sensor.tof_distances
    payload = {
        "tof": {
            "center":      round(tof[0], 2),
            "left_front":  round(tof[1], 2),
            "right_front": round(tof[2], 2),
            "left_side":   round(tof[3], 2),
            "right_side":  round(tof[4], 2),
            "tof_ok":      sensor.tof_ok,
        },
        "imu": {
            "pitch": round(sensor.gyro_pitch, 1),
            "roll":  round(sensor.gyro_roll,  1),
            "yaw":   round(sensor.gyro_yaw,   1),
        },
        "ldr":   round(sensor.ldr_value, 2),
        "speed": round(sensor.speed, 2),
        "nav": {
            "action":     cmd.action     if cmd else "N/A",
            "reason":     cmd.reason     if cmd else "",
            "confidence": round(cmd.confidence, 2) if cmd else 0.0,
        },
        "fusion": {
            "terrain":    fusion.terrain_class      if fusion else "N/A",
            "slope":      fusion.slope_detected     if fusion else False,
            "depth_conf": round(fusion.depth_confidence, 2) if fusion else 0.0,
        },
        "ts": round(time.time(), 3),
    }

    with _state_lock:
        _latest_payload = payload

    sio.emit("sensor_update", payload)


# ── MJPEG stream endpoint ────────────────────────────────────

def _frame_generator():
    while True:
        with _state_lock:
            frame = _latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.2)   # ~5 FPS to the browser


@app.route("/video_feed")
def video_feed():
    return Response(_frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── HTML dashboard ───────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>MIRA Dashboard</title>
  <meta http-equiv="refresh" content="0">
  <style>
    body { background:#111; color:#eee; font-family:monospace; margin:0; padding:12px; }
    h1   { color:#4af; margin:0 0 10px; font-size:1.2rem; }
    #container { display:flex; gap:12px; flex-wrap:wrap; }
    #video-box { flex:1; min-width:640px; }
    #video-box img { width:100%; border:1px solid #444; border-radius:4px; }
    #data-box  { flex:0 0 320px; display:flex; flex-direction:column; gap:8px; }
    .card { background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:10px; }
    .card h3 { margin:0 0 6px; color:#8cf; font-size:0.85rem; text-transform:uppercase; }
    .row { display:flex; justify-content:space-between; font-size:0.82rem; padding:2px 0; }
    .val { color:#ffe; font-weight:bold; }
    #nav-action { font-size:1.4rem; font-weight:bold; padding:8px; text-align:center;
                  border-radius:6px; background:#222; }
    .FORWARD { color:#4f4 !important; }
    .SLOW    { color:#fa0 !important; }
    .STOP    { color:#f44 !important; }
    .TURN_LEFT, .TURN_RIGHT { color:#4af !important; }
    #reason  { font-size:0.75rem; color:#aaa; text-align:center; margin-top:4px; }
  </style>
  <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
</head>
<body>
  <h1>MIRA — Live Dashboard</h1>
  <div id="container">
    <div id="video-box">
      <img src="/video_feed" alt="Camera Feed">
    </div>
    <div id="data-box">

      <div class="card">
        <div id="nav-action">—</div>
        <div id="reason"></div>
      </div>

      <div class="card">
        <h3>TOF Sensors (m)</h3>
        <div class="row"><span>Center-Front</span><span class="val" id="tof0">—</span></div>
        <div class="row"><span>Left-Front</span>  <span class="val" id="tof1">—</span></div>
        <div class="row"><span>Right-Front</span> <span class="val" id="tof2">—</span></div>
        <div class="row"><span>Left-Side</span>   <span class="val" id="tof3">—</span></div>
        <div class="row"><span>Right-Side</span>  <span class="val" id="tof4">—</span></div>
        <div class="row"><span>Status</span>      <span class="val" id="tof_ok">—</span></div>
      </div>

      <div class="card">
        <h3>IMU</h3>
        <div class="row"><span>Pitch</span><span class="val" id="pitch">—</span></div>
        <div class="row"><span>Roll</span> <span class="val" id="roll">—</span></div>
        <div class="row"><span>Yaw</span>  <span class="val" id="yaw">—</span></div>
      </div>

      <div class="card">
        <h3>Environment & Motion</h3>
        <div class="row"><span>Light (LDR)</span><span class="val" id="ldr">—</span></div>
        <div class="row"><span>Speed</span>      <span class="val" id="speed">—</span></div>
        <div class="row"><span>Terrain</span>    <span class="val" id="terrain">—</span></div>
        <div class="row"><span>Slope</span>      <span class="val" id="slope">—</span></div>
        <div class="row"><span>Depth Conf</span><span class="val" id="dconf">—</span></div>
      </div>

    </div>
  </div>

  <script>
    const s = io();
    s.on("sensor_update", function(d) {
      const t = d.tof, i = d.imu, n = d.nav, f = d.fusion;

      document.getElementById("tof0").textContent    = t.center      + " m";
      document.getElementById("tof1").textContent    = t.left_front  + " m";
      document.getElementById("tof2").textContent    = t.right_front + " m";
      document.getElementById("tof3").textContent    = t.left_side   + " m";
      document.getElementById("tof4").textContent    = t.right_side  + " m";
      document.getElementById("tof_ok").textContent  = t.tof_ok ? "✅ TOF OK" : "⚠️ ULTRA fallback";

      document.getElementById("pitch").textContent   = i.pitch + "°";
      document.getElementById("roll").textContent    = i.roll  + "°";
      document.getElementById("yaw").textContent     = i.yaw   + "°/s";

      document.getElementById("ldr").textContent     = d.ldr;
      document.getElementById("speed").textContent   = d.speed + " m/s";
      document.getElementById("terrain").textContent = f.terrain;
      document.getElementById("slope").textContent   = f.slope ? "⚠️ YES" : "—";
      document.getElementById("dconf").textContent   = f.depth_conf;

      const navEl = document.getElementById("nav-action");
      navEl.textContent = n.action;
      navEl.className   = n.action;
      document.getElementById("reason").textContent =
        n.reason + "  (conf=" + n.confidence + ")";
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── Standalone test mode (no pipeline) ──────────────────────
# When run directly, generates fake sensor data so you can test the dashboard
# without needing the full pipeline running.

def _fake_data_loop():
    import math, random
    t = 0
    class FakeSensor:
        tof_distances = [1.2, 2.0, 1.8, 0.9, 1.5]
        tof_ok = True
        gyro_pitch = 0.0
        gyro_roll  = 0.0
        gyro_yaw   = 0.0
        ldr_value  = 0.8
        ultra_front = 1.5
        speed = 0.0
    class FakeCmd:
        action = "FORWARD"
        reason = "Path clear (simulated)"
        confidence = 0.9
    class FakeFusion:
        terrain_class = "sand"
        slope_detected = False
        depth_confidence = 0.85

    sensor = FakeSensor()
    cmd    = FakeCmd()
    fusion = FakeFusion()

    # Generate a dummy canvas
    canvas = np.zeros((720, 1280, 3), np.uint8)
    cv2.putText(canvas, "MIRA — Standalone Dashboard Test", (40, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)

    while True:
        t += 0.1
        sensor.gyro_pitch = math.sin(t) * 5
        sensor.gyro_roll  = math.cos(t * 0.7) * 3
        sensor.speed      = abs(math.sin(t * 0.5)) * 0.8
        sensor.tof_distances[0] = 1.0 + 0.5 * math.sin(t * 0.3)
        cmd.action = random.choice(["FORWARD", "SLOW", "FORWARD", "FORWARD"])

        update_dashboard_state(canvas, sensor, cmd, fusion)
        time.sleep(0.2)


if __name__ == "__main__":
    print("[Dashboard] Starting standalone test mode — fake sensor data")
    print("[Dashboard] Open  http://localhost:5000  in your browser")
    t = threading.Thread(target=_fake_data_loop, daemon=True)
    t.start()
    sio.run(app, host="0.0.0.0", port=5000, debug=False)