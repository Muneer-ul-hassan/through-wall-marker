"""Wall-marker laptop server — PRODUCT server (live drone frames only).

Usage:
    python -m server.main --mode live --port 8080
    python -m server.main --mode live --port 8443 \\
        --https --cert certs/cert.pem --key certs/key.pem

The product server serves LIVE drone frames only. It refuses to start
without a real source: ``--mode live`` needs djitellopy + a reachable
Tello, and exits with a clear error otherwise — it never fakes a run.
There is NO virtual / scenario mode in this file. The virtual harness
(scenario frames, synthetic fallback, VIO ground-test feed) lives in
the test-only entry point: ``python -m tests.virtual_server``
(see tests/virtual_server.py). Ground / virtual only in this build.

Routes:
    GET  /            -> 302 to /viewer
    GET  /viewer      -> phone/viewer.html
    GET  /operator    -> laptop operator page (BRIDGE / DRONE / PERSON)
    GET  /api/status  -> JSON {bridge_valid, drones, tracks, markers,
                        uptime_s, last_frame_age_s, frames_last_10s,
                        running, paused}
    POST /api/control -> {"action": "start"|"stop"} — pause/resume the
                        pipeline; while paused the marker stream
                        honestly reports no marker (reason "paused").
    GET  /api/markers -> SSE: `data: {"type":"markers","markers":[...]}`
                        (up to 3 people, nearest first) when markers
                        exist, otherwise `data: {"type":"none",...}` at
                        2 Hz, plus a `: ping` heartbeat every 5 s.
                        JSON ONLY — no video bytes on any route.

``last_frame_age_s`` and ``frames_last_10s`` are null until the first
frame has been processed — the operator page shows "—" for them, never
a made-up 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import ssl
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# Make `server.*` and `tests.*` importable when run as `python -m server.main`.
WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server import bridge as bridge_mod          # noqa: E402
from server import depth as depth_mod            # noqa: E402
from server import fusion as fusion_mod          # noqa: E402
from server.geometry import DRONE_K              # noqa: E402

log = logging.getLogger(__name__)
FPS_DEFAULT = 10.0
DRONE_ID_DEFAULT = "drone1"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """One drone's video -> bridge -> fusion -> marker state."""

    def __init__(self, detector, drone_id=DRONE_ID_DEFAULT,
                 phone_position=None, mono_depth=None, mono_every_n=10,
                 mono_max_age_s=2.0):
        self.detector = detector
        self.drone_id = drone_id
        self.bridge = bridge_mod.Bridge()
        self.fusion = fusion_mod.Fusion(self.bridge,
                                        phone_position=phone_position)
        self._histories = {}  # person_id -> [(drone_pose, bbox, t, foot_uv)]
        # person_id -> {"foot": (3,) last foot point, "depth": last depth}
        self._person_state = {}
        self._next_person_id = 0
        self._current_person_id = None
        # Last foot point + depth, for the person association below.
        self._last_depth = None
        self._last_foot = None
        self.frames_processed = 0
        self.detections_seen = 0
        self.markers_emitted = 0
        self.methods = {}
        # Monocular depth model — third voter (server/mono_depth.py).
        # NEVER on the marker path: it runs in a background thread, at
        # most every ``mono_every_n``th detection frame (default 10 —
        # 1 Hz at the 10 fps harness rate; measured 2026-11-23, every
        # 5th frame kept the worker busy ~100% of the time and cost
        # 7-28% of pipeline fps to CPU contention on this 1.5-CPU VM,
        # while disagreements the voter arbitrates persist for many
        # frames, so 1 Hz loses nothing), on a frame copy; the
        # pipeline only ever reads the latest completed result,
        # age-gated to ``mono_max_age_s`` (2.0 s — a vote refreshes
        # roughly every 1.0-1.5 s at this rate, and the sigma is
        # inflated 0.2 m/s of age so a stale vote counts for less).
        # If the worker is busy the frame is skipped, never queued.
        # ``mono_depth`` is None by default (two-method behaviour,
        # unchanged).
        self.mono_depth = mono_depth
        self.mono_every_n = int(mono_every_n)
        self.mono_max_age_s = float(mono_max_age_s)
        self._mono_lock = threading.Lock()
        self._mono_cache = None  # {"result": dict, "t": float}
        self._mono_busy = False
        self._mono_thread = None
        self.model_voter_changed = 0  # frames where the voter changed
        self.model_votes_used = 0     # frames where a model vote was used

    def shutdown(self, timeout=3.0):
        """Wait for a running model worker to finish (up to ``timeout``
        s). Call at the end of a run: abandoning a thread mid-inference
        while the interpreter shuts down aborted the process once
        ("terminate called without an active exception", 2026-11-23).
        Never blocks the marker path — only end-of-run cleanup."""
        thread = None
        with self._mono_lock:
            thread = self._mono_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _mono_worker(self, frame_copy, drone_pose, det, t):
        """Background model run: estimate + cache, never raises."""
        try:
            result = self.mono_depth.estimate(
                frame_copy, drone_pose, DRONE_K, det["keypoints"],
                det["kpt_conf"], det["bbox_xyxy"])
            if result is not None:
                with self._mono_lock:
                    self._mono_cache = {"result": result, "t": t}
        except Exception as exc:  # a model failure must never break markers
            log.warning("Mono depth worker failed: %s", exc)
        finally:
            with self._mono_lock:
                self._mono_busy = False

    def _maybe_submit_mono(self, frame, drone_pose, det, t):
        """Submit a model run for this detection frame, if due and the
        worker is free. Never blocks; a busy worker means skip."""
        if self.mono_depth is None or self.mono_every_n <= 0:
            return
        if self.frames_processed % self.mono_every_n != 0:
            return
        with self._mono_lock:
            if self._mono_busy:
                return
            self._mono_busy = True
        thread = threading.Thread(
            target=self._mono_worker,
            args=(frame.copy(), drone_pose, det, t), daemon=True)
        with self._mono_lock:
            self._mono_thread = thread
        thread.start()

    def _cached_model_vote(self, t):
        """Latest completed model vote, age-gated, or (None, None).

        The sigma is inflated for cache age (the drone moves while the
        cached vote ages): sigma_age = 0.2 m/s * age_s, added in
        quadrature — the vote gets less trusted the older it is, and
        expires after ``mono_max_age_s``.
        """
        if self.mono_depth is None:
            return None, None, None
        with self._mono_lock:
            cache = self._mono_cache
        if cache is None:
            return None, None, None
        age = float(t) - float(cache["t"])
        if age < 0 or age > self.mono_max_age_s:
            return None, None, None
        r = cache["result"]
        sigma = float(np.sqrt(r["sigma_m"] ** 2 + (0.2 * age) ** 2))
        return float(r["depth_m"]), sigma, age

    def process_frame(self, frame, drone_pose, tracking_valid, t):
        """Run one frame through the pipeline. Returns (marker, info)."""
        self.bridge.update(drone_pose, tracking_valid, t)
        detections = self.detector.detect(frame)
        self.last_detections = detections
        self.frames_processed += 1
        info = {"t": t, "n_det": len(detections), "depth": None,
                "method": None, "sigma": None,
                "bridge_valid": self.bridge.state()["valid"],
                "marker": None}
        if detections:
            self.detections_seen += len(detections)
            det = max(detections, key=lambda d: d["conf"])
            # Monocular depth model (third voter): submit this frame's
            # run in the background if due. The marker path never
            # waits for it — see _maybe_submit_mono / _cached_model_vote.
            self._maybe_submit_mono(frame, drone_pose, det, t)
            # Foot pixel: the visible-ankle ray for a partly occluded
            # person (None for a full person — see partial_foot_uv).
            foot_uv = depth_mod.partial_foot_uv(
                det["keypoints"], det["kpt_conf"])
            # Per-person depth history (two-person scenes, 2026-11-23):
            # the history must contain views of ONE person only. The
            # max-conf detection can alternate between two people, and
            # mixing their rays triangulates a ghost point between them
            # (measured in s19: a phantom track at (0, 7.4 m) and
            # triangulated depths of 0.01-0.22 m). So associate this
            # detection with the person whose last foot point is within
            # 1.0 m of this detection's rough foot point (its ray at the
            # last known depth), and keep a SEPARATE history per person.
            # The 1.0 m gate matches the fusion's association distance
            # (server/fusion.py). A moving person moves ~0.03 m/frame,
            # so this never fires on motion alone. A single person keeps
            # exactly one history — identical to the old behaviour.
            person_id = None
            if self._last_depth is not None and self._last_foot is not None:
                rough = depth_mod.foot_point(
                    det["bbox_xyxy"], DRONE_K, drone_pose,
                    self._last_depth, foot_uv=foot_uv)
                if rough is not None:
                    for pid, st in self._person_state.items():
                        if float(np.linalg.norm(rough - st["foot"])) <= 1.0:
                            person_id = pid
                            break
            if person_id is None:
                if self._current_person_id is not None and (
                        self._last_depth is None or self._last_foot is None):
                    # No rough foot to associate with (first detection
                    # after a gap): keep the previous person.
                    person_id = self._current_person_id
                else:
                    person_id = self._next_person_id
                    self._next_person_id += 1
                    self._histories[person_id] = []
                    self._person_state[person_id] = {
                        "foot": None, "depth": None}
            self._current_person_id = person_id
            history = self._histories.setdefault(person_id, [])
            history.append((drone_pose, det["bbox_xyxy"], t, foot_uv))
            self._histories[person_id] = history[-30:]
            try:
                model_depth, model_sigma, model_age = \
                    self._cached_model_vote(t)
                detail = depth_mod.estimate_person_depth_detailed(
                    self._histories[person_id], DRONE_K["fx"],
                    model_depth=model_depth, model_sigma=model_sigma)
                depth_m, method, sigma = (detail["depth_m"],
                                          detail["method"],
                                          detail["sigma_m"])
                info.update(model_depth=model_depth,
                            model_sigma=model_sigma,
                            model_age_s=model_age,
                            voter_changed=detail["voter_changed"],
                            tri_depth=detail["tri_depth_m"],
                            bbox_depth=detail["bbox_depth_m"])
                if model_depth is not None:
                    self.model_votes_used += 1
                if detail["voter_changed"]:
                    self.model_voter_changed += 1
            except ValueError:
                depth_m, method, sigma = None, None, None
            if depth_m is not None:
                info.update(depth=depth_m, method=method, sigma=sigma)
                self.methods[method] = self.methods.get(method, 0) + 1
                joints_cam = depth_mod.joints_3d(
                    det["keypoints"], det["kpt_conf"], DRONE_K,
                    drone_pose, depth_m)
                state = self.bridge.state()
                if state["valid"] and state["T_world_drone"] is not None:
                    joints_world = bridge_mod.world_from_drone(
                        state["T_world_drone"], joints_cam)
                    # Person position: the FOOT point (ground contact),
                    # reconstructed from the bbox bottom-centre ray at
                    # the estimated depth. The median of the joints sits
                    # ~1 m above the floor and biases the displayed
                    # distance ~0.5 m short (measured 2026-11-23); the
                    # foot point matches the ground-truth definition
                    # (distance to the person's ground position).
                    # Fall back to the joint median only if the foot
                    # reconstruction fails.
                    foot = depth_mod.foot_point(
                        det["bbox_xyxy"], DRONE_K, drone_pose, depth_m,
                        foot_uv=foot_uv)
                    valid = ~np.isnan(joints_world).any(axis=1)
                    if foot is not None:
                        pos = foot
                    elif valid.any():
                        pos = np.nanmedian(joints_world[valid], axis=0)
                    else:
                        pos = None
                    if pos is not None:
                        self.fusion.update(self.drone_id, [{
                            "pos": pos, "skeleton": joints_world,
                            "sigma": sigma, "conf": det["conf"],
                            "kpt_conf": det["kpt_conf"],
                        }], t)
                        self._last_depth = depth_m
                        self._last_foot = np.asarray(pos, dtype=float)
                        self._person_state[person_id] = {
                            "foot": np.asarray(pos, dtype=float),
                            "depth": depth_m}
                    else:
                        self.fusion.update(self.drone_id, [], t)
                else:
                    self.fusion.update(self.drone_id, [], t)
            else:
                self.fusion.update(self.drone_id, [], t)
        else:
            self.fusion.update(self.drone_id, [], t)
        marker = self.fusion.marker(t)
        if marker is not None:
            self.markers_emitted += 1
        info["marker"] = marker
        return marker, info


# ---------------------------------------------------------------------------
# Operator page (laptop only) — served at GET /operator
# ---------------------------------------------------------------------------
# Self-contained HTML, inline CSS/JS, no external resources. No video,
# no image, no per-joint table, no log tail. It polls /api/status once
# a second and shows only what the server really reports: before the
# first frame every fact reads "—", never a made-up 0.

OPERATOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wall Marker — Operator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; background: #111; color: #eee; padding: 24px; max-width: 720px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: #aaa; font-size: 14px; margin-bottom: 20px; }
  .blocks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
  .block { border: 1px solid #444; border-radius: 12px; padding: 16px; text-align: center; }
  .block .name { font-size: 12px; letter-spacing: 1px; color: #aaa; margin-bottom: 8px; }
  .block .state { font-size: 20px; font-weight: 700; }
  .block.ok { border-color: #2e7d32; } .block.ok .state { color: #66bb6a; }
  .block.bad { border-color: #c62828; } .block.bad .state { color: #ef5350; }
  .block.idle { border-color: #444; } .block.idle .state { color: #bbb; }
  #facts { background: #1c1c1c; border-radius: 12px; padding: 14px 16px; font-size: 15px; line-height: 1.9; margin-bottom: 20px; }
  #facts b { color: #fff; }
  h2 { font-size: 16px; margin-bottom: 10px; }
  #checklist { list-style: none; margin-bottom: 20px; }
  #checklist li { padding: 10px 0; border-bottom: 1px solid #2a2a2a; font-size: 16px; }
  #checklist li .tick { display: inline-block; width: 26px; font-weight: 700; }
  #checklist li.done .tick { color: #66bb6a; }
  #checklist li:not(.done) .tick { color: #777; }
  #controlBtn { width: 100%; height: 56px; border: none; border-radius: 12px; background: #fff; color: #000; font-size: 18px; font-weight: 700; cursor: pointer; }
  #controlBtn:active { background: #ddd; }
  #conn { text-align: center; color: #aaa; font-size: 14px; margin-top: 12px; }
  #conn.bad { color: #ef5350; }
</style>
</head>
<body>
<h1>Wall Marker — Operator</h1>
<p class="sub">Laptop view. Live state only — nothing here is stored or invented.</p>
<div class="blocks">
  <div class="block idle" id="blockBridge"><div class="name">BRIDGE</div><div class="state">—</div></div>
  <div class="block idle" id="blockDrone"><div class="name">DRONE</div><div class="state">—</div></div>
  <div class="block idle" id="blockPerson"><div class="name">PERSON</div><div class="state">—</div></div>
</div>
<div id="facts">
  Drones: <b id="fDrones">—</b><br>
  Tracks: <b id="fTracks">—</b><br>
  Markers: <b id="fMarkers">—</b><br>
  Uptime: <b id="fUptime">—</b><br>
  Last frame: <b id="fLastFrame">—</b><br>
  Frames (last 10 s): <b id="fFrames10">—</b>
</div>
<h2>Checklist</h2>
<ul id="checklist">
  <li id="cBridge"><span class="tick">○</span> Bridge linked</li>
  <li id="cDrone"><span class="tick">○</span> Drone frames flowing</li>
  <li id="cPerson"><span class="tick">○</span> Person found</li>
</ul>
<button id="controlBtn" type="button">Stop</button>
<div id="conn">Connecting…</div>
<script>
'use strict';
var blockBridge = document.getElementById('blockBridge');
var blockDrone = document.getElementById('blockDrone');
var blockPerson = document.getElementById('blockPerson');
var fDrones = document.getElementById('fDrones');
var fTracks = document.getElementById('fTracks');
var fMarkers = document.getElementById('fMarkers');
var fUptime = document.getElementById('fUptime');
var fLastFrame = document.getElementById('fLastFrame');
var fFrames10 = document.getElementById('fFrames10');
var cBridge = document.getElementById('cBridge');
var cDrone = document.getElementById('cDrone');
var cPerson = document.getElementById('cPerson');
var btn = document.getElementById('controlBtn');
var conn = document.getElementById('conn');
var running = true;

function setBlock(el, kind, text) {
  el.className = 'block ' + kind;
  el.querySelector('.state').textContent = text;
}
function setCheck(li, done) {
  li.className = done ? 'done' : '';
  li.querySelector('.tick').textContent = done ? '✓' : '○';
}
function dash(v) { return (v === null || v === undefined) ? '—' : String(v); }

function render(s) {
  var hasFrame = s.last_frame_age_s !== null && s.last_frame_age_s !== undefined;
  // BRIDGE: OK while the shared frame holds, LOST otherwise.
  setBlock(blockBridge, s.bridge_valid ? 'ok' : (hasFrame ? 'bad' : 'idle'),
           s.bridge_valid ? 'OK' : (hasFrame ? 'LOST' : '—'));
  // DRONE: LIVE while frames arrived in the last 10 s.
  var live = hasFrame && s.frames_last_10s !== null && s.frames_last_10s > 0;
  setBlock(blockDrone, live ? 'ok' : (hasFrame ? 'bad' : 'idle'),
           live ? 'LIVE' : (hasFrame ? 'NO VIDEO' : '—'));
  // PERSON: FOUND while a marker is live.
  var found = hasFrame && s.markers !== null && s.markers > 0;
  setBlock(blockPerson, found ? 'ok' : (hasFrame ? 'idle' : 'idle'),
           found ? 'FOUND' : (hasFrame ? 'NONE' : '—'));
  fDrones.textContent = (s.drones && s.drones.length) ? s.drones.join(', ') : '—';
  fTracks.textContent = hasFrame ? String(s.tracks) : '—';
  fMarkers.textContent = hasFrame ? String(s.markers) : '—';
  fUptime.textContent = (s.uptime_s === null || s.uptime_s === undefined) ? '—' : s.uptime_s + ' s';
  fLastFrame.textContent = hasFrame ? s.last_frame_age_s + ' s ago' : '—';
  fFrames10.textContent = hasFrame ? String(s.frames_last_10s) : '—';
  setCheck(cBridge, !!s.bridge_valid);
  setCheck(cDrone, live);
  setCheck(cPerson, found);
  running = s.running !== false;
  btn.textContent = running ? 'Stop' : 'Start';
  conn.textContent = s.paused ? 'Paused' : 'Running';
  conn.className = '';
}

function poll() {
  fetch('/api/status')
    .then(function (r) { return r.json(); })
    .then(function (s) { render(s); })
    .catch(function () {
      conn.textContent = 'No connection';
      conn.className = 'bad';
    });
}

btn.addEventListener('click', function () {
  var action = running ? 'stop' : 'start';
  fetch('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: action })
  })
    .then(function (r) { return r.json(); })
    .then(function (s) { running = !!s.running; poll(); })
    .catch(function () {
      conn.textContent = 'No connection';
      conn.className = 'bad';
    });
});

poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class ServerState:
    """Shared pipeline state for the HTTP handlers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.marker = None  # single marker (backward compatible payload)
        self.markers = []   # markers array: up to 3, nearest first
        self.bridge_valid = False
        self.drones = set()
        self.tracks = 0
        self.started = time.time()
        self.last_info = {}
        # Wall-clock frame accounting for /api/status. Both stay None
        # until the first frame has been processed — the operator page
        # shows "—" for them, never a made-up 0.
        self.last_frame_wall_t = None
        self.frame_times = deque()  # wall times of frames, last 10 s only
        self.running = True   # pipeline control (operator Start/Stop)
        self.paused = False   # True while the pipeline is stopped

    def update_from_pipeline(self, pipeline, marker, info, wall_t,
                             scenario_t):
        """Record one processed frame. Called by the pipeline loop
        (live loop in the product server, virtual loop in the
        test-only entry point)."""
        with self.lock:
            self.marker = marker
            self.markers = pipeline.fusion.markers(scenario_t)
            self.bridge_valid = info["bridge_valid"]
            self.drones = {pipeline.drone_id}
            self.tracks = len(pipeline.fusion.tracks)
            self.last_info = info
            self.last_frame_wall_t = wall_t
            self.frame_times.append(wall_t)
            cutoff = wall_t - 10.0
            while self.frame_times and self.frame_times[0] < cutoff:
                self.frame_times.popleft()
            self.paused = False

    def mark_paused(self):
        """Clear the marker state while the pipeline is stopped, so
        the SSE stream honestly reports no marker (reason "paused")
        instead of re-sending the last live one."""
        with self.lock:
            self.marker = None
            self.markers = []
            self.paused = True

    def status_payload(self):
        """The /api/status JSON. ``last_frame_age_s`` and
        ``frames_last_10s`` are None until the first frame."""
        with self.lock:
            if self.last_frame_wall_t is None:
                last_age = None
                frames_10s = None
            else:
                now = time.time()
                last_age = round(now - self.last_frame_wall_t, 1)
                cutoff = now - 10.0
                frames_10s = sum(1 for t in self.frame_times
                                 if t >= cutoff)
            return {
                "bridge_valid": self.bridge_valid,
                "drones": sorted(self.drones),
                "tracks": self.tracks,
                "markers": len(self.markers),
                "uptime_s": round(time.time() - self.started, 1),
                "last_frame_age_s": last_age,
                "frames_last_10s": frames_10s,
                "running": self.running,
                "paused": self.paused,
            }


def make_handler(state, viewer_path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout for pipeline output
            pass

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/viewer")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path == "/viewer":
                if viewer_path.exists():
                    self._send(200, viewer_path.read_bytes(), "text/html")
                else:
                    note = ("<html><body><h1>Wall-marker viewer</h1>"
                            "<p>phone/viewer.html has not been built yet.</p>"
                            "<p>The server is running: /api/status and "
                            "/api/markers are live.</p></body></html>")
                    self._send(200, note, "text/html")
            elif self.path == "/operator":
                self._send(200, OPERATOR_HTML, "text/html")
            elif self.path == "/api/status":
                self._send(200, json.dumps(state.status_payload()),
                           "application/json")
            elif self.path == "/api/markers":
                self._serve_sse()
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/control":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {}
                action = body.get("action")
                if action == "start":
                    with state.lock:
                        state.running = True
                    self._send(200, json.dumps({"running": True}),
                               "application/json")
                elif action == "stop":
                    with state.lock:
                        state.running = False
                    self._send(200, json.dumps({"running": False}),
                               "application/json")
                else:
                    self._send(400, json.dumps(
                        {"error": "action must be 'start' or 'stop'"}),
                        "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_ping = time.time()
            try:
                while True:
                    with state.lock:
                        markers = list(state.markers)
                        bridge_valid = state.bridge_valid
                        paused = state.paused
                        n_drones = len(state.drones)
                    if markers:
                        # Markers array: up to 3 people, nearest first
                        # (the list is already ordered by fusion), with
                        # the drone count when more than one drone is
                        # feeding the pipeline. The single-marker
                        # payload stays accepted by the viewer; the
                        # stream now sends the array.
                        payload = json.dumps({
                            "type": "markers",
                            "markers": markers,
                            "drones": n_drones,
                            "timestamp": time.time(),
                        })
                    else:
                        if paused:
                            reason = "paused"
                        elif not bridge_valid:
                            reason = "bridge_invalid"
                        else:
                            reason = "no_person"
                        payload = json.dumps({
                            "type": "none",
                            "bridge_valid": bridge_valid,
                            "reason": reason,
                            "timestamp": time.time(),
                        })
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if time.time() - last_ping >= 5.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
                    time.sleep(0.5)
            except OSError:
                # Browser disconnected (BrokenPipe / reset /
                # WinError 10053 ConnectionAbortedError):
                # close quietly, no traceback.
                pass

    return Handler


class RobustThreadingHTTPServer(ThreadingHTTPServer):
    def get_request(self):
        while True:
            try:
                return super().get_request()
            except (ssl.SSLError, OSError):
                continue

    def handle_error(self, request, client_address):
        # A phone browser closing the viewer mid-request aborts the
        # connection (WinError 10053). That is normal, not a server
        # fault: swallow it instead of dumping a traceback.
        # (2026-11-24 live run: the 10053 traceback in the log came
        # from this path, not from _serve_sse.)
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError,
                            ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def run_server(state, viewer_path, host, port, use_https, cert, key):
    handler = make_handler(state, viewer_path)
    httpd = RobustThreadingHTTPServer((host, port), handler)
    if use_https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    scheme = "https" if use_https else "http"
    log.info("Serving %s://%s:%d/  (viewer at /viewer, operator at "
             "/operator, SSE at /api/markers)",
             scheme, host, port)
    httpd.serve_forever()


# SafeFlightController lives at module level so tests can import it.
# It is the other AI's controller (commit d89be655), kept as-is except
# for the two completions marked below (land idempotency, Not-joystick
# guidance) -- there is ONE flight controller in this file.
class SafeFlightController:
    def __init__(self, tello, poll_interval_s=0.5, ground_grace_s=2.0,
                 ground_height_cm=10):
        self.tello = tello
        self.lock = threading.Lock()
        self.status = "LANDED"  # LANDED, TAKING_OFF, HOVERING, MOVING, LANDING
        self.last_action = "Ready"
        # Ground watchdog (2026-11-25): the drone can land itself —
        # low-battery failsafe, blinking red light — without telling
        # us, so status stays HOVERING and [T] is ignored ("Takeoff
        # ignored: drone already HOVERING"). A daemon thread polls
        # the height while HOVERING; ~2 s near 0 cm means it is on
        # the ground, so status resets to LANDED and [T] works again.
        # It never fires while TAKING_OFF / MOVING / LANDING, and
        # never resets while the drone is actually airborne (height
        # clearly > 0). Only starts when the tello reports a height
        # (djitellopy does; the test fakes without get_height do not).
        self._wd_poll_s = float(poll_interval_s)
        self._wd_grace_s = float(ground_grace_s)
        self._wd_ground_cm = int(ground_height_cm)
        self._wd_stop = threading.Event()
        self._wd_ground_since = None
        if tello is not None and getattr(tello, "get_height", None) is not None:
            threading.Thread(target=self._ground_watchdog, daemon=True).start()

    def _ground_watchdog(self):
        """Poll the height; reset HOVERING -> LANDED when the drone
        has been on the ground for ~2 s. Every height read is
        guarded: a failed or missing read is skipped, never counted
        toward the ground streak."""
        while not self._wd_stop.is_set():
            self._wd_stop.wait(self._wd_poll_s)
            if self._wd_stop.is_set():
                return
            if self.status != "HOVERING":
                self._wd_ground_since = None
                continue
            try:
                height = self.tello.get_height()
            except Exception:
                self._wd_ground_since = None
                continue
            if height is None:
                self._wd_ground_since = None
                continue
            try:
                height = int(height)
            except (TypeError, ValueError):
                self._wd_ground_since = None
                continue
            if height > self._wd_ground_cm:
                self._wd_ground_since = None
                continue
            now = time.monotonic()
            if self._wd_ground_since is None:
                self._wd_ground_since = now
                continue
            if now - self._wd_ground_since < self._wd_grace_s:
                continue
            # On the ground for ~2 s while HOVERING: re-check under
            # the lock (a command may have started meanwhile) and
            # reset so [T] works again.
            if self.lock.acquire(blocking=False):
                try:
                    if self.status == "HOVERING":
                        self.status = "LANDED"
                        self.last_action = "Landed (ground watchdog)"
                        log.info("Ground watchdog: drone is on the ground "
                                 "(height %d cm) but status was HOVERING "
                                 "-- status reset to LANDED, [T] works "
                                 "again", height)
                finally:
                    self.lock.release()
            self._wd_ground_since = None

    def send_command(self, cmd_name, func, *args):
        if not self.tello:
            return
        if not self.lock.acquire(blocking=False):
            log.warning("Drone busy (%s). Skipped command '%s'", self.last_action, cmd_name)
            return

        def _worker():
            try:
                self.last_action = cmd_name
                if cmd_name == "takeoff":
                    if self.status != "LANDED":
                        log.warning("Takeoff ignored: drone already %s", self.status)
                        return
                    self.status = "TAKING_OFF"
                    log.info(">>> TAKEOFF starting... Please wait (do NOT press other keys)")
                    self.tello.takeoff()
                    self.status = "HOVERING"
                    self.last_action = "Hovering"
                    log.info(">>> Tello is airborne and HOVERING! Safe to move with [W/S/A/D]")
                elif cmd_name == "land":
                    if self.status == "LANDED":
                        log.debug("Land ignored: drone already LANDED")
                        return
                    self.status = "LANDING"
                    log.info(">>> LAND command sent...")
                    self.tello.land()
                    self.status = "LANDED"
                    self.last_action = "Landed"
                    log.info(">>> Tello LANDED safely.")
                else:
                    if self.status != "HOVERING":
                        log.warning("Cannot '%s': Drone is %s (Takeoff first with [T])", cmd_name, self.status)
                        return
                    self.status = "MOVING"
                    log.info(">>> Executing flight move: %s...", cmd_name)
                    func(*args)
                    self.status = "HOVERING"
                    self.last_action = "Hovering"
            except Exception as exc:
                log.error("Flight command '%s' returned error: %s", cmd_name, exc)
                msg = str(exc).lower()
                if "not joystick" in msg or "no valid imu" in msg:
                    # The drone is no longer under SDK control (the Tello
                    # app grabbed it mid-flight: "Not joystick") or it is
                    # down (the IMU is invalid when not flying). Either
                    # way the HOVERING state cannot be trusted: reset to
                    # LANDED so [T] works again instead of being ignored
                    # as "already HOVERING". (2026-11-24 live run: a
                    # forward 30 failed with both errors in a row and the
                    # drone was on the ground while the controller still
                    # said HOVERING, so five [T] presses were ignored.)
                    kind = "TAKEOFF" if cmd_name == "takeoff" else "MOVE"
                    log.error(
                        "%s REJECTED (%s): the Tello app on the phone "
                        "is holding the controls, or the drone is down. "
                        "Force-close the Tello app on the phone, reboot "
                        "the drone, then restart the server and press "
                        "[T] again.", kind, exc)
                    self.status = "LANDED"
                elif self.status == "TAKING_OFF":
                    self.status = "LANDED"
                elif self.status == "MOVING":
                    self.status = "HOVERING"
                self.last_action = f"Error: {exc}"
            finally:
                self.lock.release()

        threading.Thread(target=_worker, daemon=True).start()

def live_pose_step(vio, frame, start_pose, t):
    """Advance the live pose estimate by one frame.

    Returns (vio, pose, tracking_valid). Mirrors the virtual harness
    (tests/virtual_server.py): the VIO verdict passes through
    honestly. The start pose is only the VIO initialisation pose --
    it is NEVER returned as a valid pose. Before VIO initialises,
    and whenever VIO reports invalid or raises, tracking_valid is
    False (pose is the start pose, kept only so the per-person depth
    history stays well-formed; the bridge ignores it while invalid,
    so no marker is emitted).
    """
    if vio == "pending":
        from server.vio import MonocularVO
        vio = MonocularVO(start_pose=start_pose)
    if vio is not None and vio != "pending":
        try:
            pose, tracking_valid, _diag = vio.update(frame, t)
            return vio, pose, tracking_valid
        except Exception as exc:
            log.debug("VIO update failed: %s", exc)
    return vio, start_pose, False


def dispatch_drone_key(key, flight_ctrl):
    """Map one keyboard key to a flight command. Returns True if the key
    was a flight key, False otherwise (unknown key / no key / no drone).

    Key map (2026-11-25, the user's own choice): Q or T takeoff, E land,
    L turn right, J turn left. L is deliberately NOT land any more: an
    accidental L press mid-flight used to drop the drone. Q used to
    close the camera window — which only hid it (the program never
    crashed; the window went away and keys stopped working). Closing
    the window is now done with its X button (the caller watches for
    it); Ctrl+C in the console is the full exit and lands the drone.
    """
    tello = getattr(flight_ctrl, "tello", None)
    if tello is None:
        return False
    if key in (ord("q"), ord("t")):  # Takeoff
        flight_ctrl.send_command("takeoff", tello.takeoff)
    elif key == ord("e"):  # Land
        flight_ctrl.send_command("land", tello.land)
    elif key in (ord("w"), 82):  # W / Forward
        flight_ctrl.send_command("forward 30", tello.move_forward, 30)
    elif key in (ord("s"), 84):  # S / Back
        flight_ctrl.send_command("back 30", tello.move_back, 30)
    elif key in (ord("a"), 81):  # A / Left
        flight_ctrl.send_command("left 30", tello.move_left, 30)
    elif key in (ord("d"), 83):  # D / Right
        flight_ctrl.send_command("right 30", tello.move_right, 30)
    elif key == ord("i"):  # Up
        flight_ctrl.send_command("up 30", tello.move_up, 30)
    elif key == ord("k"):  # Down
        flight_ctrl.send_command("down 30", tello.move_down, 30)
    elif key == ord("j"):  # Rotate Left
        flight_ctrl.send_command("turn_left 30", tello.rotate_counter_clockwise, 30)
    elif key == ord("l"):  # Rotate Right
        flight_ctrl.send_command("turn_right 30", tello.rotate_clockwise, 30)
    else:
        return False
    return True


def format_status_line(battery, flight_status, bridge_valid, vio_diag,
                       vio_initialized, n_frames):
    """One-line console status for the live loop (~1 per 5 s).

    Battery % over time is the record the 2026-11-25 self-landing
    needed: a degraded battery sags under load, so the % across
    successive lines proves it (or rules it out) instead of a guess.
    The drone's own red light + shutdown is its fault/critical-battery
    signal — this code cannot switch the drone off or change its
    light — but the % trace shows whether the battery was sagging
    when it happened. Pure function (no I/O) so the tests can check
    the exact wording.
    """
    if battery == "" or battery is None:
        bat = "—"
    else:
        bat = f"{battery}%"
    bridge = "valid" if bridge_valid else "invalid"
    tracked = vio_diag.get("n_tracked", 0)
    inliers = vio_diag.get("inliers", 0)
    n_map = vio_diag.get("n_map", 0)
    source = vio_diag.get("pose_source", "—")
    if vio_initialized:
        vio_s = (f"init pose={source} tracked={tracked} "
                 f"inliers={inliers} map={n_map}")
    else:
        reason = vio_diag.get("init_fail") or "waiting"
        kf_age = vio_diag.get("init_kf_age_frames", "—")
        vio_s = (f"uninit init={reason} tracked={tracked} "
                 f"inliers={inliers} kf_age={kf_age}f")
    return (f"STATUS bat={bat} flight={flight_status} bridge={bridge} "
            f"vio={vio_s} frames={n_frames}")


def maybe_reset_vio_at_takeoff(vio, prev_status, status, start_pose):
    """Reset the VIO when the drone takes off (once per takeoff), so
    the keyframe is an airborne frame at the known start pose — not
    the pre-takeoff ground frame the server started with. Live
    2026-11-25: the keyframe was ~2.5 min stale (server started
    02:58:49, takeoff 03:01:23) and LK from it cannot survive the
    1.3 m rise, so init could never fire in any run. Only resets
    while the VIO is uninitialised; a working bridge is never wiped.
    Returns True if a reset happened. The caller keeps prev_status.
    """
    if (status == "HOVERING" and prev_status != "HOVERING"
            and vio is not None and vio != "pending"
            and not getattr(vio, "initialized", True)):
        vio.reset(start_pose)
        log.info("VIO reset at takeoff: new keyframe on next frame "
                 "(was a pre-takeoff ground frame)")
        return True
    return False


# ---------------------------------------------------------------------------
# CLI — product server: live drone frames only
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Wall-marker laptop server (live drone frames only)")
    ap.add_argument("--mode", choices=["live", "virtual"], default="live",
                    help="live only; --mode virtual is refused — use the "
                    "test-only entry point: python -m tests.virtual_server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--cert", default="certs/cert.pem")
    ap.add_argument("--key", default="certs/key.pem")
    # Refuse virtual mode before argparse runs: the old command line
    # (--mode virtual --scenario ...) no longer exists here, so a bare
    # --mode virtual would otherwise die in argparse without pointing
    # at the replacement entry point.
    raw = list(argv) if argv is not None else sys.argv[1:]
    if "virtual" in raw or "--scenario" in raw:
        print("ERROR: --mode virtual is not served by the product server "
              "(server/main.py serves live drone frames only). Use the "
              "test-only entry point instead: python -m tests.virtual_server "
              "--scenario s1_basic --port 8080",
              file=sys.stderr)
        return 2
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.mode == "virtual":
        print("ERROR: --mode virtual is not served by the product server "
              "(server/main.py serves live drone frames only). Use the "
              "test-only entry point instead: python -m tests.virtual_server "
              "--scenario s1_basic --port 8080",
              file=sys.stderr)
        return 2

    try:
        import djitellopy  # noqa: F401
    except ImportError:
        print("ERROR: --mode live needs djitellopy, which is not "
              "installed (real-run only). Use the test-only entry point "
              "for virtual runs: python -m tests.virtual_server "
              "--scenario s1_basic --port 8080, or install djitellopy "
              "and retry with a real Tello.",
              file=sys.stderr)
        return 2
    from server.video_source import TelloVideoSource
    try:
        source = TelloVideoSource()
    except RuntimeError as exc:
        print(f"ERROR: --mode live: {exc}", file=sys.stderr)
        return 2

    from server.detector import PoseDetector
    from server.geometry import COCO_EDGES
    import cv2

    def draw_drone_detections(frame, detections):
        vis = frame.copy()
        for det in detections:
            bbox = det.get("bbox_xyxy")
            conf = det.get("conf", 0.0)
            kpts = det.get("keypoints")
            kpt_conf = det.get("kpt_conf")
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"Person {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if kpts is not None:
                for i1, i2 in COCO_EDGES:
                    if kpt_conf is None or (kpt_conf[i1] > 0.3 and kpt_conf[i2] > 0.3):
                        p1 = (int(kpts[i1][0]), int(kpts[i1][1]))
                        p2 = (int(kpts[i2][0]), int(kpts[i2][1]))
                        cv2.line(vis, p1, p2, (0, 255, 0), 2)
                for i, (x, y) in enumerate(kpts):
                    if kpt_conf is None or kpt_conf[i] > 0.3:
                        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
        return vis

    detector = PoseDetector()
    detector.warmup()

    state = ServerState()
    pipeline = Pipeline(detector, drone_id="drone1")
    stop = threading.Event()
    vio = "pending"
    show_drone_window = True

    flight_ctrl = SafeFlightController(getattr(source, "tello", None))

    def pipeline_loop():
        nonlocal vio, show_drone_window
        t0 = time.monotonic()
        n_frames = 0
        last_status_t = time.monotonic()
        prev_flight_status = flight_ctrl.status
        last_battery = ""
        while not stop.is_set():
            if not state.running:
                state.mark_paused()
                time.sleep(0.1)
                continue
            item = source.read()
            if item is None:
                time.sleep(0.05)
                continue
            frame, pose, tracking_valid, meta = item
            src_pose = pose  # source start pose (for the takeoff reset)
            # Wall-clock scenario time (the real loop runs ~4-5 fps,
            # not the 10 fps the frame counter assumed; tracks stayed
            # "live" ~2x too long in wall time).
            t = time.monotonic() - t0
            # NO fallback: a failed VIO must never be papered over
            # with the frozen start pose. The start pose is only the
            # VIO initialisation pose; feeding it back as "tracking
            # valid" validated the bridge on a fiction frame and
            # emitted a garbage marker (2026-11-23 live run:
            # scrambled skeleton, 2.2 m for a person 0.5 m away).
            # With tracking invalid the bridge stays invalid and the
            # viewer honestly shows "Tracking lost" until VIO
            # initialises on real motion (takeoff + doorway flight).
            vio, pose, tracking_valid = live_pose_step(vio, frame, pose, t)
            marker, info = pipeline.process_frame(
                frame, pose, tracking_valid, t)
            state.update_from_pipeline(pipeline, marker, info,
                                       time.time(), t)
            n_frames += 1
            if meta.get("battery", "") != "":
                last_battery = meta.get("battery")
            # Takeoff reset (2026-11-25): the VIO keyframe must be an
            # airborne frame at the known start pose, not the
            # pre-takeoff ground frame the server started with — the
            # keyframe was ~2.5 min stale in every live run and init
            # could never fire from it.
            maybe_reset_vio_at_takeoff(vio, prev_flight_status,
                                       flight_ctrl.status, src_pose)
            prev_flight_status = flight_ctrl.status
            # Periodic console status (about one line per 5 s, never
            # per frame): battery % over time (a sag under load shows
            # up here), bridge/tracking state, VIO state + numbers,
            # frames processed.
            now = time.monotonic()
            if now - last_status_t >= 5.0:
                last_status_t = now
                vio_diag = {}
                if vio is not None and vio != "pending":
                    try:
                        vio_diag = vio.last_diag
                    except Exception:
                        vio_diag = {}
                log.info("%s", format_status_line(
                    last_battery, flight_ctrl.status,
                    info.get("bridge_valid", False), vio_diag,
                    bool(getattr(vio, "initialized", False)), n_frames))

            if show_drone_window:
                try:
                    vis = draw_drone_detections(frame, getattr(pipeline, "last_detections", []))

                    # HUD Top Bar: Drone flight state + battery
                    bat = meta.get("battery", "")
                    bat_str = f" | BAT: {bat}%" if bat != "" else ""
                    if flight_ctrl.status == "HOVERING":
                        hud_color = (0, 255, 0)     # Green: Ready
                    elif flight_ctrl.status == "LANDED":
                        hud_color = (0, 165, 255)   # Amber: Landed
                    else:
                        hud_color = (0, 255, 255)   # Yellow: Busy moving/taking off

                    hud_text = f"DRONE: [{flight_ctrl.status}]  Action: {flight_ctrl.last_action}{bat_str}"
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (20, 20, 20), -1)
                    cv2.putText(vis, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, hud_color, 2)

                    # HUD Bottom Bar: Key prompt
                    cv2.rectangle(vis, (0, vis.shape[0] - 28), (vis.shape[1], vis.shape[0]), (20, 20, 20), -1)
                    cv2.putText(vis, "[T/Q] Takeoff  [E] Land  [W/S/A/D] Move 30cm  [I/K] Up/Down  [J/L] Turn",
                                (10, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)

                    cv2.imshow("Tello Drone Camera (Live Detection View)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    dispatch_drone_key(key, flight_ctrl)
                    # Closing the window with its X button hides it (the
                    # old Q key did this; Q is takeoff now, 2026-11-25).
                    # The server keeps running either way.
                    try:
                        if cv2.getWindowProperty(
                                "Tello Drone Camera (Live Detection View)",
                                cv2.WND_PROP_VISIBLE) < 1:
                            show_drone_window = False
                            cv2.destroyAllWindows()
                    except Exception:
                        pass
                except Exception:
                    pass

            time.sleep(1.0 / FPS_DEFAULT)
        stop.set()

    thread = threading.Thread(target=pipeline_loop, daemon=True)
    thread.start()
    viewer_path = WALL_MARKER_DIR / "phone" / "viewer.html"
    try:
        run_server(state, viewer_path, args.host, args.port,
                   args.https, args.cert, args.key)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if flight_ctrl.tello:
            # Land on exit only if airborne and no command is in
            # flight: holding the controller lock proves it is idle,
            # so this cannot double-send with a landing already in
            # progress (the original bug's land spam).
            if flight_ctrl.lock.acquire(timeout=2.0):
                try:
                    if flight_ctrl.status != "LANDED":
                        log.info("Program exiting: landing drone safely...")
                        flight_ctrl.tello.land()
                        flight_ctrl.status = "LANDED"
                except Exception:
                    pass
                finally:
                    flight_ctrl.lock.release()
            else:
                log.warning("Program exiting: a flight command is still "
                            "in flight; drone left hovering -- land it "
                            "from the Tello app.")
        source.close()
        pipeline.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
