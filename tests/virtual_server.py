"""TEST-ONLY entry point: virtual server on scenario frames.

Usage:
    python -m tests.virtual_server --scenario s1_basic --port 8080
    python -m tests.virtual_server --scenario s1_basic --no-serve
    python -m tests.virtual_server --scenario s1_basic --port 8080 --vio
    python -m tests.virtual_server --scenario s1_basic --port 8080 \\
        --https --cert certs/cert.pem --key certs/key.pem

This is NOT the product server. It runs the real pipeline
(detector -> depth -> bridge -> fusion) on scenario frames from
``tests/scenarios.py`` at 10 fps and serves the viewer + operator
page + SSE marker stream using the same HTTP handler as the product
server (``server.main``). It exists so the virtual harness and the
ground tests keep working after the product server was split out
(``server/main.py`` serves live drone frames only and refuses
virtual mode).

Do NOT use this entry point for anything the user sees as the
product: it serves synthetic frames by design. Ground / virtual
only — no flight, no motors, no real drone commands.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Make `server.*` and `tests.*` importable when run as
# `python -m tests.virtual_server`.
WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.detector import PoseDetector         # noqa: E402
from server.geometry import DRONE_K              # noqa: E402
from server.main import (FPS_DEFAULT, Pipeline, ServerState,  # noqa: E402
                         run_server)
from server.video_source import SyntheticSource  # noqa: E402

log = logging.getLogger(__name__)
DRONE_ID_DEFAULT = "drone1"


# ---------------------------------------------------------------------------
# Inline synthetic frames (fallback until tests/scenarios.py exists)
# ---------------------------------------------------------------------------

def _look_at_R(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation for a camera at ``eye`` looking at ``target``.

    Camera looks along +Z (camera frame), +Y up.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    z_cam = target - eye
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(np.asarray(up, dtype=float), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return np.stack([x_cam, y_cam, z_cam])


def _find_bus_jpg():
    """Locate ultralytics' bundled bus.jpg (person source for the fallback)."""
    try:
        import ultralytics
        pkg = Path(ultralytics.__file__).resolve().parent
        for cand in (pkg / "assets" / "bus.jpg", pkg / "bus.jpg"):
            if cand.exists():
                return cand
    except Exception:
        pass
    return None


def build_inline_frames(detector, n_frames=12, fps=FPS_DEFAULT):
    """Synthetic fallback frames: a person cutout pasted on a plain room.

    The cutout comes from bus.jpg (largest person detection). Drone
    poses sweep laterally (+/-0.4 m) so triangulation gets a baseline.
    The person is placed at world (0, 0, 4.0), 1.7 m tall; the billboard
    is scaled per frame by the true drone-person distance, so the
    bbox-height / triangulation numbers are self-consistent. This is a
    SMOKE path only — the real scenarios live in tests/scenarios.py.
    """
    import cv2
    W, H = DRONE_K["width"], DRONE_K["height"]
    person_pos = np.array([0.0, 0.0, 4.0])
    frames = []
    bus = _find_bus_jpg()
    cutout = None
    if bus is not None:
        img = cv2.imread(str(bus))
        if img is not None:
            dets = detector.detect(img)
            if dets:
                d = max(dets, key=lambda x: x["conf"])
                x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
                cutout = img[max(0, y1):y2, max(0, x1):x2]
                log.info("Inline fallback: person cutout %s from %s "
                         "(bbox %s)", cutout.shape, bus, d["bbox_xyxy"])
    if cutout is None:
        log.warning("Inline fallback: no person cutout available "
                    "(bus.jpg missing or undetected); frames will show "
                    "no person and the pipeline will emit no markers.")
    for i in range(n_frames):
        t = i / fps
        x = -0.4 + 0.8 * (i / max(1, n_frames - 1))
        eye = np.array([x, 1.3, 1.5])
        R = _look_at_R(eye, person_pos + np.array([0.0, 0.85, 0.0]))
        pose = (R, eye)
        # Procedural room: plaster wall, checker floor, doorway.
        frame = np.full((H, W, 3), (196, 190, 178), dtype=np.uint8)
        floor_y = int(H * 0.62)
        for yy in range(floor_y, H, 40):
            for xx in range(0, W, 40):
                if ((xx // 40) + (yy // 40)) % 2 == 0:
                    frame[yy:yy + 40, xx:xx + 40] = (150, 140, 125)
        cv2.rectangle(frame, (int(W * 0.42), 0), (int(W * 0.58), floor_y),
                      (120, 110, 95), -1)  # doorway opening
        cv2.rectangle(frame, (60, 90), (220, 190), (90, 70, 60), -1)
        cv2.rectangle(frame, (700, 70), (880, 200), (70, 90, 110), -1)
        if cutout is not None:
            dist = float(np.linalg.norm(eye - person_pos))
            h_px = int(DRONE_K["fy"] * 1.7 / dist)
            w_px = int(h_px * cutout.shape[1] / cutout.shape[0])
            resized = cv2.resize(cutout, (w_px, h_px),
                                 interpolation=cv2.INTER_AREA)
            cx = W // 2 + int(40 * math.sin(i * 0.7))
            x0 = max(0, min(W - w_px, cx - w_px // 2))
            y0 = max(0, min(H - h_px, floor_y - h_px + 8))
            frame[y0:y0 + h_px, x0:x0 + w_px] = resized
        frames.append((frame, pose, True,
                       {"person_world": person_pos.tolist(),
                        "inline_fallback": True}))
    return frames


def load_scenario_frames(scenario, detector, smooth_look=False):
    """Scenario frames from tests/scenarios.py, or the inline fallback.

    ``smooth_look=True`` selects the smoothed-look variant (yaw/pitch
    slew-rate limited to 60 deg/s) — used by the ``--vio`` path, where
    the frames must be a flight a real Tello can fly; the GT-pose path
    keeps the original trajectories.
    """
    try:
        from tests import scenarios  # noqa: F401
        frames = scenarios.get_frames(scenario, smooth_look=smooth_look)
        log.info("Loaded scenario %s from tests/scenarios.py (%d frames)",
                 scenario, len(frames))
        return frames
    except Exception as exc:  # harness not built yet (or scenario missing)
        log.warning("tests/scenarios.py unavailable (%s); using the inline "
                    "synthetic fallback for scenario %s", exc, scenario)
        return build_inline_frames(detector)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_frame_line(idx, info):
    depth = info["depth"]
    dstr = (f"{depth:.2f}m [{info['method']} σ={info['sigma']:.2f}]"
            if depth is not None else "no depth")
    bridge = "OK" if info["bridge_valid"] else "LOST"
    marker = info["marker"]
    if marker is None:
        mstr = "marker NONE"
    else:
        mstr = (f"marker YES dist={marker['distance_m']:.2f}m "
                f"unc={marker['uncertainty_m']:.2f}m"
                + (" STALE" if marker["stale"] else ""))
    if info.get("voter_changed"):
        mstr += " +MODEL-VOTE"
    return (f"f{idx:03d} t={info['t']:.1f}s det={info['n_det']} "
            f"depth={dstr} bridge={bridge} {mstr}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Wall-marker VIRTUAL server (test-only entry point)")
    ap.add_argument("--scenario", default="s1_basic")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--no-serve", action="store_true",
                    help="run the frames once, print per-frame output, exit")
    ap.add_argument("--vio", action="store_true",
                    help="feed the bridge VIO-estimated poses "
                    "(server/vio.py) instead of the scenario ground-truth "
                    "poses. The VIO is initialised with the first frame's "
                    "pose as the known start pose (takeoff beside the "
                    "phone); from then on only the video frames drive it.")
    ap.add_argument("--mono-depth", action="store_true",
                    help="enable the monocular depth model (Depth Anything "
                    "V2 Small) as a third depth voter (server/mono_depth.py). "
                    "It runs in a background thread and never blocks the "
                    "marker path; see README.md.")
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--cert", default="certs/cert.pem")
    ap.add_argument("--key", default="certs/key.pem")
    ap.add_argument("--drone-id", default=DRONE_ID_DEFAULT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    detector = PoseDetector()
    detector.warmup()
    mono = None
    if args.mono_depth:
        from server.mono_depth import MonoDepthModel
        mono = MonoDepthModel()
        mono.load()
        log.info("Mono depth model loaded: %s", mono.model_id)
    frames = load_scenario_frames(args.scenario, detector,
                                  smooth_look=args.vio)
    if not frames:
        print(f"ERROR: scenario {args.scenario} produced no frames",
              file=sys.stderr)
        return 1

    state = ServerState()
    pipeline = Pipeline(detector, drone_id=args.drone_id, mono_depth=mono)
    stop = threading.Event()

    vio = None
    vio_prior = None
    if args.vio:
        from server.vio import MonocularVO
        # Synthetic Tello state feed (virtual harness only): the GT
        # trajectory stands in for the drone's own state packets, so
        # the yaw/velocity/h+tof prior code path is exercised without
        # a real drone. In a live run the prior is fed from the real
        # Tello state stream (djitellopy UDP 8890, timestamped on
        # the laptop) — see server/tello_state.py and
        # docs/TELLO_CALIBRATION.md. The harness calibration is the
        # matching one (velocity prior ON); the default real-run
        # calibration stays UNCALIBRATED (velocity priors OFF) until
        # a real calibration is run.
        from server.tello_state import StatePrior, SyntheticStateProvider
        _poses = [p for _, p, _, _ in frames]
        _times = [m["t"] for _, _, _, m in frames]
        _provider = SyntheticStateProvider(
            _poses, _times, velocity_unit="dm/s", velocity_frame="global")
        vio_prior = StatePrior(calibration=_provider.calibration_for())
        for _pkt in _provider.packets():
            vio_prior.add(_pkt)
        # Known start pose: the first frame's pose (takeoff beside
        # the phone). Created lazily on the first frame below.
        vio = "pending"

    def pipeline_loop():
        nonlocal vio
        source = SyntheticSource(frames)
        while not stop.is_set():
            if not state.running:
                # Operator pressed Stop: hold the last frame, clear
                # the marker state so the stream honestly reports no
                # marker, and wait for Start.
                state.mark_paused()
                time.sleep(0.1)
                continue
            item = source.read()
            if item is None:
                if args.no_serve:
                    break
                source.reset()
                continue
            frame, pose, tracking_valid, meta = item
            if vio == "pending":
                from server.vio import MonocularVO
                vio = MonocularVO(start_pose=pose)
            if vio is not None and vio != "pending":
                pose, tracking_valid, _diag = vio.update(
                    frame, pipeline.frames_processed / args.fps,
                    state=vio_prior)
            t = pipeline.frames_processed / args.fps
            marker, info = pipeline.process_frame(
                frame, pose, tracking_valid, t)
            state.update_from_pipeline(pipeline, marker, info,
                                       time.time(), t)
            print(format_frame_line(pipeline.frames_processed - 1, info),
                  flush=True)
            if not args.no_serve:
                time.sleep(1.0 / args.fps)
        stop.set()

    if args.no_serve:
        pipeline_loop()
        pipeline.shutdown()
        print(f"SUMMARY frames={pipeline.frames_processed} "
              f"detections={pipeline.detections_seen} "
              f"markers={pipeline.markers_emitted} "
              f"methods={pipeline.methods} "
              f"model_votes={pipeline.model_votes_used} "
              f"model_voter_changed={pipeline.model_voter_changed}")
        if mono is not None:
            print(f"MONO-DEPTH {mono.stats()}")
        return 0

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
        pipeline.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
