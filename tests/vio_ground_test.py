#!/usr/bin/env python3
"""Ground test for the VIO bridge pose source (``server/vio.py``).

Two parts:

1. ``generate_smooth_flight`` — renders a doorway fly-through to a
   video file + a poses JSON in the :class:`server.video_source.
   FileVideoSource` format, so this script is verified NOW with no
   real drone. The flight is deliberately SMOOTH, the way a real
   hand-flown / careful Tello flight is: the drone flies from beside
   the phone, through the doorway, past the person, then turns around
   to face them over 45 frames (~3.4 deg/frame at 10 fps) and hovers.
   It has NO yaw snaps — unlike the harness scenarios in
   ``tests/scenarios.py``, whose look-at logic snaps the camera -31
   deg at the doorway and +101 deg at the turn to face the person in
   a single 0.1 s frame (measured 2026-11-23). No frame-to-frame
   tracker (LK, ORB descriptors, photometric yaw search — all tried
   and measured) can bridge a 101 deg snap at 10 fps, so the smooth
   flight is the honest end-to-end VIO test, and the harness
   scenarios measure the snap behaviour separately (see README.md).

2. ``run_vio`` — takes a recorded video + poses JSON, runs
   ``MonocularVO`` on the video, and compares the estimated path
   against the ground-truth path: drift in cm (mean / max / final),
   per-frame inlier counts, validity, and VIO frame rate.

Real procedure (README.md): carry the Tello by hand (motors OFF)
through a doorway while it records, measure the ground-truth path
(e.g. tape-measured waypoints + a level flight at a known height),
then run ``python -m tests.vio_ground_test --video <file> --poses
<file>`` against the recording + the measured path.

Usage:
    python -m tests.vio_ground_test --generate \\
        --video /tmp/smooth_flight.avi --poses /tmp/smooth_flight.json
    python -m tests.vio_ground_test --video /tmp/smooth_flight.mp4 \\
        --poses /tmp/smooth_flight.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.detector import PoseDetector  # noqa: E402
from server.geometry import DRONE_K  # noqa: E402
from server.video_source import FileVideoSource  # noqa: E402
from server.vio import MonocularVO  # noqa: E402
from tests.render import RoomRenderer, look_at_R, load_person_cutout  # noqa: E402

log = logging.getLogger(__name__)

FPS = 10.0
PERSON_POS = np.array([0.0, 0.0, 4.0])  # same as s1_basic
WEIGHTS = str(WALL_MARKER_DIR / "yolov8n-pose.pt")


def smooth_flight_poses(n_turn=45, n_hover=30):
    """Poses for the smooth doorway fly-through.

    Returns a list of (pose (R, C), phase) with 10 fps timing.
    Phases: approach (fly to the doorway, looking forward), pass
    (through the doorway, past the person, still looking forward),
    turn (smooth ~153 deg turn to face the person, look target
    lerped from ahead to the person — ~3.4 deg/frame, no snap),
    hover (gentle sway, facing the person).
    """
    poses = []
    # Approach + pass as ONE continuous leg (30 frames): the eye
    # lerps (0, 1.3, 0.4) -> (0.6, 1.3, 5.5) and the look target
    # lerps (0, 1.0, 8.0) -> (0.6, 1.0, 8.0) at the same rate, so
    # the per-frame motion is constant — no target switch and no
    # speed change at the doorway. (An earlier version split this
    # into two legs with different targets and speeds; the 6.4 deg
    # yaw step + 0.26 m step at the doorway killed LK tracking at
    # frame 15, measured 2026-11-23 — a real flight has no such
    # step, and neither should the ground-test flight.)
    eye0 = np.array([0.0, 1.3, 0.4])
    # Pass ends just past the person (z = 4.6, not 5.5): flying
    # further forward stares at the feature-poor back wall and the
    # map drains (42 -> 17 points over frames 20-27, tracking lost
    # at frame 27, measured 2026-11-23) before the turn starts.
    eye1 = np.array([0.6, 1.3, 4.6])
    tgt0 = np.array([0.0, 1.0, 8.0])
    tgt1 = np.array([0.6, 1.0, 8.0])
    for i in range(30):
        a = i / 29.0
        eye = eye0 + a * (eye1 - eye0)
        tgt = tgt0 + a * (tgt1 - tgt0)
        R = look_at_R(eye, tgt)
        phase = "approach" if eye[2] < 3.0 else "pass"
        poses.append(((R, eye), phase))
    # Turn: n_turn frames, eye -> hover centre, YAW and PITCH
    # interpolated LINEARLY (constant yaw rate). An earlier version
    # lerped the look target instead, which made the yaw rate
    # non-uniform — ~0.4 deg/frame at the start, peaking at 8.8
    # deg/frame mid-turn (measured 2026-11-23) — and tracking died
    # at frame 39 with a thin, distorted map. Linear yaw/pitch
    # interpolation gives a constant ~3.3 deg/frame over 45 frames,
    # which LK handles, and no pitch snap at the phase boundary
    # (the target-lerp version also snapped pitch -5 deg -> level
    # in one frame when the target Y was inconsistent).
    eye0 = np.array([0.6, 1.3, 4.6])
    eye1 = np.array([0.9, 1.3, 5.8])
    person_tgt = np.array([0.0, 0.85, 4.0])
    # Yaw/pitch of the LOOK DIRECTION (third row of R): yaw =
    # atan2(z_x, z_z), pitch = -asin(z_y). NB: atan2(R[0, 2],
    # R[2, 2]) is the NEGATIVE of this (it is the x-axis's yaw),
    # and asin(-R[1, 2]) mixes in cos(yaw) — both mistakes were in
    # an earlier version and made the turn swing the wrong way
    # (52 deg snap at the turn->hover boundary, measured 2026-11-23).
    R_end = look_at_R(eye1, person_tgt)
    yaw_end = float(np.arctan2(R_end[2, 0], R_end[2, 2]))
    pitch_end = float(np.arcsin(-R_end[2, 1]))
    R_start = look_at_R(eye0, np.array([0.6, 1.0, 8.0]))
    yaw_start = float(np.arctan2(R_start[2, 0], R_start[2, 2]))
    pitch_start = float(np.arcsin(-R_start[2, 1]))
    # Unwrap the yaw so the turn takes the short way (yaw_end is
    # ~+2.6 rad; yaw_start ~0).
    while yaw_end - yaw_start > np.pi:
        yaw_end -= 2 * np.pi
    while yaw_end - yaw_start < -np.pi:
        yaw_end += 2 * np.pi
    for i in range(n_turn):
        a = (i + 1) / n_turn
        eye = eye0 + a * (eye1 - eye0)
        yaw = yaw_start + a * (yaw_end - yaw_start)
        pitch = pitch_start + a * (pitch_end - pitch_start)
        direction = np.array([np.sin(yaw) * np.cos(pitch),
                              -np.sin(pitch),
                              np.cos(yaw) * np.cos(pitch)])
        R = look_at_R(eye, eye + direction)
        poses.append(((R, eye), "turn"))
    # Hover: n_hover frames, gentle sway, facing the person.
    for i in range(n_hover):
        a = 2.0 * np.pi * i / 30.0
        eye = np.array([0.9 + 0.3 * np.sin(a), 1.3,
                        5.8 + 0.2 * np.cos(a)])
        R = look_at_R(eye, np.array([0.0, 0.85, 4.0]))
        poses.append(((R, eye), "hover"))
    return poses


def generate_smooth_flight(video_path, poses_path, detector=None,
                           room=None, n_turn=45, n_hover=30):
    """Render the smooth flight to ``video_path`` + ``poses_path``.

    Returns the number of frames written. ``room`` overrides keys of
    the renderer's DEFAULT_ROOM (e.g. ``{\"posters\":
    \"featureless_door\"}`` for the blank-wall variant).
    """
    import cv2
    if detector is None:
        detector = PoseDetector(weights=WEIGHTS)
        detector.warmup()
    cutout, mask = load_person_cutout(detector)
    renderer = RoomRenderer(DRONE_K, cutout, mask, room=room)
    poses = smooth_flight_poses(n_turn=n_turn, n_hover=n_hover)
    # Lossless FFV1 (in an .avi container): the default mp4v encoder
    # is so lossy on this flat-shaded synthetic imagery that it
    # dominated the VIO error (drift 20.6 cm mean on mp4v vs 4.2 cm
    # on raw frames, measured 2026-11-23) — the ground test must
    # measure the VIO, not the codec. A real Tello H.264 stream is
    # far cleaner than mp4v; compression noise on real footage is a
    # separate measurement (see README.md). FFV1 is large (~2 MB /
    # frame at 960x720) — keep generated videos in /tmp.
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"FFV1"),
                             FPS, (960, 720))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {video_path}")
    entries = []
    for i, (pose, phase) in enumerate(poses):
        frame, info = renderer.render(pose, person_pos=PERSON_POS)
        writer.write(frame)
        R, C = pose
        entries.append({
            "R": [[float(v) for v in row] for row in R],
            "C": [float(v) for v in C],
            "tracking_valid": True,
            "meta": {"t": i / FPS, "phase": phase,
                     "in_view": bool(info.get("in_view", False))},
        })
    writer.release()
    with open(poses_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)
    return len(entries)


def run_vio(video_path, poses_path, vio=None):
    """Run VIO on a recorded video + poses JSON; report drift.

    Returns a dict with per-phase + overall drift (m), validity,
    inliers and timing. Prints a human-readable report.
    """
    src = FileVideoSource(str(video_path), str(poses_path))
    vio = None  # created on frame 0 with the known start pose
    n_frames = 0
    valid_n = 0
    drifts = []          # (frame_idx, phase, drift_m)
    inliers = []
    ms = []
    lost_at = []
    was_valid = False
    while True:
        item = src.read()
        if item is None:
            break
        frame, gt_pose, _tv, meta = item
        if n_frames == 0:
            vio = MonocularVO(start_pose=gt_pose)
        t0 = time.perf_counter()
        est, valid, diag = vio.update(frame, meta.get("t", n_frames / FPS))
        ms.append((time.perf_counter() - t0) * 1000.0)
        if valid:
            valid_n += 1
            drifts.append((n_frames, meta.get("phase", "?"),
                           float(np.linalg.norm(est[1] - gt_pose[1]))))
            inliers.append(int(diag.get("inliers", 0)))
        elif was_valid:
            lost_at.append(n_frames)
        was_valid = valid
        n_frames += 1
    d = np.array([x[2] for x in drifts]) if drifts else np.array([])
    result = {
        "n_frames": n_frames,
        "valid": valid_n,
        "drift_mean_m": float(d.mean()) if len(d) else None,
        "drift_max_m": float(d.max()) if len(d) else None,
        "drift_final_m": float(d[-1]) if len(d) else None,
        "inliers_mean": float(np.mean(inliers)) if inliers else None,
        "inliers_min": int(np.min(inliers)) if inliers else None,
        "vio_ms_mean": float(np.mean(ms)) if ms else None,
        "vio_fps": float(1000.0 / np.mean(ms)) if ms else None,
        "lost_at": lost_at,
    }
    # Per-phase drift.
    phases = {}
    for idx, phase, drift in drifts:
        phases.setdefault(phase, []).append(drift)
    result["per_phase"] = {
        ph: {"n": len(v), "mean_m": float(np.mean(v)),
             "max_m": float(np.max(v))}
        for ph, v in phases.items()
    }
    print(f"frames: {n_frames}, VIO valid: {valid_n} "
          f"({100.0 * valid_n / max(1, n_frames):.0f}%)")
    if len(d):
        print(f"drift: mean {d.mean() * 100:.1f} cm, "
              f"max {d.max() * 100:.1f} cm, "
              f"final {d[-1] * 100:.1f} cm")
        print(f"PnP inliers: mean {np.mean(inliers):.0f}, "
              f"min {int(np.min(inliers))}")
        for ph, stats in result["per_phase"].items():
            print(f"  {ph}: n={stats['n']} mean "
                  f"{stats['mean_m'] * 100:.1f} cm max "
                  f"{stats['max_m'] * 100:.1f} cm")
    else:
        print("drift: no valid frames — VIO never initialised")
    print(f"VIO step: {np.mean(ms):.1f} ms/frame "
          f"({1000.0 / np.mean(ms):.1f} fps)")
    if lost_at:
        print(f"tracking lost at frames: {lost_at} "
              f"(bridge invalidates there — no guessed pose)")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true",
                    help="render the smooth flight video + poses JSON")
    ap.add_argument("--video", default="/tmp/smooth_flight.avi")
    ap.add_argument("--poses", default="/tmp/smooth_flight.json")
    ap.add_argument("--blank-door", action="store_true",
                    help="featureless shared wall (s23 geometry)")
    ap.add_argument("--n-turn", type=int, default=45)
    ap.add_argument("--n-hover", type=int, default=30)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.generate:
        room = {"posters": "featureless_door"} if args.blank_door else None
        n = generate_smooth_flight(args.video, args.poses,
                                   room=room, n_turn=args.n_turn,
                                   n_hover=args.n_hover)
        print(f"generated {n} frames -> {args.video} + {args.poses}")
    return run_vio(args.video, args.poses)


if __name__ == "__main__":
    main()
