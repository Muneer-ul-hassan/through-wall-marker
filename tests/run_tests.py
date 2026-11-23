#!/usr/bin/env python3
"""End-to-end virtual test runner for the wall-marker system.

Runs the REAL pipeline in-process — ``server.detector``, ``server.depth``,
``server.bridge``, ``server.fusion`` and ``server.main.Pipeline`` — on the
synthetic frames from ``tests.scenarios``. Only the video frames and the
drone poses are simulated (that is the point of virtual testing); no part
of the pipeline itself is mocked.

Per scenario and per frame it records: detection (yes/no), joint errors
(marker skeleton vs ground-truth skeleton), distance error, skeleton
scale error, marker shown (yes/no), and the depth method used
(triangulation vs bbox-height). Metrics are printed per scenario with
PASS/FAIL against the SPEC.md thresholds, and written to
``tests/TEST-REPORT.md``. Exit code is non-zero if any scenario fails.

Metric definitions (all ground truth comes from the scenario definition,
never from the detector output):
* detection rate: frames with a detection / frames where the whole person
  is inside the drone frame and >= 80 px tall (``meta["in_view"]``).
  Frames where the person is only partly visible (e.g. while the drone
  flies past them) are excluded from every metric — the renderer never
  pastes a half-visible person, and those frames are counted and stated
  in the report, not silently dropped.
* joint error: Euclidean distance between each marker-skeleton joint and
  the matching ground-truth joint, pooled over all joints and all frames
  where a marker was shown on an in-view frame. Joints the marker
  reports as null are skipped (never invented). For two-person
  scenarios the marker is scored against the NEAREST ground-truth
  person (by foot point); a marker more than 1.0 m from BOTH persons
  is counted as a merged/false marker.
* distance error: |marker distance_m - GT phone->person distance|
  (nearest person, for two-person scenarios).
* skeleton scale error: |marker skeleton height - GT height| /
  GT height, where height = max joint y - min joint y over the
  joints the marker actually reports.
* worst-frame metrics (adversarial suite): the single largest joint
  error and the single largest distance error in the scenario — means
  hide bad frames, worst values do not.
* marker-while-bridge-invalid: any frame where a marker was emitted
  while the bridge was invalid. Must be 0 in every scenario — the
  bridge is the only thing that makes a marker trustworthy.
* false markers (empty rooms): markers emitted with no person in the
  scene.
* marker-after-loss (s5): markers emitted at/after the tracking-loss
  frame (frame 36).
* loss+recovery (s21): markers during the loss window must be 0, and
  markers must only resume at/after the recovery frame.
* multi-drone: fused mean joint error (hover frames 25-59) must be <=
  the best single-drone mean joint error over the same frames. The
  single-drone numbers come from re-running each drone's frames through
  its own pipeline (same frames, re-rendered deterministically).

Also runs the no-drone-video test (SPEC.md): starts the real HTTP server
from ``server.main`` on a test port and checks /viewer (exactly one
<video> tag, no drone-stream reference), /api/markers (JSON-only SSE
payloads, no video content-type anywhere), /api/status, and that no
other routes exist (a probe list must all 404).

The report writer is APPEND-MODE: the original 2026-11-23 results
stay intact at the top of tests/TEST-REPORT.md, and each run adds (or
replaces) a dated "## Extended tests" section below them. A regression
comparison checks the re-run original-7 numbers against the recorded
2026-11-23 numbers.

Usage:
    python -m tests.run_tests [--scenarios s1_basic s2_offcentre ...]
                              [--port 18765] [--skip-video-test]
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import logging
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server import bridge as bridge_mod          # noqa: E402
from server import fusion as fusion_mod          # noqa: E402
from server.detector import PoseDetector         # noqa: E402
from server.geometry import DRONE_K, project_point  # noqa: E402
from server.main import Pipeline, ServerState, make_handler  # noqa: E402
from server.tello_state import (StatePrior, SyntheticStateProvider,
                                yaw_pitch_roll_from_R)  # noqa: E402
from server.vio import MonocularVO               # noqa: E402
from tests import scenarios                      # noqa: E402

log = logging.getLogger(__name__)

WEIGHTS = str(WALL_MARKER_DIR / "yolov8n-pose.pt")
VIEWER_PATH = WALL_MARKER_DIR / "phone" / "viewer.html"
REPORT_PATH = Path(__file__).resolve().parent / "TEST-REPORT.md"

# SPEC.md PASS thresholds — do not move these.
TH_DETECTION_RATE = 0.90
TH_JOINT_ERR_M = 0.60
TH_DIST_ERR_M = 0.60
TH_SCALE_ERR = 0.25
# A marker more than this far from every ground-truth person is a
# merged/false marker (two-person scenarios).
TH_MERGED_MARKER_M = 1.0

HOVER_START = 25  # first hover frame (multi-drone comparison window: 25-59)

# Recorded results for the original 7 scenarios (from the original
# TEST-REPORT.md table) — regression comparison only. These numbers
# are history, not targets: the re-run must reproduce them.
#
# RE-BASED 2026-11-23 (VIO fix round 3): the bbox-height detector
# calibration (server/depth.py BBOX_HEIGHT_FRACTION = 0.89) is a
# deliberate, measured fix — the YOLO bbox spans ~0.89 of the
# person's height, so H = 1.7 m overestimated depth by ~13%. It
# IMPROVED the GT numbers (s1 joint 38.6 -> 36.1 cm, s4 38.9 -> 35.6,
# s5 43.5 -> 35.7, s7 37.1 -> 34.1; dist 11.0 -> 8.7, 10.3 -> 8.0,
# 16.1 -> 9.1, 9.4 -> 6.6) and all 27 scenarios still PASS, so the
# baseline is re-based to the post-calibration numbers below. The
# pre-calibration numbers are preserved in the report's original
# section (above the "## Extended tests" marker) and in the round-3
# narrative. The scale error rose slightly on the same runs (s1
# 11.9 -> 13.1%, s5 9.7 -> 13.5%) — a side effect of the corrected
# depth on the cross-check; still PASS, and the joint error (the
# primary metric) improved everywhere. If a future run MISMATCHes
# this baseline, that is a real drift to investigate.
RECORDED_2026_11_23 = {
    "s1_basic": {"detection_rate": 1.0, "joint_err_mean": 0.361,
                 "dist_err_mean": 0.087, "scale_err_mean": 0.131,
                 "passed": True},
    "s2_offcentre": {"detection_rate": 1.0, "joint_err_mean": 0.355,
                     "dist_err_mean": 0.009, "scale_err_mean": 0.141,
                     "passed": True},
    "s3_close": {"detection_rate": 1.0, "joint_err_mean": 0.376,
                 "dist_err_mean": 0.134, "scale_err_mean": 0.113,
                 "passed": True},
    "s4_dim": {"detection_rate": 1.0, "joint_err_mean": 0.356,
               "dist_err_mean": 0.080, "scale_err_mean": 0.137,
               "passed": True},
    "s5_tracking_loss": {"detection_rate": 1.0, "joint_err_mean": 0.357,
                         "dist_err_mean": 0.091, "scale_err_mean": 0.135,
                         "passed": True},
    "s6_no_person": {"detection_rate": None, "false_markers": 0,
                     "passed": True},
    "s7_two_drones": {"detection_rate": 1.0, "joint_err_mean": 0.341,
                      "dist_err_mean": 0.066, "scale_err_mean": 0.137,
                      "passed": True},
}

# Recorded 2026-11-23 VIO round-2 results (from TEST-REPORT.md, VIO
# tests table, 2026-11-23 11:41 UTC run) — kept so the round-3
# comparison table in write_report can show round 2 vs round 3.
# Joint/dist/drift in metres; the round-2 run had no per-frame yaw
# measurement (that was added for round 3).
VIO_ROUND2_2026_11_23 = {
    "s1_basic": {"vio_valid_frames": 56, "vio_drift_mean_m": 0.098,
                 "vio_drift_max_m": 0.352, "vio_inliers_mean": 60,
                 "vio_fps": 9.1, "vio_pipeline_fps": 5.2,
                 "joint_err_mean": 0.718, "dist_err_mean": 0.359,
                 "n_markers": 10, "passed": False},
    "s23_blank_door": {"vio_valid_frames": 56, "vio_drift_mean_m": 0.133,
                       "vio_drift_max_m": 0.209, "vio_inliers_mean": 35,
                       "vio_fps": 14.8, "vio_pipeline_fps": 6.6,
                       "joint_err_mean": 0.817, "dist_err_mean": 0.140,
                       "n_markers": 11, "passed": False},
    "s20_fast": {"vio_valid_frames": 56, "vio_drift_mean_m": 0.079,
                 "vio_drift_max_m": 0.477, "vio_inliers_mean": 36,
                 "vio_fps": 14.5, "vio_pipeline_fps": 7.0,
                 "joint_err_mean": 0.386, "dist_err_mean": 0.182,
                 "n_markers": 11, "passed": True},
    "s4_dim": {"vio_valid_frames": 56, "vio_drift_mean_m": 0.047,
               "vio_drift_max_m": 0.215, "vio_inliers_mean": 62,
               "vio_fps": 15.9, "vio_pipeline_fps": 6.6,
               "joint_err_mean": 0.552, "dist_err_mean": 0.257,
               "n_markers": 10, "passed": True},
}

# Things found and fixed while building the extended suite (2026-11-23).
# Recorded here so the report carries the fix history with the numbers.
FIXES_EXTENDED_2026_11_23 = [
    "Bridge.reacquire() implemented (2026-11-23): tracking loss used to "
    "invalidate the bridge permanently (reacquire() raised "
    "NotImplementedError), so a loss+recovery scenario could never pass. "
    "A fresh valid pose after a loss now starts a NEW chain (uncertainty "
    "clock restarts at 0.10 m) via Bridge.reacquire(), wired into "
    "Bridge.update(). The virtual harness proves the state-machine logic "
    "(marker hidden during loss, returns only after re-establishment + a "
    "fresh detection); real relocalisation is still unproven and must "
    "gate reacquire in the real system. Required by s21_loss_recovery.",
    "Triangulation cross-check (2026-11-23): estimate_person_depth now "
    "compares the triangulated depth against the per-frame bbox-height "
    "depth and trusts the bbox-height estimate when they disagree by "
    ">25% (relative). Measured: static scenes agree within ~14% max "
    "(mean 11-13% across s1/s2/s3/s4/s7/s11/s12/s25), while a moving "
    "person (s18: up to 59%), an occluded person (s15: up to 93%) and "
    "mixed two-person rays (s19: up to 100%) disagree far beyond it. "
    "The triangulation residual does NOT catch this — a moving person's "
    "rays can genuinely nearly meet at a wrong point (measured residual "
    "<= 0.15 while the depth was 1.3 m off) — so the cross-check is the "
    "signal, not the residual. The bbox-height estimate is per-frame "
    "(no static-scene assumption) with a measured +20-34 cm bias, so "
    "it is the safe fallback.",
    "Per-person depth histories in Pipeline (2026-11-23): the depth "
    "history must contain views of ONE person only. The max-conf "
    "detection can alternate between two people, and mixing their rays "
    "triangulated a ghost point between them (measured in s19: a phantom "
    "track at (0, 7.4 m) and triangulated depths of 0.01-0.22 m). Each "
    "person now keeps a separate history, associated by a rough foot "
    "point within 1.0 m of that person's last foot (the 1.0 m gate "
    "matches the fusion's association distance). A single person keeps "
    "exactly one history, so single-person behaviour is unchanged. A "
    "moving person moves ~0.03 m/frame, so the association never fires "
    "on motion alone.",
    "Ankle ray for partial detections (2026-11-23): when the detector "
    "misses >= 3 joints, the person is partly occluded and the bbox is "
    "the visible part's box — its bottom-centre is laterally offset and "
    "its height is short, so both depth methods share a common-mode bias "
    "the cross-check cannot catch (measured in s15: triangulation 3.11 m "
    "and bbox-height 2.86 m agreed, yet the true depth was 2.51 m). The "
    "triangulation and foot point then use the visible-ankle ray "
    "(partial_foot_uv), a stable world point that does not move with the "
    "occlusion; the ankle-derived foot is projected straight down to "
    "y = 0 (ground contact). Measured separation: full-person frames "
    "miss 0-1 joints, occluded frames (s15) miss 3-9, so the gate never "
    "fires on a full person and the original 7 scenarios are untouched.",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _marker_joints(marker):
    """Marker skeleton as a (17, 3) array with NaN for null joints."""
    out = np.full((17, 3), np.nan, dtype=float)
    for j, p in enumerate(marker["skeleton"]):
        if p[0] is None or p[1] is None or p[2] is None:
            continue
        out[j] = p
    return out


def joint_errors(marker, gt_skeleton_world):
    """Per-joint Euclidean errors (m) for the joints the marker reports."""
    mj = _marker_joints(marker)
    gt = np.asarray(gt_skeleton_world, dtype=float)
    errs = []
    for j in range(17):
        if np.isnan(mj[j]).any():
            continue
        errs.append(float(np.linalg.norm(mj[j] - gt[j])))
    return errs


def scale_error(marker, gt_skeleton_world):
    """|marker skeleton height - GT height| / GT height (or None)."""
    mj = _marker_joints(marker)
    gt = np.asarray(gt_skeleton_world, dtype=float)
    valid = ~np.isnan(mj).any(axis=1)
    if valid.sum() < 2:
        return None
    h_marker = float(mj[valid][:, 1].max() - mj[valid][:, 1].min())
    h_gt = float(gt[:, 1].max() - gt[:, 1].min())
    if h_gt <= 0:
        return None
    return abs(h_marker - h_gt) / h_gt


def _marker_foot(marker):
    """Approximate foot point (3,) of a marker, or None.

    Mean of the ankle joints when available, else the lowest reported
    joint; y is set to 0 (ground contact). Used only to pick the
    nearest ground-truth person (two-person scenarios) and to flag
    merged markers — never as a measurement itself.
    """
    mj = _marker_joints(marker)
    if not np.isnan(mj[15]).any() and not np.isnan(mj[16]).any():
        foot = np.mean([mj[15], mj[16]], axis=0)
    else:
        valid = ~np.isnan(mj).any(axis=1)
        if not valid.any():
            return None
        foot = mj[valid][np.argmin(mj[valid][:, 1])]
    return np.array([foot[0], 0.0, foot[2]], dtype=float)


def _score_marker(marker, meta):
    """Score one marker against ground truth.

    Returns dict with joint_errs, dist_err, dist_err_signed, scale_err,
    merged (bool: marker > 1.0 m from every GT person), nearest_dist_m.
    The marker is scored against the NEAREST ground-truth person (by
    foot point); for single-person scenarios that is the only person,
    so the numbers are identical to the original scoring.
    """
    out = {"joint_errs": None, "dist_err": None, "scale_err": None,
           "dist_err_signed": None, "merged": False,
           "nearest_dist_m": None}
    persons_gt = meta.get("persons_gt") or []
    persons_world = meta.get("persons_world") or []
    if marker is None or not persons_gt:
        return out
    phone_pos = np.asarray(meta.get("phone_position", (0.0, 1.5, 0.0)),
                           dtype=float)
    foot = _marker_foot(marker)
    if foot is not None:
        dfoot = [float(np.linalg.norm(
            foot - np.asarray(pw, dtype=float))) for pw in persons_world]
        k = int(np.argmin(dfoot))
        out["nearest_dist_m"] = dfoot[k]
        if dfoot[k] > TH_MERGED_MARKER_M:
            out["merged"] = True
    else:
        k = 0
    out["joint_errs"] = joint_errors(marker, persons_gt[k])
    gt_dist = float(np.linalg.norm(
        np.asarray(persons_world[k], dtype=float) - phone_pos))
    out["dist_err"] = abs(marker["distance_m"] - gt_dist)
    out["dist_err_signed"] = marker["distance_m"] - gt_dist
    out["scale_err"] = scale_error(marker, persons_gt[k])
    return out


# ---------------------------------------------------------------------------
# Single-drone scenario run
# ---------------------------------------------------------------------------

def run_single_scenario(name, detector, mono=None):
    """Run one single-drone scenario through the real pipeline.

    ``mono`` is an optional loaded ``MonoDepthModel``: when given, the
    pipeline runs the model voter in its background thread (see
    server/main.py) and the rows record the model vote, its error vs
    ground truth, and whether it changed the outcome. When ``mono``
    is None the run is the legacy two-method baseline. Every run
    times ``process_frame`` so latency / FPS before and after the
    model can be compared on the same path.

    Returns a dict with per-frame rows and aggregate metrics.
    """
    cfg = scenarios.ALL_SCENARIOS[name]
    phone_pos = np.asarray(cfg.get("phone", (0.0, 1.5, 0.0)), dtype=float)
    pipeline = Pipeline(detector, drone_id="drone1",
                        phone_position=phone_pos, mono_depth=mono)
    rows = []
    frame_ms = []
    loss = cfg.get("tracking_loss", cfg.get("tracking_loss_frame"))
    for frame, pose, tracking_valid, meta in scenarios.iter_frames(
            name, detector=detector, drone_id="drone1"):
        t = meta["t"]
        t0 = time.perf_counter()
        marker, info = pipeline.process_frame(frame, pose, tracking_valid, t)
        frame_ms.append((time.perf_counter() - t0) * 1000.0)
        # True foot depth in this drone frame (for the measured depth bias).
        true_depth = None
        if meta["person_world"] is not None:
            p = project_point(DRONE_K, pose[0], pose[1],
                              np.array(meta["person_world"], dtype=float))
            true_depth = None if p is None else p[2]
        score = _score_marker(marker, meta)
        model_depth = info.get("model_depth")
        row = {
            "frame": meta["frame"], "t": t,
            "in_view": bool(meta["in_view"]),
            "detected": info["n_det"] > 0,
            "method": info["method"],
            "depth_m": info["depth"],
            "true_depth_m": true_depth,
            "model_depth_m": model_depth,
            "model_err_m": (None if model_depth is None or true_depth is None
                            else model_depth - true_depth),
            "voter_changed": bool(info.get("voter_changed")),
            "marker": marker is not None,
            "stale": bool(marker["stale"]) if marker else False,
            "bridge_valid": bool(info["bridge_valid"]),
            "joint_errs": score["joint_errs"],
            "dist_err": score["dist_err"],
            "scale_err": score["scale_err"],
            "dist_err_signed": score["dist_err_signed"],
            "merged": bool(score["merged"]),
            "nearest_dist_m": score["nearest_dist_m"],
        }
        rows.append(row)
        if meta["frame"] % 10 == 0:
            log.info("  %s f%02d det=%d method=%s marker=%s",
                     name, meta["frame"], info["n_det"], info["method"],
                     "YES" if marker else "no")
    pipeline.shutdown()  # never abandon a model worker mid-inference
    result = _aggregate(name, rows, loss)
    result["frame_ms_mean"] = float(np.mean(frame_ms)) if frame_ms else None
    result["pipeline_fps"] = (float(1000.0 / np.mean(frame_ms))
                              if frame_ms else None)
    return result


def _aggregate(name, rows, loss):
    """Aggregate per-frame rows into scenario metrics + PASS/FAIL."""
    cfg = scenarios.ALL_SCENARIOS[name]
    n_persons = len(cfg.get("persons", [1])) if "persons" in cfg else (
        0 if cfg.get("person") is None else 1)
    in_view = [r for r in rows if r["in_view"]]
    n_view = len(in_view)
    n_det = sum(1 for r in in_view if r["detected"])
    detection_rate = (n_det / n_view) if n_view else None

    all_joint_errs = []
    dist_errs = []
    dist_errs_signed = []
    scale_errs = []
    depth_bias = {"bbox_height": [], "triangulation": []}
    model_errs = []
    voter_changed = 0
    for r in rows:
        if r.get("model_err_m") is not None:
            model_errs.append(r["model_err_m"])
        if r.get("voter_changed"):
            voter_changed += 1
        if r["method"] in depth_bias and r["depth_m"] is not None \
                and r.get("true_depth_m") is not None:
            depth_bias[r["method"]].append(r["depth_m"] - r["true_depth_m"])
        if not r["in_view"] or not r["marker"]:
            continue
        if r["joint_errs"]:
            all_joint_errs.extend(r["joint_errs"])
        if r["dist_err"] is not None:
            dist_errs.append(r["dist_err"])
        if r.get("dist_err_signed") is not None:
            dist_errs_signed.append(r["dist_err_signed"])
        if r["scale_err"] is not None:
            scale_errs.append(r["scale_err"])

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    def _median(xs):
        return float(np.median(xs)) if xs else None

    def _max(xs):
        return float(np.max(xs)) if xs else None

    methods = Counter(r["method"] for r in rows if r["method"])
    first_tri = next((r["frame"] for r in rows
                      if r["method"] == "triangulation"), None)
    n_markers = sum(1 for r in rows if r["marker"])
    n_stale = sum(1 for r in rows if r["marker"] and r["stale"])
    n_not_in_view = sum(1 for r in rows if not r["in_view"])
    false_markers = n_markers if n_persons == 0 else 0
    marker_while_bridge_invalid = sum(
        1 for r in rows if r["marker"] and not r["bridge_valid"])
    merged_markers = sum(1 for r in rows if r["marker"] and r["merged"])

    # Tracking-loss metrics: permanent loss (int) vs loss window (tuple).
    marker_after_loss = 0
    marker_during_loss = 0
    first_marker_after_recovery = None
    if isinstance(loss, tuple):
        loss_start, loss_end = loss
        marker_during_loss = sum(
            1 for r in rows
            if r["marker"] and loss_start <= r["frame"] < loss_end)
        resumed = [r["frame"] for r in rows
                   if r["marker"] and r["frame"] >= loss_end]
        first_marker_after_recovery = min(resumed) if resumed else None
    elif isinstance(loss, int):
        marker_after_loss = sum(
            1 for r in rows if r["marker"] and r["frame"] >= loss)

    checks = {}
    if detection_rate is not None:
        checks["detection_rate"] = detection_rate >= TH_DETECTION_RATE
    if all_joint_errs:
        checks["joint_error"] = _mean(all_joint_errs) <= TH_JOINT_ERR_M
    if dist_errs:
        checks["distance_error"] = _mean(dist_errs) <= TH_DIST_ERR_M
    if scale_errs:
        checks["scale_error"] = _mean(scale_errs) <= TH_SCALE_ERR
    if n_persons == 0:
        checks["false_markers"] = false_markers == 0
    if isinstance(loss, int):
        checks["marker_after_loss"] = marker_after_loss == 0
    if isinstance(loss, tuple):
        checks["marker_during_loss"] = marker_during_loss == 0
        checks["marker_resumes_after_recovery"] = (
            first_marker_after_recovery is not None
            and first_marker_after_recovery >= loss[1])
    if n_persons > 1:
        checks["no_merged_markers"] = merged_markers == 0
    # A marker while the bridge is invalid is never acceptable, in any
    # scenario — the bridge is the only thing that makes a marker
    # trustworthy.
    checks["marker_while_bridge_invalid"] = (
        marker_while_bridge_invalid == 0)
    # A scenario with a person but no usable marker measurements at all
    # cannot pass: the joint/distance checks simply have no data.
    if n_persons > 0 and not all_joint_errs:
        checks["marker_present"] = False

    passed = bool(checks) and all(checks.values())
    return {
        "name": name,
        "desc": cfg["desc"],
        "rows": rows,
        "n_frames": len(rows),
        "n_in_view": n_view,
        "n_not_in_view": n_not_in_view,
        "n_detected_in_view": n_det,
        "detection_rate": detection_rate,
        "joint_err_mean": _mean(all_joint_errs),
        "joint_err_median": _median(all_joint_errs),
        "joint_err_max": _max(all_joint_errs),
        "n_joint_samples": len(all_joint_errs),
        "dist_err_mean": _mean(dist_errs),
        "dist_err_median": _median(dist_errs),
        "dist_err_max": _max(dist_errs),
        "dist_err_signed_mean": _mean(dist_errs_signed),
        "n_dist_samples": len(dist_errs),
        "depth_bias_m": {m: (_mean(v) if v else None)
                         for m, v in depth_bias.items()},
        "depth_bias_n": {m: len(v) for m, v in depth_bias.items()},
        "model_err_mean": _mean(model_errs),
        "model_err_abs_mean": (_mean([abs(v) for v in model_errs])
                               if model_errs else None),
        "model_err_max": _max([abs(v) for v in model_errs]),
        "model_n": len(model_errs),
        "voter_changed_frames": voter_changed,
        "scale_err_mean": _mean(scale_errs),
        "methods": dict(methods),
        "first_triangulation_frame": first_tri,
        "n_markers": n_markers,
        "n_stale_markers": n_stale,
        "false_markers": false_markers,
        "marker_after_loss": marker_after_loss,
        "marker_during_loss": marker_during_loss,
        "first_marker_after_recovery": first_marker_after_recovery,
        "marker_while_bridge_invalid": marker_while_bridge_invalid,
        "merged_markers": merged_markers,
        "n_persons": n_persons,
        "checks": checks,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# VIO scenarios (2026-11-23): the same Pipeline, but the bridge is fed
# VIO-estimated poses instead of ground truth. The GT-pose runs above
# are untouched. The VIO is initialised with the first frame's GT pose
# as the known start pose (the drone takes off beside the phone — the
# session geometry the scale method relies on); from then on only the
# video frames drive it. The `loss` passed to _aggregate is the VIO
# loss frame (first invalid frame after initialisation), so
# marker-after-loss is measured against the VIO losing tracking.
# Smoothed trajectories (2026-11-23 fix round): the VIO runs use the
# smoothed-look variant (``smooth_look=True``) — yaw/pitch slew-rate
# limited to 60 deg/s, a real Tello yaw rate — because the harness
# look-at logic snaps -31 deg at the doorway and +101 deg at the turn
# in a single 0.1 s frame, which no frame-to-frame tracker can bridge.
# Positions are unchanged and the GT-pose scenarios still run the
# original trajectories, so their recorded numbers must still
# reproduce exactly.
# ---------------------------------------------------------------------------

VIO_SCENARIOS = ["s1_basic", "s23_blank_door", "s20_fast", "s4_dim"]

# Depth-model voter suite (2026-11-23): the standard scenarios re-run
# with the monocular model voter enabled, plus the adversarial ones a
# model could help or hurt (dim / dim+noise / glare / occluded /
# moving / far / fast / blank-door) and the two-drone fusion case.
MONO_SCENARIOS = ["s1_basic", "s2_offcentre", "s3_close", "s4_dim",
                  "s13_dim_noise", "s14_glare", "s15_occluded",
                  "s18_moving", "s11_large_room_far", "s20_fast",
                  "s23_blank_door", "s17_edge", "s7_two_drones"]


def run_vio_scenario(name, detector, mono=None, vio_kwargs=None,
                     prior_offsets_deg=(0.0, 0.0, 0.0)):
    """Run one scenario through the real pipeline with VIO poses.

    ``mono`` is an optional loaded ``MonoDepthModel`` (same voter as
    the GT-pose runs); the VIO regression is measured with the voter
    on, so a voter regression on the VIO path cannot hide.
    ``vio_kwargs`` are passed to ``MonocularVO`` (e.g. ``pnp_mode``).
    ``prior_offsets_deg`` is a (yaw, pitch, roll) offset in degrees
    applied to the synthetic state packets BEFORE they are buffered
    — a prior-sensitivity test only (0, 0, 0) = the unperturbed
    harness prior. It simulates a real Tello attitude that is off
    by a known amount; it never touches the video or the GT poses.
    Returns the _aggregate result dict plus VIO fields (drift,
    inliers, frame rates). The result's ``name`` is suffixed
    ``(VIO)`` so it never collides with the GT-pose result.
    """
    cfg = scenarios.ALL_SCENARIOS[name]
    phone_pos = np.asarray(cfg.get("phone", (0.0, 1.5, 0.0)), dtype=float)
    pipeline = Pipeline(detector, drone_id="drone1",
                        phone_position=phone_pos, mono_depth=mono)
    rows = []
    vio_rows = []  # valid frames: (frame, drift_m, inliers, vio_ms, yaw_err_deg)
    all_frame_ms = []  # every frame: VIO step + process_frame, ms
    vio = None
    vio_loss_frame = None
    vio_init_frame = None
    ever_valid = False
    # Synthetic Tello state feed (harness only): the GT trajectory
    # stands in for the drone's own state packets (djitellopy UDP
    # 8890), so the yaw/velocity/h+tof prior code path is exercised
    # without a real drone. The packets are quantised like the real
    # ones (int deg, int raw velocity, int cm) and the provider's
    # calibration is the matching one — so the velocity prior is ON
    # here (it is the calibrated configuration). The default
    # real-run calibration stays UNCALIBRATED (velocity priors OFF)
    # until a real calibration is run (see tools/calibrate_tello.py
    # and docs/TELLO_CALIBRATION.md). The pose-source flag
    # (vision / attitude-aided / dead-reckoned / invalid) is
    # recorded per frame so the bridge never silently presents an
    # attitude-only pose as visual tracking.
    frames = scenarios.get_frames(
        name, detector=detector, drone_id="drone1", smooth_look=True)
    _poses = [p for _, p, _, _ in frames]
    _times = [m["t"] for _, _, _, m in frames]
    _provider = SyntheticStateProvider(
        _poses, _times, velocity_unit="dm/s", velocity_frame="global")
    _prior = StatePrior(calibration=_provider.calibration_for())
    _pkts = _provider.packets()
    if any(abs(o) > 0 for o in prior_offsets_deg):
        # Sensitivity test: shift the attitude the VIO sees, not the
        # video or the truth. Yaw wraps at +/-180; pitch/roll are
        # re-quantised to int degrees like the real packets.
        _dy, _dp, _dr = prior_offsets_deg
        _pkts = [dataclasses.replace(
            p,
            yaw=int(round(((p.yaw + _dy + 180.0) % 360.0) - 180.0)),
            pitch=int(round(p.pitch + _dp)),
            roll=int(round(p.roll + _dr))) for p in _pkts]
    for _pkt in _pkts:
        _prior.add(_pkt)
    pose_sources = {}
    for frame, pose, tracking_valid, meta in frames:
        t = meta["t"]
        if vio is None:
            # Known start pose: the drone takes off beside the phone.
            vio = MonocularVO(start_pose=pose, **(vio_kwargs or {}))
        t0 = time.perf_counter()
        est_pose, vio_valid, diag = vio.update(frame, t, state=_prior)
        vio_ms = diag.get("vio_ms", 0.0)
        pose_sources[diag.get("pose_source", "?")] = (
            pose_sources.get(diag.get("pose_source", "?"), 0) + 1)
        marker, info = pipeline.process_frame(
            frame, est_pose, vio_valid, t)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0
        all_frame_ms.append(pipeline_ms)
        if vio_valid and vio_init_frame is None:
            vio_init_frame = meta["frame"]
        if vio_valid:
            ever_valid = True
            drift = float(np.linalg.norm(est_pose[1] - pose[1]))
            # Yaw error of the VIO pose vs GT (wrapped, deg): the
            # diagnosed cause of the round-2 joint-error gap is turn
            # yaw error corrupting depth triangulation, so it is
            # measured per valid frame, not inferred from drift.
            _yaw_est = yaw_pitch_roll_from_R(est_pose[0])[0]
            _yaw_gt = yaw_pitch_roll_from_R(pose[0])[0]
            yaw_err = abs((_yaw_est - _yaw_gt + 180.0) % 360.0 - 180.0)
            vio_rows.append((meta["frame"], drift,
                             int(diag.get("inliers", 0)), vio_ms, yaw_err))
        elif ever_valid and vio_loss_frame is None:
            vio_loss_frame = meta["frame"]
        # True foot depth in this drone frame (GT pose — the reference
        # for the measured depth bias, same as the GT-pose runs).
        true_depth = None
        if meta["person_world"] is not None:
            p = project_point(DRONE_K, pose[0], pose[1],
                              np.array(meta["person_world"], dtype=float))
            true_depth = None if p is None else p[2]
        score = _score_marker(marker, meta)
        model_depth = info.get("model_depth")
        row = {
            "frame": meta["frame"], "t": t,
            "in_view": bool(meta["in_view"]),
            "detected": info["n_det"] > 0,
            "method": info["method"],
            "depth_m": info["depth"],
            "true_depth_m": true_depth,
            "model_depth_m": model_depth,
            "model_err_m": (None if model_depth is None or true_depth is None
                            else model_depth - true_depth),
            "voter_changed": bool(info.get("voter_changed")),
            "marker": marker is not None,
            "stale": bool(marker["stale"]) if marker else False,
            "bridge_valid": bool(info["bridge_valid"]),
            "joint_errs": score["joint_errs"],
            "dist_err": score["dist_err"],
            "scale_err": score["scale_err"],
            "dist_err_signed": score["dist_err_signed"],
            "merged": bool(score["merged"]),
            "nearest_dist_m": score["nearest_dist_m"],
        }
        rows.append(row)
        if meta["frame"] % 10 == 0:
            log.info("  %s (VIO) f%02d det=%d method=%s marker=%s "
                     "vio_valid=%s inl=%d",
                     name, meta["frame"], info["n_det"], info["method"],
                     "YES" if marker else "no",
                     "YES" if vio_valid else "no",
                     int(diag.get("inliers", 0)))
    pipeline.shutdown()  # never abandon a model worker mid-inference
    result = _aggregate(name, rows, vio_loss_frame)
    result["name"] = f"{name} (VIO)"
    result["vio_base"] = name
    result["model_err_mean"] = (
        float(np.mean([r["model_err_m"] for r in rows
                       if r.get("model_err_m") is not None]))
        if any(r.get("model_err_m") is not None for r in rows) else None)
    result["model_err_abs_mean"] = (
        float(np.mean([abs(r["model_err_m"]) for r in rows
                       if r.get("model_err_m") is not None]))
        if any(r.get("model_err_m") is not None for r in rows) else None)
    result["model_err_max"] = (
        float(np.max([abs(r["model_err_m"]) for r in rows
                      if r.get("model_err_m") is not None]))
        if any(r.get("model_err_m") is not None for r in rows) else None)
    result["model_n"] = sum(1 for r in rows
                            if r.get("model_err_m") is not None)
    result["voter_changed_frames"] = sum(
        1 for r in rows if r.get("voter_changed"))
    if vio_rows:
        drifts = np.array([r[1] for r in vio_rows])
        yaw_errs = np.array([r[4] for r in vio_rows])
        result["vio_valid_frames"] = len(vio_rows)
        result["vio_drift_mean_m"] = float(drifts.mean())
        result["vio_drift_max_m"] = float(drifts.max())
        result["vio_drift_final_m"] = float(drifts[-1])
        result["vio_inliers_mean"] = float(
            np.mean([r[2] for r in vio_rows]))
        result["vio_inliers_min"] = int(min(r[2] for r in vio_rows))
        result["vio_ms_mean"] = float(np.mean([r[3] for r in vio_rows]))
        result["vio_fps"] = float(1000.0 / np.mean([r[3] for r in vio_rows]))
        result["vio_yaw_err_mean_deg"] = float(yaw_errs.mean())
        result["vio_yaw_err_max_deg"] = float(yaw_errs.max())
        # Yaw error on the frames that actually produced a marker:
        # those are the frames whose yaw error reaches the joint
        # error (via the depth-triangulation ray), so the mean over
        # all valid frames understates the mechanism.
        _marker_frames = {r["frame"] for r in rows if r["marker"]}
        _marker_yaw = [r[4] for r in vio_rows
                       if r[0] in _marker_frames]
        result["vio_yaw_err_marker_frames_deg"] = (
            float(np.mean(_marker_yaw)) if _marker_yaw else None)
        result["vio_yaw_err_marker_frames_max_deg"] = (
            float(np.max(_marker_yaw)) if _marker_yaw else None)
    else:
        result["vio_valid_frames"] = 0
        result["vio_drift_mean_m"] = None
        result["vio_drift_max_m"] = None
        result["vio_drift_final_m"] = None
        result["vio_inliers_mean"] = None
        result["vio_inliers_min"] = None
        result["vio_ms_mean"] = None
        result["vio_fps"] = None
        result["vio_yaw_err_mean_deg"] = None
        result["vio_yaw_err_max_deg"] = None
        result["vio_yaw_err_marker_frames_deg"] = None
        result["vio_yaw_err_marker_frames_max_deg"] = None
    # Per-frame rows for analysis (not written to the report):
    # frame -> yaw error (deg) on valid frames, and the full rows.
    result["_vio_yaw_by_frame"] = {r[0]: r[4] for r in vio_rows}
    result["_rows"] = rows
    # Full-pipeline rate with VIO (VIO step + process_frame), over
    # ALL frames — the bridge and the detector still run on invalid
    # frames, and relocalisation attempts cost time. This is the
    # number that matters for real-time use (the 10 fps harness
    # needs < 100 ms/frame); the VIO step alone is reported
    # separately above.
    result["vio_pipeline_ms_mean"] = (
        float(np.mean(all_frame_ms)) if all_frame_ms else None)
    result["vio_pipeline_fps"] = (
        float(1000.0 / np.mean(all_frame_ms)) if all_frame_ms else None)
    result["vio_init_frame"] = vio_init_frame
    result["vio_loss_frame"] = vio_loss_frame
    result["vio_pose_sources"] = pose_sources
    return result


# ---------------------------------------------------------------------------
# s7: two drones (original, unchanged)
# ---------------------------------------------------------------------------

def run_s7(detector, mono=None):
    """Run s7: fused (shared bridge + fusion, two pipelines) + the two
    single-drone comparison runs over the hover frames 25-59.

    ``mono`` is an optional loaded ``MonoDepthModel`` shared by both
    pipelines (one model instance; each pipeline runs its own
    background worker against it).

    Returns the fused aggregate plus per-drone comparison numbers.
    """
    # --- fused run: one bridge + one fusion shared by both pipelines ---
    bridge = bridge_mod.Bridge()
    fusion = fusion_mod.Fusion(bridge)
    p1 = Pipeline(detector, drone_id="drone1", mono_depth=mono)
    p1.bridge = bridge
    p1.fusion = fusion
    p2 = Pipeline(detector, drone_id="drone2", mono_depth=mono)
    p2.bridge = bridge
    p2.fusion = fusion

    frames1 = scenarios.get_frames("s7_two_drones", detector=detector,
                                   drone_id="drone1")
    frames2 = scenarios.get_frames("s7_two_drones", detector=detector,
                                   drone_id="drone2")
    rows = []
    frame_ms = []
    for (f1, pose1, tv1, meta1), (f2, pose2, tv2, meta2) in zip(frames1, frames2):
        assert meta1["frame"] == meta2["frame"]
        t = meta1["t"]
        t0 = time.perf_counter()
        _, info1 = p1.process_frame(f1, pose1, tv1, t)
        marker, info2 = p2.process_frame(f2, pose2, tv2, t)
        frame_ms.append((time.perf_counter() - t0) * 1000.0)
        # marker from p2: computed after BOTH drones fed this timestamp.
        in_view = bool(meta1["in_view"] or meta2["in_view"])
        true_depth = None
        if meta1["person_world"] is not None:
            p = project_point(DRONE_K, pose2[0], pose2[1],
                              np.array(meta1["person_world"], dtype=float))
            true_depth = None if p is None else p[2]
        score = _score_marker(marker, meta1)
        model_depth = info2.get("model_depth")
        row = {
            "frame": meta1["frame"], "t": t, "in_view": in_view,
            "detected": info1["n_det"] > 0 or info2["n_det"] > 0,
            "method": info2["method"] or info1["method"],
            "depth_m": info2["depth"],
            "true_depth_m": true_depth,
            "model_depth_m": model_depth,
            "model_err_m": (None if model_depth is None or true_depth is None
                            else model_depth - true_depth),
            "voter_changed": bool(info2.get("voter_changed")),
            "marker": marker is not None,
            "stale": bool(marker["stale"]) if marker else False,
            "bridge_valid": bool(info2["bridge_valid"]),
            "joint_errs": score["joint_errs"],
            "dist_err": score["dist_err"],
            "scale_err": score["scale_err"],
            "dist_err_signed": score["dist_err_signed"],
            "merged": bool(score["merged"]),
            "nearest_dist_m": score["nearest_dist_m"],
        }
        rows.append(row)
        if meta1["frame"] % 10 == 0:
            log.info("  s7 f%02d det1=%d det2=%d marker=%s",
                     meta1["frame"], info1["n_det"], info2["n_det"],
                     "YES" if marker else "no")

    p1.shutdown()
    p2.shutdown()  # never abandon a model worker mid-inference
    fused = _aggregate("s7_two_drones", rows, None)
    # process_frame timing for the fused run (BOTH drones' calls per
    # timestamp), so the latency / FPS comparison covers s7 too.
    fused["frame_ms_mean"] = float(np.mean(frame_ms)) if frame_ms else None
    fused["pipeline_fps"] = (float(1000.0 / np.mean(frame_ms))
                             if frame_ms else None)
    # Per-drone detection rates (in-view of that drone only).
    for label, frames in (("drone1", frames1), ("drone2", frames2)):
        pass  # computed below from the re-run rows instead

    # --- single-drone comparison runs (hover frames only, same frames) ---
    hover_joint_errs = {}
    hover_det = {}
    for drone_id in ("drone1", "drone2"):
        pipe = Pipeline(detector, drone_id=drone_id)
        errs = []
        det_count = 0
        view_count = 0
        for frame, pose, tv, meta in scenarios.iter_frames(
                "s7_two_drones", detector=detector, drone_id=drone_id):
            if meta["frame"] < HOVER_START:
                continue
            marker, info = pipe.process_frame(frame, pose, tv, meta["t"])
            if meta["in_view"]:
                view_count += 1
                det_count += 1 if info["n_det"] > 0 else 0
            if marker is not None and meta["in_view"]:
                s = _score_marker(marker, meta)
                if s["joint_errs"]:
                    errs.extend(s["joint_errs"])
        hover_joint_errs[drone_id] = (
            float(np.mean(errs)) if errs else None)
        hover_det[drone_id] = (det_count, view_count)

    # Fused joint error over the same hover window, for the comparison.
    fused_hover_errs = []
    for r in rows:
        if r["frame"] >= HOVER_START and r["in_view"] and r["marker"] \
                and r["joint_errs"]:
            fused_hover_errs.extend(r["joint_errs"])
    fused_hover_mean = (float(np.mean(fused_hover_errs))
                        if fused_hover_errs else None)
    best_single = min(v for v in hover_joint_errs.values() if v is not None)
    fused["s7_fused_hover_joint_err_m"] = fused_hover_mean
    fused["s7_single_joint_err_m"] = dict(hover_joint_errs)
    fused["s7_single_detection"] = {k: {"detected": v[0], "in_view": v[1]}
                                    for k, v in hover_det.items()}
    fused["s7_best_single_joint_err_m"] = best_single
    fused["checks"]["fused_le_best_single"] = (
        fused_hover_mean is not None and fused_hover_mean <= best_single)
    fused["passed"] = bool(fused["checks"]) and all(fused["checks"].values())
    return fused


# ---------------------------------------------------------------------------
# Multi-drone scenarios (4 and 5 drones, generalised from run_s7)
# ---------------------------------------------------------------------------

def run_multi(name, detector):
    """Run a multi-drone scenario (3+ drones) through the real pipeline.

    One shared bridge + one shared fusion for all drones (the COVINS/CCM
    pattern); per-drone pipelines feed it in timestamp order. The marker
    is taken from the last pipeline processed at each timestamp, i.e.
    after ALL drones fed that timestamp. Single-drone comparison runs
    over the hover frames give the fused-vs-best-single check. Frame
    streams are generators (not lists) so 5 drones do not hold 5x60
    frames in memory at once.
    """
    cfg = scenarios.ALL_SCENARIOS[name]
    n = len(cfg["hovers"])
    phone_pos = np.asarray(cfg.get("phone", (0.0, 1.5, 0.0)), dtype=float)
    bridge = bridge_mod.Bridge()
    fusion = fusion_mod.Fusion(bridge, phone_position=phone_pos)
    pipes = []
    for i in range(n):
        p = Pipeline(detector, drone_id=f"drone{i + 1}",
                     phone_position=phone_pos)
        p.bridge = bridge
        p.fusion = fusion
        pipes.append(p)

    rows = []
    per_drone = {f"drone{i + 1}": {"in_view": 0, "detected": 0,
                                  "in_view_hover": 0, "detected_hover": 0}
                 for i in range(n)}
    for frame_tuple in scenarios.iter_frames_multi(name, detector=detector):
        t = frame_tuple[0][3]["t"]
        marker = None
        infos = []
        for i, (f, pose, tv, meta) in enumerate(frame_tuple):
            m, info = pipes[i].process_frame(f, pose, tv, t)
            infos.append(info)
            marker = m  # last pipeline: after ALL drones fed timestamp t
            d = per_drone[f"drone{i + 1}"]
            if meta["in_view"]:
                d["in_view"] += 1
                if info["n_det"] > 0:
                    d["detected"] += 1
            if meta["frame"] >= HOVER_START and meta["in_view"]:
                d["in_view_hover"] += 1
                if info["n_det"] > 0:
                    d["detected_hover"] += 1
        meta1 = frame_tuple[0][3]
        in_view = any(ft[3]["in_view"] for ft in frame_tuple)
        detected = any(inf["n_det"] > 0 for inf in infos)
        method = infos[-1]["method"] or next(
            (inf["method"] for inf in reversed(infos) if inf["method"]),
            None)
        true_depth = None
        if meta1["person_world"] is not None:
            last_pose = frame_tuple[-1][1]
            p = project_point(DRONE_K, last_pose[0], last_pose[1],
                              np.array(meta1["person_world"], dtype=float))
            true_depth = None if p is None else p[2]
        score = _score_marker(marker, meta1)
        row = {
            "frame": meta1["frame"], "t": t, "in_view": in_view,
            "detected": detected,
            "method": method,
            "depth_m": infos[-1]["depth"],
            "true_depth_m": true_depth,
            "marker": marker is not None,
            "stale": bool(marker["stale"]) if marker else False,
            "bridge_valid": bool(infos[-1]["bridge_valid"]),
            "joint_errs": score["joint_errs"],
            "dist_err": score["dist_err"],
            "scale_err": score["scale_err"],
            "dist_err_signed": score["dist_err_signed"],
            "merged": bool(score["merged"]),
            "nearest_dist_m": score["nearest_dist_m"],
        }
        rows.append(row)
        if meta1["frame"] % 10 == 0:
            log.info("  %s f%02d det=%s marker=%s", name, meta1["frame"],
                     [inf["n_det"] for inf in infos],
                     "YES" if marker else "no")

    for p in pipes:
        p.shutdown()  # never abandon a model worker mid-inference
    fused = _aggregate(name, rows, None)
    fused["multi_per_drone_detection"] = per_drone

    # --- single-drone comparison runs (hover frames only) ---
    hover_joint_errs = {}
    hover_det = {}
    for i in range(n):
        drone_id = f"drone{i + 1}"
        pipe = Pipeline(detector, drone_id=drone_id,
                        phone_position=phone_pos)
        errs = []
        det_count = 0
        view_count = 0
        for frame, pose, tv, meta in scenarios.iter_frames(
                name, detector=detector, drone_id=drone_id):
            if meta["frame"] < HOVER_START:
                continue
            marker, info = pipe.process_frame(frame, pose, tv, meta["t"])
            if meta["in_view"]:
                view_count += 1
                det_count += 1 if info["n_det"] > 0 else 0
            if marker is not None and meta["in_view"]:
                s = _score_marker(marker, meta)
                if s["joint_errs"]:
                    errs.extend(s["joint_errs"])
        hover_joint_errs[drone_id] = (
            float(np.mean(errs)) if errs else None)
        hover_det[drone_id] = (det_count, view_count)

    fused_hover_errs = []
    for r in rows:
        if r["frame"] >= HOVER_START and r["in_view"] and r["marker"] \
                and r["joint_errs"]:
            fused_hover_errs.extend(r["joint_errs"])
    fused_hover_mean = (float(np.mean(fused_hover_errs))
                        if fused_hover_errs else None)
    singles = [v for v in hover_joint_errs.values() if v is not None]
    best_single = min(singles) if singles else None
    fused["multi_fused_hover_joint_err_m"] = fused_hover_mean
    fused["multi_single_joint_err_m"] = dict(hover_joint_errs)
    fused["multi_single_detection"] = {
        k: {"detected": v[0], "in_view": v[1]}
        for k, v in hover_det.items()}
    fused["multi_best_single_joint_err_m"] = best_single
    fused["checks"]["fused_le_best_single"] = (
        fused_hover_mean is not None and best_single is not None
        and fused_hover_mean <= best_single)
    fused["passed"] = bool(fused["checks"]) and all(fused["checks"].values())
    return fused


# ---------------------------------------------------------------------------
# No-drone-video test (real HTTP server from server.main)
# ---------------------------------------------------------------------------

VIDEO_ENDPOINT_HINTS = ("/api/video", "drone-stream", "video.mp4", ".m3u8",
                        "mjpeg", "/stream", "/frames", "rtsp://", "webrtc")
UNKNOWN_ROUTES = ("/video", "/stream", "/frames", "/api/video",
                  "/api/frames", "/drone", "/api/drone", "/snapshot",
                  "/mjpeg", "/api/markers/1", "/admin", "/api/status/extra")


def no_drone_video_test(port=18765, sse_seconds=3.0):
    """Start the real server; assert the phone never receives drone video.

    Returns (passed, findings dict).
    """
    findings = {"checks": {}, "notes": []}
    state = ServerState()
    handler = make_handler(state, VIEWER_PATH)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    actual_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    findings["port"] = actual_port
    try:
        def get(path, timeout=5.0):
            conn = http.client.HTTPConnection("127.0.0.1", actual_port,
                                              timeout=timeout)
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            headers = dict(resp.getheaders())
            conn.close()
            return resp.status, headers, body

        # / -> 302 to /viewer
        status, headers, _ = get("/")
        findings["checks"]["root_redirects_to_viewer"] = (
            status == 302 and headers.get("Location") == "/viewer")

        # /viewer: exactly one <video>, no drone-stream reference.
        status, headers, body = get("/viewer")
        html = body.decode("utf-8", errors="replace")
        n_video = html.lower().count("<video")
        findings["checks"]["viewer_200_html"] = (
            status == 200 and "text/html" in headers.get("Content-Type", ""))
        findings["checks"]["exactly_one_video_tag"] = n_video == 1
        findings["video_tag_count"] = n_video
        findings["checks"]["no_img_tag"] = "<img" not in html.lower()
        findings["checks"]["no_fetch_or_xhr"] = (
            "fetch(" not in html and "XMLHttpRequest" not in html)
        hints = [h for h in VIDEO_ENDPOINT_HINTS
                 if h.lower() in html.lower()]
        findings["checks"]["no_drone_stream_reference"] = not hints
        findings["video_hints_found"] = hints
        findings["checks"]["uses_getusermedia"] = "getUserMedia" in html
        findings["checks"]["no_video_content_type"] = (
            "video/" not in headers.get("Content-Type", ""))

        # /api/status: JSON with the SPEC shape, including the two
        # operator fields (null until the first frame — never a
        # made-up 0).
        status, headers, body = get("/api/status")
        try:
            payload = json.loads(body.decode("utf-8"))
            ok_status = (status == 200
                         and "application/json" in headers.get("Content-Type", "")
                         and set(("bridge_valid", "drones", "tracks",
                                  "uptime_s", "last_frame_age_s",
                                  "frames_last_10s")) <= set(payload))
            ok_status_fields_null = (
                payload.get("last_frame_age_s") is None
                and payload.get("frames_last_10s") is None)
        except Exception:
            ok_status = False
            ok_status_fields_null = False
        findings["checks"]["status_json_shape"] = ok_status
        findings["checks"]["status_new_fields_null_before_first_frame"] = (
            ok_status_fields_null)

        # /operator: laptop page, 200 HTML, no video, no image, no
        # video content-type; the facts line reads "—" before the
        # first frame (the state here has processed no frames).
        status, headers, body = get("/operator")
        op_html = body.decode("utf-8", errors="replace")
        findings["checks"]["operator_200_html"] = (
            status == 200 and "text/html" in headers.get("Content-Type", ""))
        findings["checks"]["operator_no_video_tag"] = "<video" not in op_html.lower()
        findings["checks"]["operator_no_img_tag"] = "<img" not in op_html.lower()
        findings["checks"]["operator_no_video_content_type"] = (
            "video/" not in headers.get("Content-Type", ""))
        findings["checks"]["operator_has_three_blocks"] = all(
            w in op_html for w in ("BRIDGE", "DRONE", "PERSON"))
        findings["checks"]["operator_facts_dash_before_first_frame"] = (
            "—" in op_html)

        # /api/markers: JSON-only SSE payloads for a few seconds.
        conn = http.client.HTTPConnection("127.0.0.1", actual_port, timeout=6)
        conn.request("GET", "/api/markers")
        resp = conn.getresponse()
        ctype = resp.headers.get("Content-Type", "")
        findings["checks"]["sse_content_type"] = (
            resp.status == 200 and "text/event-stream" in ctype)
        findings["checks"]["sse_no_video_content_type"] = "video/" not in ctype
        events = []
        deadline = time.time() + sse_seconds
        buf = b""
        while time.time() < deadline:
            try:
                chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.decode("utf-8", errors="replace").splitlines():
                    if line.startswith("data:"):
                        events.append(line[5:].strip())
        conn.close()
        parsed = []
        for ev in events:
            try:
                parsed.append(json.loads(ev))
            except Exception:
                parsed.append(None)
        findings["sse_events_seen"] = len(events)
        findings["checks"]["sse_events_received"] = len(events) >= 1
        findings["checks"]["sse_all_json"] = (
            len(events) > 0 and all(p is not None for p in parsed))
        findings["checks"]["sse_types_json_only"] = all(
            isinstance(p, dict) and p.get("type") in ("marker", "markers", "none")
            for p in parsed if p is not None)

        # No other routes exist: probe list must all 404.
        not_found = []
        for path in UNKNOWN_ROUTES:
            status, _, _ = get(path)
            not_found.append((path, status))
        findings["unknown_route_statuses"] = dict(not_found)
        findings["checks"]["unknown_routes_all_404"] = all(
            s == 404 for _, s in not_found)
    finally:
        httpd.shutdown()
        httpd.server_close()
    findings["passed"] = all(findings["checks"].values())
    return findings["passed"], findings


# ---------------------------------------------------------------------------
# Report (APPEND-MODE: the 2026-11-23 results stay intact on top)
# ---------------------------------------------------------------------------

def _fmt_m(x):
    return "n/a" if x is None else f"{x * 100:.1f} cm"


def _fmt_pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _scenario_detail_lines(r):
    """Per-scenario detail lines (shared by the original + new sections)."""
    lines = []
    lines.append(f"### {r['name']} — {r['desc']}")
    lines.append("")
    lines.append(f"- Frames: {r['n_frames']} @ 10 fps; in-view frames "
                 f"(whole person in frame, >= 80 px tall): "
                 f"{r['n_in_view']}; frames excluded from the metrics "
                 f"(person partly out of frame, e.g. drone flying past "
                 f"them): {r['n_not_in_view']}.")
    lines.append(f"- Markers emitted: {r['n_markers']} "
                 f"({r['n_stale_markers']} stale); joint-error samples: "
                 f"{r['n_joint_samples']}; distance-error samples: "
                 f"{r['n_dist_samples']}.")
    lines.append(f"- Worst single-frame joint error: "
                 f"{_fmt_m(r['joint_err_max'])}; worst single-frame "
                 f"distance error: {_fmt_m(r['dist_err_max'])} "
                 f"(means hide bad frames — these do not).")
    lines.append(f"- Markers emitted while the bridge was invalid: "
                 f"{r['marker_while_bridge_invalid']} (must be 0).")
    if r["n_persons"] > 1:
        lines.append(f"- Merged markers (marker > 1.0 m from BOTH "
                     f"people): {r['merged_markers']} (must be 0).")
    if r["name"] in ("s6_no_person", "s24_empty_new_room"):
        lines.append(f"- False markers (empty room): {r['false_markers']}.")
    if r["marker_after_loss"]:
        lines.append(f"- Markers at/after the tracking-loss frame: "
                     f"{r['marker_after_loss']} (must be 0).")
    if r["marker_during_loss"] or r["first_marker_after_recovery"] is not None:
        lines.append(f"- Markers during the loss window: "
                     f"{r['marker_during_loss']} (must be 0); first "
                     f"marker after recovery: frame "
                     f"{r['first_marker_after_recovery']}.")
    lines.append(f"- Depth methods: {r['methods']}; first "
                 f"triangulation frame: "
                 f"{r['first_triangulation_frame']}.")
    bb = r["depth_bias_m"].get("bbox_height")
    tb = r["depth_bias_m"].get("triangulation")
    lines.append(f"- Measured depth bias (estimated - true foot depth): "
                 f"bbox-height "
                 f"{'n/a' if bb is None else f'{bb * 100:+.0f} cm (n={r['depth_bias_n']['bbox_height']})'}; "
                 f"triangulation "
                 f"{'n/a' if tb is None else f'{tb * 100:+.0f} cm (n={r['depth_bias_n']['triangulation']})'}.")
    sd = r.get("dist_err_signed_mean")
    lines.append(f"- Mean signed distance error (marker - GT): "
                 f"{'n/a' if sd is None else f'{sd * 100:+.0f} cm'} "
                 f"(negative = marker distance shorter than the true "
                 f"phone->person distance).")
    if r.get("frame_ms_mean") is not None:
        lines.append(f"- Pipeline latency (process_frame, incl. detector): "
                     f"{r['frame_ms_mean']:.1f} ms/frame mean "
                     f"({r['pipeline_fps']:.1f} fps).")
    if r.get("model_n"):
        lines.append(f"- Monocular model vote (Depth Anything V2 Small, "
                     f"floor-anchored): depth error vs GT foot depth "
                     f"mean {r['model_err_mean'] * 100:+.0f} cm, "
                     f"mean |err| {r['model_err_abs_mean'] * 100:.0f} cm, "
                     f"worst {r['model_err_max'] * 100:.0f} cm "
                     f"(n={r['model_n']} frames with a vote); voter "
                     f"changed the outcome on "
                     f"{r['voter_changed_frames']} frame(s).")
    if "s7_fused_hover_joint_err_m" in r:
        lines.append(
            f"- Fused mean joint error, hover frames 25-59: "
            f"{_fmt_m(r['s7_fused_hover_joint_err_m'])}; single-drone "
            f"mean joint error over the same frames: drone1 "
            f"{_fmt_m(r['s7_single_joint_err_m']['drone1'])}, drone2 "
            f"{_fmt_m(r['s7_single_joint_err_m']['drone2'])}; best "
            f"single: {_fmt_m(r['s7_best_single_joint_err_m'])}. "
            f"Single-drone detection (hover frames): "
            f"{r['s7_single_detection']}.")
    if "multi_fused_hover_joint_err_m" in r:
        singles = ", ".join(
            f"{k} {_fmt_m(v)}"
            for k, v in r["multi_single_joint_err_m"].items())
        lines.append(
            f"- Fused mean joint error, hover frames 25-59: "
            f"{_fmt_m(r['multi_fused_hover_joint_err_m'])}; single-drone "
            f"mean joint error over the same frames: {singles}; best "
            f"single: {_fmt_m(r['multi_best_single_joint_err_m'])}. "
            f"Single-drone detection (hover frames): "
            f"{r['multi_single_detection']}.")
        lines.append(
            f"- Per-drone detection in the fused run (all frames / "
            f"hover frames): {r['multi_per_drone_detection']}.")
    for check, ok in r["checks"].items():
        lines.append(f"- check `{check}`: {'PASS' if ok else 'FAIL'}")
    lines.append(f"- **Result: {'PASS' if r['passed'] else 'FAIL'}**")
    lines.append("")
    return lines


def write_report(results, video_findings, runtime_s, started_at,
                 scenarios_run, vio_results=None, mono_results=None,
                 mono_vio_results=None, mono_bench=None, mono_stats=None,
                 real_only=None):
    """Append a dated extended section to tests/TEST-REPORT.md.

    The original 2026-11-23 report (everything above the first
    "## Extended tests" marker) is kept verbatim. Re-running replaces
    the previous extended section, so repeated runs stay idempotent.
    """
    import cv2
    import numpy
    import ultralytics

    existing = ""
    if REPORT_PATH.exists():
        existing = REPORT_PATH.read_text(encoding="utf-8")
    marker = "## Extended tests"
    if marker in existing:
        head = existing[:existing.index(marker)].rstrip() + "\n\n"
    else:
        head = existing.rstrip() + "\n\n"

    lines = []
    lines.append(f"## Extended tests — {started_at} (UTC)")
    lines.append("")
    lines.append("New rooms/places, multi-drone (4 and 5 drones), and "
                 "adversarial scenarios on top of the original 7. "
                 "Thresholds unchanged from SPEC.md (detection >= 90%, "
                 "mean joint error <= 60 cm, mean |distance error| "
                 "<= 60 cm, skeleton scale error <= 25%, false markers "
                 "= 0, marker-after-loss = 0, fused error <= best "
                 "single-drone error).")
    lines.append("")
    lines.append(f"Scenarios run: {', '.join(scenarios_run)}")
    lines.append("")
    lines.append("## Environment (extended run)")
    lines.append("")
    lines.append(f"- Python {sys.version.split()[0]}")
    lines.append(f"- opencv {cv2.__version__}, numpy {numpy.__version__}, "
                 f"ultralytics {ultralytics.__version__}")
    lines.append(f"- Weights: `yolov8n-pose.pt` (in `wall-marker/`), "
                 f"detector conf 0.5, imgsz 640, CPU")
    lines.append(f"- Total runtime: {runtime_s:.1f} s "
                 f"({runtime_s / 60:.1f} min), one inference process")
    lines.append("- Person billboard: largest person detection from "
                 "ultralytics' bundled `bus.jpg`, cut out with grabCut "
                 "(mask eroded 2x), pasted at the TRUE projected position. "
                 "Two-person scenarios paste the same cutout twice at "
                 "different positions (identical twins — documented).")
    lines.append("- Room model: SPEC.md default (wall z = 3.0 m, doorway "
                 "x in [-0.6, 0.6] m, phone fixed at (0, 1.5, 0)) unless "
                 "a scenario overrides it (see its description). "
                 "Projection `v = cy - fy*y/z`.")
    lines.append("")
    lines.append("## Extended results")
    lines.append("")
    lines.append("| Scenario | Detection rate | Mean joint err | Worst joint err | "
                 "Mean |dist err| | Worst dist err | Scale err | "
                 "False markers | Bridge-invalid markers | Result |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        dr = ("n/a" if r["detection_rate"] is None
              else f"{r['detection_rate'] * 100:.0f}% "
                   f"({r['n_detected_in_view']}/{r['n_in_view']})")
        lines.append(
            f"| {r['name']} | {dr} | {_fmt_m(r['joint_err_mean'])} | "
            f"{_fmt_m(r['joint_err_max'])} | {_fmt_m(r['dist_err_mean'])} | "
            f"{_fmt_m(r['dist_err_max'])} | {_fmt_pct(r['scale_err_mean'])} | "
            f"{r['false_markers']} | {r['marker_while_bridge_invalid']} | "
            f"{'PASS' if r['passed'] else 'FAIL'} |")
    lines.append("")
    lines.append("Thresholds (SPEC.md, not moved): detection rate >= 90%, "
                 "mean joint error <= 60 cm, mean |distance error| <= 60 cm, "
                 "skeleton scale error <= 25%, false markers = 0 (empty "
                 "rooms), marker-after-loss = 0 (s5), marker-during-loss = 0 "
                 "(s21), no merged markers (s19), marker-while-bridge-"
                 "invalid = 0 (all scenarios), fused error <= best "
                 "single-drone error (multi-drone). Worst-frame errors are "
                 "reported, not thresholded — they are the adversarial "
                 "signal, not the pass criterion.")
    lines.append("")
    # Regression comparison for the original 7.
    lines.append("### Regression: original 7 vs the recorded 2026-11-23 run")
    lines.append("")
    lines.append("The original 7 scenarios were re-run in this extended "
                 "run. Rendering and inference are deterministic, so the "
                 "numbers must reproduce the recorded run (tolerance "
                 "0.5 cm on means, identical detection counts and "
                 "PASS/FAIL). Any drift means the refactor changed the "
                 "original rendering — that would be a bug, not noise.")
    lines.append("")
    lines.append("| Scenario | Detection (recorded / re-run) | Mean joint "
                 "(recorded / re-run) | Mean |dist| (recorded / re-run) | "
                 "Scale (recorded / re-run) | Match |")
    lines.append("|---|---|---|---|---|---|")
    for name in scenarios.list_scenarios():
        rec = RECORDED_2026_11_23[name]
        r = next((x for x in results if x["name"] == name), None)
        if r is None:
            lines.append(f"| {name} | — | — | — | — | NOT RUN |")
            continue
        def _cmp(key, fmt):
            rv, nv = rec.get(key), r.get(key)
            if rv is None or nv is None:
                return ("n/a", "n/a", rv == nv)
            return (fmt(rv), fmt(nv), abs(rv - nv) <= 0.005)
        dr = (f"{rec['detection_rate'] * 100:.0f}%" if rec.get("detection_rate") is not None else "n/a",
              f"{r['detection_rate'] * 100:.0f}%" if r["detection_rate"] is not None else "n/a",
              rec.get("detection_rate") == r["detection_rate"])
        j = _cmp("joint_err_mean", lambda v: f"{v * 100:.1f} cm")
        d = _cmp("dist_err_mean", lambda v: f"{v * 100:.1f} cm")
        s = _cmp("scale_err_mean", lambda v: f"{v * 100:.1f}%")
        match = (dr[2] and j[2] and d[2] and s[2]
                 and rec.get("passed") == r["passed"])
        lines.append(
            f"| {name} | {dr[0]} / {dr[1]} | {j[0]} / {j[1]} | "
            f"{d[0]} / {d[1]} | {s[0]} / {s[1]} | "
            f"{'MATCH' if match else 'MISMATCH'} |")
    lines.append("")
    for r in results:
        lines.extend(_scenario_detail_lines(r))
    lines.append("## No-drone-video test (extended run)")
    lines.append("")
    passed, findings = video_findings
    if not findings.get("checks"):
        lines.append("- SKIPPED (--skip-video-test); no findings recorded.")
        lines.append("")
    else:
        lines.append(f"- Server started on 127.0.0.1:{findings['port']} "
                     f"(real `server.main` handler).")
        lines.append(f"- `/viewer`: {findings['video_tag_count']} `<video>` tag(s); "
                     f"getUserMedia present: "
                     f"{findings['checks']['uses_getusermedia']}; "
                     f"video-endpoint hints found: "
                     f"{findings['video_hints_found'] or 'none'}.")
        lines.append(f"- `/api/markers`: {findings['sse_events_seen']} SSE events "
                     f"read in 3 s; all parse as JSON with type marker/none: "
                     f"{findings['checks']['sse_all_json'] and findings['checks']['sse_types_json_only']}; "
                     f"content-type text/event-stream, no video content-type "
                     f"anywhere: {findings['checks']['sse_no_video_content_type']}.")
        lines.append(f"- Unknown routes probed ({len(UNKNOWN_ROUTES)}): all 404: "
                     f"{findings['checks']['unknown_routes_all_404']} "
                     f"({findings['unknown_route_statuses']}).")
        for check, ok in findings["checks"].items():
            lines.append(f"- check `{check}`: {'PASS' if ok else 'FAIL'}")
        lines.append(f"- **Result: {'PASS' if passed else 'FAIL'}**")
        lines.append("")
    lines.append("## UI server part — real-only tests (extended run)")
    lines.append("")
    if not real_only or not real_only[1].get("tests"):
        lines.append("- SKIPPED (--skip-real-only); no findings recorded.")
        lines.append("")
    else:
        ro_passed, ro = real_only
        for name, f in ro["tests"].items():
            lines.append(f"- **{name}**: "
                         f"{'PASS' if f.get('passed') else 'FAIL'}")
            for check, ok in (f.get("checks") or {}).items():
                lines.append(f"  - check `{check}`: "
                             f"{'PASS' if ok else 'FAIL'}")
            if "events_seen" in f:
                lines.append(f"  - SSE events read: {f['events_seen']}; "
                             f"types seen: {f.get('event_types')}")
            if "runs" in f:
                for label, run in f["runs"].items():
                    lines.append(f"  - product server --mode {label}: "
                                 f"exit {run['returncode']}")
            if "result" in f:
                r = f["result"]
                lines.append(
                    f"  - viewer canvas {r.get('canvasW')}x{r.get('canvasH')}: "
                    f"{r.get('nonTransparentPixels')} non-transparent "
                    f"pixels after 60 s of none events; markers held: "
                    f"{r.get('markersLen')}; distance pills: "
                    f"{r.get('distancesChildren')}; last SSE event "
                    f"{r.get('lastEventAgeMs')} ms before the read; "
                    f"console errors: {f.get('console_errors') or 'none'}.")
            if f.get("error"):
                lines.append(f"  - error: {f['error']}")
        lines.append(f"- **Result: {'PASS' if ro_passed else 'FAIL'}**")
        lines.append("")
    lines.append("## Found and fixed (extended suite)")
    lines.append("")
    for fix in FIXES_EXTENDED_2026_11_23:
        lines.append(f"- {fix}")
    lines.append("")
    lines.append("## Honest limits — extended scenarios (what this still "
                 "does NOT prove)")
    lines.append("")
    lines.append("- Virtual poses are exact: the bridge is fed the true "
                 "drone pose, so no SLAM drift is simulated. The bridge "
                 "uncertainty (0.10 + 0.02 s/s, capped at 1.0 m) is a model, "
                 "not a measurement. The research report's end-to-end "
                 "budget (±20-60 cm at 3-6 m with good depth, ±0.5-2 m "
                 "monocular/Tello) still stands for the real system.")
    lines.append("- s23_blank_door stresses a featureless wall at the "
                 "doorway, but the virtual bridge does not do visual "
                 "feature tracking at all (it is fed true poses), so the "
                 "real VIO bridge stress of a blank wall is NOT measured "
                 "here — the scenario only proves detection/rendering "
                 "still works with a blank wall. A real VIO bridge could "
                 "still lose tracking at a blank doorway.")
    lines.append("- s21_loss_recovery proves the state-machine logic "
                 "(marker hidden during loss, returns only after "
                 "re-establishment + a fresh detection). The virtual "
                 "reacquire stands in for a real relocalisation step "
                 "with the true pose; a real relocalisation (re-observing "
                 "a mapped area / doorway loop closure) is not built or "
                 "tested here.")
    lines.append("- s16_sideways is an approximation: a sideways body is "
                 "rendered as a narrowed frontal billboard, not a true "
                 "profile (one shoulder hidden, different arm pose). The "
                 "measured joint errors are a lower bound on real "
                 "sideways difficulty.")
    lines.append("- s18_moving: the person walks slowly (0.28 m/s) and "
                 "triangulation assumes a static scene. See the scenario "
                 "section for the measured numbers; the residual is the "
                 "honest signal that the static assumption broke.")
    lines.append("- s15_occluded: the occluder is a flat wardrobe plane "
                 "drawn after the billboard; a real occluder (doorframe "
                 "edge, furniture) would also change the person's visible "
                 "shape in 3D. Detection of a half-hidden person is "
                 "measured, not assumed.")
    lines.append("- s13_dim_noise uses synthetic Gaussian noise (sigma 12 "
                 "on 0-255 after a 0.4x brightness factor) — real low-light "
                 "noise is signal-dependent and coloured; this is a harder "
                 "test than brightness alone, but not a real night shot.")
    lines.append("- s14_glare uses a bright window quad + a Gaussian bloom "
                 "(radius 200 px, strength 0.35) — a synthetic glare model, "
                 "not a real sunlit window with lens flare.")
    lines.append("- Two-person scenarios paste the SAME cutout twice "
                 "(identical twins). Real second people differ in height, "
                 "clothing and pose; the merge check (no midpoint marker) "
                 "is the point of the scenario, not the realism of the "
                 "second person.")
    lines.append("- One static room pair per scenario, no occlusion beyond "
                 "s15, no motion blur, no network latency, no live-phone "
                 "rendering check (the viewer projection is verified by "
                 "code inspection + the GT phone projection in the "
                 "scenarios, not by a live phone).")
    lines.append("- Detection at 640 px on CPU.")
    lines.append("")
    lines.append("## Method used per frame (extended scenarios)")
    lines.append("")
    lines.append("Per scenario, frames using triangulation vs bbox-height "
                 "are counted in the table above. In every person scenario "
                 "the pipeline switches to triangulation as soon as the "
                 "drone's motion baseline reaches 0.3 m (a few frames after "
                 "the person first appears) and stays there; bbox-height is "
                 "used only for the first 1-3 detections. Exact per-frame "
                 "method sequences:")
    lines.append("")
    for r in results:
        seq = []
        last = None
        start = 0
        for row in r["rows"]:
            m = row["method"] or "-"
            if m != last:
                if last is not None:
                    seq.append(f"f{start}-{row['frame'] - 1}:{last}")
                last, start = m, row["frame"]
        if last is not None:
            seq.append(f"f{start}-{r['rows'][-1]['frame']}:{last}")
        lines.append(f"- {r['name']}: " + (", ".join(seq) if seq else "no detections"))
    lines.append("")
    if vio_results:
        lines.append("## VIO tests")
        lines.append("")
        lines.append("The same Pipeline, but the bridge is fed "
                     "VIO-estimated poses (``server/vio.py``) instead of "
                     "ground truth. The VIO is initialised with the first "
                     "frame's GT pose as the known start pose (the drone "
                     "takes off beside the phone); from then on only the "
                     "video frames drive it. The GT-pose runs above are "
                     "untouched — the VIO results are a separate table so "
                     "the regression comparison still reads the GT runs.")
        lines.append("")
        lines.append("VIO pose drift is reported against the 60 cm "
                     "joint-error budget (SPEC.md) — there is no separate "
                     "VIO threshold, and none was invented. Frame rate is "
                     "the VIO step alone and the full pipeline with VIO "
                     "(VIO + detector + depth + bridge + fusion) on CPU; "
                     "the 10 fps harness needs < 100 ms/frame.")
        lines.append("")
        lines.append("| Scenario (VIO) | VIO valid | Drift mean | Drift max | "
                     "Yaw err mean | Yaw err max | Yaw err, marker frames | "
                     "PnP inliers mean/min | VIO fps | Pipeline fps | "
                     "Detection | Mean joint err | Mean |dist err| | "
                     "Markers | Bridge-invalid markers | Result |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in vio_results:
            dr = ("n/a" if r["detection_rate"] is None
                  else f"{r['detection_rate'] * 100:.0f}% "
                       f"({r['n_detected_in_view']}/{r['n_in_view']})")
            _ym = r.get("vio_yaw_err_mean_deg")
            _yx = r.get("vio_yaw_err_max_deg")
            _yfm = r.get("vio_yaw_err_marker_frames_deg")
            lines.append(
                f"| {r['vio_base']} | {r['vio_valid_frames']}/"
                f"{r['n_frames']} | {_fmt_m(r['vio_drift_mean_m'])} | "
                f"{_fmt_m(r['vio_drift_max_m'])} | "
                f"{'n/a' if _ym is None else f'{_ym:.1f} deg'} | "
                f"{'n/a' if _yx is None else f'{_yx:.1f} deg'} | "
                f"{'n/a' if _yfm is None else f'{_yfm:.1f} deg'} | "
                f"{r['vio_inliers_mean']:.0f}/{r['vio_inliers_min']} | "
                f"{r['vio_fps']:.1f} | {r['vio_pipeline_fps']:.1f} | "
                f"{dr} | {_fmt_m(r['joint_err_mean'])} | "
                f"{_fmt_m(r['dist_err_mean'])} | {r['n_markers']} | "
                f"{r['marker_while_bridge_invalid']} | "
                f"{'PASS' if r['passed'] else 'FAIL'} |")
        lines.append("")
        # Round 3 vs round 2: the round-2 VIO table is replaced by
        # this run's table above, so the round-2 numbers are recorded
        # here (from TEST-REPORT.md, VIO tests, 2026-11-23 11:41 UTC
        # run) to keep the comparison readable in one place.
        lines.append("### VIO fix round 3 — constrained PnP (round 2 vs round 3)")
        lines.append("")
        lines.append("Round 2 (2026-11-23) left joint error over the "
                     "60 cm budget in s1_basic (71.8 cm) and "
                     "s23_blank_door (81.7 cm) with turn yaw error "
                     "1.0-1.2 deg mean, 4.0-4.5 deg max corrupting "
                     "depth triangulation. Round 3 constrains the PnP "
                     "itself with the Tello attitude prior — see "
                     "``server/vio.py`` ``pnp_mode`` — instead of "
                     "correcting the yaw after the fact (tried and "
                     "reverted in round 2: drift improved, joint "
                     "error worsened to 111 cm, because changing R "
                     "without re-optimising C breaks consistency).")
        lines.append("")
        lines.append("| Scenario | Joint err, round 2 / round 3 | "
                     "|Dist err|, round 2 / round 3 | Drift mean, "
                     "round 2 / round 3 | Markers, round 2 / round 3 |")
        lines.append("|---|---|---|---|---|")
        for r in vio_results:
            rec = VIO_ROUND2_2026_11_23.get(r["vio_base"])
            if rec is None:
                continue
            lines.append(
                f"| {r['vio_base']} | "
                f"{_fmt_m(rec['joint_err_mean'])} / "
                f"{_fmt_m(r['joint_err_mean'])} | "
                f"{_fmt_m(rec['dist_err_mean'])} / "
                f"{_fmt_m(r['dist_err_mean'])} | "
                f"{_fmt_m(rec['vio_drift_mean_m'])} / "
                f"{_fmt_m(r['vio_drift_mean_m'])} | "
                f"{rec['n_markers']} / {r['n_markers']} |")
        lines.append("")
        lines.append("### VIO vs ground-truth poses (same scenarios)")
        lines.append("")
        lines.append("| Scenario | Joint err, GT poses | Joint err, VIO poses | "
                     "|Dist err|, GT poses | |Dist err|, VIO poses | "
                     "Markers, GT / VIO |")
        lines.append("|---|---|---|---|---|---|")
        for r in vio_results:
            g = next((x for x in results if x["name"] == r["vio_base"]),
                     None)
            lines.append(
                f"| {r['vio_base']} | "
                f"{_fmt_m(g['joint_err_mean']) if g else 'n/a'} | "
                f"{_fmt_m(r['joint_err_mean'])} | "
                f"{_fmt_m(g['dist_err_mean']) if g else 'n/a'} | "
                f"{_fmt_m(r['dist_err_mean'])} | "
                f"{g['n_markers'] if g else 'n/a'} / {r['n_markers']} |")
        lines.append("")
        for r in vio_results:
            lines.extend(_scenario_detail_lines(r))
            _ym = r.get("vio_yaw_err_mean_deg")
            _yx = r.get("vio_yaw_err_max_deg")
            _yfm = r.get("vio_yaw_err_marker_frames_deg")
            _ym_s = "n/a" if _ym is None else f"{_ym:.1f} deg"
            _yx_s = "n/a" if _yx is None else f"{_yx:.1f} deg"
            _yfm_s = "n/a" if _yfm is None else f"{_yfm:.1f} deg"
            lines.append(f"- VIO: initialised at frame "
                         f"{r['vio_init_frame']}; first loss at frame "
                         f"{r['vio_loss_frame']}; valid "
                         f"{r['vio_valid_frames']}/{r['n_frames']} frames; "
                         f"drift mean {_fmt_m(r['vio_drift_mean_m'])}, "
                         f"max {_fmt_m(r['vio_drift_max_m'])}, final "
                         f"{_fmt_m(r['vio_drift_final_m'])}; yaw error "
                         f"mean {_ym_s}, max {_yx_s}, "
                         f"on marker frames {_yfm_s}; "
                         f"PnP inliers "
                         f"mean {r['vio_inliers_mean']:.0f}, min "
                         f"{r['vio_inliers_min']}; VIO step "
                         f"{r['vio_ms_mean']:.1f} ms/frame "
                         f"({r['vio_fps']:.1f} fps); full pipeline with "
                         f"VIO {r['vio_pipeline_ms_mean']:.1f} ms/frame "
                         f"({r['vio_pipeline_fps']:.1f} fps).")
            lines.append("")
        lines.append("### Honest limits — VIO (what this still does NOT prove)")
        lines.append("")
        lines.append("- The harness trajectories have yaw snaps the VIO "
                     "cannot track: the look-at logic snaps the camera "
                     "-31 deg at the doorway (frame 13->14) and +101 deg "
                     "at the turn to face the person (frame 19->20) in a "
                     "single 0.1 s frame. Measured 2026-11-23: LK carried "
                     "only 36 of 86 points across the doorway (19 of them "
                     "hallucinated onto points that had left the frame); "
                     "ORB descriptor matching found 0 geometrically "
                     "correct matches across the doorway snap on this "
                     "repetitive checker floor, and a photometric yaw "
                     "search found the wrong yaw (-54 deg vs -31 deg GT; "
                     "+33 deg vs +101 deg GT) because near-wall parallax "
                     "dominates the photometric error. A real Tello flight "
                     "yaws at ~30-60 deg/s (1-2 deg/frame at 30 fps), not "
                     "300-1000 deg/s — the snaps are a harness artifact, "
                     "and the real-run procedure is to fly and turn "
                     "smoothly (see README.md).")
        lines.append("- The smooth-flight ground test "
                     "(``tests/vio_ground_test.py``) is the end-to-end "
                     "VIO validation on a realistic flight (no snaps): "
                     "approach + pass + most of the turn track with "
                     "9.2 cm mean drift, then tracking is lost mid-turn "
                     "when the map drains in rotation-dominant motion "
                     "(the camera translates only ~2.7 cm/frame while "
                     "yawing 3.4 deg/frame, so new points cannot reach "
                     "the 0.15 m triangulation baseline before leaving "
                     "the frame). The bridge invalidates correctly there "
                     "— no guessed pose, no marker.")
        lines.append("- VIO reacquire in this harness proves the "
                     "state-machine logic only (same caveat as s21): a "
                     "fresh VIO pose after loss re-establishes the bridge "
                     "only when VIO reports solid re-tracking. In the real "
                     "system reacquire must be gated on genuine "
                     "relocalisation (re-observing a mapped area / doorway "
                     "loop closure), which is not built here.")
        lines.append("- No loop closure and no bundle adjustment: odometry "
                     "drift grows with distance travelled, and the map "
                     "distorts non-rigidly (measured: the map's "
                     "reprojection error at the GT pose grows to ~11 px "
                     "by the turn, which the PnP absorbs as pose drift). "
                     "The ground-test drift (9.2 cm mean, 33.6 cm max) "
                     "is the honest measure of that.")
        lines.append("- The VIO step runs on CPU at ~40-90 fps alone; the "
                     "full pipeline with VIO is dominated by the YOLO "
                     "detector (see the pipeline fps column).")
        lines.append("")
    if mono_results is not None:
        lines.append("## Depth-model tests (monocular third voter)")
        lines.append("")
        lines.append("Model: Depth Anything V2 Small "
                     "(`depth-anything/Depth-Anything-V2-Small-hf`, "
                     "transformers, fp32, CPU) — 24.8M params, 95 MB "
                     "weights (Apache-2.0). The alternative, ZoeDepth "
                     "(`Intel/zoedepth-nyu-kitti`, DPT-Large backbone, "
                     "metric output), is ~1.37 GB fp32 — ~14x larger "
                     "than DA-V2-Small and larger than the ~1.2 GB free "
                     "RAM on this machine — so it was excluded on "
                     "measured download size / RAM before any inference "
                     "run, and its CPU inference time was not measured. "
                     "DA-V2-Small outputs RELATIVE depth only; its "
                     "metric scale comes from a MEASURED per-frame "
                     "anchor — a robust linear fit true_z = a * model "
                     "+ b over floor pixels, where the true depth "
                     "follows from the ray / y = 0 plane intersection "
                     "using the known drone pose (camera height above "
                     "the floor). No hard-coded scale constant anywhere: "
                     "the fit is accepted only with >= 20 floor samples, "
                     "residual RMS <= 0.35 m and |Pearson r| >= 0.8, "
                     "otherwise there is no model vote that frame. The "
                     "person depth is the fitted value at the median "
                     "model value over the visible keypoint pixels.")
        lines.append("")
        lines.append("Voting rule (server/depth.py: "
                     "``fuse_with_model_votes``): the model NEVER "
                     "replaces triangulation as the primary method. "
                     "When triangulation and bbox-height agree "
                     "(<= 25%), the model stays OUT — there is "
                     "nothing to vote on, and its measured error is "
                     "larger than triangulation's. When "
                     "they disagree (> 25%), the model arbitrates: it "
                     "backs triangulation (outcome changes vs the "
                     "legacy bbox-height fallback) or bbox-height (same "
                     "outcome), and if it agrees with neither the "
                     "legacy fallback stands. Pure arbitration: the "
                     "winner's depth is returned EXACTLY, never fused "
                     "with the model. Fusing the model depth into the "
                     "winner was tried first and measured to hurt — "
                     "in s15_occluded it took joint error 59.3 -> "
                     "64.5 cm (agree-case fusion) and 59.3 -> 64.3 cm "
                     "(disagree-case refinement), because the occluder "
                     "biases the model vote the same way it biases "
                     "the bbox. Without triangulation the model does "
                     "nothing (bbox-height alone). The model sigma "
                     "carries a floor of "
                     "0.15 * depth (server/mono_depth.py): the "
                     "floor-fit residual alone understates the "
                     "model's real error (probe mean |err| 5-47 cm "
                     "clean, 57-71 cm occluded, vs a ~0.2 m fit "
                     "sigma) — the floor keeps the vote's stated "
                     "uncertainty honest. The model runs in a "
                     "background thread, at "
                     "most every 10th detection frame (1 Hz at the 10 "
                     "fps harness rate — every 5th frame kept the "
                     "worker busy ~100% of the time and cost 7-28% of "
                     "pipeline fps to CPU contention on this 1.5-CPU "
                     "VM, while the disagreements it arbitrates persist "
                     "for many frames, so 1 Hz loses nothing), on a "
                     "256x192 copy; the pipeline reads only the "
                     "latest completed vote, age-gated to 2.0 s with "
                     "the sigma inflated 0.2 m/s of age. The marker "
                     "path never waits for it. The model's torch "
                     "thread count is left at the process default so "
                     "the YOLO detector's threading is identical with "
                     "the voter on or off.")
        lines.append("")
        if mono_bench:
            lines.append("### Model choice — measured size / CPU latency")
            lines.append("")
            lines.append("| Model | Params | Weights | Licence | "
                         "CPU inference (this machine) |")
            lines.append("|---|---|---|---|---|")
            lines.append("| Depth Anything V2 Small | 24.8M | 95 MB | "
                         "Apache-2.0 | "
                         + "; ".join(
                             f"{s}x{int(s) * 3 // 4}: {v:.0f} ms"
                             for s, v in sorted(mono_bench.items()))
                         + " (single frame, process-default thread "
                           "count; the pipeline uses 256x192) |")
            lines.append("| ZoeDepth (nyu-kitti) | ~345M (DPT-Large) | "
                         "~1.37 GB fp32 | MIT | not measured — excluded "
                         "on size / RAM (see above) |")
            lines.append("")
            lines.append("Model calls / votes over the whole depth-model "
                         f"suite: {mono_stats}.")
            lines.append("")
        lines.append("### Baseline (two methods) vs model voter — same scenarios, GT poses")
        lines.append("")
        lines.append("| Scenario | Joint err base / +model | "
                     "|Dist err| base / +model | Model depth err "
                     "mean |worst| (n) | Voter changed (frames) | "
                     "Pipeline fps base / +model | Result base / +model |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in mono_results:
            g = next((x for x in results if x["name"] == r["name"]), None)
            lines.append(
                f"| {r['name']} | "
                f"{_fmt_m(g['joint_err_mean']) if g else 'n/a'} / "
                f"{_fmt_m(r['joint_err_mean'])} | "
                f"{_fmt_m(g['dist_err_mean']) if g else 'n/a'} / "
                f"{_fmt_m(r['dist_err_mean'])} | "
                f"{'n/a' if r.get('model_err_mean') is None else f'{r['model_err_mean'] * 100:+.0f} cm | {r['model_err_max'] * 100:.0f} cm (n=' + str(r['model_n']) + ')'} | "
                f"{r['voter_changed_frames']} | "
                f"{'n/a' if not g or g.get('pipeline_fps') is None else f'{g['pipeline_fps']:.1f}'} / "
                f"{'n/a' if r.get('pipeline_fps') is None else f'{r['pipeline_fps']:.1f}'} | "
                f"{'PASS' if g and g['passed'] else 'FAIL'} / "
                f"{'PASS' if r['passed'] else 'FAIL'} |")
        lines.append("")
        lines.append("Per-scenario detail (model runs) follows; the "
                     "baseline detail is in the extended-results section "
                     "above. The model-vote error is the floor-anchored "
                     "model depth minus the GT foot depth, pooled over "
                     "frames where a vote was available — it is the "
                     "model's own error, not the fused pipeline error.")
        lines.append("")
        for r in mono_results:
            lines.extend(_scenario_detail_lines(r))
        if mono_vio_results:
            lines.append("### Model voter on the VIO path (same 4 VIO scenarios)")
            lines.append("")
            lines.append("| Scenario (VIO) | VIO valid | Joint err "
                         "VIO-only / VIO+model | |Dist err| VIO-only / "
                         "VIO+model | Model err mean |worst| | Voter "
                         "changed | Pipeline fps VIO-only / VIO+model | "
                         "Result VIO-only / VIO+model |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in mono_vio_results:
                g = next((x for x in (vio_results or [])
                          if x["name"] == r["name"]), None)
                lines.append(
                    f"| {r['vio_base']} | {r['vio_valid_frames']}/"
                    f"{r['n_frames']} | "
                    f"{_fmt_m(g['joint_err_mean']) if g else 'n/a'} / "
                    f"{_fmt_m(r['joint_err_mean'])} | "
                    f"{_fmt_m(g['dist_err_mean']) if g else 'n/a'} / "
                    f"{_fmt_m(r['dist_err_mean'])} | "
                    f"{'n/a' if r.get('model_err_mean') is None else f'{r['model_err_mean'] * 100:+.0f} cm | {r['model_err_max'] * 100:.0f} cm (n=' + str(r['model_n']) + ')'} | "
                    f"{r['voter_changed_frames']} | "
                    f"{'n/a' if not g or g.get('vio_pipeline_fps') is None else f'{g['vio_pipeline_fps']:.1f}'} / "
                    f"{'n/a' if r.get('vio_pipeline_fps') is None else f'{r['vio_pipeline_fps']:.1f}'} | "
                    f"{'PASS' if g and g['passed'] else 'FAIL'} / "
                    f"{'PASS' if r['passed'] else 'FAIL'} |")
            lines.append("")
            for r in mono_vio_results:
                lines.extend(_scenario_detail_lines(r))
        lines.append("### Honest limits — depth model (what this still does NOT prove)")
        lines.append("")
        lines.append("- The floor anchor needs visible floor in the "
                     "lower frame: a drone looking level or up, or a "
                     "floor with no texture / a mirror floor, gives no "
                     "fit and therefore no vote — the pipeline then "
                     "falls back to the two-method behaviour silently "
                     "(by design: no vote is better than a guessed "
                     "scale). The virtual floor is a high-contrast "
                     "checker; a real plain floor may fit worse — the "
                     "fit RMS / correlation gates are the honest check, "
                     "and their pass rate on real footage is unmeasured.")
        lines.append("- s17_edge produced NO model vote at all (0 of 4 "
                     "calls): the floor-fit residual RMS there is "
                     "0.74-0.75 m, above the 0.35 m gate (|r| = 0.81 "
                     "passes, barely). The drone sits deep in room B "
                     "(z ~ 3.8 m) looking at a fixed point, so the "
                     "floor samples span 2.5-6.9 m of true depth and "
                     "the model's output is not affine in depth over "
                     "that range — the gate working as designed, not a "
                     "bug: a bad fit must not vote. The s17 mono run "
                     "therefore reproduces its baseline exactly. "
                     "Restricting the fit to nearer floor (as the VIO "
                     "scale does) might recover votes there; it was "
                     "NOT done — it would change the fit for every "
                     "scenario and needs its own measurement round.")
        lines.append("- The model vote is sampled at the keypoint "
                     "pixels: an occluder in front of the person can "
                     "contaminate it (measured in s15_occluded: worst "
                     "model error 137 cm on one frame). The outlier "
                     "drop in the voting rule is what keeps a bad model "
                     "frame out of the fused depth; the raw model error "
                     "is still reported above, not hidden.")
        lines.append("- The model is per-frame (no static-scene "
                     "assumption), which is why it can arbitrate a "
                     "moving person — but its own error (~15-47 cm "
                     "mean |err| in the scenarios above) is larger "
                     "than triangulation's in static scenes, so it is "
                     "a pure arbitrator, not a refiner: it only ever "
                     "flips the choice between triangulation and "
                     "bbox-height, and it flipped that choice on 0 "
                     "frames in this suite (see the voter-changed "
                     "column — the mono runs reproduce the baseline "
                     "exactly). Its value is for the case the virtual "
                     "suite does not contain: triangulation right and "
                     "bbox-height badly wrong on the same frame (e.g. "
                     "a crouching or half-out-of-frame person whose "
                     "bbox height collapses). The flip logic itself is "
                     "proven by direct checks of "
                     "``fuse_with_model_votes`` (disagree-backs-tri "
                     "returns triangulation with "
                     "voter_changed=True).")
        lines.append("- The background worker makes the model runs "
                     "timing-dependent: which frames carry a vote "
                     "depends on worker speed on the day, so the "
                     "model-run numbers reproduce approximately, not "
                     "exactly, unlike the deterministic baseline. The "
                     "regression comparison above still reads the "
                     "baseline (model-off) runs.")
        lines.append("- Latency is measured on this shared 1.5-CPU VM "
                     "with the YOLO detector running concurrently; a "
                     "real laptop (more cores, no shared-CPU throttle) "
                     "will differ. The pipeline fps numbers are "
                     "process_frame only (rendering excluded).")
        lines.append("")
    REPORT_PATH.write_text(head + "\n".join(lines), encoding="utf-8")
    log.info("Extended report section written to %s", REPORT_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Wall-marker virtual tests")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="subset of scenarios to run (default: all 23)")
    ap.add_argument("--port", type=int, default=18765,
                    help="test port for the no-drone-video server")
    ap.add_argument("--skip-video-test", action="store_true")
    ap.add_argument("--skip-real-only", action="store_true",
                    help="skip the real-only UI server tests "
                    "(tests/test_real_only.py)")
    ap.add_argument("--skip-vio", action="store_true",
                    help="skip the VIO scenarios (GT-pose runs only)")
    ap.add_argument("--skip-mono", action="store_true",
                    help="skip the monocular depth-model voter suite")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    started = time.time()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    names = args.scenarios or scenarios.list_all_scenarios()
    detector = PoseDetector(weights=WEIGHTS)
    detector.warmup()
    log.info("Detector ready; running scenarios: %s", ", ".join(names))

    results = []
    for name in names:
        log.info("Running %s ...", name)
        if name == "s7_two_drones":
            results.append(run_s7(detector))
        elif "hovers" in scenarios.ALL_SCENARIOS[name]:
            results.append(run_multi(name, detector))
        else:
            results.append(run_single_scenario(name, detector))
        r = results[-1]
        log.info("%s: detection=%s joint=%s worst_joint=%s dist=%s "
                 "worst_dist=%s scale=%s bridge_inv=%d -> %s",
                 name,
                 "n/a" if r["detection_rate"] is None
                 else f"{r['detection_rate']:.2f}",
                 _fmt_m(r["joint_err_mean"]), _fmt_m(r["joint_err_max"]),
                 _fmt_m(r["dist_err_mean"]), _fmt_m(r["dist_err_max"]),
                 _fmt_pct(r["scale_err_mean"]),
                 r["marker_while_bridge_invalid"],
                 "PASS" if r["passed"] else "FAIL")

    vio_results = []
    if not args.skip_vio:
        for name in VIO_SCENARIOS:
            if name not in names:
                continue
            log.info("Running %s (VIO) ...", name)
            r = run_vio_scenario(name, detector)
            vio_results.append(r)
            log.info("%s (VIO): vio_valid=%d/%d drift=%s worst=%s "
                     "inliers=%.0f vio_fps=%.1f pipeline_fps=%.1f "
                     "detection=%s joint=%s dist=%s markers=%d "
                     "bridge_inv=%d -> %s",
                     name, r["vio_valid_frames"], r["n_frames"],
                     _fmt_m(r["vio_drift_mean_m"]),
                     _fmt_m(r["vio_drift_max_m"]),
                     r["vio_inliers_mean"] or 0,
                     r["vio_fps"] or 0, r["vio_pipeline_fps"] or 0,
                     "n/a" if r["detection_rate"] is None
                     else f"{r['detection_rate']:.2f}",
                     _fmt_m(r["joint_err_mean"]),
                     _fmt_m(r["dist_err_mean"]), r["n_markers"],
                     r["marker_while_bridge_invalid"],
                     "PASS" if r["passed"] else "FAIL")

    # --- Monocular depth-model voter suite (GT + VIO, model on) ---
    mono_results = []
    mono_vio_results = []
    mono_bench = None
    mono_stats = None
    if not args.skip_mono:
        from server.mono_depth import MonoDepthModel
        mono = MonoDepthModel()
        mono.load()
        log.info("Mono depth model loaded: %s", mono.model_id)
        # CPU inference benchmark on one real scenario frame, at the
        # three input sizes the probe measured (same frame, same
        # weights — only the resize changes).
        bench_frames = scenarios.get_frames(
            "s1_basic", detector=detector, drone_id="drone1")
        bench_frame = bench_frames[30][0]
        mono_bench = {}
        for size in (256, 336, 518):
            mono.input_size = size
            mono.predict(bench_frame)  # warm
            mss = [mono.predict(bench_frame)[1] for _ in range(5)]
            mono_bench[size] = float(np.mean(mss))
            log.info("Mono bench %dx%d: %.0f ms mean over 5 runs",
                     size, size * 3 // 4, mono_bench[size])
        mono.input_size = 256
        for name in MONO_SCENARIOS:
            if name not in names:
                continue
            log.info("Running %s (mono voter) ...", name)
            if name == "s7_two_drones":
                r = run_s7(detector, mono=mono)
            else:
                r = run_single_scenario(name, detector, mono=mono)
            mono_results.append(r)
            log.info("%s (mono): joint=%s dist=%s model_err=%s "
                     "worst_model=%s voter_changed=%d fps=%.1f -> %s",
                     name, _fmt_m(r["joint_err_mean"]),
                     _fmt_m(r["dist_err_mean"]),
                     _fmt_m(r["model_err_mean"]),
                     _fmt_m(r["model_err_max"]),
                     r["voter_changed_frames"],
                     r.get("pipeline_fps") or 0,
                     "PASS" if r["passed"] else "FAIL")
        if not args.skip_vio:
            for name in VIO_SCENARIOS:
                if name not in names:
                    continue
                log.info("Running %s (VIO + mono voter) ...", name)
                r = run_vio_scenario(name, detector, mono=mono)
                mono_vio_results.append(r)
                log.info("%s (VIO+mono): vio_valid=%d/%d joint=%s "
                         "dist=%s model_err=%s voter_changed=%d "
                         "pipe_fps=%.1f -> %s",
                         name, r["vio_valid_frames"], r["n_frames"],
                         _fmt_m(r["joint_err_mean"]),
                         _fmt_m(r["dist_err_mean"]),
                         _fmt_m(r["model_err_mean"]),
                         r["voter_changed_frames"],
                         r["vio_pipeline_fps"] or 0,
                         "PASS" if r["passed"] else "FAIL")
        mono_stats = mono.stats()
        log.info("Mono depth stats: %s", mono_stats)

    video_findings = (True, {"port": args.port, "checks": {},
                             "video_tag_count": 0, "video_hints_found": [],
                             "sse_events_seen": 0,
                             "unknown_route_statuses": {}})
    if not args.skip_video_test:
        log.info("Running no-drone-video test on port %d ...", args.port)
        video_findings = no_drone_video_test(port=args.port)
        log.info("No-drone-video test: %s",
                 "PASS" if video_findings[0] else "FAIL")

    real_only = (True, {"tests": {}})
    if not args.skip_real_only:
        from tests.test_real_only import run_all as run_real_only
        log.info("Running real-only UI server tests ...")
        real_only = run_real_only(port=args.port + 100)
        log.info("Real-only tests: %s",
                 "PASS" if real_only[0] else "FAIL")

    runtime_s = time.time() - started
    write_report(results, video_findings, runtime_s, started_at, names,
                 vio_results=vio_results, mono_results=mono_results or None,
                 mono_vio_results=mono_vio_results or None,
                 mono_bench=mono_bench, mono_stats=mono_stats,
                 real_only=real_only)

    print("\n=== WALL-MARKER VIRTUAL TESTS ===")
    for r in results:
        dr = ("n/a" if r["detection_rate"] is None
              else f"{r['detection_rate'] * 100:.0f}%")
        print(f"{r['name']:26s} det={dr:>5s} "
              f"joint={_fmt_m(r['joint_err_mean']):>8s} "
              f"worstJ={_fmt_m(r['joint_err_max']):>8s} "
              f"dist={_fmt_m(r['dist_err_mean']):>8s} "
              f"worstD={_fmt_m(r['dist_err_max']):>8s} "
              f"scale={_fmt_pct(r['scale_err_mean']):>6s} "
              f"false={r['false_markers']} bridgeInv="
              f"{r['marker_while_bridge_invalid']} "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    for r in results:
        if "s7_fused_hover_joint_err_m" in r:
            print(f"{'s7 fused-vs-single':26s} fused(hover)="
                  f"{_fmt_m(r['s7_fused_hover_joint_err_m'])} best-single="
                  f"{_fmt_m(r['s7_best_single_joint_err_m'])}")
        if "multi_fused_hover_joint_err_m" in r:
            print(f"{r['name'] + ' fused-vs-single':26s} fused(hover)="
                  f"{_fmt_m(r['multi_fused_hover_joint_err_m'])} "
                  f"best-single="
                  f"{_fmt_m(r['multi_best_single_joint_err_m'])}")
    for r in vio_results:
        dr = ("n/a" if r["detection_rate"] is None
              else f"{r['detection_rate'] * 100:.0f}%")
        print(f"{r['name']:26s} vio_valid="
              f"{r['vio_valid_frames']}/{r['n_frames']} "
              f"drift={_fmt_m(r['vio_drift_mean_m']):>8s} "
              f"worstD={_fmt_m(r['vio_drift_max_m']):>8s} "
              f"yaw={r.get('vio_yaw_err_mean_deg') or 0:.1f}deg/"
              f"{r.get('vio_yaw_err_max_deg') or 0:.1f}deg "
              f"yawMk={r.get('vio_yaw_err_marker_frames_deg') or 0:.1f}deg "
              f"inl={r['vio_inliers_mean']:.0f} "
              f"vio_fps={r['vio_fps']:.1f} "
              f"pipe_fps={r['vio_pipeline_fps']:.1f} "
              f"det={dr:>5s} "
              f"joint={_fmt_m(r['joint_err_mean']):>8s} "
              f"dist={_fmt_m(r['dist_err_mean']):>8s} "
              f"markers={r['n_markers']} bridgeInv="
              f"{r['marker_while_bridge_invalid']} "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    for r in mono_results:
        g = next((x for x in results if x["name"] == r["name"]), None)
        print(f"{r['name'] + ' (mono)':26s} "
              f"joint={_fmt_m(r['joint_err_mean']):>8s} "
              f"(base {_fmt_m(g['joint_err_mean']) if g else 'n/a'}) "
              f"dist={_fmt_m(r['dist_err_mean']):>8s} "
              f"(base {_fmt_m(g['dist_err_mean']) if g else 'n/a'}) "
              f"model_err={_fmt_m(r['model_err_mean']):>8s} "
              f"worstM={_fmt_m(r['model_err_max']):>8s} "
              f"voter_changed={r['voter_changed_frames']} "
              f"fps={r.get('pipeline_fps') or 0:.1f} "
              f"(base {g.get('pipeline_fps') or 0:.1f}) "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    for r in mono_vio_results:
        g = next((x for x in vio_results if x["name"] == r["name"]), None)
        print(f"{r['name'] + ' (VIO+mono)':26s} "
              f"joint={_fmt_m(r['joint_err_mean']):>8s} "
              f"(VIO-only {_fmt_m(g['joint_err_mean']) if g else 'n/a'}) "
              f"dist={_fmt_m(r['dist_err_mean']):>8s} "
              f"model_err={_fmt_m(r['model_err_mean']):>8s} "
              f"voter_changed={r['voter_changed_frames']} "
              f"pipe_fps={r['vio_pipeline_fps']:.1f} "
              f"(VIO-only {g['vio_pipeline_fps']:.1f}) "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    if mono_bench:
        print(f"mono bench (ms, single frame): "
              + ", ".join(f"{s}px={v:.0f}" for s, v in sorted(mono_bench.items()))
              + f"; mono stats: {mono_stats}")
    print(f"no-drone-video test: {'PASS' if video_findings[0] else 'FAIL'}")
    print(f"real-only UI server tests: {'PASS' if real_only[0] else 'FAIL'}")
    print(f"total runtime: {runtime_s:.1f} s")
    print(f"report: {REPORT_PATH}")

    all_ok = (all(r["passed"] for r in results)
              and all(r["passed"] for r in vio_results)
              and all(r["passed"] for r in mono_results)
              and all(r["passed"] for r in mono_vio_results)
              and video_findings[0]
              and real_only[0])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
