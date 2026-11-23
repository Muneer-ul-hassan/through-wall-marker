#!/usr/bin/env python3
"""Vanishing-point yaw tests (server/vp_yaw.py) — three passes.

Pass 1 — VP yaw vs GT yaw per frame on the four VIO scenarios
(s1_basic, s23_blank_door, s20_fast, s4_dim): availability, error
stats, 2-VP fraction, vertical-VP rejection rate, ms/frame. The
hint is the GT yaw and gravity comes from the GT pitch/roll, so
this isolates the VP measurement itself. Pass 1 also measures the
half-resolution variant (480x360, K halved) to see whether the
~60 ms/frame cost can be cut for a real integration. Pass 1b
repeats the full-resolution measurement with a noisy hint (sigma
2 deg, the magnetometer) and noisy pitch/roll (sigma 1 deg) to
show the measurement does not depend on a perfect prior.

Pass 2 — fusion: the run_vio_scenario loop replicated with a
FusedPrior (a StatePrior subclass defined below) whose R_prior(t)
is the circular 1/sigma^2-weighted mean of the magnetometer yaw
and the VP yaw (sigma_prior 2 deg, sigma_vp from pass 1), gated
to fuse only when |VP yaw - prior yaw| < 10 deg, else the prior
alone. Joint error vs the round-3 VIO baseline (s1 40.5, s23
32.6, s20 36.1, s4 36.3 cm).

Pass 3 — bad-prior rescue: pass 2 with prior_offsets_deg =
(2, -1, 1) (the round-3 fragility case: 80.0/93.9/76.4/79.8 cm,
all FAIL). The VP yaw is computed from the frame, so it is
unperturbed by the packet offset; the fusion should pull the
joint error back under the 60 cm budget.

The fusion lives in THIS test script only. It is wired into
server/vio.py only if pass 2/3 measure clearly better (see the
wiring decision in tests/TEST-REPORT.md).

Usage:
    python tests/vp_yaw_test.py [--passes 1|2|3|all]
                                [--scenarios s1_basic ...]
                                [--out tests/vp_yaw_results.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.detector import PoseDetector  # noqa: E402
from server.geometry import DRONE_K, project_point  # noqa: E402
from server.main import Pipeline  # noqa: E402
from server.tello_state import (R_from_yaw_pitch_roll, StatePrior,
                                SyntheticStateProvider,
                                yaw_pitch_roll_from_R)  # noqa: E402
from server.vio import MonocularVO  # noqa: E402
from server.vp_yaw import estimate_vp_yaw  # noqa: E402
from tests import scenarios  # noqa: E402
from tests.run_tests import _aggregate, _score_marker  # noqa: E402

log = logging.getLogger(__name__)

WEIGHTS = str(WALL_MARKER_DIR / "yolov8n-pose.pt")
RESULTS_PATH = Path(__file__).resolve().parent / "vp_yaw_results.json"

VIO_SCENARIOS = ["s1_basic", "s23_blank_door", "s20_fast", "s4_dim"]
# Round-3 VIO baselines (tests/TEST-REPORT.md): mean joint error, cm.
BASELINE_JOINT_CM = {"s1_basic": 40.5, "s23_blank_door": 32.6,
                     "s20_fast": 36.1, "s4_dim": 36.3}
# Round-3 fragility (prior offset yaw+2, pitch-1, roll+1): cm.
FRAGILE_JOINT_CM = {"s1_basic": 80.0, "s23_blank_door": 93.9,
                    "s20_fast": 76.4, "s4_dim": 79.8}
TH_JOINT_ERR_M = 0.60  # SPEC.md budget — do not move.
SIGMA_PRIOR_DEG = 2.0  # magnetometer yaw sigma (matches vio.py default)
FUSE_GATE_DEG = 10.0   # fuse only when |VP yaw - prior yaw| < this


def _wrap_deg(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def _circ_weighted_mean_deg(angles_deg, weights):
    a = np.radians(np.asarray(angles_deg, dtype=float))
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w * np.sin(a)))
    c = float(np.sum(w * np.cos(a)))
    return math.degrees(math.atan2(s, c))


class FusedPrior(StatePrior):
    """StatePrior with an optional VP yaw fused into R_prior(t).

    The test loop computes the VP yaw for the current frame and
    calls ``set_vp_yaw`` before ``vio.update``; every R_prior read
    in the VIO then sees the fused yaw. Pitch/roll, velocity and
    height come from the base prior unchanged. The fusion is the
    circular 1/sigma^2-weighted mean of the magnetometer yaw and
    the VP yaw, gated: if the VP yaw disagrees with the prior by
    more than ``gate_deg`` it is ignored (a spurious VP must never
    drag the pose away from the prior).
    """

    def __init__(self, *args, sigma_vp_deg=1.0,
                 sigma_prior_deg=SIGMA_PRIOR_DEG,
                 gate_deg=FUSE_GATE_DEG, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_vp_deg = float(sigma_vp_deg)
        self.sigma_prior_deg = float(sigma_prior_deg)
        self.gate_deg = float(gate_deg)
        self._vp_yaw_deg = None
        self.fused = False       # last R_prior call fused the VP yaw
        self.gate_fired = False  # last R_prior call rejected the VP yaw

    def set_vp_yaw(self, yaw_deg):
        self._vp_yaw_deg = None if yaw_deg is None else float(yaw_deg)

    def R_prior(self, t):
        att = self.attitude(t)
        if att is None:
            return None
        yaw, pitch, roll = att
        self.fused = False
        self.gate_fired = False
        if self._vp_yaw_deg is not None:
            if abs(_wrap_deg(self._vp_yaw_deg - yaw)) < self.gate_deg:
                w_p = 1.0 / self.sigma_prior_deg ** 2
                w_v = 1.0 / self.sigma_vp_deg ** 2
                yaw = _circ_weighted_mean_deg(
                    [yaw, self._vp_yaw_deg], [w_p, w_v])
                self.fused = True
            else:
                self.gate_fired = True
        return R_from_yaw_pitch_roll(
            yaw, pitch, roll, tilt_deg=self.calibration.camera_tilt_deg)


# ---------------------------------------------------------------------------
# Pass 1 — VP yaw vs GT yaw per frame
# ---------------------------------------------------------------------------

def _half_k(K):
    return {"fx": K["fx"] / 2.0, "fy": K["fy"] / 2.0,
            "cx": K["cx"] / 2.0, "cy": K["cy"] / 2.0,
            "width": K["width"] // 2, "height": K["height"] // 2}


def pass1(scenario_names, rng_seed=0):
    """Per-frame VP yaw vs GT yaw. Returns (summary dict, rows)."""
    import cv2
    rng = np.random.default_rng(rng_seed)
    summaries = {}
    all_errs = []
    rows_out = []
    for name in scenario_names:
        frames = scenarios.get_frames(name, smooth_look=True)
        errs, errs_half, avail, avail_half = [], [], 0, 0
        nvp2, rej, g_vp, g_prior = 0, 0, 0, 0
        ms, ms_half = [], []
        for frame, pose, _tv, _meta in frames:
            ypr = yaw_pitch_roll_from_R(pose[0])
            gt_yaw, gt_pitch, gt_roll = (float(v) for v in ypr)
            # Full resolution, GT hint + GT pitch/roll (isolates the
            # VP measurement from prior noise).
            t0 = time.perf_counter()
            r = estimate_vp_yaw(frame, DRONE_K, yaw_hint_deg=gt_yaw,
                                pitch_roll_deg=(gt_pitch, gt_roll))
            ms.append((time.perf_counter() - t0) * 1000.0)
            if r is not None:
                avail += 1
                nvp2 += 1 if r["horizontal_vps"] == 2 else 0
                rej += 1 if r["vertical_vp_rejected"] else 0
                g_vp += 1 if r["gravity_source"] == "vp" else 0
                g_prior += 1 if r["gravity_source"] == "prior" else 0
                err = _wrap_deg(r["yaw_deg"] - gt_yaw)
                errs.append(err)
                all_errs.append(err)
            # Half resolution (480x360, K halved), same hint.
            small = cv2.resize(frame, (480, 360),
                               interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            rh = estimate_vp_yaw(small, _half_k(DRONE_K),
                                 yaw_hint_deg=gt_yaw,
                                 pitch_roll_deg=(gt_pitch, gt_roll))
            ms_half.append((time.perf_counter() - t0) * 1000.0)
            if rh is not None:
                avail_half += 1
                errs_half.append(_wrap_deg(rh["yaw_deg"] - gt_yaw))
        e = np.array(errs)
        eh = np.array(errs_half)
        summaries[name] = {
            "frames": len(frames), "avail": avail,
            "avail_pct": 100.0 * avail / len(frames),
            "err_mean_deg": float(e.mean()) if len(e) else None,
            "err_abs_mean_deg": float(np.abs(e).mean()) if len(e) else None,
            "err_std_deg": float(e.std()) if len(e) else None,
            "err_p90_deg": float(np.percentile(np.abs(e), 90)) if len(e) else None,
            "err_max_deg": float(np.abs(e).max()) if len(e) else None,
            "n_vp2": nvp2, "vertical_vp_rejected": rej,
            "gravity_vp": g_vp, "gravity_prior": g_prior,
            "ms_mean": float(np.mean(ms)),
            "ms_p90": float(np.percentile(ms, 90)),
            "half_avail": avail_half,
            "half_err_abs_mean_deg": (float(np.abs(eh).mean())
                                      if len(eh) else None),
            "half_err_max_deg": (float(np.abs(eh).max())
                                 if len(eh) else None),
            "half_ms_mean": float(np.mean(ms_half)),
        }
        rows_out.append({"scenario": name,
                         "errs_deg": [float(x) for x in errs],
                         "errs_half_deg": [float(x) for x in errs_half]})
        log.info("pass1 %s: avail=%d/%d abs_mean=%.2f max=%.2f "
                 "nVP2=%d rej=%d ms=%.1f | half avail=%d abs_mean=%s max=%s ms=%.1f",
                 name, avail, len(frames),
                 summaries[name]["err_abs_mean_deg"],
                 summaries[name]["err_max_deg"], nvp2, rej,
                 summaries[name]["ms_mean"], avail_half,
                 ("%.2f" % summaries[name]["half_err_abs_mean_deg"]
                  if summaries[name]["half_err_abs_mean_deg"] is not None
                  else "n/a"),
                 ("%.2f" % summaries[name]["half_err_max_deg"]
                  if summaries[name]["half_err_max_deg"] is not None
                  else "n/a"),
                 summaries[name]["half_ms_mean"])
    pooled_std = float(np.std(np.array(all_errs))) if all_errs else None
    return {"per_scenario": summaries, "pooled_std_deg": pooled_std}, rows_out


def pass1b(scenario_names, rng_seed=7):
    """Pass 1 with a realistic noisy prior: hint = GT yaw + N(0, 2
    deg) (the magnetometer), pitch/roll = GT + N(0, 1 deg)."""
    rng = np.random.default_rng(rng_seed)
    out = {}
    for name in scenario_names:
        frames = scenarios.get_frames(name, smooth_look=True)
        errs, avail = [], 0
        for frame, pose, _tv, _meta in frames:
            ypr = yaw_pitch_roll_from_R(pose[0])
            gt_yaw, gt_pitch, gt_roll = (float(v) for v in ypr)
            hint = _wrap_deg(gt_yaw + rng.normal(0.0, 2.0))
            pr = (gt_pitch + rng.normal(0.0, 1.0),
                  gt_roll + rng.normal(0.0, 1.0))
            r = estimate_vp_yaw(frame, DRONE_K, yaw_hint_deg=hint,
                                pitch_roll_deg=pr)
            if r is None:
                continue
            avail += 1
            errs.append(_wrap_deg(r["yaw_deg"] - gt_yaw))
        e = np.array(errs)
        out[name] = {
            "frames": len(frames), "avail": avail,
            "err_abs_mean_deg": float(np.abs(e).mean()) if len(e) else None,
            "err_max_deg": float(np.abs(e).max()) if len(e) else None,
        }
        log.info("pass1b %s: avail=%d/%d abs_mean=%.2f max=%.2f",
                 name, avail, len(frames), out[name]["err_abs_mean_deg"],
                 out[name]["err_max_deg"])
    return out


# ---------------------------------------------------------------------------
# Pass 2/3 — VIO fusion run (run_vio_scenario loop, FusedPrior)
# ---------------------------------------------------------------------------

def run_vio_vp_fusion(name, detector, sigma_vp_deg,
                      prior_offsets_deg=(0.0, 0.0, 0.0), mono=None):
    """Mirror of tests.run_tests.run_vio_scenario, with the VP yaw
    fused into the prior the VIO sees. Returns the _aggregate
    result dict plus VIO and VP-fusion fields."""
    cfg = scenarios.ALL_SCENARIOS[name]
    phone_pos = np.asarray(cfg.get("phone", (0.0, 1.5, 0.0)), dtype=float)
    pipeline = Pipeline(detector, drone_id="drone1",
                        phone_position=phone_pos, mono_depth=mono)
    rows = []
    vio_rows = []
    all_frame_ms = []
    vio = None
    vio_loss_frame = None
    vio_init_frame = None
    ever_valid = False
    vp_stats = {"avail": 0, "fused": 0, "gate_fired": 0, "frames": 0,
                "vp_yaw_errs": []}
    frames = scenarios.get_frames(
        name, detector=detector, drone_id="drone1", smooth_look=True)
    _poses = [p for _, p, _, _ in frames]
    _times = [m["t"] for _, _, _, m in frames]
    _provider = SyntheticStateProvider(
        _poses, _times, velocity_unit="dm/s", velocity_frame="global")
    _prior = FusedPrior(calibration=_provider.calibration_for(),
                        sigma_vp_deg=sigma_vp_deg)
    _pkts = _provider.packets()
    if any(abs(o) > 0 for o in prior_offsets_deg):
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
        vp_stats["frames"] += 1
        # VP yaw for THIS frame, hinted by the prior attitude at t
        # (magnetometer yaw) and validated against its pitch/roll.
        att = _prior.attitude(t)
        vp_yaw = None
        if att is not None:
            _yaw_pr, _pitch_pr, _roll_pr = att
            r = estimate_vp_yaw(
                frame, DRONE_K,
                yaw_hint_deg=_wrap_deg(_yaw_pr),
                pitch_roll_deg=(_pitch_pr, _roll_pr))
            if r is not None:
                vp_stats["avail"] += 1
                vp_yaw = r["yaw_deg"]
                _yaw_gt = yaw_pitch_roll_from_R(pose[0])[0]
                vp_stats["vp_yaw_errs"].append(
                    _wrap_deg(r["yaw_deg"] - _yaw_gt))
        _prior.set_vp_yaw(vp_yaw)
        if vio is None:
            vio = MonocularVO(start_pose=pose)
        t0 = time.perf_counter()
        est_pose, vio_valid, diag = vio.update(frame, t, state=_prior)
        vio_ms = diag.get("vio_ms", 0.0)
        if _prior.fused:
            vp_stats["fused"] += 1
        if _prior.gate_fired:
            vp_stats["gate_fired"] += 1
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
            _yaw_est = yaw_pitch_roll_from_R(est_pose[0])[0]
            _yaw_gt = yaw_pitch_roll_from_R(pose[0])[0]
            yaw_err = abs((_yaw_est - _yaw_gt + 180.0) % 360.0 - 180.0)
            vio_rows.append((meta["frame"], drift,
                             int(diag.get("inliers", 0)), vio_ms, yaw_err))
        elif ever_valid and vio_loss_frame is None:
            vio_loss_frame = meta["frame"]
        true_depth = None
        if meta["person_world"] is not None:
            p = project_point(DRONE_K, pose[0], pose[1],
                              np.array(meta["person_world"], dtype=float))
            true_depth = None if p is None else p[2]
        score = _score_marker(marker, meta)
        row = {
            "frame": meta["frame"], "t": t,
            "in_view": bool(meta["in_view"]),
            "detected": info["n_det"] > 0,
            "method": info["method"],
            "depth_m": info["depth"],
            "true_depth_m": true_depth,
            "model_depth_m": info.get("model_depth"),
            "model_err_m": (None if info.get("model_depth") is None
                            or true_depth is None
                            else info.get("model_depth") - true_depth),
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
    pipeline.shutdown()
    result = _aggregate(name, rows, vio_loss_frame)
    result["name"] = f"{name} (VIO+VP)"
    result["vio_base"] = name
    if vio_rows:
        drifts = np.array([r[1] for r in vio_rows])
        yaw_errs = np.array([r[4] for r in vio_rows])
        result["vio_valid_frames"] = len(vio_rows)
        result["vio_drift_mean_m"] = float(drifts.mean())
        result["vio_drift_max_m"] = float(drifts.max())
        result["vio_drift_final_m"] = float(drifts[-1])
        result["vio_inliers_mean"] = float(np.mean([r[2] for r in vio_rows]))
        result["vio_inliers_min"] = int(min(r[2] for r in vio_rows))
        result["vio_ms_mean"] = float(np.mean([r[3] for r in vio_rows]))
        result["vio_fps"] = float(1000.0 / np.mean([r[3] for r in vio_rows]))
        result["vio_yaw_err_mean_deg"] = float(yaw_errs.mean())
        result["vio_yaw_err_max_deg"] = float(yaw_errs.max())
        _marker_frames = {r["frame"] for r in rows if r["marker"]}
        _marker_yaw = [r[4] for r in vio_rows if r[0] in _marker_frames]
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
    result["vio_pipeline_ms_mean"] = (
        float(np.mean(all_frame_ms)) if all_frame_ms else None)
    result["vio_pipeline_fps"] = (
        float(1000.0 / np.mean(all_frame_ms)) if all_frame_ms else None)
    result["vio_init_frame"] = vio_init_frame
    result["vio_loss_frame"] = vio_loss_frame
    result["vio_pose_sources"] = pose_sources
    # VP-fusion fields.
    vpe = np.array(vp_stats["vp_yaw_errs"]) if vp_stats["vp_yaw_errs"] else np.array([])
    result["vp_avail_frames"] = vp_stats["avail"]
    result["vp_fused_frames"] = vp_stats["fused"]
    result["vp_gate_fired_frames"] = vp_stats["gate_fired"]
    result["vp_yaw_err_abs_mean_deg"] = (
        float(np.abs(vpe).mean()) if len(vpe) else None)
    result["vp_yaw_err_max_deg"] = (
        float(np.abs(vpe).max()) if len(vpe) else None)
    result["sigma_vp_deg"] = float(sigma_vp_deg)
    return result


def _joint_mean_cm(result):
    """Mean joint error in cm from an _aggregate result dict."""
    # _aggregate stores joint error under "joint_err_mean" (m) — fall
    # back to the per-row pooled joints if the key differs.
    if result.get("joint_err_mean") is not None:
        return float(result["joint_err_mean"]) * 100.0
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="VP yaw tests (passes 1-3)")
    ap.add_argument("--passes", nargs="*", default=["all"],
                    choices=["1", "1b", "2", "3", "all"],
                    help="which passes to run (default: all)")
    ap.add_argument("--scenarios", nargs="*", default=VIO_SCENARIOS,
                    help="subset of VIO scenarios")
    ap.add_argument("--out", default=str(RESULTS_PATH),
                    help="results JSON path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    passes = {"1", "1b", "2", "3"} if "all" in args.passes else set(args.passes)
    results = {"scenarios": args.scenarios}
    sigma_vp = None
    if "1" in passes:
        log.info("=== PASS 1: VP yaw vs GT yaw ===")
        p1, rows = pass1(args.scenarios)
        results["pass1"] = p1
        results["pass1_rows"] = rows
        sigma_vp = p1["pooled_std_deg"]
        log.info("pass1 pooled VP yaw sigma = %.2f deg", sigma_vp)
    if "1b" in passes:
        log.info("=== PASS 1b: VP yaw vs GT yaw, noisy prior ===")
        results["pass1b"] = pass1b(args.scenarios)
    if "2" in passes or "3" in passes:
        if sigma_vp is None:
            # Pass 1 was skipped: use the measured pooled sigma from
            # the last full run if present, else a conservative 1.0.
            sigma_vp = 1.0
            if Path(args.out).exists():
                try:
                    prev = json.loads(Path(args.out).read_text())
                    sigma_vp = float(prev["pass1"]["pooled_std_deg"])
                except Exception:
                    pass
            log.info("sigma_vp (no pass 1 this run) = %.2f deg", sigma_vp)
        detector = PoseDetector(weights=WEIGHTS)
        detector.warmup()
        if "2" in passes:
            log.info("=== PASS 2: VIO fusion (clean prior) ===")
            p2 = {}
            for name in args.scenarios:
                log.info("Running fusion %s ...", name)
                r = run_vio_vp_fusion(name, detector, sigma_vp)
                jm = _joint_mean_cm(r)
                p2[name] = {
                    "joint_mean_cm": jm,
                    "baseline_joint_cm": BASELINE_JOINT_CM.get(name),
                    "detection_rate": r.get("detection_rate"),
                    "vio_valid_frames": r.get("vio_valid_frames"),
                    "vio_drift_mean_m": r.get("vio_drift_mean_m"),
                    "vio_yaw_err_mean_deg": r.get("vio_yaw_err_mean_deg"),
                    "vio_yaw_err_marker_frames_deg": r.get(
                        "vio_yaw_err_marker_frames_deg"),
                    "vio_pipeline_fps": r.get("vio_pipeline_fps"),
                    "vp_avail_frames": r.get("vp_avail_frames"),
                    "vp_fused_frames": r.get("vp_fused_frames"),
                    "vp_gate_fired_frames": r.get("vp_gate_fired_frames"),
                    "vp_yaw_err_abs_mean_deg": r.get(
                        "vp_yaw_err_abs_mean_deg"),
                    "vp_yaw_err_max_deg": r.get("vp_yaw_err_max_deg"),
                    "pass": (jm is not None and jm <= 60.0),
                }
                log.info("%s: joint=%.1f cm (baseline %.1f) "
                         "yaw_err=%.2f deg fps=%.1f vp_fused=%d/%d -> %s",
                         name, jm, BASELINE_JOINT_CM.get(name),
                         r.get("vio_yaw_err_mean_deg"),
                         r.get("vio_pipeline_fps"),
                         r.get("vp_fused_frames"), r.get("vp_avail_frames"),
                         "PASS" if p2[name]["pass"] else "FAIL")
            results["pass2"] = p2
        if "3" in passes:
            log.info("=== PASS 3: bad-prior rescue (offset +2/-1/+1) ===")
            p3 = {}
            for name in args.scenarios:
                log.info("Running rescue %s ...", name)
                r = run_vio_vp_fusion(name, detector, sigma_vp,
                                      prior_offsets_deg=(2.0, -1.0, 1.0))
                jm = _joint_mean_cm(r)
                p3[name] = {
                    "joint_mean_cm": jm,
                    "fragile_joint_cm": FRAGILE_JOINT_CM.get(name),
                    "detection_rate": r.get("detection_rate"),
                    "vio_valid_frames": r.get("vio_valid_frames"),
                    "vio_drift_mean_m": r.get("vio_drift_mean_m"),
                    "vio_yaw_err_mean_deg": r.get("vio_yaw_err_mean_deg"),
                    "vio_yaw_err_marker_frames_deg": r.get(
                        "vio_yaw_err_marker_frames_deg"),
                    "vio_pipeline_fps": r.get("vio_pipeline_fps"),
                    "vp_avail_frames": r.get("vp_avail_frames"),
                    "vp_fused_frames": r.get("vp_fused_frames"),
                    "vp_gate_fired_frames": r.get("vp_gate_fired_frames"),
                    "vp_yaw_err_abs_mean_deg": r.get(
                        "vp_yaw_err_abs_mean_deg"),
                    "vp_yaw_err_max_deg": r.get("vp_yaw_err_max_deg"),
                    "pass": (jm is not None and jm <= 60.0),
                }
                log.info("%s: joint=%.1f cm (fragile %.1f, budget 60) "
                         "yaw_err=%.2f deg vp_fused=%d/%d -> %s",
                         name, jm, FRAGILE_JOINT_CM.get(name),
                         r.get("vio_yaw_err_mean_deg"),
                         r.get("vp_fused_frames"), r.get("vp_avail_frames"),
                         "PASS" if p3[name]["pass"] else "FAIL")
            results["pass3"] = p3
    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info("results written to %s", args.out)
    # Plain summary.
    print("\n=== VP YAW SUMMARY ===")
    if "pass1" in results:
        for name, s in results["pass1"]["per_scenario"].items():
            print(f"pass1 {name}: avail {s['avail']}/{s['frames']} "
                  f"abs_mean {s['err_abs_mean_deg']:.2f} deg "
                  f"max {s['err_max_deg']:.2f} deg nVP2 {s['n_vp2']} "
                  f"vp_rej {s['vertical_vp_rejected']} "
                  f"ms {s['ms_mean']:.1f} | half-res avail {s['half_avail']} "
                  f"abs_mean {s['half_err_abs_mean_deg']} "
                  f"max {s['half_err_max_deg']} ms {s['half_ms_mean']:.1f}")
        print(f"pass1 pooled VP yaw sigma = "
              f"{results['pass1']['pooled_std_deg']:.2f} deg")
    if "pass1b" in results:
        for name, s in results["pass1b"].items():
            print(f"pass1b {name}: avail {s['avail']}/{s['frames']} "
                  f"abs_mean {s['err_abs_mean_deg']:.2f} deg "
                  f"max {s['err_max_deg']:.2f} deg")
    if "pass2" in results:
        for name, s in results["pass2"].items():
            print(f"pass2 {name}: joint {s['joint_mean_cm']:.1f} cm "
                  f"(baseline {s['baseline_joint_cm']}) "
                  f"yaw_err {s['vio_yaw_err_mean_deg']:.2f} deg "
                  f"fps {s['vio_pipeline_fps']:.1f} "
                  f"vp_fused {s['vp_fused_frames']}/{s['vp_avail_frames']} "
                  f"-> {'PASS' if s['pass'] else 'FAIL'}")
    if "pass3" in results:
        for name, s in results["pass3"].items():
            print(f"pass3 {name}: joint {s['joint_mean_cm']:.1f} cm "
                  f"(fragile {s['fragile_joint_cm']}, budget 60) "
                  f"yaw_err {s['vio_yaw_err_mean_deg']:.2f} deg "
                  f"vp_fused {s['vp_fused_frames']}/{s['vp_avail_frames']} "
                  f"-> {'PASS' if s['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
