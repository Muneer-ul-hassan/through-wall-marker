#!/usr/bin/env python3
"""Calibrate the Tello state fields (report §6.9) from a real recording.

DJI publishes NO accuracy spec for any state field, and its own docs
disagree about the velocity unit (SDK 3.0 says vgx/vgy/vgz are dm/s and
agx cm/s^2; older material says cm/s and 0.001g — a 10x error if
assumed wrong), and the velocity frame (body vs global) is
undocumented. Do NOT integrate any velocity before running this.

What it does (report §6.9):
  1. Velocity UNIT: fly 1 m in a straight line (tape-measured).
     Integrate vgx/vgy/vgz over the flight assuming cm/s and dm/s;
     the correct unit integrates to ~1 m, the wrong one to ~0.1 m
     or ~10 m.
  2. Velocity FRAME: yaw mid-translation (turn 90 deg while moving).
     Integrate assuming the velocity is in the BODY frame (rotate
     each sample by the yaw at that time) vs the GLOBAL frame (use
     it as-is); the correct frame lands on the known endpoint, the
     wrong one is off by the yaw angle.
  3. Yaw accuracy: compare the unwrapped state yaw against a visual
     yaw (the VIO's yaw, or the known GT yaw from the flight plan)
     — reports RMS / max error and whether the yaw-jump gate
     (>10 deg/frame) ever fired.
  4. Height: compare h and tof against the tape-measured height
     (h is barometric, tof is the IR time-of-flight; tof reads 10 cm
     when out of range — that sentinel is filtered).
  5. VPS-blind: flag any stretch where the velocity reads ~0 while
     the drone is actually moving (the visual odometry / GT shows
     motion) — the VPS is blind there and the velocity is unusable.
  6. Latency: cross-correlate a sharp yaw step (a fast yaw command)
     between the state stream and the video/VIO yaw; the lag at
     peak correlation is the state latency (the state has no
     timestamp — it is timestamped on the laptop, so the latency
     is the offset between the two streams).

Input: a recording directory with
  * ``state.jsonl`` — one JSON object per line: {"t": <laptop time s>,
    "state": "<djitellopy state string>"} (see tools/record_tello.py,
    or capture with ``nc -u -l 8890`` + a timestamp).
  * ``ground_truth.json`` — the known path: {"segments": [...]} where
    each segment is {"kind": "straight", "distance_m": 1.0,
    "t_start": ..., "t_end": ..., "direction_deg": 0} or
    {"kind": "yaw_turn", "angle_deg": 90, "t_start": ..., "t_end": ...}
    or {"kind": "hover", "height_m": 1.3, "t_start": ..., "t_end": ...}.
    For the yaw-accuracy check, an optional "yaw_gt_deg" series:
    {"t": [...], "yaw_deg": [...]} (from the VIO or the flight plan).
  * optional ``video_yaw.json`` — {"t": [...], "yaw_deg": [...]} for
    the latency cross-correlation (the VIO's yaw per frame).

Output: a ``TelloCalibration`` as JSON (``--out``) + a human-readable
report on stdout. The JSON is the single source of truth — copy it
into the code (``server/tello_state.py`` ``TelloCalibration``) only
after reviewing the report. The default in the code stays
UNCALIBRATED (velocity priors OFF) until this has been run on a
real recording.

Synthetic validation: ``--self-test`` generates synthetic logs with
KNOWN unit/frame (dm/s/global and cm/s/body) and verifies this
script recovers them, and synthetic logs with a KNOWN yaw noise
sigma (1.0 and 2.5 deg) and a KNOWN doorway bias (+3 deg during
the turn) and verifies the yaw-sigma measurement, the doorway-bias
measurement, and the yaw gate (sigma <= 1.5 deg AND |doorway bias|
<= 1.5 deg — the real test may proceed only on PASS). Run it after
any change to this script:
  python3 tools/calibrate_tello.py --self-test
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.tello_state import (TelloCalibration, TelloStatePacket,  # noqa: E402
                                parse_state_string, unwrap_deg)

# The four hypotheses for the velocity (unit x frame). Exactly one
# should land on the known endpoint; the others are off by 10x or
# by the yaw angle.
VELOCITY_HYPOTHESES = [
    ("cm/s", "body"),
    ("cm/s", "global"),
    ("dm/s", "body"),
    ("dm/s", "global"),
]

YAW_JUMP_GATE_DEG = 10.0  # per 0.1 s frame (matches StatePrior)

# Yaw gate for the REAL test (round-3 fragility finding): the 1-DoF
# yaw PnP needs the attitude prior accurate to ~1 deg — a
# (yaw+2, pitch-1, roll+1) deg prior offset fails s1/s20/s4 at
# 81.6-90.4 cm joint error. The real test may proceed ONLY if the
# measured yaw sigma is <= 1.5 deg AND the doorway bias (mean signed
# yaw error during the doorway turn minus during straight flight) is
# <= 1.5 deg in magnitude. Otherwise the magnetometer is not good
# enough as a yaw prior and the real test must NOT proceed — fix the
# yaw source first (re-calibrate, move the flight line away from
# metal, or add a third yaw measurement). The gate is evaluated in
# run() and printed plainly by main().
YAW_GATE_SIGMA_DEG = 1.5
YAW_GATE_BIAS_DEG = 1.5


def load_state_log(path):
    """Read state.jsonl -> (times, packets)."""
    times, packets = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pkt = parse_state_string(rec["state"], float(rec["t"]))
            times.append(float(rec["t"]))
            packets.append(pkt)
    return np.array(times), packets


def _vel_ms(pkt, unit):
    """Raw (vgx, vgy, vgz) -> m/s under the assumed unit."""
    scale = 0.01 if unit == "cm/s" else 0.1  # dm/s
    return np.array([pkt.vgx, pkt.vgy, pkt.vgz], dtype=float) * scale


def _rotate_yaw(v_body, yaw_deg):
    """Body-frame velocity (x=forward, y=right) -> world (x=right,
    z=forward) using the yaw. The DJI body frame is x=forward,
    y=right, z=up; the world frame here is x=right, y=up, z=forward
    (the geometry convention). A body velocity (vx, vy) at yaw psi
    (clockwise from +Z) maps to world (vx*sin(psi)+vy*cos(psi),
    vx*cos(psi)-vy*sin(psi)) in (x, z)."""
    psi = math.radians(yaw_deg)
    vx, vy = float(v_body[0]), float(v_body[1])
    wx = vx * math.sin(psi) + vy * math.cos(psi)
    wz = vx * math.cos(psi) - vy * math.sin(psi)
    return np.array([wx, 0.0, wz])


def integrate_segment(times, packets, t0, t1, unit, frame):
    """Integrate the velocity over [t0, t1] under one hypothesis.
    Returns the displacement (dx, dy, dz) in metres (world frame)."""
    idx = [i for i, t in enumerate(times) if t0 <= t <= t1]
    if len(idx) < 2:
        return np.zeros(3)
    disp = np.zeros(3)
    yaws = unwrap_deg(np.array([packets[i].yaw for i in idx], dtype=float))
    for k in range(len(idx) - 1):
        i = idx[k]
        dt = float(times[idx[k + 1]] - times[i])
        if dt <= 0 or dt > 1.0:
            continue
        v = _vel_ms(packets[i], unit)
        if frame == "body":
            vw = _rotate_yaw(v, yaws[k])
        else:  # global: DJI-x -> world-Z, DJI-y -> world-X
            vw = np.array([v[1], v[2], v[0]])
        disp += vw * dt
    return disp


def calibrate_unit_frame(times, packets, segments):
    """Test each (unit, frame) hypothesis on the straight segments
    (and the yaw-turn segments, which discriminate the frame).
    Returns {hypothesis: [displacement error per segment]}."""
    results = {}
    for unit, frame in VELOCITY_HYPOTHESES:
        errs = []
        for seg in segments:
            if seg["kind"] not in ("straight", "yaw_turn"):
                continue
            disp = integrate_segment(times, packets, seg["t_start"],
                                     seg["t_end"], unit, frame)
            if seg["kind"] == "straight":
                # Known distance along the segment direction.
                d = math.radians(seg.get("direction_deg", 0.0))
                want = np.array([math.sin(d), 0.0, math.cos(d)]) \
                    * seg["distance_m"]
                errs.append(float(np.linalg.norm(disp - want)))
            else:
                # Yaw turn while translating: the endpoint is the
                # known displacement (if given), else just the
                # magnitude (the frame error shows up as a wrong
                # direction, which the magnitude alone hides — so
                # a yaw_turn segment needs "distance_m" +
                # "direction_deg" to discriminate).
                if "distance_m" in seg:
                    d = math.radians(seg.get("direction_deg", 0.0))
                    want = np.array([math.sin(d), 0.0, math.cos(d)]) \
                        * seg["distance_m"]
                    errs.append(float(np.linalg.norm(disp - want)))
        results[(unit, frame)] = errs
    return results


def yaw_accuracy(times, packets, yaw_gt):
    """Compare the unwrapped state yaw to a ground-truth yaw series.
    Returns (rms_deg, max_deg, n_jump_rejects)."""
    gt_t = np.array(yaw_gt["t"], dtype=float)
    gt_y = unwrap_deg(np.array(yaw_gt["yaw_deg"], dtype=float))
    yaw = unwrap_deg(np.array([p.yaw for p in packets], dtype=float))
    # Interpolate the GT yaw onto the state times.
    gt_i = np.interp(times, gt_t, gt_y)
    err = np.array([((a - b + 180.0) % 360.0 - 180.0)
                    for a, b in zip(yaw, gt_i)])
    # Yaw-jump gate: how many state steps exceed 10 deg / 0.1 s.
    steps = np.abs(np.diff(yaw))
    dt = np.diff(times)
    rate = np.where(dt > 0, steps / np.maximum(dt, 1e-9), 0.0)
    n_jumps = int(np.sum(rate > YAW_JUMP_GATE_DEG / 0.1))
    return float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err))), n_jumps


def yaw_doorway_bias(times, packets, yaw_gt, segments):
    """Doorway bias: mean SIGNED yaw error during the doorway turn
    minus mean signed yaw error during straight flight.

    A magnetometer biased by metal near the doorway (door frame,
    rebar, motors) shows up as a yaw error that exists ONLY near
    the doorway — the overall RMS can still look fine while the
    turn — the one place the yaw prior matters most — is off by
    several degrees. Returns {"turn_mean_deg", "straight_mean_deg",
    "bias_deg"} or None when there is no yaw GT or no turn segment.
    The bias is signed: +3 deg means the state yaw reads 3 deg high
    through the doorway relative to straight flight.
    """
    gt_t = np.array(yaw_gt["t"], dtype=float)
    gt_y = unwrap_deg(np.array(yaw_gt["yaw_deg"], dtype=float))
    yaw = unwrap_deg(np.array([p.yaw for p in packets], dtype=float))
    gt_i = np.interp(times, gt_t, gt_y)
    err = np.array([((a - b + 180.0) % 360.0 - 180.0)
                    for a, b in zip(yaw, gt_i)])
    turn_idx, straight_idx = [], []
    for i, t in enumerate(times):
        for seg in segments:
            if seg["t_start"] <= t <= seg["t_end"]:
                if seg["kind"] == "yaw_turn":
                    turn_idx.append(i)
                elif seg["kind"] == "straight":
                    straight_idx.append(i)
                break
    if not turn_idx or not straight_idx:
        return None
    turn_mean = float(np.mean(err[turn_idx]))
    straight_mean = float(np.mean(err[straight_idx]))
    return {"turn_mean_deg": turn_mean,
            "straight_mean_deg": straight_mean,
            "bias_deg": turn_mean - straight_mean}


def height_check(times, packets, segments):
    """Compare h and tof against the tape-measured hover height."""
    rows = []
    for seg in segments:
        if seg["kind"] != "hover":
            continue
        idx = [i for i, t in enumerate(times)
               if seg["t_start"] <= t <= seg["t_end"]]
        if not idx:
            continue
        h = np.array([packets[i].h for i in idx], dtype=float) / 100.0
        tof = np.array([packets[i].tof for i in idx], dtype=float)
        tof = tof[tof != 10] / 100.0  # 10 cm is the out-of-range sentinel
        want = float(seg["height_m"])
        rows.append({
            "t_start": seg["t_start"], "height_m": want,
            "h_mean_m": float(np.mean(h)) if len(h) else None,
            "h_err_m": float(np.mean(h) - want) if len(h) else None,
            "tof_mean_m": float(np.mean(tof)) if len(tof) else None,
            "tof_err_m": float(np.mean(tof) - want) if len(tof) else None,
            "tof_n": int(len(tof)),
        })
    return rows


def vps_blind_check(times, packets, segments):
    """Flag segments where the velocity reads ~0 while the GT shows
    motion (the VPS is blind — e.g. over a featureless floor)."""
    blind = []
    for seg in segments:
        if seg["kind"] != "straight":
            continue
        idx = [i for i, t in enumerate(times)
               if seg["t_start"] <= t <= seg["t_end"]]
        if len(idx) < 3:
            continue
        v = np.array([np.linalg.norm(_vel_ms(packets[i], "dm/s"))
                      for i in idx])
        # The drone covers distance_m in (t_end - t_start); the
        # mean speed should be ~distance/duration.
        want_speed = seg["distance_m"] / max(seg["t_end"] - seg["t_start"], 1e-9)
        mean_speed = float(np.mean(v))
        if mean_speed < 0.25 * want_speed:
            blind.append({"t_start": seg["t_start"],
                          "want_speed_m_s": want_speed,
                          "mean_speed_m_s": mean_speed})
    return blind


def latency_check(times, packets, video_yaw):
    """Cross-correlate a sharp yaw step between the state yaw and
    the video/VIO yaw. Returns the lag (s) at peak correlation
    (positive = the state lags the video)."""
    if video_yaw is None:
        return None
    vt = np.array(video_yaw["t"], dtype=float)
    vy = unwrap_deg(np.array(video_yaw["yaw_deg"], dtype=float))
    sy = unwrap_deg(np.array([p.yaw for p in packets], dtype=float))
    # Resample both onto a common 100 Hz grid.
    t0 = max(times[0], vt[0])
    t1 = min(times[-1], vt[-1])
    if t1 - t0 < 1.0:
        return None
    grid = np.arange(t0, t1, 0.01)
    s_i = np.interp(grid, times, sy)
    v_i = np.interp(grid, vt, vy)
    # Correlate the yaw RATES (the step is a spike in the rate).
    sr = np.diff(s_i)
    vr = np.diff(v_i)
    sr = sr - sr.mean()
    vr = vr - vr.mean()
    corr = np.correlate(sr, vr, mode="full")
    lags = np.arange(-len(vr) + 1, len(vr)) * 0.01
    lag = float(lags[int(np.argmax(corr))])
    return lag


def run(rec_dir):
    rec_dir = Path(rec_dir)
    times, packets = load_state_log(rec_dir / "state.jsonl")
    gt = json.loads((rec_dir / "ground_truth.json").read_text())
    segments = gt.get("segments", [])
    video_yaw = None
    vy_path = rec_dir / "video_yaw.json"
    if vy_path.exists():
        video_yaw = json.loads(vy_path.read_text())

    report = {"n_packets": len(packets),
              "t_span_s": float(times[-1] - times[0]) if len(times) else 0.0}
    # 1-2. Velocity unit + frame.
    uf = calibrate_unit_frame(times, packets, segments)
    best = None
    for hyp, errs in uf.items():
        mean_err = float(np.mean(errs)) if errs else float("inf")
        report.setdefault("velocity_hypotheses", {})[f"{hyp[0]}/{hyp[1]}"] = {
            "mean_endpoint_err_m": mean_err,
            "per_segment_err_m": [round(e, 3) for e in errs],
        }
        if best is None or mean_err < best[1]:
            best = (hyp, mean_err)
    report["velocity_best"] = {"unit": best[0][0], "frame": best[0][1],
                               "mean_endpoint_err_m": best[1]}
    # The best hypothesis must be clearly better than the runner-up
    # (a 10x unit error is unambiguous; a frame error is ~the yaw
    # angle). If the two best are within 20%, the test did not
    # discriminate — re-fly with a bigger yaw mid-translation.
    errs_sorted = sorted((float(np.mean(e)) if e else float("inf"), h)
                         for h, e in uf.items())
    report["velocity_discriminated"] = (
        len(errs_sorted) > 1 and errs_sorted[1][0] > 1.2 * max(errs_sorted[0][0], 1e-9))
    # 3. Yaw accuracy.
    if "yaw_gt_deg" in gt:
        rms, mx, n_jumps = yaw_accuracy(times, packets, gt["yaw_gt_deg"])
        report["yaw"] = {"rms_deg": rms, "max_deg": mx,
                         "jump_rejects": n_jumps}
        # Doorway bias: a magnetometer error that exists only near
        # the doorway is invisible in the overall RMS — measure it
        # separately (turn-segment error minus straight error).
        bias = yaw_doorway_bias(times, packets, gt["yaw_gt_deg"],
                                segments)
        report["yaw_doorway_bias"] = bias
        # YAW GATE (round-3 fragility finding): the real test may
        # proceed ONLY if sigma <= 1.5 deg AND |doorway bias| <=
        # 1.5 deg. Otherwise the magnetometer is not good enough as
        # a yaw prior for the 1-DoF PnP — the real test must NOT
        # proceed until the yaw source is fixed.
        report["yaw_gate"] = {
            "sigma_deg": rms,
            "doorway_bias_deg": (None if bias is None
                                 else bias["bias_deg"]),
            "criterion": ("sigma <= 1.5 deg AND |doorway bias| "
                          "<= 1.5 deg"),
            "pass": (rms <= YAW_GATE_SIGMA_DEG
                     and (bias is None
                          or abs(bias["bias_deg"]) <= YAW_GATE_BIAS_DEG)),
        }
    # 4. Height.
    report["height"] = height_check(times, packets, segments)
    # 5. VPS-blind.
    report["vps_blind_segments"] = vps_blind_check(times, packets, segments)
    # 6. Latency.
    lag = latency_check(times, packets, video_yaw)
    report["state_latency_s"] = lag
    return report


def _synthetic_log(tmpdir, unit="dm/s", frame="global",
                   yaw_noise_sigma_deg=0.0, doorway_bias_deg=0.0):
    """Generate a synthetic recording with a KNOWN unit/frame, for
    --self-test. A 1 m straight flight (x) + a 90 deg yaw turn while
    translating 0.5 m, at 10 Hz, with the velocity quantised like
    the real packet (int raw units).

    ``yaw_noise_sigma_deg`` adds KNOWN Gaussian noise to the state
    yaw (before the 1 deg quantisation) so the yaw-sigma
    measurement can be proven: the script must recover a sigma
    close to the injected one. ``doorway_bias_deg`` adds a constant
    offset to the state yaw ONLY during the yaw-turn segment (the
    doorway), so the doorway-bias measurement can be proven: the
    script must recover a bias close to the injected one and fail
    the yaw gate. Both default to 0 (the original perfect-yaw log).
    """
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 6.0, 0.1)
    # Segment 1 (0-3 s): fly +Z 1 m at 0.33 m/s, yaw 0.
    # Segment 2 (3-6 s): yaw to 90 deg while translating +X 0.5 m.
    lines = []
    gt_segments = []
    yaw_gt = {"t": [], "yaw_deg": []}
    for ti in t:
        if ti < 3.0:
            pos = np.array([0.0, 1.3, ti / 3.0 * 1.0])
            vel_world = np.array([0.0, 0.0, 1.0 / 3.0])
            yaw = 0.0
        else:
            u = (ti - 3.0) / 3.0
            yaw = 90.0 * u
            pos = np.array([u * 0.5, 1.3, 1.0])
            vel_world = np.array([0.5 / 3.0, 0.0, 0.0])
        # State yaw: GT yaw + injected noise + doorway bias (turn
        # segment only), then 1 deg quantisation like the real packet.
        yaw_state = yaw + rng.normal(0.0, yaw_noise_sigma_deg)
        if ti >= 3.0:
            yaw_state += doorway_bias_deg
        yaw_state = int(round(yaw_state))
        # Velocity in the requested frame, in the requested unit.
        if frame == "global":
            v_raw = vel_world  # DJI-x=world-Z, DJI-y=world-X
            v_dji = np.array([vel_world[2], vel_world[0], vel_world[1]])
        else:
            psi = math.radians(yaw)
            # world -> body: vx_body = vz*cos + vx*sin, vy_body = ...
            vx_b = vel_world[2] * math.cos(psi) + vel_world[0] * math.sin(psi)
            vy_b = -vel_world[2] * math.sin(psi) + vel_world[0] * math.cos(psi)
            v_dji = np.array([vx_b, vy_b, 0.0])
        scale = 10.0 if unit == "dm/s" else 100.0  # raw units per m/s
        vq = np.round(v_dji * scale).astype(int)
        # Real SDK 2.0 format: semicolon-separated key:value pairs.
        state = (f"pitch:0;roll:0;yaw:{yaw_state};"
                 f"vgx:{vq[0]};vgy:{vq[1]};vgz:{vq[2]};"
                 f"tof:130;h:130;bat:80;time:0;baro:0.0;"
                 f"agx:0;agy:0;agz:0;")
        lines.append(json.dumps({"t": float(ti), "state": state}))
        yaw_gt["t"].append(float(ti))
        yaw_gt["yaw_deg"].append(float(yaw))
    (tmpdir / "state.jsonl").write_text("\n".join(lines) + "\n")
    gt = {"segments": [
        {"kind": "straight", "distance_m": 1.0, "t_start": 0.0,
         "t_end": 3.0, "direction_deg": 0.0},
        {"kind": "yaw_turn", "angle_deg": 90, "t_start": 3.0,
         "t_end": 6.0, "distance_m": 0.5, "direction_deg": 90.0},
        {"kind": "hover", "height_m": 1.3, "t_start": 0.0, "t_end": 6.0},
    ], "yaw_gt_deg": yaw_gt}
    (tmpdir / "ground_truth.json").write_text(json.dumps(gt))
    return tmpdir


def self_test():
    import tempfile
    for unit, frame in [("dm/s", "global"), ("cm/s", "body")]:
        with tempfile.TemporaryDirectory() as d:
            rec = _synthetic_log(Path(d), unit=unit, frame=frame)
            rep = run(rec)
            best = rep["velocity_best"]
            ok = best["unit"] == unit and best["frame"] == frame
            print(f"synthetic {unit}/{frame}: recovered "
                  f"{best['unit']}/{best['frame']} "
                  f"(err {best['mean_endpoint_err_m']:.3f} m) "
                  f"discriminated={rep['velocity_discriminated']} "
                  f"yaw_rms={rep.get('yaw', {}).get('rms_deg')} deg "
                  f"-> {'OK' if ok else 'FAIL'}")
            if not ok:
                return 1
    print("self-test: unit/frame hypotheses recovered")
    # Yaw-sigma proof: synthetic logs with a KNOWN yaw noise sigma
    # (and a KNOWN doorway bias) — the script must recover them.
    # The 1 deg packet quantisation adds ~0.29 deg in quadrature,
    # so the recovered RMS is sqrt(sigma^2 + 0.29^2); the tolerance
    # below brackets that.
    # The doorway-bias estimator's own noise is the difference of
    # two segment means: sigma_diff ~= 0.26 * sigma_yaw (30 packets
    # per segment), so the bias tolerance widens with the noise.
    # (Measured on seed 0: noise 2.5 deg gives a 1.28 deg spurious
    # bias with zero injected — pure sampling noise of the two
    # means, not a script error. On a real run with sigma ~1 deg
    # the bias estimate is good to ~+/-0.3 deg.)
    yaw_cases = [
        # (noise sigma, doorway bias, expect gate, rms range, bias range)
        (1.0, 0.0, True, (0.7, 1.4), (-0.8, 0.8)),
        (2.5, 0.0, False, (2.2, 2.9), (-1.4, 1.4)),
        (1.0, 3.0, False, (0.7, 2.6), (2.4, 3.6)),
    ]
    for noise, bias_deg, expect_pass, rms_rng, bias_rng in yaw_cases:
        with tempfile.TemporaryDirectory() as d:
            rec = _synthetic_log(Path(d), unit="dm/s", frame="global",
                                 yaw_noise_sigma_deg=noise,
                                 doorway_bias_deg=bias_deg)
            rep = run(rec)
            rms = rep["yaw"]["rms_deg"]
            bias = rep["yaw_doorway_bias"]["bias_deg"]
            gate = rep["yaw_gate"]["pass"]
            ok = (rms_rng[0] <= rms <= rms_rng[1]
                  and bias_rng[0] <= bias <= bias_rng[1]
                  and gate == expect_pass)
            print(f"synthetic yaw noise={noise} deg bias={bias_deg} deg: "
                  f"recovered rms={rms:.2f} deg "
                  f"doorway_bias={bias:.2f} deg "
                  f"gate={'PASS' if gate else 'FAIL'} "
                  f"-> {'OK' if ok else 'FAIL'}")
            if not ok:
                return 1
    print("self-test: yaw sigma + doorway bias recovered, gate correct")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", nargs="?",
                    help="recording directory (state.jsonl + ground_truth.json)")
    ap.add_argument("--out", help="write the TelloCalibration JSON here")
    ap.add_argument("--self-test", action="store_true",
                    help="validate on synthetic logs with known unit/frame")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.recording:
        ap.error("a recording directory is required (or --self-test)")
    rep = run(args.recording)
    print(json.dumps(rep, indent=2))
    # Yaw gate verdict, in plain words (round-3 fragility finding):
    # the real test may proceed ONLY if the yaw sigma is <= 1.5 deg
    # AND the doorway bias is <= 1.5 deg. This is separate from the
    # velocity calibration below — a good velocity unit does not
    # excuse a bad yaw prior.
    gate = rep.get("yaw_gate")
    if gate is not None:
        if gate["pass"]:
            print(f"\nYAW GATE: PASS — sigma {gate['sigma_deg']:.2f} deg, "
                  f"doorway bias {gate['doorway_bias_deg']:.2f} deg "
                  f"(criterion: {gate['criterion']}). The yaw prior "
                  f"is good enough for the real test.")
        else:
            print(f"\nYAW GATE: FAIL — sigma {gate['sigma_deg']:.2f} deg, "
                  f"doorway bias {gate['doorway_bias_deg']:.2f} deg "
                  f"(criterion: {gate['criterion']}). The real test "
                  f"must NOT proceed — fix the yaw source first "
                  f"(re-calibrate, move the flight line away from "
                  f"metal, or add a third yaw measurement).")
    # Build the calibration from the report (only if the velocity
    # hypothesis was discriminated and the yaw is sane).
    if rep["velocity_discriminated"] and rep.get("yaw", {}).get("rms_deg", 99) < 5.0:
        cal = TelloCalibration(
            velocity_unit=rep["velocity_best"]["unit"],
            velocity_frame=rep["velocity_best"]["frame"],
            yaw_sigma_deg=max(rep["yaw"]["rms_deg"], 1.0),
            video_latency_s=rep["state_latency_s"] or 0.0,
            calibrated=True)
        if args.out:
            Path(args.out).write_text(json.dumps({
                "velocity_unit": cal.velocity_unit,
                "velocity_frame": cal.velocity_frame,
                "yaw_sigma_deg": cal.yaw_sigma_deg,
                "video_latency_s": cal.video_latency_s,
                "calibrated": True}, indent=2))
            print(f"\ncalibration written to {args.out} — review it, "
                  f"then copy it into server/tello_state.py")
        else:
            print("\ncalibration (review before use):",
                  json.dumps({"velocity_unit": cal.velocity_unit,
                              "velocity_frame": cal.velocity_frame,
                              "yaw_sigma_deg": cal.yaw_sigma_deg,
                              "video_latency_s": cal.video_latency_s,
                              "calibrated": True}, indent=2))
    else:
        print("\nNOT calibrated: the velocity hypothesis was not "
              "discriminated, or the yaw error is too large — "
              "re-fly the calibration manoeuvre (see "
              "docs/TELLO_CALIBRATION.md).")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
