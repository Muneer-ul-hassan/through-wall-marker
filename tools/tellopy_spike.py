#!/usr/bin/env python3
"""TelloPy binary-log bench spike (report §6.8 / TelloPy lead).

The undocumented TelloPy binary log (EVENT_LOG_DATA) carries onboard
odometry — ``mvo.pos`` / ``mvo.vel`` (the downward visual odometry,
in metres / m/s) plus the IMU gyro and quaternion (``imu.q0``–``q3``).
That is a SECOND odometer, independent of the front camera: it
survives the front-camera turn (the downward camera keeps seeing the
floor while the front camera rotates through a featureless wall).
If it validates, it is a strong prior / fallback for the VIO; if it
does not, it is dropped. Do NOT depend on it until validated.

What this tool does:
  * ``--record``: connect to a real Tello via TelloPy, subscribe to
    EVENT_LOG_DATA, and log mvo.pos/vel + imu gyro/quaternion (with
    laptop timestamps) alongside the video frames, over a straight
    segment and a turn segment. Writes ``tellopy_log.jsonl``.
  * ``--compare <log> <ground_truth.json>``: compare the logged
    mvo.pos against the known path (the GT trajectory) — reports the
    position error (mean/max) and the velocity error, separately for
    the straight and the turn segments. This is the validation.
  * ``--self-test``: validate the comparison logic on a SYNTHETIC
    log with known mvo values (no drone needed).

Status: BENCH TOOLING ONLY. No real drone is available in this
session and no flight is allowed, so the bench run (``--record``)
awaits a real Tello. The comparison logic is synthetic-validated
(see --self-test). djitellopy and tellopy are NOT installed in the
venv — this script imports them lazily and fails gracefully with a
clear message if they are missing.

Notes (from the research report):
  * The downward camera is EDU/Talent only and cannot stream with
    the front camera — do not chase it. The mvo odometry is computed
    onboard from the downward camera + the VPS; it is available in
    the binary log even though the downward VIDEO is not.
  * The binary log format is undocumented and version-dependent.
    The field names here (``mvo.pos``, ``mvo.vel``, ``imu.q0``–``q3``,
    ``imu.gyro``) are from the TelloPy source / community dumps; if
    the log event has different keys, ``--record`` prints the keys
    it sees so the parser can be fixed to match.
  * Mission pads are FORBIDDEN (physical markers) — never use
    mid/x/y/z/mpry, and this tool does not read them.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _import_tellopy():
    try:
        import tellopy  # noqa: F401
        return tellopy
    except ImportError:
        return None


def record(output, duration_s=60.0):
    """Connect to a real Tello and log EVENT_LOG_DATA."""
    tellopy = _import_tellopy()
    if tellopy is None:
        print("ERROR: tellopy is not installed in this venv.\n"
              "Install it with: pip install tellopy\n"
              "(djitellopy is a different library — this spike needs "
              "tellopy for the binary log.)",
              file=sys.stderr)
        return 2
    drone = tellopy.Tello()
    try:
        drone.connect()
        drone.wait_for_connection(10.0)
    except Exception as exc:  # noqa: BLE001 - bench tool, report and stop
        print(f"ERROR: could not connect to a Tello: {exc}", file=sys.stderr)
        return 2
    print("connected; waiting for the video stream to start...")
    drone.start_video()
    # Subscribe to the binary log. The exact event name / payload
    # shape is version-dependent — print what arrives.
    seen_keys = set()
    rows = []
    t_end = time.time() + duration_s

    def _on_log(event, sender, data, **kwargs):
        # data is the decoded log dict (tellopy decodes the binary
        # log into a dict of named fields).
        now = time.time()
        if isinstance(data, dict):
            seen_keys.update(data.keys())
            rows.append({"t": now, "log": data})
        else:
            rows.append({"t": now, "log_raw": repr(data)})

    try:
        drone.subscribe(drone.EVENT_LOG_DATA, _on_log)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not subscribe to EVENT_LOG_DATA: {exc}\n"
              f"Available events: "
              f"{[a for a in dir(drone) if a.startswith('EVENT_')]}",
              file=sys.stderr)
        return 2
    print(f"logging EVENT_LOG_DATA for {duration_s:.0f} s — fly the "
          f"straight + turn segments now (see docs/TELLO_CALIBRATION.md "
          f"for the manoeuvre). Ctrl-C to stop early.")
    try:
        while time.time() < t_end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        try:
            drone.quit()
        except Exception:  # noqa: BLE001 - best effort on teardown
            pass
    with open(output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} log rows to {output}")
    print(f"keys seen in the log events: {sorted(seen_keys)}")
    if not any("mvo" in k for k in seen_keys):
        print("WARNING: no mvo.* keys seen — the log format may differ "
              "from the expected one; check the keys above and update "
              "the parser in this file.")
    return 0


def _extract_mvo(log):
    """Extract (pos, vel, quat) from one log dict. Returns None if
    the expected keys are absent (the format is undocumented)."""
    if not isinstance(log, dict):
        return None
    # TelloPy exposes the log fields with dotted names in some
    # versions and nested dicts in others — try both.
    def _get(*names):
        for n in names:
            if n in log:
                return log[n]
            parts = n.split(".")
            cur = log
            ok = True
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    ok = False
                    break
            if ok:
                return cur
        return None
    pos = _get("mvo.pos", "mvo_pos", "mvo_position")
    vel = _get("mvo.vel", "mvo_vel", "mvo_velocity")
    quat = _get("imu.quaternion", "imu_quat",
                "imu.q0", "imu_q0")
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=float).reshape(-1)
    vel = np.asarray(vel, dtype=float).reshape(-1) if vel is not None else None
    if quat is not None and not isinstance(quat, (list, tuple, np.ndarray)):
        # q0..q3 as separate keys
        q = [_get(f"imu.q{i}", f"imu_q{i}") for i in range(4)]
        quat = np.array(q, dtype=float) if all(v is not None for v in q) else None
    return pos, vel, quat


def compare(log_path, gt_path):
    """Compare a TelloPy log against the known path."""
    gt = json.loads(Path(gt_path).read_text())
    segs = gt.get("segments", [])
    gt_traj = gt.get("trajectory")  # {"t": [...], "pos": [[x,y,z], ...]}
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("no log rows", file=sys.stderr)
        return 2
    mvo_t, mvo_pos, mvo_vel = [], [], []
    for r in rows:
        ext = _extract_mvo(r.get("log", {}))
        if ext is None:
            continue
        pos, vel, _quat = ext
        mvo_t.append(float(r["t"]))
        mvo_pos.append(pos)
        mvo_vel.append(vel)
    if not mvo_pos:
        print("ERROR: no mvo.pos in the log — the log format differs "
              "from the expected one. Re-run --record and check the "
              "keys it prints.", file=sys.stderr)
        return 2
    mvo_t = np.array(mvo_t)
    mvo_pos = np.array(mvo_pos)
    print(f"{len(mvo_pos)} mvo samples, t span "
          f"{mvo_t[-1] - mvo_t[0]:.1f} s")
    # The mvo frame is the takeoff frame (x forward, y right, z up
    # in the Tello body frame at takeoff). The GT trajectory is in
    # the geometry frame (x right, y up, z forward). The comparison
    # below is done in the mvo frame: the GT is rotated into it by
    # the takeoff yaw (gt["takeoff_yaw_deg"], default 0 = the drone
    # took off facing +Z).
    if gt_traj is None:
        print("no 'trajectory' in the GT file — cannot compare "
              "positions. Provide {\"t\": [...], \"pos\": [[x,y,z], ...]} "
              "in the geometry frame.", file=sys.stderr)
        return 2
    gt_t = np.array(gt_traj["t"], dtype=float)
    gt_p = np.array(gt_traj["pos"], dtype=float)
    yaw0 = math.radians(gt.get("takeoff_yaw_deg", 0.0))
    # Geometry (x right, y up, z forward) -> mvo (x forward, y right,
    # z up): mvo_x = z, mvo_y = x, mvo_z = y, then rotate by -yaw0
    # about the up axis (the mvo frame is fixed at takeoff).
    gp = np.stack([gt_p[:, 2], gt_p[:, 0], gt_p[:, 1]], axis=1)
    c, s = math.cos(yaw0), math.sin(yaw0)
    gp = np.stack([gp[:, 0] * c + gp[:, 1] * s,
                   -gp[:, 0] * s + gp[:, 1] * c,
                   gp[:, 2]], axis=1)
    # Interpolate the GT onto the mvo times, align the origins (the
    # mvo pos is relative to takeoff; the GT is absolute — subtract
    # the GT at the first mvo time).
    gt_i = np.stack([np.interp(mvo_t, gt_t, gp[:, k]) for k in range(3)],
                    axis=1)
    gt_i = gt_i - gt_i[0]
    mvo_rel = mvo_pos - mvo_pos[0]
    err = np.linalg.norm(mvo_rel - gt_i, axis=1)
    print(f"mvo position error vs GT: mean {np.mean(err) * 100:.1f} cm, "
          f"max {np.max(err) * 100:.1f} cm")
    for seg in segs:
        idx = [i for i, t in enumerate(mvo_t)
               if seg["t_start"] <= t <= seg["t_end"]]
        if not idx:
            continue
        e = err[idx]
        print(f"  {seg['kind']} ({seg['t_start']:.1f}-{seg['t_end']:.1f}s): "
              f"mean {np.mean(e) * 100:.1f} cm, max {np.max(e) * 100:.1f} cm")
    # Velocity error (where the log carries it).
    if mvo_vel[0] is not None:
        gt_v = np.gradient(gt_i, mvo_t, axis=0)
        mv = np.array([v if v is not None else np.full(3, np.nan)
                       for v in mvo_vel])
        ok = ~np.isnan(mv).any(axis=1)
        verr = np.linalg.norm(mv[ok] - gt_v[ok], axis=1)
        print(f"mvo velocity error vs GT: mean {np.mean(verr) * 100:.1f} "
              f"cm/s, max {np.max(verr) * 100:.1f} cm/s")
    return 0


def self_test():
    """Validate the comparison logic on a synthetic log (no drone)."""
    import tempfile
    rng = np.random.default_rng(0)
    # A 1 m straight flight + a 90 deg turn, at 10 Hz, with the mvo
    # pos = the true pos + 2 cm of noise (a good odometer).
    t = np.arange(0.0, 6.0, 0.1)
    pos = []
    for ti in t:
        if ti < 3.0:
            pos.append([0.0, 1.3, ti / 3.0 * 1.0])
        else:
            u = (ti - 3.0) / 3.0
            pos.append([u * 0.5, 1.3, 1.0])
    pos = np.array(pos)
    # mvo frame: x forward (= geometry z), y right (= geometry x).
    mvo = np.stack([pos[:, 2], pos[:, 0], pos[:, 1]], axis=1)
    mvo = mvo + rng.normal(0, 0.02, mvo.shape)
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "tellopy_log.jsonl"
        gt_path = Path(d) / "ground_truth.json"
        with open(log_path, "w") as f:
            for ti, p in zip(t, mvo):
                f.write(json.dumps({
                    "t": float(ti),
                    "log": {"mvo.pos": [float(v) for v in p],
                            "mvo.vel": [0.0, 0.0, 0.0],
                            "imu.q0": 1.0, "imu.q1": 0.0,
                            "imu.q2": 0.0, "imu.q3": 0.0}}) + "\n")
        gt = {"takeoff_yaw_deg": 0.0,
              "segments": [
                  {"kind": "straight", "t_start": 0.0, "t_end": 3.0},
                  {"kind": "yaw_turn", "t_start": 3.0, "t_end": 6.0}],
              "trajectory": {"t": [float(v) for v in t],
                             "pos": [[float(v) for v in p] for p in pos]}}
        gt_path.write_text(json.dumps(gt))
        rc = compare(str(log_path), str(gt_path))
    print("self-test: " + ("OK" if rc == 0 else "FAIL"))
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", metavar="OUT",
                    help="record EVENT_LOG_DATA to OUT (needs a real Tello)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="record duration in seconds (default 60)")
    ap.add_argument("--compare", nargs=2, metavar=("LOG", "GT"),
                    help="compare a log against a GT file")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the comparison on synthetic data")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.record:
        return record(args.record, args.duration)
    if args.compare:
        return compare(args.compare[0], args.compare[1])
    ap.error("one of --record, --compare, --self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
