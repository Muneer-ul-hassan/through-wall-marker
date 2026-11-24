"""Tracker-robustness regression tests (2026-11-25).

Standalone: run from the project root with the project venv::

    ~/workspace/tello-search/.venv/bin/python -m pytest \\
        tests/test_tracker_robustness.py -q

No drone, no flight, no network — the VIO runs on the synthetic
textured room from tests/test_relocalisation.py, driven through
the real ``MonocularVO``. NOT wired into tests/run_tests.py (that
runner is the 27-scenario virtual suite; these are live-path unit
tests).

Root cause (measured, 2026-11-25, 04:31 live run + synthetic
reproduction): the map collapsed 482 -> 15 -> 0 and PnP failed
twice with ``pnp-failed`` because of TWO bugs in ``server/vio.py``
``_track`` — NOT because the tracker "just loses it":
1. The relocalisation anchor (``_good_*``) was overwritten on
   EVERY accepted frame, including attitude-aided (rotation-only)
   frames that carry ZERO 3D points. Measured: one 30-deg turn
   dropped the anchor from 130 3D points to 0 in 3 frames, and
   once the anchor has 0 3D points ``_try_relocalise`` can never
   recover (it needs >= 12). Fix: only a vision-PnP frame may
   become the anchor.
2. The 3-px reprojection prune ran against EVERY accepted pose,
   including rotation-only poses whose translation is a
   motion-model PREDICTION, not a measurement — a reprojection
   failure there is the pose's fault, not the point's. Measured:
   during a 30-deg turn the prune deleted 27 of 57 surviving map
   points in one frame and all 8 remaining 3D points in the next.
   Fix: prune only on vision-PnP poses.
3. ``_try_relocalise`` returned False the moment the LK anchor was
   too thin (or LK itself failed) WITHOUT ever consulting the
   keyframe ORB database, which survives independently of the live
   anchor. Fix: fall through to ``_orb_recover`` in both cases.

Plus the autopilot side (``server/autopilot.py``): blind-scan
turns (SCAN / SEARCH) are now 15 deg, not 30 — measured, a 30-deg
turn drops the map to 0 in 3 frames, while 15-deg turns keep
vision PnP alive for ~48 deg of yaw (vs ~12 deg). Targeted
correction turns (AIM / TRACK) and manual keys stay 30 deg.

What each test proves:
* the relocalisation anchor keeps >= 12 3D points through a
  30-deg turn (measured 13; before the fix it hit 0).
* the prune never fires on an attitude-aided frame (pruned == 0),
  while vision frames still prune (the prune still works where
  the pose is a measurement).
* 15-deg turns keep vision PnP alive for 45 deg of yaw
  (n_map >= 12), while 30-deg turns are already attitude-aided
  at 18 deg of yaw.
* relocalisation recovers after a turn + loss (end-to-end, via
  the surviving anchor).
* ORB recovery fires when the LK anchor is too thin (white-box:
  the anchor is thinned to 5 3D points by hand, the keyframes
  still carry the map — the test proves the new code path, not
  the synthetic room's keyframe coverage).

The recovery inlier threshold was deliberately NOT lowered: the
live relocalisation succeeded at 20 inliers, and the 04:32:38
loss had 13 inliers (which passes the 12-point thin tier) — so
no measured case shows a correct pose refused by the gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.vio import MIN_PNP_THIN_INLIERS  # noqa: E402
from tests.test_relocalisation import (  # noqa: E402
    _drive_vio, _lose_tracking, _pose_at, _render)

DRIVE_END_X = 1.95  # x of the last drive frame (14 frames x 0.15)


def _anchor_3d(vio):
    """3D points in the relocalisation anchor (0 if none)."""
    if vio._good_pts3d is None:
        return 0
    return int((~np.isnan(
        np.asarray(vio._good_pts3d, dtype=float)).any(axis=1)).sum())


def _turn_frames(vio, scene, x, yaw_step_deg, n_frames, t0=1.4):
    """Feed n_frames yawing at yaw_step_deg/frame from yaw 0.
    Returns the list of (valid, diag) per frame."""
    out = []
    for k in range(n_frames):
        yaw = yaw_step_deg * (k + 1)
        frame = _render(_pose_at(x, yaw_deg=yaw), scene)
        _est, valid, diag = vio.update(frame, t=t0 + k * 0.1)
        out.append((valid, diag))
    return out


# ---------------------------------------------------------------------------
# Fix 1 — the anchor survives attitude-aided frames
# ---------------------------------------------------------------------------

def test_good_anchor_survives_attitude_aided_frames():
    """One 30-deg turn: the anchor keeps >= 12 3D points (measured
    13; before the fix it hit 0 in 3 frames and recovery became
    impossible)."""
    scene, vio, _est = _drive_vio()
    before = _anchor_3d(vio)
    assert before >= 100, f"drive anchor too thin to test: {before}"
    frames = _turn_frames(vio, scene, DRIVE_END_X, 6.0, 5)
    assert any(d.get("pose_source") == "attitude-aided"
               for _v, d in frames), \
        "no attitude-aided frame in the turn — test is vacuous"
    after = _anchor_3d(vio)
    assert after >= MIN_PNP_THIN_INLIERS, \
        f"anchor drained to {after} 3D points (was {before})"
    print(f"PASS good_anchor_survives_attitude_aided_frames "
          f"(anchor {before} -> {after} 3D points)")


# ---------------------------------------------------------------------------
# Fix 2 — the prune runs on vision poses only
# ---------------------------------------------------------------------------

def test_prune_skipped_on_attitude_aided_frames():
    """During a 30-deg turn, every attitude-aided frame prunes
    nothing (pruned == 0), while vision frames still prune — the
    prune still works where the pose is a measurement."""
    scene, vio, _est = _drive_vio()
    frames = _turn_frames(vio, scene, DRIVE_END_X, 6.0, 5)
    aided = [d for _v, d in frames
             if d.get("pose_source") == "attitude-aided"]
    vision = [d for _v, d in frames if d.get("pose_source") == "vision"]
    assert aided, "no attitude-aided frame in the turn"
    assert all(d.get("pruned", -1) == 0 for d in aided), \
        f"prune fired on an attitude-aided frame: " \
        f"{[d.get('pruned') for d in aided]}"
    assert vision and any(d.get("pruned", 0) > 0 for d in vision), \
        "vision frames pruned nothing — the prune itself is broken"
    print(f"PASS prune_skipped_on_attitude_aided_frames "
          f"({len(aided)} attitude-aided frames, all pruned=0; "
          f"vision pruned {[d.get('pruned') for d in vision]})")


# ---------------------------------------------------------------------------
# Fix 3 — 15-deg turns keep vision alive longer than 30-deg turns
# ---------------------------------------------------------------------------

def test_fifteen_deg_turns_keep_vision_longer():
    """Four 15-deg turns (60 deg of yaw at 3 deg/frame): vision PnP
    is still alive at 45 deg of yaw (measured n_map 33). Contrast:
    30-deg turns are already attitude-aided at 18 deg of yaw
    (measured n_map 8)."""
    scene, vio, _est = _drive_vio()
    frames = _turn_frames(vio, scene, DRIVE_END_X, 3.0, 20)
    # Frame 15 is 45 deg of yaw (15 frames x 3 deg).
    valid15, diag15 = frames[14]
    assert diag15.get("pose_source") == "vision", \
        f"15-deg turns lost vision by 45 deg yaw: {diag15}"
    assert diag15.get("n_map", 0) >= MIN_PNP_THIN_INLIERS, \
        f"15-deg turns: n_map {diag15.get('n_map')} at 45 deg yaw"
    assert valid15 is True
    # Contrast: the same room, 30-deg turns (6 deg/frame).
    _scene2, vio2, _est2 = _drive_vio()
    frames30 = _turn_frames(vio2, _scene2, DRIVE_END_X, 6.0, 3)
    _v30, diag30 = frames30[2]  # 18 deg of yaw
    assert diag30.get("pose_source") == "attitude-aided", \
        f"30-deg turns still vision at 18 deg yaw: {diag30}"
    print(f"PASS fifteen_deg_turns_keep_vision_longer "
          f"(15-deg: vision at 45 deg yaw, n_map "
          f"{diag15.get('n_map')}; 30-deg: attitude-aided at 18 deg "
          f"yaw, n_map {diag30.get('n_map')})")


# ---------------------------------------------------------------------------
# End-to-end — relocalisation after a turn + loss
# ---------------------------------------------------------------------------

def test_relocalise_after_turn_loss():
    """Drive, one 30-deg turn, forced loss, then the anchor's view
    again: the in-VIO relocalisation recovers (valid True). Before
    the anchor fix the anchor was 0 and this could never recover."""
    scene, vio, _est = _drive_vio()
    _turn_frames(vio, scene, DRIVE_END_X, 6.0, 5, t0=1.4)
    assert _anchor_3d(vio) >= MIN_PNP_THIN_INLIERS, \
        "anchor too thin to test recovery"
    _lose_tracking(vio)
    # The anchor is the last vision frame of the turn (yaw ~12 deg);
    # feed that view until a relocalise attempt fires (every 5th
    # lost frame) and recovery lands.
    recovered = False
    last_diag = {}
    for k in range(6):
        _est, valid, diag = vio.update(
            _render(_pose_at(DRIVE_END_X, yaw_deg=12.0), scene),
            t=2.2 + k * 0.1)
        last_diag = diag
        if valid:
            recovered = True
            break
    assert recovered, \
        f"no relocalisation after turn + loss: {last_diag}"
    assert not vio._lost
    print(f"PASS relocalise_after_turn_loss "
          f"(relocalised={last_diag.get('relocalised')}, "
          f"inliers={last_diag.get('inliers')})")


# ---------------------------------------------------------------------------
# Fix 4 — ORB recovery when the LK anchor is too thin (white-box)
# ---------------------------------------------------------------------------

def test_orb_recovery_when_lk_anchor_too_thin():
    """Thin the anchor to 5 3D points by hand (simulating the
    drained-anchor state the 04:31 live run hit), force a loss,
    then feed a mapped view: the LK path cannot attempt PnP
    (< 12 3D), so the ORB keyframe path must fire and recover.
    Before the fix _try_relocalise returned False here without
    ever consulting the keyframes."""
    scene, vio, _est = _drive_vio()
    assert len(vio._keyframes) >= 1, "drive left no keyframes to test"
    kf_3d = [int((~np.isnan(np.asarray(kf["pts3d"], dtype=float))
                  .any(axis=1)).sum()) for kf in vio._keyframes]
    assert max(kf_3d) >= MIN_PNP_THIN_INLIERS, \
        f"keyframes too thin to test ORB recovery: {kf_3d}"
    # Thin the anchor: keep 5 3D points, NaN the rest.
    pts3d = np.asarray(vio._good_pts3d, dtype=float).copy()
    has3d = ~np.isnan(pts3d).any(axis=1)
    idx3d = np.nonzero(has3d)[0]
    assert len(idx3d) >= 12, "anchor already thin — test is vacuous"
    thinned = np.full_like(pts3d, np.nan)
    thinned[idx3d[:5]] = pts3d[idx3d[:5]]
    vio._good_pts3d = thinned
    assert _anchor_3d(vio) == 5
    _lose_tracking(vio)
    # A mapped view (x=1.5, yaw 0 — a drive keyframe view): LK
    # tracks the anchor's 2D points but only 5 have 3D, so the
    # ORB path must do the recovery.
    recovered = False
    last_diag = {}
    for k in range(6):
        _est, valid, diag = vio.update(
            _render(_pose_at(1.5), scene), t=2.2 + k * 0.1)
        last_diag = diag
        if valid:
            recovered = True
            break
    assert recovered, \
        f"ORB recovery did not fire on a thin anchor: {last_diag}"
    assert last_diag.get("relocalised") == "orb", \
        f"recovered, but not via ORB: {last_diag}"
    print(f"PASS orb_recovery_when_lk_anchor_too_thin "
          f"(inliers={last_diag.get('inliers')})")


# ---------------------------------------------------------------------------
# Runner (same shape as the other live-path test files)
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_good_anchor_survives_attitude_aided_frames,
    test_prune_skipped_on_attitude_aided_frames,
    test_fifteen_deg_turns_keep_vision_longer,
    test_relocalise_after_turn_loss,
    test_orb_recovery_when_lk_anchor_too_thin,
]


def run_all():
    for fn in ALL_TESTS:
        fn()
    print(f"tracker-robustness: {len(ALL_TESTS)}/{len(ALL_TESTS)} PASS")


if __name__ == "__main__":
    run_all()
