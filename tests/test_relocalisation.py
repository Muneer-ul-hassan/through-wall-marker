"""Saved-map relocalisation tests (2026-11-25).

Standalone: run from the project root with the project venv::

    ~/workspace/tello-search/.venv/bin/python -m pytest \\
        tests/test_relocalisation.py -q

No drone, no flight, no network — the VIO runs on a synthetic
textured room rendered in this file (checker floor with unique
per-square speckles, striped posters on the back wall, boxes for
non-planar structure), driven through the real ``MonocularVO``.
NOT wired into tests/run_tests.py (that runner is the 27-scenario
virtual suite; these are saved-map unit tests).

What each test proves:
* save/load round-trip (synthetic MapData, and a map driven from
  the real VIO): points, descriptors, validity flags, IDs, K and
  the reference pose survive unchanged.
* relocalisation: after simulated tracking loss (blank frames ->
  lk-empty -> _declare_lost), a frame viewing the same scene
  relocalises against the SAVED map, returns True, and the
  recovered pose is within 0.35 m of the VIO's own map-frame
  estimate for that view (measured 0.015-0.05 m same-view); tracking
  then resumes on the next frame. A shifted view (+0.3 m, +8 deg
  yaw) also relocalises (measured 17 inliers).
* negative case: a frame of a different scene returns False, leaves
  the VIO state untouched (still lost, same pose), and never raises.
* restart case: a FRESH VIO (never initialised) relocalises against
  a map saved by another VIO — the server-start path.
* empty map saves nothing; a missing file raises FileNotFoundError;
  garbage inputs to try_relocalise return False, never raise.

Tolerance note: the recovered pose is judged against the VIO's own
map-frame estimate, NOT ground truth. The synthetic flight is
sideways-only, so the VIO's init scale is stretched ~1.4x in x and
the whole map frame sits ~0.7 m from truth at frame 10 — that is
VIO scale drift, not a relocalisation error. Relocalisation restores
the MAP frame (so tracking + the bridge stay consistent); it does
not fix scale drift.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.geometry import DRONE_K, project_point  # noqa: E402
from server.map_store import (DEFAULT_MAP_PATH, MapData, load_map,  # noqa: E402
                              save_map, try_relocalise)
from server.vio import MonocularVO  # noqa: E402

START_POSE = (np.eye(3), np.array([0.0, 1.3, 0.0]))
RELOCALISE_TOL_M = 0.35  # vs the VIO's own map-frame estimate
N_DRIVE_FRAMES = 14


# ---------------------------------------------------------------------------
# Synthetic textured room (self-contained renderer)
# ---------------------------------------------------------------------------

def _look_at_R(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation, geometry convention (cf. tests/render)."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    z_cam = target - eye
    z_cam /= np.linalg.norm(z_cam)
    up = np.asarray(up, dtype=float)
    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return np.stack([x_cam, y_cam, z_cam])


def _build_scene(seed=7):
    """Checker floor + speckles + posters + boxes. ``seed`` picks the
    texture: a different seed is a different scene (negative case)."""
    rng = np.random.default_rng(seed)
    polys = []  # (world quad (4,3), gray)
    for ix in range(-8, 8):
        for iz in range(0, 16):
            x0, x1 = ix * 0.5, (ix + 1) * 0.5
            z0, z1 = iz * 0.5, (iz + 1) * 0.5
            shade = 85 if (ix + iz) % 2 == 0 else 195
            polys.append((np.array([[x0, 0, z0], [x1, 0, z0],
                                    [x1, 0, z1], [x0, 0, z1]], float),
                          shade))
    dots = []  # (world xyz, gray, radius_m): unique texture per square
    for ix in range(-8, 8):
        for iz in range(0, 16):
            for _ in range(4):
                dots.append((np.array([(ix + rng.random()) * 0.5, 0.0,
                                       (iz + rng.random()) * 0.5]),
                             int(rng.integers(30, 230)), 0.035))
    polys.append((np.array([[-4, 0, 7.5], [4, 0, 7.5],
                            [4, 3, 7.5], [-4, 3, 7.5]], float), 120))
    posters = []
    for px in (-3.0, -1.5, 0.0, 1.5, 3.0):
        w, h, y0 = 1.0, 1.3, 1.0
        bars, bx = [], px - w / 2
        while bx < px + w / 2 - 0.05:
            bw = float(rng.uniform(0.08, 0.25))
            bars.append((bx, min(bw, px + w / 2 - bx),
                         int(rng.integers(40, 230))))
            bx += bw
        posters.append((px, y0, w, h, bars))
    boxes = [(-2.0, 2.5, 0.6, 0.5), (2.2, 3.5, 0.5, 0.8),
             (-1.0, 5.0, 0.7, 0.4), (1.2, 1.8, 0.45, 0.6)]
    return polys, dots, posters, boxes


def _render(pose, scene):
    """Render one 960x720 BGR frame of the scene from ``pose`` (R, C)."""
    polys, dots, posters, boxes = scene
    K = DRONE_K
    W, H = int(K["width"]), int(K["height"])
    frame = np.full((H, W, 3), 60, dtype=np.uint8)
    R, C = pose

    def proj_poly(pts):
        p_cam = (R @ (np.asarray(pts, dtype=float) - C).T).T
        if np.any(p_cam[:, 2] <= 0.15):
            return None
        z = p_cam[:, 2]
        u = K["fx"] * p_cam[:, 0] / z + K["cx"]
        v = K["cy"] - K["fy"] * p_cam[:, 1] / z
        return np.stack([u, v], axis=1).astype(np.int32)

    items = [(float(np.mean((R @ (pts - C).T).T[:, 2])), pts, shade)
             for pts, shade in polys]
    items.sort(key=lambda it: -it[0])  # far to near
    for _, pts, shade in items:
        px = proj_poly(pts)
        if px is not None:
            cv2.fillPoly(frame, [px],
                         (int(shade), int(shade), int(shade)))
    for _px0, y0, w, h, bars in posters:
        for bx, bw, shade in bars:
            pts = np.array([[bx, y0, 7.49], [bx + bw, y0, 7.49],
                            [bx + bw, y0 + h, 7.49],
                            [bx, y0 + h, 7.49]])
            px = proj_poly(pts)
            if px is not None:
                cv2.fillPoly(frame, [px],
                             (int(shade), int(shade), int(shade)))
    for cx, cz, s, h in boxes:
        x0, x1, z0, z1 = cx - s / 2, cx + s / 2, cz - s / 2, cz + s / 2
        for pts, shade in [
                (np.array([[x0, h, z0], [x1, h, z0], [x1, h, z1],
                           [x0, h, z1]], float), 210),
                (np.array([[x0, 0, z1], [x1, 0, z1], [x1, h, z1],
                           [x0, h, z1]], float), 150),
                (np.array([[x1, 0, z0], [x1, 0, z1], [x1, h, z1],
                           [x1, h, z0]], float), 110)]:
            px = proj_poly(pts)
            if px is not None:
                cv2.fillPoly(frame, [px],
                             (int(shade), int(shade), int(shade)))
    for p, shade, r_m in dots:
        pr = project_point(K, R, C, p)
        if pr is None:
            continue
        u, v, z = pr
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(frame, (int(u), int(v)),
                       max(2, int(28 * r_m / z)),
                       (int(shade), int(shade), int(shade)), -1)
    return frame


def _pose_at(x, yaw_deg=0.0):
    """Camera pose at (x, 1.3, 0), looking forward, slightly down,
    optionally yawed about world Y."""
    eye = np.array([x, 1.3, 0.0])
    R = _look_at_R(eye, eye + np.array([0.0, -0.35, 3.0]))
    if yaw_deg:
        th = math.radians(yaw_deg)
        Ry = np.array([[math.cos(th), 0.0, math.sin(th)],
                       [0.0, 1.0, 0.0],
                       [-math.sin(th), 0.0, math.cos(th)]])
        R = Ry @ R
    return R, eye


def _drive_vio(n_frames=N_DRIVE_FRAMES, seed=7):
    """Drive the real VIO sideways across the synthetic room.

    Returns (scene, vio, est_poses) with est_poses[i] the VIO's own
    (R, C) estimate at frame i — the map-frame reference poses."""
    scene = _build_scene(seed=seed)
    vio = MonocularVO(start_pose=START_POSE)
    est_poses = {}
    for i in range(n_frames):
        frame = _render(_pose_at(0.15 * i), scene)
        est, valid, _diag = vio.update(frame, t=i * 0.1)
        est_poses[i] = (est, valid)
    assert vio.initialized, "synthetic drive did not initialise the VIO"
    # Frame 0 is the keyframe and is NEVER valid by design (no motion
    # observed yet); every frame after initialisation must be valid.
    assert all(v for i, (_e, v) in est_poses.items() if i > 0), \
        "synthetic drive lost tracking mid-flight"
    return scene, vio, est_poses


def _lose_tracking(vio):
    """Blank frames -> lk-empty -> _declare_lost (the blanket case)."""
    blank = np.zeros((720, 960, 3), dtype=np.uint8)
    for i in range(3):
        _, valid, _diag = vio.update(blank, t=1.4 + i * 0.1)
        assert valid is False
    assert vio._lost, "blank frames did not lose tracking"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_synthetic_map():
    """A hand-built MapData survives save -> load bit-for-bit."""
    rng = np.random.default_rng(0)
    md = MapData(
        points=rng.normal(size=(50, 3)),
        descriptors=rng.integers(0, 256, size=(50, 32), dtype=np.uint8),
        des_valid=rng.random(50) > 0.3,
        ids=np.arange(50, dtype=np.int64),
        K=dict(DRONE_K), camera_height_m=1.3,
        ref_R=np.eye(3), ref_C=np.array([0.1, 1.3, 0.2]),
        ref_frame=42,
        keyframe_R=np.stack([np.eye(3), np.eye(3)]),
        keyframe_C=np.array([[0.0, 1.3, 0.0], [0.1, 1.3, 0.2]]),
        keyframe_frame=np.array([1, 42]),
        start_R=np.eye(3), start_C=np.array([0.0, 1.3, 0.0]),
        n_frames=42, n_keyframes=2)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "map.npz"
        assert save_map(path, md) == 50
        back = load_map(path)
    assert len(back) == 50
    assert np.array_equal(back.points, md.points)
    assert np.array_equal(back.descriptors, md.descriptors)
    assert np.array_equal(back.des_valid, md.des_valid)
    assert np.array_equal(back.ids, md.ids)
    assert back.K == md.K
    assert np.array_equal(back.ref_R, md.ref_R)
    assert np.array_equal(back.ref_C, md.ref_C)
    assert back.ref_frame == 42 and back.n_frames == 42
    assert back.n_keyframes == 2
    assert np.array_equal(back.keyframe_C, md.keyframe_C)
    assert np.array_equal(back.start_R, md.start_R)
    assert np.array_equal(back.start_C, md.start_C)
    print("PASS save_load_roundtrip_synthetic_map")


def test_save_load_roundtrip_driven_vio():
    """A map exported from a real driven VIO survives save -> load."""
    _scene, vio, _est = _drive_vio()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "latest.npz"
        n = save_map(path, vio)
        assert n > 100, f"driven map too small: {n} points"
        back = load_map(path)
    assert len(back) == n
    assert back.n_descriptors >= 12, \
        f"too few described points to relocalise: {back.n_descriptors}"
    assert back.K == dict(DRONE_K)
    assert back.camera_height_m == 1.3
    assert back.ref_R.shape == (3, 3) and back.ref_C.shape == (3,)
    assert np.isfinite(back.points).all()
    assert back.n_keyframes >= 1
    # Export is read-only: saving must not disturb live tracking.
    assert vio.initialized and not vio._lost
    print(f"PASS save_load_roundtrip_driven_vio "
          f"({n} pts, {back.n_descriptors} described)")


def test_default_map_path_is_under_repo():
    """The default map path convention: maps/latest.npz in the repo."""
    assert DEFAULT_MAP_PATH.name == "latest.npz"
    assert DEFAULT_MAP_PATH.parent.name == "maps"
    assert DEFAULT_MAP_PATH.parent.parent == WALL_MARKER_DIR
    print("PASS default_map_path_is_under_repo")


# ---------------------------------------------------------------------------
# Relocalisation
# ---------------------------------------------------------------------------

def test_relocalise_same_view():
    """Lost -> same view as a mapped frame -> True, pose recovered
    within tolerance of the VIO's own map-frame estimate, and
    tracking resumes on the next frame."""
    scene, vio, est_poses = _drive_vio()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "latest.npz"
        save_map(path, vio)
        md = load_map(path)
    _lose_tracking(vio)
    diag = {}
    ok = try_relocalise(vio, _render(_pose_at(1.5), scene), md, diag)
    assert ok is True, f"same-view relocalisation failed: {diag}"
    assert diag.get("relocalised") == "saved-map"
    assert diag.get("inliers", 0) >= 12, f"diag: {diag}"
    assert vio.initialized and not vio._lost
    _R, C = vio.pose
    ref_C = est_poses[10][0][1]  # VIO's own estimate at frame 10 (x=1.5)
    err = float(np.linalg.norm(C - ref_C))
    assert err <= RELOCALISE_TOL_M, \
        f"recovered pose {err:.3f} m from map-frame estimate " \
        f"(tol {RELOCALISE_TOL_M} m)"
    # Tracking genuinely resumes: the next frame localises valid.
    _, valid, diag2 = vio.update(_render(_pose_at(1.65), scene), t=1.8)
    assert valid is True, f"tracking did not resume: {diag2}"
    print(f"PASS relocalise_same_view "
          f"(err {err:.3f} m, {diag['inliers']} inliers)")


def test_relocalise_shifted_view():
    """Lost -> view shifted +0.3 m and +8 deg yaw -> still True, pose
    within tolerance of the shifted map-frame expectation."""
    scene, vio, est_poses = _drive_vio()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "latest.npz"
        save_map(path, vio)
        md = load_map(path)
    _lose_tracking(vio)
    diag = {}
    ok = try_relocalise(vio, _render(_pose_at(1.8, yaw_deg=8.0), scene),
                        md, diag)
    assert ok is True, f"shifted-view relocalisation failed: {diag}"
    _R, C = vio.pose
    # Map-frame expectation: frame-10 estimate shifted by +0.3 m in
    # the map frame's (stretched) x — the map frame's x-scale is
    # est_x / true_x at frame 10.
    ref_C = est_poses[10][0][1]
    scale = float(ref_C[0]) / 1.5
    expected = ref_C + np.array([0.3 * scale, 0.0, 0.0])
    err = float(np.linalg.norm(C - expected))
    assert err <= RELOCALISE_TOL_M, \
        f"shifted pose {err:.3f} m from map-frame expectation " \
        f"(tol {RELOCALISE_TOL_M} m)"
    assert 0.5 <= float(C[1]) <= 2.5  # height guard held
    print(f"PASS relocalise_shifted_view "
          f"(err {err:.3f} m, {diag.get('inliers')} inliers)")


def test_relocalise_different_scene_returns_false_and_preserves_state():
    """A frame of a different scene -> False, no crash, VIO state
    byte-identical (still lost, same pose, still uninitialised-ness
    untouched) and a later update still reports invalid."""
    scene, vio, _est = _drive_vio()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "latest.npz"
        save_map(path, vio)
        md = load_map(path)
    _lose_tracking(vio)
    before_R, before_C = vio._R.copy(), vio._C.copy()
    before_lost, before_init = vio._lost, vio.initialized
    other_scene = _build_scene(seed=99)  # different texture = new room
    diag = {}
    ok = try_relocalise(vio, _render(_pose_at(1.5), other_scene), md,
                        diag)
    assert ok is False, f"different scene relocalised?! diag: {diag}"
    assert np.array_equal(vio._R, before_R)
    assert np.array_equal(vio._C, before_C)
    assert vio._lost is before_lost and vio.initialized is before_init
    _, valid, _d = vio.update(_render(_pose_at(1.65), scene), t=1.8)
    assert valid is False  # still lost, not silently "recovered"
    print("PASS relocalise_different_scene_returns_false_and_"
          "preserves_state")


def test_relocalise_fresh_vio_from_saved_map():
    """Restart case: a fresh VIO that never initialised relocalises
    against a map saved by another VIO (the server-start path)."""
    scene, vio, est_poses = _drive_vio()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "latest.npz"
        save_map(path, vio)
        md = load_map(path)
    fresh = MonocularVO(start_pose=START_POSE)
    assert not fresh.initialized
    diag = {}
    ok = try_relocalise(fresh, _render(_pose_at(1.5), scene), md, diag)
    assert ok is True, f"fresh-VIO relocalisation failed: {diag}"
    assert fresh.initialized and not fresh._lost
    _R, C = fresh.pose
    ref_C = est_poses[10][0][1]
    err = float(np.linalg.norm(C - ref_C))
    assert err <= RELOCALISE_TOL_M, f"fresh-VIO pose err {err:.3f} m"
    print(f"PASS relocalise_fresh_vio_from_saved_map (err {err:.3f} m)")


# ---------------------------------------------------------------------------
# Edge cases: empty / missing / garbage
# ---------------------------------------------------------------------------

def test_save_empty_map_writes_nothing():
    """An uninitialised VIO has no map: save returns 0, writes no file."""
    vio = MonocularVO(start_pose=START_POSE)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "latest.npz"
        assert save_map(path, vio) == 0
        assert not path.exists()
    print("PASS save_empty_map_writes_nothing")


def test_load_missing_file_raises():
    """Loading a map that was never saved is a loud error, not a
    silent empty map (the caller treats it as 'no saved map')."""
    with tempfile.TemporaryDirectory() as td:
        try:
            load_map(Path(td) / "nope.npz")
        except FileNotFoundError:
            print("PASS load_missing_file_raises")
            return
    raise AssertionError("load_map on a missing file did not raise")


def test_try_relocalise_never_raises_on_garbage():
    """None / empty / wrong-shaped inputs return False, never raise —
    a failed relocalisation is an ordinary lost frame, not a crash
    (the live pipeline thread must survive it)."""
    _scene, vio, _est = _drive_vio()
    _lose_tracking(vio)
    assert try_relocalise(vio, None, None) is False
    assert try_relocalise(vio, np.zeros((720, 960, 3), dtype=np.uint8),
                          MapData()) is False
    assert try_relocalise(vio, "not an image", MapData()) is False
    assert try_relocalise(None, np.zeros((10, 10, 3), dtype=np.uint8),
                          MapData()) is False
    assert vio._lost  # still lost, untouched by the garbage calls
    print("PASS try_relocalise_never_raises_on_garbage")


# ---------------------------------------------------------------------------
# Standalone runner (same pattern as tests/test_live_fixes.py)
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_save_load_roundtrip_synthetic_map,
        test_save_load_roundtrip_driven_vio,
        test_default_map_path_is_under_repo,
        test_relocalise_same_view,
        test_relocalise_shifted_view,
        test_relocalise_different_scene_returns_false_and_preserves_state,
        test_relocalise_fresh_vio_from_saved_map,
        test_save_empty_map_writes_nothing,
        test_load_missing_file_raises,
        test_try_relocalise_never_raises_on_garbage,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} relocalisation tests PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
