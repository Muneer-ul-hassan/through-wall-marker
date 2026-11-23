"""Vanishing-point / Manhattan compass — a THIRD yaw measurement.

Why (research report ``research_notes/vio-yaw-error-better-
methods-20261123/report.md`` §4): indoor scenes are Manhattan
worlds — most lines are parallel to one of three orthogonal axes.
Two vanishing points determine the full camera attitude, and the
yaw it gives is ABSOLUTE per frame: no chaining, no drift, no
magnetometer. The doorway turn is a textbook Manhattan scene
(vertical door frames + horizontal floor/wall edges), and a long
door-frame edge survives rotation better than a point track (its
endpoints may leave the frame while the line stays measurable).
Kottas & Roumeliotis (RSS 2013) prove one structural line + gravity
makes all degrees of freedom observable; Kim et al. (CVPR 2018)
estimate drift-free 3-DoF rotation from a single line + plane.

What it is NOT: it does not replace the magnetometer yaw or the
vision PnP. It is a third, independent yaw measurement, fused with
the other two only when it measures better (see
``tests/vp_yaw_test.py``). It needs the prior yaw to resolve a 90
deg ambiguity (a horizontal vanishing point is the world X axis or
the world Z axis — both assignments are self-consistent and differ
by exactly 90 deg), so ``yaw_hint_deg`` (the magnetometer yaw) is
required for a single yaw answer.

Method (no new dependency — the venv's OpenCV 5.1.0 exposes LSD):
  1. LSD line segments (>= 30 px), image -> normalised camera
     coordinates with K^-1.
  2. Vertical vanishing point from near-vertical lines (RANSAC +
     linear SVD refine). It also gives the gravity direction in
     camera coordinates (vision-only); a pitch/roll prior may be
     passed instead.
  3. Horizontal vanishing points from the remaining lines
     (sequential RANSAC, two VPs, each checked to lie near the
     horizon, i.e. perpendicular to gravity, and ~90 deg apart).
  4. Rotation from the VP directions (columns of R are the world
     axes in camera coordinates), yaw from
     ``tello_state.yaw_pitch_roll_from_R``. Both X/Z assignments
     are tried; the one closer to the hint wins.

Two fixes found by measurement (not by reasoning — both were
wrong in the first build and showed up as exactly +-90 deg and
10-45 deg yaw errors in the smoke tests):
  * AXIS ANGLES ARE MOD 180, NOT MOD 360. A line's vanishing
    point is the same for both directions along the line, so the
    axis angle a is only defined mod 180 deg. The first build
    averaged the two assignment yaws with a plain circular mean;
    the wrong assignment's two yaws are 180 deg apart, whose
    circular mean is garbage (it landed at exactly +-90 deg off).
    The fix: normalise each axis angle to (-90, 90], average the
    two yaws of an assignment only after bringing them onto the
    same mod-180 branch, and expand each assignment mean to its
    mod-360 aliases before comparing with the hint.
  * THE VERTICAL VP IS VALIDATED AGAINST THE PITCH/ROLL PRIOR.
    During the doorway turn, world-Z lines run toward the camera
    and look near-vertical in the image, so the vertical-VP
    RANSAC latches onto the Z VP instead of the Y VP (measured:
    the found "vertical VP" was 5-10 deg from the true Z axis
    and 90-100 deg from true Y in s1 f19/f22 and s23 f8/f25,
    which sent the yaw 10-45 deg wrong). A vertical VP that
    disagrees with the prior up direction by more than 10 deg is
    rejected: gravity comes from the prior and no lines are
    removed as "vertical" (the horizon check filters true
    vertical lines out of the horizontal VPs instead). The
    attitude prior is trusted here — the round-3 fragility was
    yaw, not pitch/roll.
  * A horizontal VP needs >= 12 supporting lines (raised from
    5): in the s4 end frames a VP from 6-9 spurious lines gave a
    yaw 27-40 deg wrong; every good frame has >= 29.

Honest limits (measured in tests/vp_yaw_test.py, not assumed):
  * A frame with too few structural lines gives NO measurement
    (the blank-door scenario is the risk) — it falls back to the
    other two yaw sources, exactly the graceful degradation the
    "Robust Visual Compass" paper implements.
  * The 90 deg X/Z ambiguity is real: without a hint there is no
    single yaw answer, so this module never invents one.
  * Lines on the person billboard are vertical world lines and
    correctly support the vertical VP; they are not filtered out.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the venv always has it
    cv2 = None

from server.tello_state import (R_from_yaw_pitch_roll,
                                yaw_pitch_roll_from_R)

# Tuning (measured on the VIO scenario frames; see tests/vp_yaw_test.py).
MIN_LINE_LEN_PX = 30.0      # LSD segments shorter than this are noise
VERTICAL_ANGLE_DEG = 35.0   # segment direction within this of image-vertical
RANSAC_PAIRS = 200          # random line pairs sampled per VP
SUPPORT_TOL = 0.015         # |l.v| in normalised coords (~0.9 deg)
MIN_VERTICAL_SUPPORT = 6    # lines needed to accept the vertical VP
MIN_HORIZONTAL_SUPPORT = 12  # lines needed to accept a horizontal VP
                            # (raised from 5: in the s4 end frames a
                            # VP from 6-9 spurious lines gave a yaw
                            # 27-40 deg wrong; every good frame has
                            # >= 29 horizontal lines)
ORTHO_TOL_DEG = 15.0        # VP directions must be this close to perpendicular
HORIZON_TOL_DEG = 12.0      # horizontal VP must lie this close to the horizon
VERTICAL_VP_GATE_DEG = 10.0 # vertical VP must agree with the pitch/roll
                            # prior within this, else it is rejected
                            # (world-Z lines look near-vertical in the
                            # image during the turn and fool the RANSAC)


def _wrap_deg(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def _k_inv(K):
    """Inverse of the pinhole matrix from a geometry K dict."""
    fx, fy = float(K["fx"]), float(K["fy"])
    cx, cy = float(K["cx"]), float(K["cy"])
    return np.array([[1.0 / fx, 0.0, -cx / fx],
                     [0.0, 1.0 / fy, -cy / fy],
                     [0.0, 0.0, 1.0]])


def detect_lines(frame_bgr, min_len_px=MIN_LINE_LEN_PX):
    """LSD segments as an (N, 4) float array [x1, y1, x2, y2].

    Returns an empty (0, 4) array when nothing is found. Lines are
    in image pixels (origin top-left, y down).
    """
    if cv2 is None:
        return np.zeros((0, 4))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD)
    out = lsd.detect(gray)
    lines = out[0]
    if lines is None:
        return np.zeros((0, 4))
    segs = np.asarray(lines, dtype=float).reshape(-1, 4)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    return segs[length >= min_len_px]


def _lines_to_normals(segs, k_inv):
    """Segments -> (homogeneous lines, kept_mask) in normalised
    camera coords.

    Each line is (a, b, c) with sqrt(a^2 + b^2) = 1, so |l . v| is
    the sine of the angle between the line and a direction v.
    ``kept_mask`` maps the returned rows back to ``segs`` rows
    (degenerate lines are dropped, which is near-impossible for
    segments >= 30 px but must not corrupt the caller's indexing).
    """
    if len(segs) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)
    p1 = np.stack([segs[:, 0], segs[:, 1], np.ones(len(segs))], axis=1)
    p2 = np.stack([segs[:, 2], segs[:, 3], np.ones(len(segs))], axis=1)
    n1 = p1 @ k_inv.T
    n2 = p2 @ k_inv.T
    lines = np.cross(n1, n2)
    nrm = np.hypot(lines[:, 0], lines[:, 1])
    ok = nrm > 1e-9
    lines = lines[ok]
    lines = lines / nrm[ok][:, None]
    return lines, ok


def _ransac_vp(lines, rng, n_pairs=RANSAC_PAIRS, tol=SUPPORT_TOL):
    """Best vanishing point (unit direction) from a set of lines.

    RANSAC over random line pairs (intersection = candidate VP),
    scored by inlier count; the winner is refined by the linear
    SVD method on its inliers. Returns (vp, inlier_mask,
    median_residual_deg) or (None, None, None) when no pair scores.
    """
    n = len(lines)
    if n < 2:
        return None, None, None
    best_v, best_inl, best_score = None, None, -1
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        v = np.cross(lines[i], lines[j])
        nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            continue
        v = v / nv
        if v[2] < 0:  # a VP direction is sign-ambiguous; fix the sign
            v = -v
        if abs(v[2]) < 1e-6:  # VP at infinity in this frame — skip
            continue
        resid = np.abs(lines @ v)
        inl = resid < tol
        score = int(inl.sum())
        if score > best_score:
            best_v, best_inl, best_score = v, inl, score
    if best_v is None or best_score < 2:
        return None, None, None
    # Linear refine on the inliers: smallest eigenvector of sum l l^T.
    inl_lines = lines[best_inl]
    _, _, vt = np.linalg.svd(inl_lines.T @ inl_lines)
    v = vt[-1]
    v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    resid = np.abs(lines @ v)
    inl = resid < tol
    med = float(np.degrees(np.median(np.arcsin(
        np.clip(resid[inl], 0.0, 1.0))))) if inl.any() else float("nan")
    return v, inl, med


def _vp_direction_to_yaw(vp_dir, gravity, K):
    """Axis angle (deg) of one horizontal VP direction, mod 180.

    ``vp_dir`` is the VP direction in camera coords (unit, z > 0);
    ``gravity`` is the gravity direction in camera coords (unit).
    The VP direction is projected onto the horizontal plane (perp.
    to gravity); the camera forward axis is projected the same way.
    The signed angle between the two projections is the yaw of that
    world axis relative to the camera — returned normalised to
    (-90, 90], because a line's vanishing point is the same for
    both directions along the line, so the axis angle is only
    defined mod 180 deg. Whether the axis is world Z (yaw = a) or
    world X (yaw = a - 90, mod 180) is the caller's 90 deg
    ambiguity, resolved by the hint.
    """
    d = np.asarray(vp_dir, dtype=float)
    g = np.asarray(gravity, dtype=float)
    d_h = d - float(d @ g) * g
    f = np.array([0.0, 0.0, 1.0])
    f_h = f - float(f @ g) * g
    nd = np.linalg.norm(d_h)
    nf = np.linalg.norm(f_h)
    if nd < 1e-6 or nf < 1e-6:
        return None
    d_h, f_h = d_h / nd, f_h / nf
    # Signed angle from f_h to d_h about the gravity axis (g points
    # down in camera coords here — the sign convention is checked
    # against GT yaw in the tests, not assumed).
    cross = np.cross(f_h, d_h)
    angle = math.degrees(math.atan2(float(cross @ g), float(f_h @ d_h)))
    return ((angle + 90.0) % 180.0) - 90.0


def estimate_vp_yaw(frame_bgr, K, yaw_hint_deg=None,
                    pitch_roll_deg=None):
    """Absolute yaw (deg) from vanishing points, or None.

    ``K`` is a geometry K dict ({fx, fy, cx, cy, ...}).
    ``yaw_hint_deg`` is the magnetometer yaw — required, because a
    horizontal VP is the world X axis or the world Z axis and both
    assignments are self-consistent 90 deg apart; the hint picks.
    ``pitch_roll_deg`` is an optional (pitch, roll) in deg: it
    supplies the gravity direction AND validates the vertical VP
    (a vertical VP more than 10 deg from the prior up direction
    is rejected — world-Z lines look near-vertical in the image
    during the turn and fool the RANSAC). Without it, gravity
    comes from the vertical VP alone (vision-only).

    Returns a dict with ``yaw_deg`` (closest to the hint),
    ``yaw_alt_deg`` (the 90 deg-swapped candidate), line/VP counts,
    the median line residual, the gravity source (``"vp"`` or
    ``"prior"``), whether the vertical VP was rejected, and the
    per-frame runtime in ms — or None when no usable VP exists.
    The yaw is wrapped to (-180, 180].
    """
    t0 = time.perf_counter()
    if cv2 is None:
        return None
    segs = detect_lines(frame_bgr)
    if len(segs) < 5:
        return None
    k_inv = _k_inv(K)
    lines, _kept = _lines_to_normals(segs, k_inv)
    if len(lines) < 5:
        return None
    rng = np.random.default_rng(0)  # deterministic RANSAC

    # --- vertical VP: near-vertical segments only -------------------
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    ang_vert = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    vert_mask = ang_vert < VERTICAL_ANGLE_DEG
    vert_idx = np.where(vert_mask)[0]
    v_dir, v_inl, v_res = None, None, None
    n_vertical = 0
    if len(vert_idx) >= MIN_VERTICAL_SUPPORT:
        v_lines, v_kept = _lines_to_normals(segs[vert_idx], k_inv)
        v_dir, v_inl, v_res = _ransac_vp(v_lines, rng)
        if v_dir is not None and v_inl is not None:
            n_vertical = int(v_inl.sum())
            if n_vertical < MIN_VERTICAL_SUPPORT:
                v_dir, v_inl = None, None
                n_vertical = 0
            else:
                # Map the vertical inliers back to segment indices:
                # v_inl indexes the kept vertical lines, v_kept
                # maps kept lines back to positions in vert_idx.
                v_inl_seg = np.zeros(len(segs), dtype=bool)
                kept_pos = np.where(v_kept)[0]
                v_inl_seg[vert_idx[kept_pos[v_inl]]] = True
                v_inl = v_inl_seg

    # --- gravity: vertical VP VALIDATED against the prior ----------
    # The image-space "near-vertical" filter is fooled during the
    # turn: world-Z lines run toward the camera and look
    # near-vertical in the image, so the RANSAC latches onto the Z
    # VP instead of the Y VP (measured on s1 f19/f22 and s23 f8/f25:
    # the found "vertical VP" was 5-10 deg from the true Z axis and
    # 90-100 deg from true Y, which sent the yaw 10-45 deg wrong).
    # The vertical VP is therefore VALIDATED against the pitch/roll
    # prior (attitude is trusted — the round-3 fragility was yaw,
    # not pitch/roll). If it disagrees by more than the gate it is
    # rejected: gravity comes from the prior and NO lines are
    # removed as "vertical" (the horizon check below filters true
    # vertical lines out of the horizontal VPs instead).
    up_prior = None
    if pitch_roll_deg is not None:
        R_p = R_from_yaw_pitch_roll(0.0, float(pitch_roll_deg[0]),
                                    float(pitch_roll_deg[1]))
        up_prior = R_p @ np.array([0.0, 1.0, 0.0])
    v_ok = False
    v_rejected = False
    if v_dir is not None:
        if up_prior is None:
            v_ok = True
        else:
            cosang = float(np.clip(v_dir @ up_prior, -1.0, 1.0))
            v_ok = math.degrees(math.acos(cosang)) <= VERTICAL_VP_GATE_DEG
            v_rejected = not v_ok
    gravity = None
    gravity_source = None
    if v_ok:
        # Vertical lines point up in the world: the VP direction is
        # the world +Y axis in camera coords, so gravity is -v.
        gravity = -v_dir / np.linalg.norm(v_dir)
        gravity_source = "vp"
    elif up_prior is not None:
        gravity = -up_prior / np.linalg.norm(up_prior)
        gravity_source = "prior"
    if gravity is None:
        return None

    # --- horizontal VPs: sequential RANSAC with horizon check -------
    # Lines are removed as "vertical" only when the vertical VP was
    # accepted; on rejection the full set is used (a rejected VP's
    # inliers are usually Z lines, not vertical ones). Each RANSAC
    # round's VP must lie near the horizon (|d.g| small — this
    # rejects VPs built from true vertical lines) and ~90 deg from
    # any VP already accepted; rejected VPs have their inliers
    # removed anyway so the next round can find the real family.
    if v_ok and v_inl is not None and n_vertical > 0:
        keep = np.ones(len(segs), dtype=bool)
        keep[v_inl] = False
        h_segs = segs[keep]
    else:
        h_segs = segs
    h_lines, _ = _lines_to_normals(h_segs, k_inv)
    if len(h_lines) < MIN_HORIZONTAL_SUPPORT:
        return None
    h_dirs, h_supports, h_resids = [], [], []
    rest = h_lines
    for _ in range(3):  # at most 3 RANSAC attempts, at most 2 VPs
        if len(rest) < MIN_HORIZONTAL_SUPPORT or len(h_dirs) >= 2:
            break
        d, inl, res = _ransac_vp(rest, rng)
        if d is None or inl is None or int(inl.sum()) < MIN_HORIZONTAL_SUPPORT:
            break
        near_horizon = (abs(float(d @ gravity))
                        <= math.sin(math.radians(HORIZON_TOL_DEG)))
        orthogonal = all(abs(float(d @ d0))
                         <= math.sin(math.radians(ORTHO_TOL_DEG))
                         for d0 in h_dirs)
        if near_horizon and orthogonal:
            h_dirs.append(d)
            h_supports.append(int(inl.sum()))
            h_resids.append(res)
        rest = rest[~inl]
    if not h_dirs:
        return None

    # --- yaw from the horizontal VP(s) --------------------------------
    # Each VP gives an AXIS angle a in (-90, 90] (mod 180 deg — a
    # line's vanishing point is the same for both directions along
    # the line). The axis is world Z -> yaw = a (mod 180); the axis
    # is world X -> yaw = a - 90 (mod 180). With two VPs the two
    # assignments differ by exactly 90 deg (mod 180), and each
    # assignment's two yaws AGREE mod 180 (the wrong assignment
    # would give yaws 180 deg apart — that is why a plain circular
    # mean of the raw angles fails: the mean of two yaws 180 deg
    # apart is garbage, which is exactly the +-90 deg error the
    # first smoke test showed). The hint (magnetometer yaw, good
    # to a few deg) picks among the mod-360 aliases; the other
    # assignment is reported as the 90 deg-swapped alt.
    if yaw_hint_deg is None:
        return None  # the 90 deg ambiguity cannot be resolved alone
    cand = []  # per VP: axis angle a in (-90, 90]
    for d in h_dirs:
        a = _vp_direction_to_yaw(d, gravity, K)
        if a is None:
            continue
        cand.append(a)
    if not cand:
        return None

    def _mean_agreeing(y1, y2):
        # Mean of two yaws that agree mod 180: bring y2 onto y1's
        # branch first, else the mean of (89, -89) is 0, not 89.
        y2b = y2 + 180.0 * round((y1 - y2) / 180.0)
        return (y1 + y2b) / 2.0

    if len(cand) == 1:
        a = cand[0]
        pairs = [(a, a - 90.0), (a - 90.0, a)]  # (yaw_if_Z, yaw_if_X)
    else:
        a1, a2 = cand[0], cand[1]
        mA = _mean_agreeing(a1 - 90.0, a2)  # assignment A: VP1=X, VP2=Z
        mB = _mean_agreeing(a1, a2 - 90.0)  # assignment B: VP1=Z, VP2=X
        pairs = [(mA, mB), (mB, mA)]
    best = None  # (err, yaw_deg, yaw_alt_deg)
    for mean_yaw, alt_yaw in pairs:
        for k in (0.0, 180.0):  # mod-180 mean -> mod-360 aliases
            y = mean_yaw + k
            err = abs(_wrap_deg(y - yaw_hint_deg))
            if best is None or err < best[0]:
                best = (err, y, alt_yaw + k)
    _, yaw_deg, yaw_alt_deg = best
    ms = (time.perf_counter() - t0) * 1000.0
    resids = [r for r in h_resids if r is not None and np.isfinite(r)]
    if v_res is not None and np.isfinite(v_res):
        resids.append(v_res)
    return {
        "yaw_deg": float(_wrap_deg(yaw_deg)),
        "yaw_alt_deg": float(_wrap_deg(yaw_alt_deg)),
        "hint_err_deg": float(abs(_wrap_deg(yaw_deg - yaw_hint_deg))),
        "n_lines": int(len(segs)),
        "n_vertical": int(n_vertical),
        "n_horizontal": int(sum(h_supports)),
        "horizontal_vps": int(len(h_dirs)),
        "vp_residual_deg": (float(np.median(resids)) if resids else None),
        "gravity_source": gravity_source,
        "vertical_vp_rejected": bool(v_rejected),
        "ms": float(ms),
    }
