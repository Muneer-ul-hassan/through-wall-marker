"""Rotation-only estimation for the VIO: NEC two-view rotation +
rotation averaging (research report §1-§2, 2026-11-23).

STATUS — MEASURED AND REJECTED, 2026-11-23 (VIO fix round 3). Do not
wire this in without re-measuring: on this data the NEC has a
systematic per-pair bias (signed mean −0.37 deg s1, −0.26 deg s23;
−0.75 deg/frame during the turn) that chaining turns into 27.8 deg
mean absolute yaw error, and long-range pairs are worse (lag 5:
1.16/1.67 deg, lag 10: 1.55/3.51 deg), so averaging cannot help.
The λ_min formulation shows the same bias, so it is in the NEC
objective / the near-planar two-plane scene, not the solver. The
winner was the 1-DoF yaw PnP + the bbox-height calibration instead.
This module is kept as reference (and its two real bugs — the
Shonan gauge is RIGHT multiplication, and GTSAM 4.3.0's
``run(min_p, max_p)`` signature — are documented in the code) so
the NEC is not re-tried blindly. Full numbers: TEST-REPORT.md
"VIO fix round 3 — constrained PnP + bbox calibration".

Why it was built: the 1-DoF yaw PnP and the pose-graph yaw prior
both estimate yaw from the same coupled system that is failing — a
3D map triangulated on a ~2.7 cm baseline, chained frame-to-frame
(hence the 4.0-4.5 deg max yaw error). These two methods break both
couplings:

1. NEC (Normal Epipolar Constraint, Kneip & Lynen, ICCV 2013): the
   relative rotation between two views is recovered from BEARING
   vectors alone — translation drops out of the problem entirely, so
   there is no map, no triangulation and no baseline in the rotation
   problem. Pure rotation is its sweet spot (where the essential
   matrix fails). Here it is solved as a joint non-linear least
   squares over (relative rotation R, translation direction t):
   residuals r_i = (f_i x R f'_i) . t, which vanish at the true R
   (all epipolar-plane normals are perpendicular to t). Huber loss
   makes it robust to LK mismatches. This is equivalent to Kneip's
   lambda_min(M(R)) objective at the optimum, but smoother to
   optimise (the eigenvalue objective is non-smooth at crossings).

2. Rotation averaging (Shonan, Dellaert et al., ECCV 2020): the
   pairwise relative rotations are collected into a view graph and
   the absolute rotations of all frames are solved jointly, so
   per-pair errors are distributed over the window instead of
   accumulating frame-to-frame (chaining is what turns 1 deg
   per-frame errors into a 4.0-4.5 deg max). GTSAM's ShonanAveraging3
   (pip install gtsam, 4.3.0, cp312 wheel) does this; a SciPy
   Levenberg-Marquardt averaging is the fallback if GTSAM is absent.

The averaged absolute rotations then feed the translation-only PnP
(``server.vio._pnp_transonly``) as the known rotation — the
known-rotation problem (Chng et al., arXiv:2010.01872): with R fixed,
only the 3-DoF position drifts, exactly as VINS-Mono's 4-DoF pose
graph but with yaw also fixed by the averaged rotation instead of a
magnetometer.

Conventions: all rotations here are GEOMETRY-frame (Y-up) camera
rotations R (world -> camera), matching ``server.vio``'s external
frame. A relative rotation R_ij returned by ``nec_rotation`` maps
frame-j bearings into the frame-i camera frame, i.e.
R_ij = R_i @ R_j.T. The averaging graph uses the same convention.
"""

from __future__ import annotations

import math

import numpy as np


def bearings(uv, Kmat):
    """Pixel coordinates (N, 2) -> unit bearing vectors (N, 3)."""
    uv = np.asarray(uv, dtype=float)
    fx, fy, cx, cy = Kmat[0, 0], Kmat[1, 1], Kmat[0, 2], Kmat[1, 2]
    b = np.stack([(uv[:, 0] - cx) / fx,
                  (uv[:, 1] - cy) / fy,
                  np.ones(len(uv))], axis=1)
    n = np.linalg.norm(b, axis=1, keepdims=True)
    return b / np.maximum(n, 1e-12)


def _rot_from_axis_angle(theta):
    """Axis-angle vector (3,) -> rotation matrix (3, 3)."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(theta, dtype=float).reshape(3, 1))
    return R


def _epipolar_normals(f1, f2, R):
    """n_i = f_i x (R f'_i), (N, 3)."""
    Rf2 = (R @ f2.T).T
    return np.cross(f1, Rf2)


def _lambda_min_and_vector(f1, f2, R):
    """Smallest eigenvalue of M(R) = sum n_i n_i^T and its eigenvector.

    M's smallest eigenvector is the translation direction t at the
    optimum (all epipolar-plane normals are perpendicular to t).
    """
    n = _epipolar_normals(f1, f2, R)
    M = n.T @ n
    w, v = np.linalg.eigh(M)
    return float(w[0]), v[:, 0]


def nec_rotation(f1, f2, R_init=None, t_init=None, huber_delta=0.02,
                 max_nfev=60):
    """Relative rotation between two views from bearings only (NEC).

    ``f1``, ``f2`` are (N, 3) unit bearing vectors in the two views'
    camera frames (same correspondences, same order). Returns
    ``(R, info)`` where R (3, 3) maps frame-2 bearings into the
    frame-1 camera frame (R = R_1 @ R_2.T for absolute rotations
    R_1, R_2), and ``info`` carries ``lambda_min`` (the NEC
    objective at the solution — 0 at the true rotation, up to scale),
    ``n_inliers`` (epipolar residual < 3 * huber_delta), ``rms``
    (RMS epipolar residual of the inliers) and ``n``. Returns
    ``(None, info)`` when there are too few correspondences (< 8)
    or the solve fails. ``R_init`` (default identity) is the
    starting rotation — a warm start matters: with the tiny
    inter-frame translation of a hovering drone the NEC objective
    is flat, so starting from the previous relative rotation (or
    the attitude prior's) keeps the solver in the true basin;
    a cold start from identity can settle in a wrong minimum
    (measured 2026-11-23: 1.2 deg mean / 7.4 deg max from identity
    on scenario frames vs 0.005 deg on synthetic data with a good
    start). The solve is two-stage: a plain least-squares pass
    (finds the basin) then a Huber pass (rejects LK mismatches).
    """
    from scipy.optimize import least_squares

    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    n = len(f1)
    info = {"n": n, "lambda_min": None, "n_inliers": 0, "rms": None}
    if n < 8 or len(f2) != n:
        return None, info
    R0 = np.eye(3) if R_init is None else np.asarray(R_init, dtype=float)
    # Translation-direction init: the lambda_min eigenvector of
    # M(R_init) — the best t for the starting rotation.
    _lam0, t0 = _lambda_min_and_vector(f1, f2, R0)
    if t_init is not None:
        t0 = np.asarray(t_init, dtype=float)
        t0 = t0 / max(np.linalg.norm(t0), 1e-12)

    def residuals(x):
        R = _rot_from_axis_angle(x[:3]) @ R0
        t = x[3:6]
        tn = np.linalg.norm(t)
        if tn < 1e-12:
            return np.ones(n)
        t = t / tn
        nrm = _epipolar_normals(f1, f2, R)
        return nrm @ t

    x0 = np.concatenate([np.zeros(3), t0])
    try:
        # Stage 1: plain least squares from the warm start (finds
        # the basin — the Huber loss would flatten the gradient
        # while the residuals are still large).
        sol1 = least_squares(residuals, x0, loss="linear",
                             max_nfev=max_nfev,
                             xtol=1e-10, ftol=1e-10)
        # Stage 2: Huber from the stage-1 solution (outlier
        # rejection once the residuals are at inlier scale).
        sol = least_squares(residuals, sol1.x, loss="huber",
                            f_scale=huber_delta, max_nfev=max_nfev,
                            xtol=1e-10, ftol=1e-10)
    except Exception:
        return None, info
    R = _rot_from_axis_angle(sol.x[:3]) @ R0
    lam, _t = _lambda_min_and_vector(f1, f2, R)
    nrm = _epipolar_normals(f1, f2, R)
    r = np.abs(nrm @ _t)
    # Epipolar residual scale: |n_i| is ~sin(parallax); normalise
    # by |n_i| so the threshold is an angle-like quantity.
    scale = np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)
    rn = r / scale
    inl = rn < 3.0 * huber_delta
    info["lambda_min"] = lam
    info["n_inliers"] = int(inl.sum())
    info["rms"] = (float(np.sqrt(np.mean(rn[inl] ** 2)))
                   if inl.any() else None)
    info["t"] = _t
    return R, info


def rotation_average_gtsam(pairs, n_frames, prior_sigma_deg=0.1,
                           pair_sigma_deg=1.0):
    """Average relative rotations into absolute rotations (Shonan).

    ``pairs`` is a list of ``(i, j, R_ij)`` with R_ij = R_i @ R_j.T
    (the NEC convention: R_ij maps frame-j bearings into the
    frame-i camera frame). ``n_frames`` is the number of frames
    (indices 0..n_frames-1). The returned rotations are RELATIVE to
    frame 0: R_k^rel = R_k @ R_0.T (the gauge freedom of rotation
    averaging is RIGHT multiplication by a common rotation — the
    measurements R_i @ R_j.T are invariant under R_k -> R_k @ Q —
    so the gauge is removed by right-multiplying by the inverse of
    the recovered frame-0 rotation). The caller composes with the
    known start pose as R_k = R_k^rel @ R_0. Returns a list of
    (n_frames) rotations or None if GTSAM is unavailable or the
    solve fails. The Shonan solver is globally optimal for rotation
    averaging (with a certificate), so a single bad pair is outvoted
    by the graph instead of chaining.
    """
    try:
        import gtsam
        from gtsam import BinaryMeasurementRot3, Rot3, ShonanAveraging3
        from gtsam import noiseModel
    except Exception:
        return None
    if n_frames < 2 or not pairs:
        return None
    pair_noise = noiseModel.Isotropic.Sigma(
        3, math.radians(pair_sigma_deg))
    measurements = []
    for (i, j, R_ij) in pairs:
        if i == j:
            continue
        # BinaryMeasurementRot3(i, j, measured) with measured the
        # relative rotation satisfying R_j = R_i @ measured, i.e.
        # measured = R_i.T @ R_j = (R_i @ R_j.T).T = R_ij.T.
        # (Verified against known rotations in tests/nec_test.py.)
        measured = Rot3(np.asarray(R_ij, dtype=float).T)
        measurements.append(BinaryMeasurementRot3(
            int(i), int(j), measured, pair_noise))
    if not measurements:
        return None
    try:
        shonan = ShonanAveraging3(measurements)
        # run(min_p, max_p): Shonan optimises over Stiefel manifolds
        # St(p, 3) for p = 3..10, certifying global optimality at
        # each level; returns (Values, cost).
        values, _cost = shonan.run(3, 10)
    except Exception:
        return None
    out = []
    for k in range(n_frames):
        try:
            out.append(np.asarray(values.atRot3(k).matrix()))
        except Exception:
            return None
    # Remove the gauge freedom (RIGHT multiplication — see the
    # docstring): return rotations relative to frame 0.
    R0inv = out[0].T
    return [R @ R0inv for R in out]


def rotation_average_scipy(pairs, n_frames, prior_sigma_deg=0.1,
                           pair_sigma_deg=1.0, max_nfev=1000):
    """SciPy fallback for rotation averaging (no GTSAM needed).

    Levenberg-Marquardt over the absolute rotations' axis-angle
    vectors (frame 0 fixed at identity). Residuals are the
    Log-map of R_ij^-1 @ (R_i @ R_j.T) per pair (i.e. the
    disagreement between the measured relative rotation and the
    one implied by the absolute rotations), plus a strong prior
    residual on frame 0. Same convention as
    ``rotation_average_gtsam``. Initialised from the CHAINED
    composition of the pairs (a zero init does not converge on a
    60-frame chain — measured 2026-11-23: 15 deg mean error from
    zero vs the chained solution). Returns a list of absolute
    rotations or None on failure.
    """
    from scipy.optimize import least_squares

    if n_frames < 2 or not pairs:
        return None

    def log_R(R):
        import cv2
        v, _ = cv2.Rodrigues(R)
        return v.reshape(3)

    # Chained init: compose the consecutive pairs from frame 0.
    pair_by_j = {}
    for (i, j, R_ij) in pairs:
        if j == i + 1:
            pair_by_j[j] = R_ij
    chained = [np.eye(3)]
    for k in range(1, n_frames):
        if k in pair_by_j:
            chained.append(pair_by_j[k].T @ chained[-1])
        else:
            chained.append(chained[-1])

    def pack(thetas):
        Rs = [np.eye(3)]
        for th in thetas:
            Rs.append(_rot_from_axis_angle(th))
        return Rs

    def residuals(x):
        thetas = x.reshape(n_frames - 1, 3)
        Rs = pack(thetas)
        res = []
        sig = math.radians(pair_sigma_deg)
        for (i, j, R_ij) in pairs:
            if i == j:
                continue
            pred = Rs[i] @ Rs[j].T
            dR = np.asarray(R_ij, dtype=float).T @ pred
            res.append(log_R(dR) / sig)
        return np.concatenate(res) if res else np.zeros(1)

    x0 = np.concatenate([log_R(R) for R in chained[1:]])
    try:
        sol = least_squares(residuals, x0, loss="huber", f_scale=3.0,
                            max_nfev=max_nfev)
    except Exception:
        return None
    return pack(sol.x.reshape(n_frames - 1, 3))


def rotation_average(pairs, n_frames, **kwargs):
    """Rotation averaging: GTSAM Shonan first, SciPy LM as fallback."""
    out = rotation_average_gtsam(pairs, n_frames, **kwargs)
    if out is not None:
        return out, "gtsam"
    out = rotation_average_scipy(pairs, n_frames, **kwargs)
    if out is not None:
        return out, "scipy"
    return None, None
