"""Laptop-side MONOCULAR visual odometry (VIO) from the drone video alone.

The Tello reports no pose, so on the real run the drone pose fed to the
bridge must come from video. This module is that pose source: feature
tracking (GoodFeaturesToTrack + Lucas-Kanade optical flow, cv2),
essential-matrix relative motion (``cv2.findEssentialMat`` +
``recoverPose``) for initialisation, then per-frame pose from a
hand-rolled Levenberg-Marquardt PnP (``_lm_pnp`` + ``_pnp_ransac``)
against a growing 3D map, with new map points triangulated from known
absolute poses. The hand-rolled solver exists because
``cv2.solvePnP`` / ``cv2.solvePnPRansac`` are unreliable on this
near-planar map (see _lm_pnp — mirror solution, 5-inlier RANSAC).

Scale — the one thing monocular VO cannot observe (SPEC.md research
report). The scale method used here, exactly:
  1. The session geometry is KNOWN: the drone takes off beside the
     phone from a known start pose (passed in as ``start_pose`` — in
     the virtual harness that is the first scenario frame's pose; in
     the real run it is the measured takeoff pose beside the phone)
     and flies LEVEL at a known, roughly constant height above a flat
     floor (``camera_height_m``, default 1.3 m — the takeoff hover
     height, measured/confirmed at session start).
  2. INITIALISATION ONLY: the first two-view pair with enough
     parallax is triangulated with a UNIT baseline. Floor points in
     that reconstruction (keyframe pixels in the lower 40% of the
     image — the floor is what a level, slightly-down-looking camera
     sees there) have a known true world height (y = 0), so the true
     baseline is ``s = C_y / (C_y - median_y_unit)`` where ``y_unit``
     is the reconstructed world height of the floor points in unit-
     baseline coordinates and ``C_y`` is the known camera height.
     That one scale factor turns the unit-baseline reconstruction
     into metres and seeds the map in absolute scale.
  3. AFTER initialisation the scale is CARRIED BY THE MAP: every new
     frame is localised by PnP against map points that are already in
     metres, and every new map point is triangulated from two poses
     that are already absolute. No per-step scale is ever re-estimated
     from motion, so scale cannot silently drift step by step — but
     PnP noise still accumulates as a slow random walk, which is why
     the tests MEASURE drift against ground truth instead of assuming
     it away (see tests/vio_ground_test.py and TEST-REPORT.md).

Assumptions (stated, not hidden):
  * level flight at roughly constant height over a flat floor;
  * the floor is visible in the lower part of the image at
    initialisation and carries trackable texture (the synthetic
    checker floor does; a polished featureless floor would not);
  * the scene is mostly static (moving people break triangulation
    of NEW points only — PnP against the existing map is unaffected);
  * no loop closure: this is odometry, so drift grows with distance
    flown. That is the honest limit. A windowed local BA (point
    positions only, poses held fixed) refines the map on keyframes,
    and a keyframe-descriptor database supports relocalisation
    after a loss — but there is no global loop closure.

Rotation-dominant motion (report §1, 2026-11-23): pure rotation
has no parallax, so no new map point can be triangulated and the
map drains. This module therefore keeps untriangulated tracks as
2D BEARINGS (inverse depth d=0, GRANITE-style) instead of deleting
them for failing a baseline gate, triangulates only when the
parallax angle at the point reaches PARALLAX_MIN_DEG (5 deg)
across keyframes (deferred triangulation), and estimates ROTATION
ONLY from the bearings (homography) when the 3D map is too thin
for PnP, freezing translation (or advancing it by the calibrated
velocity prior). An optional Tello state prior
(server.tello_state) supplies yaw/pitch/roll + h/tof as priors
(velocity only when calibrated) and bounds a dead-reckoned bridge
across brief visual gaps. Every returned pose is flagged
(``diag["pose_source"]``: vision / attitude-aided /
dead-reckoned / invalid) so the bridge never silently presents an
attitude-only pose as visual tracking.

Failure behaviour (hard rule, same as the bridge):
  * ``tracking_valid`` is True ONLY when the current pose comes from
    a PnP (or initialisation / essential-matrix fallback) with at
    least ``MIN_PNP_INLIERS`` inliers AND an inlier ratio of at
    least ``MIN_PNP_RATIO`` of the map points used. The numbers are
    justified below (see MIN_PNP_INLIERS): in normal flight the
    harness measures 80-250 PnP inliers; a lost track collapses to
    single digits. A pose that fails the gate is NEVER returned as
    valid — the caller (``server.main.Pipeline``) feeds
    ``tracking_valid=False`` to the bridge, which invalidates it and
    hides the marker. No guessed pose, ever.
  * A jump guard rejects any one-frame pose jump over ``MAX_JUMP_M``
    (1.0 m in 0.1 s is faster than this drone flies) or a camera
    height outside [0.5, 2.5] m: both are PnP failure signatures,
    not real motion.
  * RECOVERY: after a loss the map and the last good frame are kept.
    Each later frame tries to relocalise by LK-tracking the last
    good frame's map points into the current frame and running PnP
    on them. If that succeeds, tracking resumes and the bridge may
    reacquire — a NEW chain (``Bridge.reacquire`` restarts the
    uncertainty clock). This is the same caveat as the existing s21
    scenario: the harness proves the state-machine logic, NOT a real
    relocalisation. In the REAL system reacquire must be gated on a
    genuine relocalisation (re-observing a mapped area / a doorway
    loop closure), never on a bare "tracking is valid again" flag —
    and a re-initialisation from scratch is FORBIDDEN here: it would
    silently teleport the drone into a new arbitrary world frame.
    If relocalisation fails, VIO stays invalid for the rest of the
    session (the operator restarts the session / re-flies the bridge).

Coordinate conventions: exactly ``server.geometry`` — a pose is
``(R, C)`` with R the world-to-camera rotation and C the camera centre
in world coordinates; intrinsics are ``DRONE_K`` by default.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np

from .geometry import DRONE_K
from .tello_state import (DEAD_RECKON_MAX_M, DEAD_RECKON_MAX_S,
                          POSE_ATTITUDE_AIDED, POSE_DEAD_RECKONED,
                          POSE_INVALID, POSE_VISION, R_from_yaw_pitch_roll,
                          StatePrior, TelloCalibration, TelloStatePacket,
                          yaw_pitch_roll_from_R)

log = logging.getLogger(__name__)

# --- Tunables (chosen, then measured — see the justification notes) -----
MAX_FEATURES = 800
# GoodFeaturesToTrack: quality 0.005 keeps weak-but-real corners on the
# checker floor / poster edges; minDistance 6 px spreads them out.
GFTT_QUALITY = 0.005
GFTT_MIN_DISTANCE = 6
# LK pyramid: 21 px window, 4 levels — handles the ~50-150 px/frame
# flow of the doorway approach at 10 fps, and the faster hover flow.
# Level 4 (not 3) is measured, 2026-11-23: at the doorway the median
# flow is ~90 px/frame (2 m/s approach + 6 deg/frame yaw slew, fx 800)
# and the checker floor squares are ~100-150 px apart, so a 3-level
# pyramid (8x) leaves the flow at ~11 px on the top level and LK
# locks onto neighbouring identical corners — forward-backward-good
# tracks at f15->f16 were 24/44 at level 3 vs 43/51 at level 4
# (offline LK test, tests/scenarios s1 smoothed). Level 5 loses
# points (36 kept vs 55 at level 4, f15->f16): the 32x image washes
# the corners out. A 31 px window was also tried and is worse
# (38/48 good at f15->f16).
LK_WIN = (21, 21)
LK_MAX_LEVEL = 4
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
# Forward-backward LK check: a track is kept only if tracking it back
# to the previous frame returns within this distance of its start
# pixel. Measured 2026-11-23 (offline LK test, s1 smoothed): good
# tracks have a forward-backward error of 0.00-0.02 px (median), bad
# ones (LK locked onto a neighbouring identical checker corner) 0.8 px
# to 165 px — the 1.0 px cut sits in the empty gap. This is the
# standard KLT outlier rejection, and it matters here because the
# checker floor is periodic: a wrong-corner track looks exactly like
# a right one to a one-way LK (same corner appearance), so one-way
# status=1 cannot be trusted. Costs a second LK call per frame.
LK_FB_MAX_ERR_PX = 1.0
# Motion-model LK prediction (2026-11-23 fix round): the frame pair
# f19->f20 on the smoothed s1 flight kills one-way LK outright
# (2-4 forward-backward-good tracks out of ~90 at EVERY pyramid
# setting — the near floor flows up to ~500 px/frame as the drone
# passes the person at 0.6 m, and the yaw rate steps 0 -> 6 deg/frame
# as the turn starts). The tracker therefore predicts each point's
# current pixel from a constant-velocity motion model (last two
# accepted poses) — map points by projecting their 3D at the
# predicted pose, fresh points by the predicted relative rotation —
# and passes that as the LK initial guess (OPTFLOW_USE_INITIAL_FLOW).
# The prediction only INITIALISES the search: the final track is
# still the LK measurement, verified by the forward-backward check
# and the PnP RANSAC, so a wrong prediction can lose tracking but can
# never fabricate a valid pose. If the predicted track yields too few
# points, _track retries with plain (zero-motion-init) LK.
LK_PREDICT_MIN_TRACKED = 12  # == MIN_PNP_THIN_INLIERS (defined below)
# Initialisation: wait for this median pixel displacement between the
# keyframe and the current frame before attempting the essential
# matrix. Forward motion at ~0.17 m/frame gives floor points ~45 px of
# flow per frame, so this fires on frame 2-3 — with a baseline of
# ~0.15-0.3 m, which is what the essential matrix needs to be well
# conditioned. Waiting longer hurts: LK mismatches on the repetitive
# checker floor grow with displacement (measured: essential inliers
# fall from 62/88 at frame 3 to 31/88 at frame 5).
INIT_MIN_DISPLACEMENT_PX = 15.0
INIT_MIN_TRACKED = 40
# Essential-matrix inliers needed to accept an initialisation. A
# two-view RANSAC on 40+ tracked points that returns < 30 inliers is
# a degenerate / textureless pair, not a pose.
INIT_MIN_INLIERS = 30
# PnP validity gate. JUSTIFICATION: in the virtual harness a healthy
# frame localises with 80-250 inliers out of 150-350 map points
# (measured in tests/vio_ground_test.py runs — see TEST-REPORT.md);
# when tracking is genuinely lost (blank wall, violent motion) the
# inlier count collapses to < 10 and the ratio to < 0.1. The gate
# (20 inliers, 25% ratio) sits in the empty gap between those two
# regimes: it cannot be crossed by a handful of points that happen to
# agree on a wrong pose (e.g. all on one wall plane), and it is far
# below any healthy frame, so it does not flappingly drop good poses.
# If the measured distributions ever overlap this gap, the threshold
# must be re-justified from new measurements, not silently moved.
MIN_PNP_INLIERS = 20
MIN_PNP_RATIO = 0.25
# Thin-map tier of the PnP gate. JUSTIFICATION (measured 2026-11-23):
# the map thins to ~20 points mid-turn (feature-poor back-wall view),
# and a 19/19 unanimous consensus was REFUSED by the 20-inlier gate
# — _track skipped PnP entirely and tracking was lost at frame 31
# of the smooth flight, even though the PnP on those 19 points
# converges to a 6.1 cm pose with all 19 within 3 px. A thin map can
# still give a trustworthy pose if nearly all of it agrees, so a
# pose is also accepted with >= 12 inliers at >= 50% ratio. 12 is 2x
# the PnP minimum (6 points); the 50% ratio + the cheirality, jump
# and height guards are what keep a wrong pose out, and genuinely
# lost frames still measure < 10 inliers at < 0.1 ratio.
MIN_PNP_THIN_INLIERS = 12
MIN_PNP_THIN_RATIO = 0.5


def _accept_pnp(n_inliers, n_map):
    """Two-tier PnP acceptance gate (see MIN_PNP_INLIERS)."""
    if (n_inliers >= MIN_PNP_INLIERS
            and n_inliers >= MIN_PNP_RATIO * max(1, n_map)):
        return True
    return (n_inliers >= MIN_PNP_THIN_INLIERS
            and n_inliers >= MIN_PNP_THIN_RATIO * max(1, n_map))
PNP_REPROJECTION_ERROR_PX = 3.0
# Triangulation gate: a new map point is committed only once the
# PARALLAX ANGLE subtended at the point by its origin pose and the
# current pose reaches this value (report §1.2(a), GRANITE §IV-C:
# "only triangulate if parallax > ~5 deg"). This REPLACES the old
# 0.15 m absolute baseline gate (NEW_POINT_BASELINE_M, removed
# 2026-11-23): an absolute distance gate is depth-blind — a far
# point needs far more baseline than a near one to reach the same
# parallax, so a fixed 0.15 m admits noisy far points and starves
# near ones. The parallax angle is the quantity that actually
# conditions the triangulation (depth error grows as 1/sin(angle)).
# A point that has not reached the gate is NOT deleted — it is kept
# as a 2D bearing (inverse depth d=0, GRANITE-style) and constrains
# rotation in the rotation-only mode, and is re-tried for
# triangulation on every later frame (deferred triangulation,
# report §1.2(b) / DT-SLAM). Measured on the old gate (smooth
# flight, 2026-11-23): 0.25 m tracked 34/105 frames (drift 4.1 cm
# mean) and the map drained mid-turn; 0.15 m tracked 55/105 (drift
# 9.2 cm mean, 33.6 cm max); 0.10 m tracked the same 55 but drift
# grew to 19.1 cm mean (57.0 cm max) from triangulation noise.
# A 3 deg gate was also tried (2026-11-23): it admits noisier
# points (~7 cm depth error at 3 m) that corrupt the map and the
# PnP — s1 max drift 47 cm (vs 32 cm at 5 deg), s4 max 57 cm
# (vs 35 cm), s20 loses tracking at f52 (48/60 vs 56/60).
PARALLAX_MIN_DEG = 5.0
# Reprojection error a freshly triangulated point must meet in the
# current frame to be committed to the map (px). A point triangulated
# at barely-5-deg parallax can still be noisy; this rejects the bad
# ones at birth instead of letting the prune catch them a frame later.
TRI_REPROJ_MAX_PX = 3.0
# Replenish fresh (no-3D-yet) features when the tracked count falls
# below this.
REPLENISH_BELOW = 120
RELOCALISE_EVERY = 5
# Tracking-keyframe rule (report §1.2(f), GRANITE-style — see
# reset()): a new keyframe on this much rotation since the last
# keyframe, or this many frames, or when the 3D map thins below
# this count. Keyframes anchor the windowed local BA and the
# keyframe-descriptor relocalisation database.
KEYFRAME_ROT_DEG = 10.0
KEYFRAME_MAX_GAP = 10
KEYFRAME_MIN_MAP = 20
# Minimum spacing between keyframes (frames). During the turn the
# rotation rule alone fires every ~3 frames (3.4 deg/frame), so
# the BA would run on nearly every frame (measured 15-218 ms per
# BA call — the 218 ms frame at s1 f45) and dominate the step
# time. 5 frames is ~17 deg of turn, still dense enough for the
# windowed BA and the relocalisation database.
KEYFRAME_MIN_GAP = 5
# Windowed local BA (report §4 fix 4): keyframes + points per solve.
BA_WINDOW_KEYFRAMES = 5
# BA size cap. 150 points / 463 obs cost up to 218 ms (s1 f45);
# 80 points keeps the call ~30-60 ms while still refining the
# points that matter (the most recently observed). The cap keeps
# the worst-case VIO step near the 38-83 ms/frame measured before
# the BA was added.
BA_MAX_POINTS = 80
# ORB recovery (doorway / turn snaps): descriptor matching does
# not need small inter-frame motion, so it bridges the yaw snaps
# that kill LK (measured 2026-11-23: -31 deg at the doorway,
# +101 deg at the turn to face the person; LK carried only 36
# of 86 points across the doorway, 19 of them hallucinated
# onto points that had left the frame). ORB matched f13->f14
# with 39% of ratio-good matches geometrically correct —
# enough for the PnP RANSAC gate. It does NOT bridge f19->f20
# (0% correct: the two views barely overlap), which is a
# harness-trajectory limit, not a matcher limit — see README.
ORB_NFEATURES = 1000
ORB_RATIO = 0.75
ORB_ASSOC_PX = 3.0  # ORB keypoint <-> tracked point association
# Relocalisation is attempted only every Nth lost frame: a
# failed attempt runs the 60-round RANSAC (~100 ms), so
# trying every frame while lost for 45 frames dominated the
# mean VIO time (84 ms/frame) even though steady-state
# tracking is ~18 ms/frame (profiled 2026-11-23). The drone
# keeps moving while lost, so re-trying 5x/second at 10 fps
# loses nothing — the anchor is the last good frame either way.  # re-detect fresh features only when the
# tracked set is thin; 350 re-ran goodFeaturesToTrack (~9 ms)
# on nearly every frame for no benefit — the map holds ~70
# points and PnP needs only 20 inliers (measured 2026-11-23)
# One-frame jump guard: the Tello cannot move this far in 0.1 s
# (even the fast s20 hover peaks at ~0.5 m/frame).
MAX_JUMP_M = 1.0
CAM_HEIGHT_RANGE_M = (0.5, 2.5)
# Floor candidates at initialisation: keyframe pixels in the lower
# 40% of the image (v > 0.6 * height) — the floor is what a level,
# slightly-down-looking camera sees there.
FLOOR_PIXEL_V_FRAC = 0.6
# Sanity range for the initialisation baseline (metres). The drone
# covers ~0.17 m/frame on approach, so a real init baseline is
# 0.15-0.6 m; anything outside is a degenerate essential matrix.
INIT_BASELINE_RANGE_M = (0.05, 2.0)


def _K_matrix(K):
    return np.array([[K["fx"], 0.0, K["cx"]],
                     [0.0, K["fy"], K["cy"]],
                     [0.0, 0.0, 1.0]], dtype=float)


# ---------------------------------------------------------------------------
# Y-up camera (geometry.py) vs OpenCV's Y-down convention.
#
# ``server.geometry`` uses a Y-UP camera frame with image rows
# top-origin: ``v = cy - fy*y/z``. OpenCV's essential-matrix, PnP and
# triangulation routines assume the standard pinhole ``v = cy + fy*Y/z``
# with Y DOWN. Those two are the SAME image: writing ``Y = -y`` gives
# ``v = cy + fy*Y/z`` with the original pixels and the original K, so
# NO image flip is needed (a flip was tried first, 2026-11-23, and it
# broke the geometry: triangulated floor points came out mirrored and
# the essential translation picked up a spurious Y component). What
# changes is only the rotation: the OpenCV camera frame is the geometry
# frame with Y negated, ``M = diag(1, -1, 1)``, so ``R_opencv = M @
# R_orig`` (M is its own inverse) and the camera centre C is unchanged.
# Every internal pose is stored in the OpenCV frame; only ``pose`` /
# ``update()`` return values are converted back to the geometry
# convention. Triangulated points are in the OpenCV camera frame and
# are converted to world with the INTERNAL R — converting them with
# the geometry R mirrors them in Y (measured 2026-11-23: floor points
# at y_unit +5.7 instead of -5.7, which made the floor scale negative
# and initialisation fail).
# ---------------------------------------------------------------------------
_M_FLIP = np.diag([1.0, -1.0, 1.0])


def _to_internal_R(R_orig):
    """Geometry-convention R (Y-up) -> OpenCV-internal R (Y-down)."""
    return _M_FLIP @ np.asarray(R_orig, dtype=float)


def _to_external_R(R_internal):
    """OpenCV-internal R (Y-down) -> geometry-convention R (Y-up)."""
    return _M_FLIP @ np.asarray(R_internal, dtype=float)


def _wrap_deg(a):
    """Wrap degrees to [-180, 180)."""
    return (float(a) + 180.0) % 360.0 - 180.0


def _reproj_residuals(R, C, obj, img, Kmat):
    """Flattened reprojection residuals (px) for a pose (internal frame).

    Exactly the residual used by ``_lm_pnp`` (same z clamp at
    0.05 m): factored out so the 1-DoF yaw PnP shares it instead of
    duplicating the projection. ``obj``/``img`` are (N, 3)/(N, 2).
    """
    fx, fy, cx, cy = Kmat[0, 0], Kmat[1, 1], Kmat[0, 2], Kmat[1, 2]
    Pc = (R @ (np.asarray(obj, dtype=float) - C).T).T
    z = np.maximum(Pc[:, 2], 0.05)
    uv = np.stack([fx * Pc[:, 0] / z + cx,
                   fy * Pc[:, 1] / z + cy], axis=1)
    return (uv - np.asarray(img, dtype=float)).reshape(-1)


def _proj_matrix(Kmat, R, C):
    """Projection matrix K [R | -R C] for triangulation."""
    t = -np.asarray(R, dtype=float) @ np.asarray(C, dtype=float)
    return Kmat @ np.hstack([np.asarray(R, dtype=float),
                             t.reshape(3, 1)])


def _lm_jacobian(R, C, obj, Kmat):
    """Analytic Jacobian of the reprojection residuals in ``_lm_pnp``.

    Residual rows are interleaved (u - u_obs, v - v_obs) with
    u = fx*X/Z + cx, v = fy*Y/Z + cy, Pc = R (P - C) = (X, Y, Z).
    Rotation increment dR ~= I + [delta]x is applied on the LEFT
    (camera frame), so dPc/d(delta) = delta x Pc; centre increment
    dPc/dC = -R. Verified against finite differences 2026-11-23
    (max abs difference < 1e-6 on random poses/points).
    """
    fx, fy = Kmat[0, 0], Kmat[1, 1]
    Pc = (R @ (obj - C).T).T
    Z = np.maximum(Pc[:, 2], 0.05)
    X, Y = Pc[:, 0], Pc[:, 1]
    Z2 = Z ** 2
    n = len(obj)
    J = np.zeros((2 * n, 6))
    # rotation columns (delta_x, delta_y, delta_z)
    J[0::2, 0] = -fx * X * Y / Z2
    J[1::2, 0] = -fy * (Y * Y + Z2) / Z2
    J[0::2, 1] = fx * (X * X + Z2) / Z2
    J[1::2, 1] = fy * X * Y / Z2
    J[0::2, 2] = -fx * Y / Z
    J[1::2, 2] = fy * X / Z
    # centre columns: dPc/dC_k = -R[:, k]
    for k in range(3):
        J[0::2, 3 + k] = fx * (-R[0, k] * Z + X * R[2, k]) / Z2
        J[1::2, 3 + k] = fy * (-R[1, k] * Z + Y * R[2, k]) / Z2
    return J


def _lm_pnp(obj, img, Kmat, R_init, C_init, iters=30, huber_px=3.0,
            R_prior=None, prior_sigmas_deg=None):
    """Levenberg-Marquardt PnP from an initial pose (R_init, C_init).

    Minimises the Huber-robust reprojection error over the 6 pose
    parameters (rotation-vector increment in the camera frame +
    camera-centre increment in the world frame), numeric Jacobian,
    adaptive damping. Returns ``(R, C)`` in the OpenCV-internal
    frame, or None if the solve fails. It deliberately returns the
    rotation MATRIX, not a rotation vector: the internal frame differs
    from the geometry frame by M = diag(1,-1,1), a 180 deg rotation,
    so internal rotations sit at angle ~pi, where
    ``cv2.Rodrigues`` round-trips are numerically unstable (measured
    2026-11-23: a converged LM pose with C = [0.00, 1.30, 0.88] came
    back from a Rodrigues round-trip as C = [-0.89, -1.30, 0.00] —
    the rotation vector of a ~pi rotation flipped axis). The LM itself
    works in matrix space (dR @ R) and is unaffected; only the
    conversion was broken. Tiny Rodrigues increments (1e-6 rad) in
    the Jacobian are fine.

    This exists because ``cv2.solvePnP`` is unreliable on this data
    (measured 2026-11-23): on a nearly-planar map (most points on the
    floor plane y = 0) plain ITERATIVE solvePnP converges to the
    MIRROR solution — a ~180 deg yaw flip whose C is nevertheless
    within centimetres of the truth and whose reprojection error is
    ~0.9 px, because the mirror negates both y and z of every point
    and the ratios y/z, x/z survive. It is only caught by cheirality
    (0 of 66 points in front of the camera). ``solvePnPRansac`` is
    worse: 5 inliers + garbage on the same data, for every flag.
    Starting LM from the previous pose (inter-frame motion is small)
    stays in the true basin, and the Huber loss bounds the influence
    of LK mismatches.

    Prior factor (round 3, 2026-11-23): with ``R_prior`` given (the
    Tello attitude prior at this frame time, internal frame), three
    extra residual rows pull the yaw/pitch/roll toward the prior,
    weighted by ``prior_sigmas_deg`` (default (2, 1, 1) deg — the
    prior is a real sensor, not truth: yaw is magnetometer-fused
    and quantised to whole degrees, so it is a spring, not a clamp).
    This is a one-node pose graph: visual factors + one prior
    factor, solved jointly — the pose and the centre stay
    consistent because C is re-optimised WITH the constrained R
    (unlike the reverted post-hoc yaw correction, which changed R
    without re-optimising C). With ``R_prior=None`` the behaviour
    is exactly the round-2 solver (no prior rows at all).
    """
    obj = np.asarray(obj, dtype=float)
    img = np.asarray(img, dtype=float)
    if len(obj) < 6:
        return None
    R = np.asarray(R_init, dtype=float).copy()
    C = np.asarray(C_init, dtype=float).copy()

    def residuals(R_, C_):
        return _reproj_residuals(R_, C_, obj, img, Kmat)

    # Prior residuals (None = no prior factor, round-2 behaviour).
    if R_prior is not None:
        _sig = np.radians(np.asarray(
            prior_sigmas_deg if prior_sigmas_deg is not None
            else (2.0, 1.0, 1.0), dtype=float))
        _ypr_pr = np.asarray(
            yaw_pitch_roll_from_R(_to_external_R(R_prior)), dtype=float)

        def prior_residual(R_):
            _ypr = np.asarray(
                yaw_pitch_roll_from_R(_to_external_R(R_)), dtype=float)
            _d = np.array([_wrap_deg(_ypr[i] - _ypr_pr[i])
                           for i in range(3)])
            return np.radians(_d) / _sig

        def prior_jacobian(R_):
            # d(prior residual)/d(rotation increment): the LM's
            # rotation increment is a LEFT multiplication dR @ R in
            # the internal frame (same convention as _lm_jacobian),
            # so the numeric derivative uses that exact update.
            # Central differences, h = 1e-6 rad.
            _Jp = np.zeros((3, 6))
            _h = 1e-6
            for _k in range(3):
                _e = np.zeros(3)
                _e[_k] = _h
                _dR, _ = cv2.Rodrigues(_e)
                _rp = prior_residual(_dR @ R_)
                _dRm, _ = cv2.Rodrigues(-_e)
                _rm = prior_residual(_dRm @ R_)
                _Jp[:, _k] = (_rp - _rm) / (2.0 * _h)
            return _Jp
    else:
        prior_residual = None

    def cost(R_, C_):
        r = residuals(R_, C_).reshape(-1, 2)
        e = np.linalg.norm(r, axis=1)
        total = float(np.sum(np.where(
            e <= huber_px, 0.5 * e ** 2, huber_px * (e - 0.5 * huber_px))))
        if prior_residual is not None:
            _rp = prior_residual(R_)
            total += 0.5 * float(_rp @ _rp)
        return total

    lam = 1e-3
    cur = cost(R, C)
    for _ in range(iters):
        r = residuals(R, C)
        r2 = r.reshape(-1, 2)
        e = np.linalg.norm(r2, axis=1)
        w = np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-9))
        w2 = np.repeat(w, 2)
        # Analytic Jacobian (numeric finite differences were ~7x
        # slower and no more accurate); see _lm_jacobian.
        J = _lm_jacobian(R, C, obj, Kmat)
        Jw = J * w2[:, None]
        rw = r * w2
        if prior_residual is not None:
            Jp = prior_jacobian(R)
            rp = prior_residual(R)
            J_full = np.vstack([Jw, Jp])
            r_full = np.concatenate([rw, rp])
            A = J_full.T @ J_full
            g = J_full.T @ r_full
        else:
            A = Jw.T @ Jw
            g = Jw.T @ rw
        step = np.zeros(6)
        accepted = False
        for _ in range(12):
            try:
                step = np.linalg.solve(
                    A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            dR, _ = cv2.Rodrigues(step[:3])
            Rn, Cn = dR @ R, C + step[3:]
            nc = cost(Rn, Cn)
            if nc < cur:
                R, C, cur = Rn, Cn, nc
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 10.0
        if not accepted or float(np.linalg.norm(step)) < 1e-9:
            break
    return R, C


def _lm_pnp_yaw1dof(obj, img, Kmat, R_prior, C_init, iters=30,
                    huber_px=3.0):
    """1-DoF yaw PnP (round 3, option A): pitch/roll FIXED from the
    Tello attitude prior, yaw + camera centre solved jointly by LM.

    ``R_prior`` is the Tello attitude prior at this frame time
    (geometry frame). The rotation is parametrised as
    ``R(yaw, pitch_prior, roll_prior)`` and the LM optimises
    (yaw, Cx, Cy, Cz) — 4 parameters, not 6. The turn yaw error
    that corrupts depth triangulation comes from the 6-DoF PnP
    trading yaw against a noisy near-planar map; fixing pitch/roll
    removes the two degrees of freedom the map noise uses to drag
    the yaw, and the centre is re-optimised WITH the constrained
    rotation, so R-C consistency is kept (unlike the reverted
    post-hoc yaw correction, which changed R without re-optimising
    C). The yaw is initialised from the prior yaw with a +/-6 deg
    coarse grid (guards a local minimum when the prior is a few
    degrees off); the Huber loss and the numeric central-difference
    Jacobian mirror ``_lm_pnp``. Returns ``(R, C)`` in the
    OpenCV-internal frame, or None if the solve fails.
    """
    obj = np.asarray(obj, dtype=float)
    img = np.asarray(img, dtype=float)
    if len(obj) < 6:
        return None
    yaw_pr, pitch_pr, roll_pr = yaw_pitch_roll_from_R(R_prior)

    def R_of(psi_deg):
        return _to_internal_R(R_from_yaw_pitch_roll(
            psi_deg, pitch_pr, roll_pr))

    def cost(psi_deg, C_):
        r = _reproj_residuals(
            R_of(psi_deg), C_, obj, img, Kmat).reshape(-1, 2)
        e = np.linalg.norm(r, axis=1)
        return float(np.sum(np.where(
            e <= huber_px, 0.5 * e ** 2, huber_px * (e - 0.5 * huber_px))))

    # Coarse yaw grid around the prior yaw (cheap: 7 cost evals).
    psi = float(yaw_pr)
    cur = cost(psi, C_init)
    for d in (-6.0, -4.0, -2.0, 2.0, 4.0, 6.0):
        c = cost(yaw_pr + d, C_init)
        if c < cur:
            psi, cur = yaw_pr + d, c
    C = np.asarray(C_init, dtype=float).copy()
    lam = 1e-3
    h_psi = 1e-6  # rad (central-difference step for the yaw column)
    h_C = 1e-6    # m (central-difference step for the centre columns)
    for _ in range(iters):
        r = _reproj_residuals(R_of(psi), C, obj, img, Kmat)
        r2 = r.reshape(-1, 2)
        e = np.linalg.norm(r2, axis=1)
        w = np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-9))
        w2 = np.repeat(w, 2)
        # Numeric Jacobian, central differences: columns
        # (yaw [rad], Cx, Cy, Cz). The yaw derivative is per radian
        # (the step is applied in degrees, divided by the radian
        # step), matching the LM's rotation-increment units.
        J = np.zeros((2 * len(obj), 4))
        _hp_deg = math.degrees(h_psi)
        J[:, 0] = (_reproj_residuals(R_of(psi + _hp_deg), C, obj, img,
                                     Kmat)
                   - _reproj_residuals(R_of(psi - _hp_deg), C, obj, img,
                                       Kmat)) / (2.0 * h_psi)
        for k in range(3):
            Cp = C.copy()
            Cp[k] += h_C
            Cm = C.copy()
            Cm[k] -= h_C
            J[:, 1 + k] = (_reproj_residuals(
                R_of(psi), Cp, obj, img, Kmat)
                - _reproj_residuals(R_of(psi), Cm, obj, img, Kmat)
                ) / (2.0 * h_C)
        Jw = J * w2[:, None]
        rw = r * w2
        A = Jw.T @ Jw
        g = Jw.T @ rw
        step = np.zeros(4)
        accepted = False
        for _ in range(12):
            try:
                step = np.linalg.solve(
                    A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            psi_n = psi + math.degrees(step[0])
            Cn = C + step[1:]
            nc = cost(psi_n, Cn)
            if nc < cur:
                psi, C, cur = psi_n, Cn, nc
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 10.0
        if not accepted or float(np.linalg.norm(step)) < 1e-9:
            break
    return R_of(psi), C


def _pnp_yaw1dof(obj, img, Kmat, R_prior, C_init, thresh_px=3.0):
    """Robust 1-DoF yaw PnP: LM on all points, then one refinement
    on its inliers (the same fast-path pattern as ``_pnp_ransac``,
    without the random rounds — the yaw grid already removes the
    main local minimum, and the Huber loss bounds outlier
    influence). Returns ``(R, C, n_inliers)`` (internal frame) or
    None. ``R_prior`` is in the geometry frame."""
    obj = np.asarray(obj, dtype=float)
    img = np.asarray(img, dtype=float)
    n = len(obj)
    if n < 6:
        return None
    sol = _lm_pnp_yaw1dof(obj, img, Kmat, R_prior, C_init)
    if sol is None:
        return None
    inl = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
    best = (int(inl.sum()), sol[0], sol[1])
    if int(inl.sum()) >= 6:
        ref = _lm_pnp_yaw1dof(obj[inl], img[inl], Kmat, R_prior, sol[1])
        if ref is not None:
            inl2 = _reproj_inliers(ref[0], ref[1], obj, img, Kmat,
                                   thresh_px)
            if int(inl2.sum()) >= best[0]:
                best = (int(inl2.sum()), ref[0], ref[1])
    if best[0] < 4:
        return None
    return best[1], best[2], best[0]


def _lm_pnp_transonly(obj, img, Kmat, R_fixed, C_init, iters=30,
                      huber_px=3.0):
    """Translation-only PnP (round 3, option A2): rotation FIXED from
    the Tello attitude prior, centre solved by LM (3 params).

    With the full attitude known, only the 3-DoF position drifts —
    the analogue of VINS-Mono's 4-DoF pose graph (Qin et al. 2018),
    where roll/pitch come from the IMU so drift lives in (x, y, z,
    yaw); here the prior also fixes yaw, so the drift lives in
    (x, y, z) alone. The centre columns of ``_lm_jacobian`` are the
    exact analytic Jacobian for this solve (rotation held fixed), so
    no numeric Jacobian is needed. Huber loss and adaptive damping
    mirror ``_lm_pnp``. Returns ``(R_fixed, C)`` in the
    OpenCV-internal frame, or None if the solve fails.
    """
    obj = np.asarray(obj, dtype=float)
    img = np.asarray(img, dtype=float)
    if len(obj) < 6:
        return None
    R = np.asarray(R_fixed, dtype=float).copy()
    C = np.asarray(C_init, dtype=float).copy()

    def cost(C_):
        r = _reproj_residuals(R, C_, obj, img, Kmat).reshape(-1, 2)
        e = np.linalg.norm(r, axis=1)
        return float(np.sum(np.where(
            e <= huber_px, 0.5 * e ** 2, huber_px * (e - 0.5 * huber_px))))

    lam = 1e-3
    cur = cost(C)
    for _ in range(iters):
        r = _reproj_residuals(R, C, obj, img, Kmat)
        r2 = r.reshape(-1, 2)
        e = np.linalg.norm(r2, axis=1)
        w = np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-9))
        w2 = np.repeat(w, 2)
        # Centre columns only (rotation fixed): exact analytic
        # Jacobian, same convention as _lm_pnp.
        J = _lm_jacobian(R, C, obj, Kmat)[:, 3:6]
        Jw = J * w2[:, None]
        rw = r * w2
        A = Jw.T @ Jw
        g = Jw.T @ rw
        step = np.zeros(3)
        accepted = False
        for _ in range(12):
            try:
                step = np.linalg.solve(
                    A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            Cn = C + step
            nc = cost(Cn)
            if nc < cur:
                C, cur = Cn, nc
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 10.0
        if not accepted or float(np.linalg.norm(step)) < 1e-9:
            break
    return R, C


def _pnp_transonly(obj, img, Kmat, R_prior, C_init, thresh_px=3.0):
    """Robust translation-only PnP: LM on all points, then one
    refinement on its inliers (the same fast-path pattern as
    ``_pnp_ransac``). ``R_prior`` is the attitude prior in the
    GEOMETRY frame; the returned R is in the internal frame.
    Returns ``(R, C, n_inliers)`` or None."""
    obj = np.asarray(obj, dtype=float)
    img = np.asarray(img, dtype=float)
    n = len(obj)
    if n < 6:
        return None
    R_fixed = _to_internal_R(R_prior)
    sol = _lm_pnp_transonly(obj, img, Kmat, R_fixed, C_init)
    if sol is None:
        return None
    inl = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
    best = (int(inl.sum()), sol[0], sol[1])
    if int(inl.sum()) >= 6:
        ref = _lm_pnp_transonly(obj[inl], img[inl], Kmat, R_fixed, sol[1])
        if ref is not None:
            inl2 = _reproj_inliers(ref[0], ref[1], obj, img, Kmat,
                                   thresh_px)
            if int(inl2.sum()) >= best[0]:
                best = (int(inl2.sum()), ref[0], ref[1])
    if best[0] < 4:
        return None
    return best[1], best[2], best[0]


def _reproj_inliers(R, C, obj, img, Kmat, thresh_px):
    """Boolean inlier mask for a PnP pose (reprojection error < thresh).

    Points behind the camera (z <= 0) are NEVER inliers — this is
    what rejects the mirror solution (see _lm_pnp).
    """
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    Pcam = (R @ (np.asarray(obj, dtype=float) - C).T).T
    z = Pcam[:, 2]
    good = z > 1e-6
    uv = np.full((len(obj), 2), np.inf)
    uv[good] = np.stack([
        Kmat[0, 0] * Pcam[good, 0] / z[good] + Kmat[0, 2],
        Kmat[1, 1] * Pcam[good, 1] / z[good] + Kmat[1, 2]], axis=1)
    err = np.linalg.norm(uv - np.asarray(img, dtype=float), axis=1)
    return err < thresh_px


def _reproj_err(R, C, obj, img, Kmat):
    """Per-point reprojection error (px) of 3D points at pose (R, C)."""
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    Pcam = (R @ (np.asarray(obj, dtype=float) - C).T).T
    z = Pcam[:, 2]
    err = np.full(len(obj), np.inf)
    good = z > 1e-6
    uv = np.stack([
        Kmat[0, 0] * Pcam[good, 0] / z[good] + Kmat[0, 2],
        Kmat[1, 1] * Pcam[good, 1] / z[good] + Kmat[1, 2]], axis=1)
    err[good] = np.linalg.norm(uv - np.asarray(img, dtype=float)[good], axis=1)
    return err


def _pnp_ransac(obj, img, Kmat, R_init, C_init, thresh_px=3.0,
                iterations=60, subset=12, rng_seed=20261123,
                R_prior=None, prior_sigmas_deg=None):
    """Robust PnP: Huber-LM from the previous pose, RANSAC fallback.

    Fast path: LM on ALL points from the previous pose (inter-frame
    motion is small, so this is the true basin), refined once on its
    inliers. If that is already >= 90% inliers, take it. Otherwise a
    few random-subset RANSAC rounds (LM from the previous pose on
    each subset) look for a cleaner consensus — 8 rounds when the
    fast path passed the validity gate, the full 60 when it failed
    it — keeping the pose with the most inliers, then a final LM
    refinement on those inliers. The 60-round version on every frame
    cost ~80 ms/frame (profiled 2026-11-23); 8 rounds recover the
    accuracy (s1 drift 4.0 cm) for ~12 ms.
    Returns ``(R, C, n_inliers)`` (OpenCV-internal frame) or ``None``.
    Deterministic (seeded RNG) so runs reproduce.

    ``R_prior`` (internal frame) + ``prior_sigmas_deg`` add the
    round-3 prior factor to every LM call (see ``_lm_pnp``); with
    both None this is exactly the round-2 solver.
    """
    obj = np.asarray(obj, dtype=float)
    img = np.asarray(img, dtype=float)
    n = len(obj)
    if n < 6:
        return None
    best = None  # (n_inliers, R, C)
    sol = _lm_pnp(obj, img, Kmat, R_init, C_init,
                  R_prior=R_prior, prior_sigmas_deg=prior_sigmas_deg)
    if sol is not None:
        inl = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
        best = (int(inl.sum()), sol[0], sol[1])
        if best[0] >= 0.9 * n:
            return best[1], best[2], best[0]
        if _accept_pnp(best[0], n):
            # Fast path already passes the gate; still run the few
            # RANSAC rounds below for a cleaner consensus (they are
            # cheap and the gate_ok flag below selects 8 vs 60).
            pass
        # One cheap refinement on the fast path's inliers: the raw
        # fast path is dragged by its outliers (accepting it directly
        # regressed s1 drift 4.0 -> 10.5 cm, measured 2026-11-23).
        if int(inl.sum()) >= 6:
            ref = _lm_pnp(obj[inl], img[inl], Kmat, sol[0], sol[1])
            if ref is not None:
                inl2 = _reproj_inliers(
                    ref[0], ref[1], obj, img, Kmat, thresh_px)
                if int(inl2.sum()) >= best[0]:
                    best = (int(inl2.sum()), ref[0], ref[1])
    # RANSAC rounds: a few find a cleaner consensus than the fast
    # path's outlier-dragged fit (s1 drift 9.5 cm fast-path-only vs
    # 4.0 cm with full RANSAC). 8 rounds (~12 ms) when the fast path
    # already passed the gate; the full 60 only when it failed it.
    gate_ok = best is not None and _accept_pnp(best[0], n)
    rounds = 8 if gate_ok else iterations
    rng = np.random.default_rng(rng_seed)
    for _ in range(rounds):
        if best is not None and best[0] >= 0.9 * n:
            break
        idx = rng.choice(n, size=min(subset, n), replace=False)
        sol = _lm_pnp(obj[idx], img[idx], Kmat, R_init, C_init, iters=10,
                      R_prior=R_prior,
                      prior_sigmas_deg=prior_sigmas_deg)
        if sol is None:
            continue
        inl = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
        cnt = int(inl.sum())
        if best is None or cnt > best[0]:
            best = (cnt, sol[0], sol[1])
    if best is None or best[0] < 4:
        return None
    inl = _reproj_inliers(best[1], best[2], obj, img, Kmat, thresh_px)
    if int(inl.sum()) >= 6:
        sol = _lm_pnp(obj[inl], img[inl], Kmat, best[1], best[2],
                      R_prior=R_prior,
                      prior_sigmas_deg=prior_sigmas_deg)
        if sol is not None:
            inl2 = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
            if int(inl2.sum()) >= best[0]:
                best = (int(inl2.sum()), sol[0], sol[1])
    return best[1], best[2], best[0]
    # The gate-failing case below is the expensive one; but even
    # when the fast path passed the gate, a few RANSAC rounds find
    # a cleaner consensus than the fast path's outlier-dragged fit
    # (s1 drift 9.5 cm fast-path-only vs 4.0 cm with full RANSAC).
    # Cap the extra rounds at 8 (~12 ms) — the full 60 run only
    # when the fast path actually failed the gate.
    extra_rounds = 8 if best is not None else iterations
    rng = np.random.default_rng(rng_seed)
    for _ in range(extra_rounds):
        if best is not None and best[0] >= 0.9 * n:
            break
        idx = rng.choice(n, size=min(subset, n), replace=False)
        sol = _lm_pnp(obj[idx], img[idx], Kmat, R_init, C_init, iters=10)
        if sol is None:
            continue
        inl = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
        cnt = int(inl.sum())
        if best is None or cnt > best[0]:
            best = (cnt, sol[0], sol[1])
    if best is None or best[0] < 4:
        return None
    inl = _reproj_inliers(best[1], best[2], obj, img, Kmat, thresh_px)
    if int(inl.sum()) >= 6:
        sol = _lm_pnp(obj[inl], img[inl], Kmat, best[1], best[2])
        if sol is not None:
            inl2 = _reproj_inliers(sol[0], sol[1], obj, img, Kmat, thresh_px)
            if int(inl2.sum()) >= best[0]:
                best = (int(inl2.sum()), sol[0], sol[1])
    return best[1], best[2], best[0]


class MonocularVO:
    """Monocular visual odometry from a drone video stream.

    Usage::

        vio = MonocularVO(start_pose=takeoff_pose)  # known session geometry
        for frame, t in video:
            pose, valid, diag = vio.update(frame, t)
            pipeline.process_frame(frame, pose, valid, t)

    ``pose`` is always a ``(R, C)`` pair (the last estimate, or the
    start pose before initialisation) — but it is only meaningful when
    ``valid`` is True. Callers MUST pass ``valid`` through to the
    bridge and never use an invalid pose as if it were a measurement.
    """

    def __init__(self, K=None, start_pose=None, camera_height_m=1.3,
                 min_pnp_inliers=MIN_PNP_INLIERS,
                 min_pnp_ratio=MIN_PNP_RATIO, state_prior=None,
                 pnp_mode="yaw1dof", yaw_prior_sigma_deg=2.0,
                 pitch_prior_sigma_deg=1.0, roll_prior_sigma_deg=1.0):
        self.K = dict(DRONE_K) if K is None else dict(K)
        # No image flip: the raw image is already OpenCV-standard in
        # the Y-down reading (see _M_FLIP); only R is converted.
        self._Kmat = _K_matrix(self.K)
        self.camera_height_m = float(camera_height_m)
        self.min_pnp_inliers = int(min_pnp_inliers)
        self.min_pnp_ratio = float(min_pnp_ratio)
        # PnP mode (round 3, 2026-11-23): how the Tello attitude
        # prior constrains the pose solve. "full" = the round-2
        # 6-DoF PnP (prior used only as the LM start, as before).
        # "yaw1dof" = 1-DoF yaw PnP (pitch/roll fixed from the prior,
        # yaw + centre solved jointly). "attitude" = translation-only
        # PnP (full attitude from the prior, centre solved) with
        # fallback to yaw1dof then full. "prior" = the 6-DoF PnP
        # with the prior as a soft factor (yaw/pitch/roll residuals
        # with sigmas below). The default is "yaw1dof" — the measured
        # winner (see tests/TEST-REPORT.md, VIO fix round 3): with
        # the bbox-height detector calibration it passes all 4 VIO
        # scenarios with 19.5-27.4 cm of margin (s1 40.5, s23 32.6,
        # s20 36.1, s4 36.3 cm). Caveat, measured the same day: it is
        # FRAGILE to prior error — a (yaw+2, pitch-1, roll+1) deg
        # prior offset fails s1/s20/s4 (81.6-90.4 cm) — because the
        # pitch/roll are fixed from the prior rather than solved.
        # The harness prior is accurate to ~1 deg (quantised GT), so
        # the harness pass holds; on a real Tello the magnetometer
        # yaw accuracy is the binding constraint, and the section 6.9
        # yaw calibration (tools/calibrate_tello.py) must show
        # sigma ≲ 1.5 deg before the real test. "full" is robust to
        # prior error but fails s1/s23 even unperturbed (71.8/81.7).
        if pnp_mode not in ("full", "yaw1dof", "attitude", "prior"):
            raise ValueError(f"unknown pnp_mode {pnp_mode!r}")
        self._pnp_mode = pnp_mode
        self._yaw_prior_sigma_deg = float(yaw_prior_sigma_deg)
        self._pitch_prior_sigma_deg = float(pitch_prior_sigma_deg)
        self._roll_prior_sigma_deg = float(roll_prior_sigma_deg)
        # Optional Tello state prior (server.tello_state.StatePrior):
        # yaw/pitch/roll + h/tof always usable (units unambiguous);
        # velocity only when calibrated (see tello_state). May be
        # attached later via set_state_prior() / update(state=...).
        self._state_prior = state_prior
        self._start_R, self._start_C = (None, None)  # internal (Y-down) R
        if start_pose is not None:
            R0, C0 = start_pose
            self._start_R = _to_internal_R(R0)
            self._start_C = np.asarray(C0, dtype=float).copy()
        self.reset(start_pose)

    def set_state_prior(self, state_prior):
        """Attach (or replace) the Tello state prior."""
        self._state_prior = state_prior

    # -- lifecycle ---------------------------------------------------------

    def reset(self, start_pose=None):
        """(Re)start the odometry at a known start pose."""
        if start_pose is not None:
            R0, C0 = start_pose
            self._start_R = _to_internal_R(R0)
            self._start_C = np.asarray(C0, dtype=float).copy()
            self._R = self._start_R.copy()
            self._C = self._start_C.copy()
        elif self._start_R is not None:
            self._R = self._start_R.copy()
            self._C = self._start_C.copy()
        else:
            self._R, self._C = np.eye(3), np.zeros(3)
        self._initialized = False
        self._lost = False
        self._prev_gray = None
        # Tracked points in the previous frame: (N, 2) pixels.
        self._prev_pts = None
        # World 3D point per tracked point, or None (not triangulated yet).
        self._prev_pts3d = None  # (N, 3) float, NaN row = no 3D yet
        # Pose of the frame where each no-3D point was first seen, and
        # its pixel there — needed to triangulate it once the baseline
        # is long enough.
        self._prev_origin_pose = None  # list of (R, C) or None per point
        self._prev_origin_uv = None    # (N, 2)
        # Keyframe (initialisation reference) state.
        self._kf_gray = None
        self._kf_pts = None
        self._kf_pose = None
        self._kf_frame = None
        # Accepted-pose history for the constant-velocity motion
        # model (LK prediction): last two (frame, R, C), internal
        # frame. _record_pose appends on every accepted pose;
        # _reset_pose_hist restarts it after a recovery jump, where
        # the pre-loss motion no longer predicts the new chain.
        self._pose_hist = []
        # Last GOOD frame (relocalisation anchor after a loss).
        self._good_gray = None
        self._good_pts = None
        self._good_pts3d = None
        self._good_pose = None
        self._n_frames = 0
        self._n_lost_frames = 0
        self._lost_frames = 0  # frames since tracking was lost
        # ORB descriptors of the last good frame's tracked points
        # (aligned with _good_pts / _good_pts3d); None until the
        # first successful track. Only points with a 3D map point
        # are usable for PnP recovery.
        self._orb = cv2.ORB_create(nfeatures=ORB_NFEATURES,
                                   scaleFactor=1.2, nlevels=8)
        self._good_des = None
        self._good_des_valid = None
        # Frame times (scenario/video clock) for the velocity prior
        # and dead reckoning: _t_cur is this frame, _t_prev the last
        # accepted frame. None until two frames have been seen.
        self._t_cur = None
        self._t_prev = None
        # Tracking keyframe (report §1.2(f), GRANITE-style): the
        # reference pose for windowed BA and keyframe relocalisation.
        # A new keyframe is set when the rotation since the last
        # keyframe exceeds KEYFRAME_ROT_DEG, or every
        # KEYFRAME_MAX_GAP frames, or when the 3D map thins below
        # KEYFRAME_MIN_MAP. (The report's negative-entropy rule is
        # the ideal; this parallax + time rule is the implemented
        # approximation — it fires on the same events: a turn, a
        # long straight, or a thinning map.)
        self._kf_track_pose = None
        self._kf_track_frame = None
        self._keyframes = []  # dicts: frame, R, C, ids, uv, pts3d, des
        # Persistent map-point IDs (for windowed BA): _prev_ids[i]
        # is the ID of _prev_pts[i]'s 3D point, or -1 if the point is
        # still a bearing (d=0, not yet triangulated). IDs are
        # assigned on triangulation and carried across frames.
        self._prev_ids = None
        self._next_point_id = 0
        # Dead-reckoning state (vision lost, state prior available):
        # the last good pose + the time/distance integrated since
        # vision was lost. Cleared on any visual recovery.
        self._dr_pose = None
        self._dr_start_t = None
        self._dr_dist_m = 0.0
        self._last_diag = {}
        # Init-wait log throttle (see _log_init_wait): last reason
        # logged and when, so a persistent init failure logs at most
        # one line per 5 s instead of one per frame.
        self._last_init_fail = None
        self._last_init_fail_t = 0.0

    # -- public API ----------------------------------------------------------

    @property
    def initialized(self):
        return self._initialized

    @property
    def pose(self):
        """Current estimated pose (R, C) — meaningful only when valid."""
        return _to_external_R(self._R), self._C.copy()

    def _external_pose(self):
        return _to_external_R(self._R), self._C.copy()

    def update(self, frame_bgr, t=None, state=None):
        """Feed one BGR frame; return (pose (R, C), tracking_valid, diag).

        ``state`` (optional) is a ``server.tello_state.StatePrior``
        (already buffered) or a single ``TelloStatePacket`` (added to
        the attached prior, or to a fresh prior if none is attached).
        The prior supplies yaw/pitch/roll + h/tof always, and
        velocity only when calibrated (see tello_state).

        ``diag`` carries ``inliers`` (PnP / essential inliers this
        frame), ``n_tracked`` (LK points carried into this frame),
        ``n_map`` (tracked points with a 3D map point), ``n_features``
        (features detected on this frame, if any),
        ``initialized``, ``vio_ms`` (this call's wall time),
        ``init_scale_m`` (the baseline scale found at initialisation,
        once known), and ``pose_source`` (``vision`` /
        ``attitude-aided`` / ``dead-reckoned`` / ``invalid`` — the
        per-pose flag so the bridge never silently presents an
        attitude-only pose as visual tracking).
        """
        t0 = time.perf_counter()
        self._n_frames += 1
        if t is not None:
            self._t_prev = self._t_cur
            self._t_cur = float(t)
        if state is not None:
            from .tello_state import StatePrior as _SP
            if isinstance(state, _SP):
                self._state_prior = state
            else:  # a single packet
                if self._state_prior is None:
                    self._state_prior = _SP()
                self._state_prior.add(state)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diag = {"frame": self._n_frames, "inliers": 0, "n_tracked": 0,
                "n_map": 0, "n_features": 0,
                "initialized": self._initialized, "vio_ms": 0.0,
                "init_scale_m": getattr(self, "_init_scale_m", None),
                "relocalised": False, "pose_source": POSE_INVALID}
        if self._prev_gray is None:
            # First frame: the keyframe. Pose = known start pose, but
            # NOT valid yet — no motion has been observed, so there is
            # nothing to track against. The bridge stays invalid until
            # the first successful initialisation.
            self._kf_gray = gray
            self._kf_frame = self._n_frames
            self._kf_pose = (self._R.copy(), self._C.copy())
            pts = self._detect(gray)
            diag["n_features"] = 0 if pts is None else len(pts)
            self._kf_pts = pts
            self._prev_gray = gray
            self._prev_pts = pts
            self._prev_pts3d = (np.full((len(pts), 3), np.nan)
                                if pts is not None else None)
            self._prev_origin_pose = ([(self._R.copy(), self._C.copy())]
                                      * (0 if pts is None else len(pts)))
            self._prev_origin_uv = (None if pts is None else pts.copy())
            self._prev_ids = (np.full(0 if pts is None else len(pts),
                                      -1, dtype=int)
                              if pts is not None else None)
            self._good_gray, self._good_pts = gray, pts
            self._good_pts3d = self._prev_pts3d
            self._good_pose = (self._R.copy(), self._C.copy())
            diag["vio_ms"] = (time.perf_counter() - t0) * 1000.0
            self._last_diag = diag
            return self._external_pose(), False, diag

        if not self._initialized:
            pose, valid, diag = self._try_initialise(gray, diag)
            self._log_init_wait(diag)
        elif self._lost:
            self._lost_frames += 1
            if self._lost_frames % RELOCALISE_EVERY == 1:
                pose, valid, diag = self._try_relocalise(gray, diag)
            else:
                # Skip this frame: relocalisation is expensive and
                # the anchor (last good frame) is unchanged, so
                # re-trying every frame buys nothing. Report invalid
                # — never a guessed pose. (Dead reckoning, if a state
                # prior is available, is attempted in _try_relocalise
                # on its frames; between attempts the pose is invalid
                # rather than silently integrated.)
                diag["n_lost_frames"] = self._n_lost_frames
                pose, valid = self._external_pose(), False
        else:
            pose, valid, diag = self._track(gray, diag)

        diag["vio_ms"] = (time.perf_counter() - t0) * 1000.0
        diag["initialized"] = self._initialized
        self._last_diag = diag
        return pose, valid, diag

    @property
    def last_diag(self):
        return dict(self._last_diag)

    # -- saved-map export / adopt (2026-11-25, saved-map relocalisation)
    # The map has no single array: live points live in _prev_*,
    # history in _keyframes (snapshots with persistent point IDs).
    # export_map_data() consolidates them by ID so server.map_store
    # can persist the map and relocalise against it after a loss,
    # instead of the VIO staying lost until a from-scratch
    # re-initialisation (which would teleport the world frame).
    # Points are in the INTERNAL (OpenCV Y-down) world frame — the
    # same frame _solve_pnp expects — so a saved map is only valid
    # in a session anchored at the same start pose (takeoff beside
    # the phone). Export is read-only: it never mutates tracking.

    def export_map_data(self):
        """Consolidate the sparse map into a plain dict for saving.

        Deduplicates by persistent point ID across the live set and
        all keyframes (oldest first, so the newest — BA-refined —
        observation of a point wins); a point keeps the newest
        VALID ORB descriptor seen for it. Bearings (no 3D yet,
        id -1 / NaN) are skipped: they cannot anchor a PnP solve.
        Returns the dict ``server.map_store.MapData.from_dict``
        consumes (points, descriptors, des_valid, ids, K,
        camera_height_m, ref_R/ref_C/ref_frame, keyframe poses,
        start_R/start_C, n_frames). Empty when uninitialised.
        """
        by_id = {}

        def _add(pts3d, des, des_valid, ids):
            if pts3d is None or ids is None:
                return
            pts3d = np.asarray(pts3d, dtype=float)
            ids = np.asarray(ids, dtype=int)
            n = min(len(pts3d), len(ids))
            for i in range(n):
                pid = int(ids[i])
                if pid < 0:
                    continue
                p = pts3d[i]
                if not np.isfinite(p).all():
                    continue
                rec = by_id.get(pid)
                if rec is None:
                    rec = {"point": p.copy(), "des": None,
                           "des_valid": False}
                    by_id[pid] = rec
                else:
                    rec["point"] = p.copy()
                if (des is not None and des_valid is not None
                        and i < len(des_valid) and bool(des_valid[i])):
                    rec["des"] = np.asarray(
                        des[i], dtype=np.uint8).copy()
                    rec["des_valid"] = True

        _add(self._prev_pts3d, None, None, self._prev_ids)
        for kf in self._keyframes:
            _add(kf.get("pts3d"), kf.get("des"), kf.get("des_valid"),
                 kf.get("ids"))
        # Descriptors for the live set: keyframes carry descriptors
        # only for points observed at a keyframe, and the init
        # keyframe carries none at all — on a short flight that
        # leaves most of the map undescribed. The live tracked set
        # is the most recently observed view, so fill any point
        # still without a descriptor from the previous frame (one
        # ORB detect, at save time only — never per frame). Measured
        # on the synthetic room: 73 -> ~127 described points.
        # Best-effort: a failure here must never lose the map.
        if (self._prev_gray is not None and self._prev_pts is not None
                and len(self._prev_pts) and self._prev_ids is not None):
            try:
                des_live, valid_live = self._orb_descriptors(
                    self._prev_gray, self._prev_pts)
                for i, pid in enumerate(
                        np.asarray(self._prev_ids, dtype=int)):
                    rec = by_id.get(int(pid))
                    if (rec is not None and not rec["des_valid"]
                            and i < len(valid_live)
                            and bool(valid_live[i])):
                        rec["des"] = np.asarray(
                            des_live[i], dtype=np.uint8).copy()
                        rec["des_valid"] = True
            except Exception as exc:  # noqa: BLE001 - best-effort only
                log.debug("export_map_data: live descriptors failed: %s",
                          exc)
        pids = sorted(by_id)
        n = len(pids)
        points = np.zeros((n, 3), dtype=float)
        descriptors = np.zeros((n, 32), dtype=np.uint8)
        des_valid = np.zeros(n, dtype=bool)
        for j, pid in enumerate(pids):
            rec = by_id[pid]
            points[j] = rec["point"]
            if rec["des"] is not None:
                descriptors[j] = rec["des"]
                des_valid[j] = True
        # Reference pose for the PnP init: newest keyframe, else the
        # last good pose, else the current pose (all internal frame).
        if self._keyframes:
            kf = self._keyframes[-1]
            ref_R, ref_C = kf["R"].copy(), kf["C"].copy()
            ref_frame = int(kf.get("frame", self._n_frames))
        elif self._good_pose is not None:
            ref_R, ref_C = (self._good_pose[0].copy(),
                            self._good_pose[1].copy())
            ref_frame = self._n_frames
        else:
            ref_R, ref_C = self._R.copy(), self._C.copy()
            ref_frame = self._n_frames
        kf_R = np.array([kf["R"] for kf in self._keyframes],
                        dtype=float).reshape(-1, 3, 3)
        kf_C = np.array([kf["C"] for kf in self._keyframes],
                        dtype=float).reshape(-1, 3)
        kf_frame = np.array([int(kf.get("frame", 0))
                             for kf in self._keyframes], dtype=int)
        start_R = (None if self._start_R is None
                   else _to_external_R(self._start_R))
        start_C = (None if self._start_C is None
                   else np.asarray(self._start_C, dtype=float).copy())
        return {
            "points": points, "descriptors": descriptors,
            "des_valid": des_valid,
            "ids": np.array(pids, dtype=np.int64),
            "K": dict(self.K),
            "camera_height_m": float(self.camera_height_m),
            "ref_R": ref_R, "ref_C": ref_C, "ref_frame": ref_frame,
            "keyframe_R": kf_R, "keyframe_C": kf_C,
            "keyframe_frame": kf_frame,
            "start_R": start_R, "start_C": start_C,
            "n_frames": int(self._n_frames),
            "n_keyframes": len(self._keyframes),
        }

    def adopt_saved_map_pose(self, pts_cur, pts3d, R_new, C_new, gray,
                             diag=None):
        """Resume tracking from a pose relocalised against a SAVED map.

        Public entry point for ``server.map_store.try_relocalise``.
        ``(R_new, C_new)`` are in the internal frame (as returned by
        ``_solve_pnp``). Unlike ``_adopt_recovered`` this also marks
        the VIO initialised: a saved map IS the initialisation — its
        world frame is the saved session's frame, so no from-scratch
        re-initialisation (and no teleport) is needed. Returns the
        ``(pose, True, diag)`` triple for ``update()``.
        """
        if diag is None:
            diag = {}
        self._initialized = True
        return self._adopt_recovered(
            np.asarray(pts_cur, dtype=np.float32),
            np.asarray(pts3d, dtype=float),
            np.asarray(R_new, dtype=float),
            np.asarray(C_new, dtype=float), gray, diag)

    # -- feature handling ----------------------------------------------------

    def _detect(self, gray, mask=None):
        """GoodFeaturesToTrack on ``gray`` -> (N, 2) pixels or None."""
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=MAX_FEATURES, qualityLevel=GFTT_QUALITY,
            minDistance=GFTT_MIN_DISTANCE, blockSize=7, mask=mask)
        if pts is None or len(pts) == 0:
            return None
        return pts.reshape(-1, 2).astype(np.float32)

    def _track_lk(self, gray_prev, pts_prev, gray_cur, init_pts=None):
        """LK-track points into the current frame.

        Returns (pts_cur (M,2), pts_prev_kept (M,2), kept_idx (M,))
        with outliers (status 0 / large flow error) dropped. Every
        surviving track also passes a forward-backward check
        (``LK_FB_MAX_ERR_PX``): the point is tracked back into the
        previous frame and must return within 1 px of its start —
        one-way LK on this repetitive checker floor locks onto
        neighbouring identical corners with status=1, so the
        backward check is what rejects those hallucinated tracks.
        ``init_pts`` (optional, (N, 2)) is the motion-model prediction
        of the points' current pixels, passed as the LK initial guess
        (``OPTFLOW_USE_INITIAL_FLOW``); without it the search starts
        at the previous pixels (zero-motion init). The backward check
        always starts from the tracked point's original previous
        pixel, so it verifies the round trip exactly.
        """
        if pts_prev is None or len(pts_prev) == 0:
            return None, None, None
        p0 = np.asarray(pts_prev, dtype=np.float32).reshape(-1, 1, 2)
        if init_pts is not None:
            p_init = np.asarray(init_pts, dtype=np.float32).reshape(-1, 1, 2)
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_cur, p0, p_init.copy(), winSize=LK_WIN,
                maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        else:
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_cur, p0, None, winSize=LK_WIN,
                maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA)
        if p1 is None or status is None:
            return None, None, None
        keep = (status.reshape(-1) == 1)
        if err is not None:
            keep &= (err.reshape(-1) < 30.0)
        # Drop points that left the frame.
        h, w = gray_cur.shape
        inside = ((p1.reshape(-1, 2)[:, 0] >= 0)
                  & (p1.reshape(-1, 2)[:, 0] < w)
                  & (p1.reshape(-1, 2)[:, 1] >= 0)
                  & (p1.reshape(-1, 2)[:, 1] < h))
        keep &= inside
        idx = np.nonzero(keep)[0]
        if len(idx) == 0:
            return None, None, None
        # Forward-backward check on the survivors: track back into
        # the previous frame; a track whose round trip does not
        # return to its start pixel is a hallucination (wrong
        # identical corner), not a measurement. The backward search
        # starts from the point's original previous pixel, so the
        # round trip is verified exactly.
        p1_kept = p1.reshape(-1, 2)[idx]
        p0_start = np.asarray(pts_prev, dtype=np.float32)[idx]
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray_cur, gray_prev, p1_kept.reshape(-1, 1, 2),
            p0_start.reshape(-1, 1, 2).copy(),
            winSize=LK_WIN, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if p0_back is not None and st_back is not None:
            fb_err = np.linalg.norm(
                p0_back.reshape(-1, 2) - p0_start, axis=1)
            fb_ok = (st_back.reshape(-1) == 1) & (
                fb_err < LK_FB_MAX_ERR_PX)
            idx = idx[fb_ok]
            if len(idx) == 0:
                return None, None, None
        pts_out = p1.reshape(-1, 2)[idx].astype(np.float32)
        # NOTE (2026-11-23): cornerSubPix re-anchoring was tried here
        # and REJECTED — it made tracking worse, not better: init
        # inliers fell 69 -> 39, init drift 1.0 cm -> 6.2 cm, and the
        # track collapsed at frame 5. The synthetic renderer's
        # flat-shaded checker + per-cell noise gives cornerSubPix
        # ambiguous gradients, so it pulls points to neighbouring
        # corners instead of re-anchoring them. LK bias is handled by
        # the per-frame map prune in _track instead. On real Tello
        # video (natural texture) cornerSubPix might help; that is
        # unproven and must be re-tested on real footage, not assumed.
        return (pts_out,
                np.asarray(pts_prev, dtype=np.float32)[idx],
                idx)

    # -- motion-model LK prediction ----------------------------------------

    def _record_pose(self, R, C, frame=None):
        """Append an accepted pose to the motion-model history."""
        if frame is None:
            frame = self._n_frames
        self._pose_hist.append(
            (int(frame), np.asarray(R, dtype=float).copy(),
             np.asarray(C, dtype=float).copy()))
        del self._pose_hist[:-2]

    def _reset_pose_hist(self, R, C, frame=None):
        """Restart the motion-model history at one pose (after a jump)."""
        if frame is None:
            frame = self._n_frames
        self._pose_hist = [(int(frame), np.asarray(R, dtype=float).copy(),
                            np.asarray(C, dtype=float).copy())]

    def _predict_pose(self):
        """Constant-velocity pose prediction for the current frame.

        From the last two accepted poses (per-frame linear velocity
        for C, per-frame relative rotation for R — both normalised by
        the frame gap, so the keyframe->init gap is handled). Returns
        ``(R_pred, C_pred, R_rel_pred)`` in the internal frame, where
        ``R_rel_pred`` is the predicted relative rotation from the
        previous frame to the current one (for fresh-point prediction),
        or None when there is no usable history — the history must
        end at the immediately previous frame, so after a loss or a
        recovery jump there is no prediction until two consecutive
        poses are accepted again. The prediction only initialises the
        LK search (see LK_PREDICT_MIN_TRACKED); it is never itself a
        pose estimate.

        Tello state priors (report §2/§6.4) override the motion
        model where they are measurements, not assumptions: if a yaw
        prior is available at the current frame time, the predicted
        rotation comes from it (the onboard yaw measures exactly the
        failing degree of freedom, and it does not suffer the
        constant-velocity lag at the start of the turn); if a
        CALIBRATED velocity is available, the predicted translation
        comes from it instead of the constant-velocity extrapolation.
        Pitch/roll come from the prior too when available; otherwise
        the motion model fills them in.
        """
        if len(self._pose_hist) < 2:
            return None
        (f0, R0, C0), (f1, R1, C1) = self._pose_hist[-2], self._pose_hist[-1]
        f = self._n_frames
        if f1 != f - 1 or f <= f1:
            return None
        gap = float(f1 - f0)
        if gap < 1.0:
            return None
        dt = float(f - f1)  # normally 1
        # Translation: calibrated velocity prior beats the
        # constant-velocity extrapolation (it is a measurement of
        # the actual motion, not an assumption of steady motion).
        C_pred = None
        if self._state_prior is not None and self._t_cur is not None:
            vel = self._state_prior.velocity(self._t_cur)
            if vel is not None and self._t_prev is not None:
                dt_s = max(float(self._t_cur - self._t_prev), 1e-9)
                C_pred = C1 + np.asarray(vel, dtype=float) * dt_s
        if C_pred is None:
            C_pred = C1 + (C1 - C0) / gap * dt
        # Rotation: the yaw prior (if available) is the absolute
        # rotation at the current frame time — a measurement, not a
        # prediction. The motion model supplies the relative
        # rotation only when no prior is available.
        R_prior = None
        if self._state_prior is not None and self._t_cur is not None:
            R_prior = self._state_prior.R_prior(self._t_cur)
        if R_prior is not None:
            # StatePrior returns the GEOMETRY-convention R (Y-up);
            # the VIO works in the internal (Y-down) frame.
            R_pred = _to_internal_R(R_prior)
            R_rel_pred = R_pred @ R1.T
            return R_pred, C_pred, R_rel_pred
        dR = R1 @ R0.T
        rvec, _ = cv2.Rodrigues(dR)
        rvec = rvec.reshape(3) / gap * dt
        R_rel_pred, _ = cv2.Rodrigues(rvec)
        R_pred = R_rel_pred @ R1
        return R_pred, C_pred, R_rel_pred

    def _predict_pixels(self, pts_prev, pts3d, R_pred, C_pred, R_rel_pred,
                        gray_shape):
        """LK initial guesses for the current frame, from the prediction.

        Map points (have 3D) are projected at the predicted pose —
        exact up to the motion-model error. Fresh points (no 3D yet)
        are predicted by the relative rotation only (pure rotation
        needs no depth); that removes the yaw component of their
        flow, which is the dominant term during the turn. Any point
        whose prediction falls behind the camera or outside the frame
        keeps its previous pixel (zero-motion init, the old
        behaviour). Returns (N, 2) float32, aligned with pts_prev.
        """
        pts_prev = np.asarray(pts_prev, dtype=float)
        h, w = gray_shape
        init = pts_prev.copy()
        # Rotation-only prediction for every point (no depth needed).
        Kinv = np.linalg.inv(self._Kmat)
        pts_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1))])
        p_cam = (R_rel_pred @ (Kinv @ pts_h.T)).T
        z = p_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv_rot = np.stack([
                self._Kmat[0, 0] * p_cam[:, 0] / z + self._Kmat[0, 2],
                self._Kmat[1, 1] * p_cam[:, 1] / z + self._Kmat[1, 2]],
                axis=1)
        ok_rot = (z > 1e-6) & (uv_rot[:, 0] >= 0) & (uv_rot[:, 0] < w) \
            & (uv_rot[:, 1] >= 0) & (uv_rot[:, 1] < h)
        init[ok_rot] = uv_rot[ok_rot]
        # Map points: full projection at the predicted pose (better
        # than rotation-only — it also predicts the translation flow).
        if pts3d is not None:
            has3d = ~np.isnan(np.asarray(pts3d, dtype=float)).any(axis=1)
            if has3d.any():
                obj = np.asarray(pts3d, dtype=float)[has3d]
                Pc = (R_pred @ (obj - C_pred).T).T
                zz = Pc[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    uv = np.stack([
                        self._Kmat[0, 0] * Pc[:, 0] / zz + self._Kmat[0, 2],
                        self._Kmat[1, 1] * Pc[:, 1] / zz + self._Kmat[1, 2]],
                        axis=1)
                ok = (zz > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < w) \
                    & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                idx3d = np.nonzero(has3d)[0][ok]
                init[idx3d] = uv[ok]
        return init.astype(np.float32)

    # -- initialisation ------------------------------------------------------

    def _try_initialise(self, gray, diag):
        """Essential-matrix initialisation against the keyframe.

        Holds the known start pose (invalid) until the first pair with
        enough parallax + inliers triangulates and floor-scales.
        """
        tracked = self._track_lk(self._kf_gray, self._kf_pts, gray)
        diag["init_kf_age_frames"] = self._n_frames - (
            self._kf_frame if self._kf_frame is not None
            else self._n_frames)
        if tracked[0] is None:
            diag["n_tracked"] = 0
            diag["init_fail"] = "lk-empty"
            return self._external_pose(), False, diag
        pts_cur, pts_kf, _ = tracked
        diag["n_tracked"] = len(pts_cur)
        flow = np.linalg.norm(pts_cur - pts_kf, axis=1)
        diag["init_flow_px"] = float(np.median(flow)) if len(flow) else 0.0
        if len(pts_cur) < INIT_MIN_TRACKED:
            # Too few points survived LK from the keyframe — a
            # feature-poor view, or a keyframe too stale to track
            # from (live 2026-11-25: the keyframe was a pre-takeoff
            # ground frame). Keep holding the start pose.
            diag["init_fail"] = "few-tracked"
            return self._external_pose(), False, diag
        if diag["init_flow_px"] < INIT_MIN_DISPLACEMENT_PX:
            # Not enough parallax yet: keep holding the start pose.
            diag["init_fail"] = "low-displacement"
            return self._external_pose(), False, diag
        try:
            E, mask = cv2.findEssentialMat(
                pts_kf, pts_cur, self._Kmat, method=cv2.RANSAC,
                prob=0.999, threshold=1.0)
        except cv2.error:
            diag["init_fail"] = "essential-failed"
            return self._external_pose(), False, diag
        if E is None or mask is None:
            diag["init_fail"] = "essential-failed"
            return self._external_pose(), False, diag
        inliers = mask.reshape(-1) == 1
        n_inliers = int(inliers.sum())
        diag["inliers"] = n_inliers
        if n_inliers < INIT_MIN_INLIERS:
            diag["init_fail"] = "few-essential-inliers"
            return self._external_pose(), False, diag
        try:
            n_pose_inliers, R_rel, t_unit, mask_pose = cv2.recoverPose(
                E, pts_kf[inliers], pts_cur[inliers], self._Kmat)
        except cv2.error:
            diag["init_fail"] = "essential-failed"
            return self._external_pose(), False, diag
        # recoverPose's retval IS the inlier count; its mask is 0/255,
        # so never sum it as a count (that over-counts 255x).
        diag["init_pose_inliers"] = int(n_pose_inliers)
        if n_pose_inliers < INIT_MIN_INLIERS:
            diag["init_fail"] = "few-pose-inliers"
            return self._external_pose(), False, diag
        # Triangulate the inlier points with a UNIT baseline.
        R_kf, C_kf = self._kf_pose
        P1 = _proj_matrix(self._Kmat, np.eye(3), np.zeros(3))
        # NB: recoverPose's t is ALREADY the projection translation
        # (P2 = K [R | t]); _proj_matrix would negate it (it takes a
        # camera centre). Passing t as a centre mirrors every point
        # behind the cameras (measured 2026-11-23: all depths -20).
        P2 = self._Kmat @ np.hstack(
            [R_rel, np.asarray(t_unit, dtype=float).reshape(3, 1)])
        pts4d = cv2.triangulatePoints(
            P1, P2, pts_kf[inliers].T.astype(float),
            pts_cur[inliers].T.astype(float))
        pts_cam = (pts4d[:3] / pts4d[3]).T  # in keyframe camera frame
        # Cheirality: points must be in front of both cameras.
        z1 = pts_cam[:, 2]
        pts_cam2 = (R_rel @ pts_cam.T).T + t_unit.reshape(1, 3)
        z2 = pts_cam2[:, 2]
        ok = (z1 > 0) & (z2 > 0)
        if int(ok.sum()) < INIT_MIN_INLIERS:
            diag["init_fail"] = "few-cheirality"
            return self._external_pose(), False, diag
        # Floor scale: keyframe pixels in the lower 40% of the image
        # are floor candidates. Their reconstructed world height in
        # unit-baseline coordinates is y_unit; the true floor is y=0
        # and the camera is C_y above it, so s = C_y/(C_y - y_unit).
        # Only the NEARER half of the floor candidates is used:
        # triangulation error grows with depth squared, so far floor
        # points (seen at a shallow angle) bias the median — measured
        # 2026-11-23: all floor points gave a +3.4% scale error on
        # the smooth flight (0.454 m vs 0.439 m GT baseline) and
        # -2.8% on s1; the nearer half gives +1.2% / -1.9%. A 3.4%
        # scale error becomes ~14 cm of drift over a 4 m flight, so
        # this cut matters. Falls back to all floor points if the
        # near half would leave fewer than 8.
        H = int(self.K["height"])
        floor = pts_kf[inliers][ok][:, 1] > FLOOR_PIXEL_V_FRAC * H
        diag["init_floor_pts"] = int(floor.sum())
        if int(floor.sum()) < 8:
            diag["init_fail"] = "few-floor-points"
            return self._external_pose(), False, diag
        floor_idx = np.nonzero(floor)[0]
        unit_depth = np.linalg.norm(pts_cam[ok][floor], axis=1)
        near = unit_depth <= float(np.median(unit_depth))
        if int(near.sum()) >= 8:
            floor_idx = floor_idx[near]
        pts_world_unit = (R_kf.T @ pts_cam[ok][floor_idx].T).T + C_kf
        y_unit = float(np.median(pts_world_unit[:, 1]))
        diag["init_floor_y"] = y_unit
        if y_unit >= -0.05:  # floor must reconstruct below the camera
            diag["init_fail"] = "floor-not-below"
            return self._external_pose(), False, diag
        scale = float(C_kf[1]) / (float(C_kf[1]) - y_unit)
        if not (INIT_BASELINE_RANGE_M[0] <= scale <= INIT_BASELINE_RANGE_M[1]):
            diag["init_scale_candidate"] = scale
            diag["init_fail"] = "scale-out-of-range"
            return self._external_pose(), False, diag
        # Compose the absolute current pose.
        R_cur = R_rel @ R_kf
        C_cur = C_kf + R_kf.T @ (-R_rel.T @ t_unit.reshape(3) * scale)
        # Map points in the world frame (absolute scale).
        pts_world = (R_kf.T @ (pts_cam[ok] * scale).T).T + C_kf
        # Adopt: the current frame becomes the tracked frame; every
        # inlier point carries its 3D map point. Non-inlier tracked
        # points are dropped (they will be replenished).
        self._R, self._C = R_cur, C_cur
        self._initialized = True
        self._lost = False
        self._init_scale_m = scale
        # Motion-model history: keyframe pose + init pose (the gap
        # is normalised per-frame by _predict_pose).
        self._pose_hist = [
            (self._kf_frame, R_kf.copy(), C_kf.copy()),
            (self._n_frames, R_cur.copy(), C_cur.copy())]
        diag["init_scale_m"] = scale
        diag["init_fail"] = None
        self._last_init_fail = None
        self._prev_gray = gray
        self._prev_pts = pts_cur[inliers][ok]
        self._prev_pts3d = pts_world
        # The initialised points are all 3D map points: assign IDs.
        n_init = len(pts_world)
        self._prev_ids = np.arange(n_init, dtype=int)
        self._next_point_id = n_init
        self._prev_origin_pose = [(R_cur.copy(), C_cur.copy())] * len(pts_world)
        self._prev_origin_uv = pts_cur[inliers][ok].copy()
        self._good_gray, self._good_pts = gray, self._prev_pts.copy()
        self._good_pts3d = pts_world.copy()
        self._good_pose = (R_cur.copy(), C_cur.copy())
        # The initialisation frame is the first tracking keyframe.
        self._kf_track_pose = (R_cur.copy(), C_cur.copy())
        self._kf_track_frame = self._n_frames
        self._keyframes = [{
            "frame": self._n_frames, "R": R_cur.copy(), "C": C_cur.copy(),
            "ids": self._prev_ids.copy(),
            "uv": self._prev_pts.copy(), "pts3d": pts_world.copy(),
            "des": None, "des_valid": None}]
        log.info("VIO initialised at frame %d: E inliers=%d, pose "
                 "inliers=%d, tracked=%d, baseline scale %.3f m "
                 "(floor median y_unit %.2f)",
                 self._n_frames, n_inliers, n_pose_inliers,
                 len(pts_cur), scale, y_unit)
        return self._external_pose(), True, diag

    def _log_init_wait(self, diag):
        """One INFO line per init-failure transition (plus a repeat at
        most every 5 s for a persistent failure) — never per frame.
        The line carries the reason and the numbers, so the next live
        run's log says WHY init is waiting instead of the viewer
        just showing 'Tracking lost' (2026-11-25 black box)."""
        reason = diag.get("init_fail")
        if not reason:
            return
        now = time.monotonic()
        if (reason == self._last_init_fail
                and now - self._last_init_fail_t < 5.0):
            return
        self._last_init_fail = reason
        self._last_init_fail_t = now
        hints = {
            "lk-empty": "no points tracked from the keyframe at all",
            "few-tracked": "point the camera at a textured scene",
            "low-displacement": "move sideways (translation), not just turn",
            "essential-failed": "turning without translating gives no "
                               "parallax -- move sideways",
            "few-essential-inliers": "turning without translating gives "
                                     "no parallax -- move sideways",
            "few-pose-inliers": "turning without translating gives no "
                                "parallax -- move sideways",
            "few-cheirality": "too few points in front of both cameras "
                              "-- move sideways",
            "few-floor-points": "point the camera so the floor is "
                                "visible in the lower frame",
            "floor-not-below": "floor reconstructed above the camera -- "
                               "check level flight at the takeoff height",
            "scale-out-of-range": "baseline scale out of range -- check "
                                  "the camera height",
        }
        log.info("VIO init waiting: %s (tracked=%d/%d flow=%.1fpx "
                 "inliers=%d floor_pts=%s/8 kf_age=%sf) -- %s",
                 reason, diag.get("n_tracked", 0), INIT_MIN_TRACKED,
                 diag.get("init_flow_px", 0.0), diag.get("inliers", 0),
                 diag.get("init_floor_pts", "?"),
                 diag.get("init_kf_age_frames", "?"),
                 hints.get(reason, ""))

    # -- tracking ------------------------------------------------------------

    def _orb_descriptors(self, gray, pts):
        """ORB descriptors for ``pts`` (tracked points, (N, 2)).

        Detects ORB on ``gray`` and associates each tracked point
        with its nearest keypoint (within ORB_ASSOC_PX). Returns
        ``(des (N, 32) uint8, valid (N,) bool)``; rows without a
        nearby keypoint are invalid. Descriptors come from the
        frame's own ORB detection, so they are a fresh measurement
        of the same physical corner — not carried LK positions.
        """
        kp, des = self._orb.detectAndCompute(gray, None)
        n = len(pts)
        out = np.zeros((n, 32), dtype=np.uint8)
        valid = np.zeros(n, dtype=bool)
        if des is None or len(kp) == 0:
            return out, valid
        kxy = np.array([k.pt for k in kp], dtype=np.float32)
        for i, xy in enumerate(pts):
            d = np.linalg.norm(kxy - xy, axis=1)
            j = int(np.argmin(d))
            if d[j] <= ORB_ASSOC_PX:
                out[i] = des[j]
                valid[i] = True
        return out, valid

    def _orb_recover(self, gray, diag):
        """Recover the pose by ORB descriptor matching to the map.

        Matches the current frame's ORB against the keyframe
        database (most recent first) and then the last good frame,
        and runs PnP from the matched keyframe's pose. This is the
        recovery path for motions too large for LK (doorway / turn
        yaw snaps) and for relocalisation after a loss (report §4
        fix 4: a keyframe-descriptor database checked on loss, so a
        loss is a gap, not the end of the run — the recovered pose
        is in the OLD map's world frame, i.e. a restart-and-merge,
        not a new arbitrary frame). It never invents a pose — the
        PnP inlier gate is the trust signal, exactly as in _track.
        Returns ``(R, C, n_inliers, pts_cur, pts3d)`` (the matched
        2D/3D points, for resuming tracking) or None.
        """
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return None
        kxy = np.array([k.pt for k in kp], dtype=np.float32)
        # Candidates: keyframes (most recent first), then the last
        # good frame. The most recent keyframe is the most likely
        # re-observation; older keyframes cover the pre-turn view.
        candidates = []
        for kf in reversed(self._keyframes):
            if kf.get("des") is not None and kf.get("pts3d") is not None:
                candidates.append((kf["des"], kf["des_valid"],
                                   kf["pts3d"], (kf["R"], kf["C"]), "keyframe"))
        if (self._good_des is not None and self._good_pts3d is not None
                and self._good_pts is not None):
            candidates.append((self._good_des, self._good_des_valid,
                               self._good_pts3d, self._good_pose, "good"))
        for des_ref, valid_ref, pts3d_ref, pose_ref, kind in candidates:
            rec = self._orb_match_one(
                des, kxy, des_ref, valid_ref, pts3d_ref, pose_ref, diag)
            if rec is not None:
                diag["recovered_from"] = kind
                return rec
        return None

    def _orb_match_one(self, des_cur, kxy_cur, des_ref, valid_ref,
                       pts3d_ref, pose_ref, diag):
        """ORB-match the current frame against ONE reference frame.

        Returns ``(R, C, n_inliers, pts_cur, pts3d)`` or None.
        """
        has3d = ~np.isnan(np.asarray(pts3d_ref, dtype=float)).any(axis=1)
        use = has3d & np.asarray(valid_ref, dtype=bool)
        if int(use.sum()) < MIN_PNP_THIN_INLIERS:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            pairs = bf.knnMatch(des_ref[use], des_cur, k=2)
        except cv2.error:
            return None
        idx_ref = np.nonzero(use)[0]
        obj, img = [], []
        for m_n in pairs:
            if len(m_n) != 2:
                continue
            m, n_ = m_n
            if m.distance < ORB_RATIO * n_.distance:
                obj.append(pts3d_ref[idx_ref[m.queryIdx]])
                img.append(kxy_cur[m.trainIdx])
        if len(obj) < MIN_PNP_THIN_INLIERS:
            diag["orb_matches"] = len(obj)
            return None
        diag["orb_matches"] = len(obj)
        Rg, Cg = pose_ref
        sol = self._solve_pnp(np.asarray(obj), np.asarray(img),
                              Rg, Cg,
                              thresh_px=PNP_REPROJECTION_ERROR_PX,
                              diag=diag)
        if sol is None:
            return None
        R_new, C_new, n_inliers = sol
        if not _accept_pnp(n_inliers, len(obj)):
            return None
        return (R_new, C_new, n_inliers,
                np.asarray(img, dtype=np.float32),
                np.asarray(obj, dtype=float))

    def _adopt_recovered(self, pts_cur, pts3d, R_new, C_new, gray, diag):
        """Resume tracking from a recovered pose + matched points.

        Sets the previous/good frame state, replenishes fresh
        features so the next frame has enough points, and refreshes
        the ORB descriptors for the new good frame. Returns the
        ``(pose, True, diag)`` triple for ``update()``.
        """
        self._R, self._C = R_new, C_new
        self._lost = False
        self._reset_pose_hist(R_new, C_new)
        self._n_lost_frames = 0
        self._lost_frames = 0
        self._dr_pose = None
        self._dr_start_t = None
        self._dr_dist_m = 0.0
        pts_cur = np.asarray(pts_cur, dtype=np.float32)
        pts3d = np.asarray(pts3d, dtype=float)
        has3d = ~np.isnan(pts3d).any(axis=1)
        # Re-adopted 3D points get fresh IDs (the recovery breaks
        # the cross-frame ID chain; the BA treats them as new points
        # from here on, which is conservative and correct).
        ids = np.full(len(pts_cur), -1, dtype=int)
        n3d = int(has3d.sum())
        if n3d:
            ids[has3d] = np.arange(self._next_point_id,
                                   self._next_point_id + n3d, dtype=int)
            self._next_point_id += n3d
        if len(pts_cur) < REPLENISH_BELOW:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for (x, y) in pts_cur.astype(int):
                cv2.circle(mask, (int(x), int(y)), 12, 0, -1)
            fresh = self._detect(gray, mask=mask)
            if fresh is not None and len(fresh):
                diag["n_features"] = len(fresh)
                pts_cur = np.vstack([pts_cur, fresh])
                pts3d = np.vstack([
                    pts3d, np.full((len(fresh), 3), np.nan)])
                ids = np.concatenate([
                    ids, np.full(len(fresh), -1, dtype=int)])
        self._prev_gray = gray
        self._prev_pts = pts_cur
        self._prev_pts3d = pts3d
        self._prev_ids = ids
        self._prev_origin_pose = [
            (R_new.copy(), C_new.copy())] * len(pts_cur)
        self._prev_origin_uv = pts_cur.copy()
        self._good_gray, self._good_pts = gray, pts_cur.copy()
        self._good_pts3d = pts3d.copy()
        self._good_pose = (R_new.copy(), C_new.copy())
        self._good_des, self._good_des_valid = self._orb_descriptors(
            gray, pts_cur)
        return self._external_pose(), True, diag

    def _solve_pnp(self, obj, img, R_init, C_init,
                   thresh_px=PNP_REPROJECTION_ERROR_PX, diag=None):
        """PnP dispatch (round 3) with a fallback chain.

        The mode is a preference order, not a single solver: each
        candidate is tried in turn and the first whose inlier count
        passes ``_accept_pnp`` is returned. A constrained solve that
        fails the gate falls back to a less-constrained one instead
        of failing the frame — a wrong prior must never cost the
        frame, it only costs the constraint. Orders:
        "attitude": translation-only (full attitude from the prior)
          -> 1-DoF yaw -> full 6-DoF.
        "yaw1dof": 1-DoF yaw -> full 6-DoF.
        "prior": prior-augmented 6-DoF -> full 6-DoF.
        "full": full 6-DoF.
        When the mode needs the prior and none is available at this
        frame time, the chain starts at "full". Returns
        ``(R, C, n_inliers)`` (internal frame) or None. ``diag``
        records the solver that actually ran (``diag["pnp_solver"]``)
        and whether the prior was used. The callers still apply
        their own ``_accept_pnp`` gate to the returned pose, so this
        inner gate only selects between solvers, never the final
        acceptance.
        """
        mode = self._pnp_mode
        R_pr = None
        if mode != "full" and self._state_prior is not None \
                and self._t_cur is not None:
            R_pr = self._state_prior.R_prior(self._t_cur)
        n = len(obj)
        if diag is not None:
            diag["pnp_mode"] = mode
            diag["pnp_prior_used"] = R_pr is not None and mode != "full"

        def _try(sol, solver):
            if sol is not None and _accept_pnp(sol[2], n):
                if diag is not None:
                    diag["pnp_solver"] = solver
                return sol
            return None

        if R_pr is not None:
            if mode == "attitude":
                got = _try(_pnp_transonly(
                    obj, img, self._Kmat, R_pr, C_init,
                    thresh_px=thresh_px), "attitude")
                if got is not None:
                    return got
                got = _try(_pnp_yaw1dof(
                    obj, img, self._Kmat, R_pr, C_init,
                    thresh_px=thresh_px), "yaw1dof")
                if got is not None:
                    return got
            elif mode == "yaw1dof":
                got = _try(_pnp_yaw1dof(
                    obj, img, self._Kmat, R_pr, C_init,
                    thresh_px=thresh_px), "yaw1dof")
                if got is not None:
                    return got
            elif mode == "prior":
                got = _try(_pnp_ransac(
                    obj, img, self._Kmat, R_init, C_init,
                    thresh_px=thresh_px,
                    R_prior=_to_internal_R(R_pr),
                    prior_sigmas_deg=(self._yaw_prior_sigma_deg,
                                      self._pitch_prior_sigma_deg,
                                      self._roll_prior_sigma_deg)),
                    "prior")
                if got is not None:
                    return got
        sol = _pnp_ransac(obj, img, self._Kmat, R_init, C_init,
                          thresh_px=thresh_px)
        if diag is not None:
            diag["pnp_solver"] = "full"
        return sol

    def _track(self, gray, diag):
        """LK-track the previous frame, then PnP against the map.

        Pose path (in order): (1) PnP against the 3D map, initialised
        from the yaw prior when one is available (it measures the
        turn directly, so the LM starts in the right basin);
        (2) ROTATION-ONLY mode (homography over all bearings) when
        the map is too thin for PnP — the turn case, where bearings
        constrain rotation but not translation (report §1.2);
        (3) ORB descriptor recovery; (4) the essential-matrix
        fallback; (5) dead reckoning from the state prior (bounded);
        (6) loss — never a guessed pose. The returned pose is
        flagged in ``diag["pose_source"]`` (vision / attitude-aided
        / dead-reckoned) so the bridge knows what it is being fed.
        """
        init_pts = None
        pred = self._predict_pose()
        if pred is not None:
            R_pred, C_pred, R_rel_pred = pred
            init_pts = self._predict_pixels(
                self._prev_pts, self._prev_pts3d, R_pred, C_pred,
                R_rel_pred, gray.shape)
            diag["lk_predicted"] = True
        tracked = self._track_lk(self._prev_gray, self._prev_pts, gray,
                                 init_pts=init_pts)
        if (tracked[0] is None or len(tracked[0]) < LK_PREDICT_MIN_TRACKED) \
                and init_pts is not None:
            # The prediction hurt more than it helped (a motion
            # discontinuity — e.g. the yaw-rate step at the start of
            # the turn, where constant velocity underpredicts the new
            # rate): retry with plain zero-motion-init LK rather than
            # losing the frame on a bad guess.
            diag["lk_predicted"] = False
            diag["lk_predict_retry"] = True
            tracked = self._track_lk(self._prev_gray, self._prev_pts, gray)
        if tracked[0] is None:
            return self._declare_lost(gray, diag, reason="lk-empty")
        pts_cur, pts_prev, idx = tracked
        diag["n_tracked"] = len(pts_cur)
        pts3d = self._prev_pts3d[idx]
        origin_pose = [self._prev_origin_pose[i] for i in idx]
        origin_uv = self._prev_origin_uv[idx]
        ids = (self._prev_ids[idx] if self._prev_ids is not None
               else np.full(len(pts_cur), -1, dtype=int))
        has3d = ~np.isnan(pts3d).any(axis=1)
        diag["n_map"] = int(has3d.sum())

        # PnP initialisation: the yaw prior (if available) is a
        # measurement of the current rotation, so the LM starts
        # there instead of at the previous pose — during the turn
        # the previous pose is up to 6 deg behind, which is the
        # difference between the true basin and a wrong one.
        R_init, C_init = self._R, self._C
        if self._state_prior is not None and self._t_cur is not None:
            R_pr = self._state_prior.R_prior(self._t_cur)
            if R_pr is not None:
                R_init = _to_internal_R(R_pr)
                diag["pnp_prior"] = True
        pose = None
        pose_source = POSE_VISION
        n_inliers = 0
        if int(has3d.sum()) >= MIN_PNP_THIN_INLIERS:
            sol = self._solve_pnp(pts3d[has3d], pts_cur[has3d],
                                  R_init, C_init,
                                  thresh_px=PNP_REPROJECTION_ERROR_PX,
                                  diag=diag)
            if sol is not None:
                R_new, C_new, n_inliers = sol
                ratio = n_inliers / max(1, int(has3d.sum()))
                diag["inliers"] = n_inliers
                diag["pnp_ratio"] = float(ratio)
                if _accept_pnp(n_inliers, int(has3d.sum())):
                    if self._pose_plausible(R_new, C_new):
                        pose = (R_new, C_new)
        if pose is None:
            # PnP failed or was rejected. First the ROTATION-ONLY
            # mode: the bearings still constrain rotation even when
            # the map is too thin to triangulate (the turn). Then
            # ORB descriptor recovery, then the essential-matrix
            # fallback, then dead reckoning. If all fail, this frame
            # is a loss — never a guessed pose.
            rot = self._rotation_only(pts_prev, pts_cur, diag)
            if rot is not None:
                R_new, C_new, n_inliers = rot
                # NOTE: diag["inliers"] is left at the PnP count (0
                # here — no PnP ran) so the inlier statistic is not
                # polluted by mixing homography inliers (2D-2D, ~400)
                # with PnP inliers (3D-2D, ~20-100); the homography
                # count is in diag["homography_inliers"].
                diag["rotation_only"] = True
                pose = (R_new, C_new)
                pose_source = POSE_ATTITUDE_AIDED
        if pose is None:
            rec = self._orb_recover(gray, diag)
            if rec is not None:
                R_new, C_new, n_inliers, m_pts, m_3d = rec
                if self._pose_plausible(R_new, C_new):
                    diag["inliers"] = n_inliers
                    diag["recovered"] = "orb"
                    return self._adopt_recovered(
                        m_pts, m_3d, R_new, C_new, gray, diag)
            pose = self._essential_fallback(gray, pts_cur, pts3d, has3d, diag)
            if pose is None:
                pose = self._dead_reckon(diag)
                if pose is None:
                    return self._declare_lost(gray, diag, reason="pnp-failed")
                pose_source = POSE_DEAD_RECKONED
        R_new, C_new = pose
        diag["inliers"] = diag.get("inliers", 0)
        diag["pose_source"] = pose_source
        self._R, self._C = R_new, C_new
        self._record_pose(R_new, C_new)
        # A visual pose (PnP or rotation-only) clears any dead
        # reckoning: the chain is re-anchored on a measurement.
        if pose_source in (POSE_VISION, POSE_ATTITUDE_AIDED):
            self._dr_pose = None
            self._dr_start_t = None
            self._dr_dist_m = 0.0
        if pose_source == POSE_DEAD_RECKONED:
            # Dead reckoning: the pose is an integration, not a
            # visual measurement. Keep the LK state aligned for
            # continuity, but do NOT touch the map — no prune, no
            # triangulation, no replenish, no keyframe. Using an
            # integrated pose to triangulate would corrupt the map
            # with points whose baseline is a guess.
            self._prev_gray = gray
            self._prev_pts = pts_cur
            self._prev_pts3d = pts3d
            self._prev_ids = ids
            self._prev_origin_pose = list(origin_pose)
            self._prev_origin_uv = origin_uv
            diag["dead_reckoned"] = True
            return self._external_pose(), True, diag
        # Prune map points that no longer reproject (LK drift / bad
        # triangulation): a point whose 3D does not reproject within
        # 3 px at the accepted pose is dropped entirely — it would
        # otherwise drag every later PnP down with it (the inlier
        # ratio fell 0.97 -> 0.28 over frames 4-10 before this prune).
        # Fresh detection (below) replaces dropped points within a
        # frame or two, so the map stays clean and current. Bearings
        # (no 3D yet) are NEVER pruned here — they have no
        # reprojection to fail, and deleting them for failing a
        # triangulation gate is exactly the map drain the rotation
        # mode exists to prevent (report §1.2(g)).
        keep = np.ones(len(pts_cur), dtype=bool)
        if int(has3d.sum()) > 0:
            err = _reproj_err(R_new, C_new, pts3d[has3d], pts_cur[has3d],
                              self._Kmat)
            keep[np.nonzero(has3d)[0]] = err <= PNP_REPROJECTION_ERROR_PX
            diag["pruned"] = int((~keep).sum())
            if not keep.all():
                pts_cur = pts_cur[keep]
                pts3d = pts3d[keep]
                has3d = has3d[keep]
                ids = ids[keep]
                origin_pose = [origin_pose[i] for i in np.nonzero(keep)[0]]
                origin_uv = origin_uv[keep]

        # Deferred triangulation (report §1.2(a)/(b)): a bearing
        # becomes a 3D map point only once the PARALLAX ANGLE at the
        # point reaches PARALLAX_MIN_DEG between its origin pose and
        # the current pose, and the triangulated point reprojects
        # within TRI_REPROJ_MAX_PX. Below the gate the bearing is
        # KEPT (it still constrains rotation) and re-tried on every
        # later frame — it is never deleted for failing the gate.
        pts3d_new = pts3d.copy()
        for j in np.nonzero(~has3d)[0]:
            Ro, Co = origin_pose[j]
            ang = self._parallax_angle_deg(
                Ro, Co, origin_uv[j], R_new, C_new, pts_cur[j])
            if ang < PARALLAX_MIN_DEG:
                continue
            p = self._triangulate_point(Ro, Co, origin_uv[j],
                                        R_new, C_new, pts_cur[j])
            if p is None:
                continue
            # Reprojection check at birth: a point triangulated at
            # barely-5-deg parallax can still be noisy.
            Pc = R_new @ (p - C_new)
            if Pc[2] <= 0:
                continue
            uv = np.array([self._Kmat[0, 0] * Pc[0] / Pc[2]
                           + self._Kmat[0, 2],
                           self._Kmat[1, 1] * Pc[1] / Pc[2]
                           + self._Kmat[1, 2]])
            if float(np.linalg.norm(uv - pts_cur[j])) > TRI_REPROJ_MAX_PX:
                continue
            pts3d_new[j] = p
            ids[j] = self._next_point_id
            self._next_point_id += 1
        # Replenish features if the tracked set is thinning.
        if len(pts_cur) < REPLENISH_BELOW:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for (x, y) in pts_cur.astype(int):
                cv2.circle(mask, (int(x), int(y)), 12, 0, -1)
            fresh = self._detect(gray, mask=mask)
            if fresh is not None and len(fresh):
                diag["n_features"] = len(fresh)
                pts_cur = np.vstack([pts_cur, fresh])
                pts3d_new = np.vstack([
                    pts3d_new,
                    np.full((len(fresh), 3), np.nan)])
                ids = np.concatenate([
                    ids, np.full(len(fresh), -1, dtype=int)])
                origin_pose = list(origin_pose) + [
                    (R_new.copy(), C_new.copy())] * len(fresh)
                origin_uv = np.vstack([origin_uv, fresh])
        self._prev_gray = gray
        self._prev_pts = pts_cur
        self._prev_pts3d = pts3d_new
        self._prev_ids = ids
        self._prev_origin_pose = list(origin_pose)
        self._prev_origin_uv = origin_uv
        self._good_gray, self._good_pts = gray, pts_cur.copy()
        self._good_pts3d = pts3d_new.copy()
        self._good_pose = (R_new.copy(), C_new.copy())
        self._good_des, self._good_des_valid = self._orb_descriptors(
            gray, pts_cur)
        # Tracking keyframe (report §1.2(f)): snapshot for the
        # windowed BA and the relocalisation database.
        self._maybe_keyframe(gray, pts_cur, pts3d_new, ids, diag)
        return self._external_pose(), True, diag

    def _essential_fallback(self, gray, pts_cur, pts3d, has3d, diag):
        """Relative motion prev->cur, scaled against the map.

        Returns a pose (R, C) or None. The scale comes from the map:
        triangulated points (unit baseline) are rescaled so their
        distances from the previous camera centre match the map
        points' distances. No map points -> no fallback.
        """
        if int(has3d.sum()) < self.min_pnp_inliers:
            return None
        try:
            E, mask = cv2.findEssentialMat(
                self._prev_pts, pts_cur, self._Kmat, method=cv2.RANSAC,
                prob=0.999, threshold=1.0)
        except cv2.error:
            return None
        if E is None or mask is None:
            return None
        inl = mask.reshape(-1) == 1
        if int(inl.sum()) < self.min_pnp_inliers:
            return None
        try:
            _, R_rel, t_unit, _ = cv2.recoverPose(
                E, self._prev_pts[inl], pts_cur[inl], self._Kmat)
        except cv2.error:
            return None
        # Triangulate the map points' correspondences with unit baseline.
        P1 = _proj_matrix(self._Kmat, np.eye(3), np.zeros(3))
        # Same NB as in _try_initialise: t is the projection
        # translation, not a camera centre.
        P2 = self._Kmat @ np.hstack(
            [R_rel, np.asarray(t_unit, dtype=float).reshape(3, 1)])
        a = self._prev_pts[inl][has3d[inl]]
        b = pts_cur[inl][has3d[inl]]
        if len(a) < self.min_pnp_inliers:
            return None
        pts4d = cv2.triangulatePoints(P1, P2, a.T.astype(float),
                                      b.T.astype(float))
        pts_unit = (pts4d[:3] / pts4d[3]).T
        ok = pts_unit[:, 2] > 0
        if int(ok.sum()) < self.min_pnp_inliers:
            return None
        d_unit = np.linalg.norm(pts_unit[ok], axis=1)
        d_map = np.linalg.norm(pts3d[inl][has3d[inl]][ok] - self._C, axis=1)
        good = (d_unit > 1e-6) & (d_map > 0)
        if int(good.sum()) < self.min_pnp_inliers:
            return None
        scale = float(np.median(d_map[good] / d_unit[good]))
        if not (0.02 <= scale <= 3.0):
            return None
        R_new = R_rel @ self._R
        C_new = self._C + self._R.T @ (-R_rel.T @ t_unit.reshape(3) * scale)
        diag["fallback"] = "essential"
        diag["fallback_scale_m"] = scale
        if not self._pose_plausible(R_new, C_new):
            return None
        return R_new, C_new

    def _triangulate_point(self, R1, C1, uv1, R2, C2, uv2):
        """Triangulate one point from two absolute poses; None if bad.

        NB (measured 2026-11-23): ``cv2.triangulatePoints`` returns
        the point in the WORLD frame of the projection matrices —
        NOT in the first camera's frame (that is only true when
        P1 = K[I|0], as in _try_initialise). An earlier version of
        this function applied ``R1.T @ p_cam + C1`` to the result,
        double-transforming it (it mirrored every new map point in Y:
        a wall point at y = 1.8 m came back at y = -0.5 m, and the
        map corrupted within 3 frames of initialisation — GT
        reprojection error 0.3 px at f03 growing to 216 px by f10).
        The fix is to use the returned point as a world point
        directly, checking cheirality by projecting it back into
        both cameras.
        """
        P1 = _proj_matrix(self._Kmat, R1, C1)
        P2 = _proj_matrix(self._Kmat, R2, C2)
        pts4d = cv2.triangulatePoints(
            P1, P2, np.asarray(uv1, dtype=float).reshape(2, 1),
            np.asarray(uv2, dtype=float).reshape(2, 1))
        w = float(pts4d[3, 0])
        if abs(w) < 1e-9:
            return None
        p_world = pts4d[:3, 0] / w
        p_cam1 = np.asarray(R1, dtype=float) @ (
            p_world - np.asarray(C1, dtype=float))
        if p_cam1[2] <= 0:
            return None
        p_cam2 = np.asarray(R2, dtype=float) @ (
            p_world - np.asarray(C2, dtype=float))
        if p_cam2[2] <= 0:
            return None
        return p_world

    def _parallax_angle_deg(self, R1, C1, uv1, R2, C2, uv2):
        """Parallax angle (deg) subtended at a point by two views.

        Provisionally triangulates the point from (R1,C1,uv1) and
        (R2,C2,uv2); the angle is the angle at the point between the
        two camera centres. Returns 0.0 if the point cannot be
        triangulated (degenerate / behind a camera) — i.e. it has no
        usable parallax yet and stays a bearing (d=0). This is the
        triangulation gate (PARALLAX_MIN_DEG): it is depth-aware,
        unlike the old absolute-baseline gate it replaces.
        """
        p = self._triangulate_point(R1, C1, uv1, R2, C2, uv2)
        if p is None:
            return 0.0
        a = np.asarray(C1, dtype=float) - p
        b = np.asarray(C2, dtype=float) - p
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _rotation_only(self, pts_prev, pts_cur, diag):
        """Rotation-only pose from 2D bearing correspondences.

        GRANITE-style rotation tracking (report §1.2(a)/(c)): when
        the 3D map is too thin for PnP (rotation-dominant motion —
        the turn), estimate the relative rotation from the bearings
        alone via a homography (pure rotation is H = K R K^-1; the
        essential matrix is degenerate when t = 0, so it cannot be
        used here). Translation comes from the motion-model
        prediction (constant velocity from the pose history) — NOT
        frozen, because freezing assumes zero velocity and the drone
        is still translating during the hover sway — or from the
        calibrated velocity prior when one is available (a real
        measurement). The bearings constrain rotation only, and
        pretending they constrain translation would be a faked pose.
        Returns ``(R_new, C_new, n_inliers)`` (internal frame) or
        None. The caller flags the pose as rotation-only /
        attitude-aided, never as a full visual PnP.
        """
        if pts_prev is None or pts_cur is None:
            return None
        pts_prev = np.asarray(pts_prev, dtype=float)
        pts_cur = np.asarray(pts_cur, dtype=float)
        if len(pts_prev) < 8 or len(pts_cur) < 8:
            return None
        try:
            H, mask = cv2.findHomography(
                pts_prev, pts_cur, cv2.RANSAC, 3.0)
        except cv2.error:
            return None
        if H is None or mask is None:
            return None
        inl = mask.reshape(-1) == 1
        n_inl = int(inl.sum())
        diag["homography_inliers"] = n_inl
        if n_inl < 8:
            return None
        try:
            _retval, Rs, _ts, _ns = cv2.decomposeHomographyMat(
                H, self._Kmat)
        except cv2.error:
            return None
        if Rs is None or len(Rs) == 0:
            return None
        # Pick the candidate rotation closest to the predicted
        # relative rotation (motion model, else the yaw prior, else
        # identity). decomposeHomographyMat returns up to 4
        # solutions; continuity with the prediction is what
        # disambiguates them (cheirality is already enforced by the
        # decomposition's normal check, and a wrong solution would
        # be a large rotation jump, which the jump guard rejects).
        pred = self._predict_pose()
        if pred is not None:
            R_rel_pred = pred[2]
        else:
            R_rel_pred = None
        best = None
        best_err = None
        for Rc in Rs:
            Rc = np.asarray(Rc, dtype=float)
            if R_rel_pred is not None:
                dR = Rc @ R_rel_pred.T
                err = float(np.arccos(np.clip(
                    (np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
            else:
                # No prediction: prefer the smallest rotation (the
                # turn is smooth, so the true relative rotation is
                # the small one among the 4 candidates).
                err = float(np.arccos(np.clip(
                    (np.trace(Rc) - 1.0) / 2.0, -1.0, 1.0)))
            if best is None or err < best_err:
                best, best_err = Rc, err
        if best is None:
            return None
        R_new = best @ self._R
        # Translation: the motion-model prediction (constant velocity
        # from the pose history), NOT frozen — freezing assumes zero
        # velocity, which is wrong whenever the drone is actually
        # translating (the hover sway is 0.08 m/frame; freezing it
        # accumulates a large error while the rotation-only mode
        # carries the pose). The motion model is a prediction, so
        # its error is bounded by the (small) hover acceleration.
        # A calibrated velocity prior overrides it (a measurement).
        C_new = self._C.copy()
        pred = self._predict_pose()
        if pred is not None:
            C_new = pred[1]
        vel = None
        if getattr(self, "_state_prior", None) is not None:
            vel = self._state_prior.velocity(self._t_cur)
        if vel is not None and self._t_prev is not None:
            dt = max(float(self._t_cur - self._t_prev), 1e-9)
            C_new = self._C + np.asarray(vel, dtype=float) * dt
            diag["rotation_only_velocity"] = True
        if not self._pose_plausible(R_new, C_new):
            return None
        return R_new, C_new, n_inl

    def _dead_reckon(self, diag):
        """Integrate the state prior while vision is down (bounded).

        Report §5.1 / §6.4: when vision is lost, integrate yaw (+
        velocity, if calibrated) + h/tof from the last good pose.
        The rotation comes from the yaw prior (a real sensor), the
        translation from the calibrated velocity (or frozen, if
        uncalibrated), and z from h/tof (direct measurements). This
        keeps the marker PRESENT with a growing uncertainty — it is
        not a faked pose — but it is bounded: if vision does not
        return within DEAD_RECKON_MAX_S or the integrated distance
        exceeds DEAD_RECKON_MAX_M, this returns None and the VIO
        invalidates. The caller flags the pose DEAD-RECKONED.
        Returns ``(R_new, C_new)`` (internal frame) or None.
        """
        if self._state_prior is None or self._t_cur is None:
            return None
        R_prior = self._state_prior.R_prior(self._t_cur)
        if R_prior is None:
            return None
        if self._dr_pose is None:
            # First dead-reckoned frame: anchor on the last good pose.
            self._dr_pose = (self._R.copy(), self._C.copy())
            self._dr_start_t = self._t_cur
            self._dr_dist_m = 0.0
        age = float(self._t_cur - self._dr_start_t)
        if age > DEAD_RECKON_MAX_S:
            return None
        if self._dr_dist_m > DEAD_RECKON_MAX_M:
            return None
        R_new = _to_internal_R(R_prior)
        C_new = self._C.copy()
        vel = self._state_prior.velocity(self._t_cur)
        if vel is not None and self._t_prev is not None:
            dt = max(float(self._t_cur - self._t_prev), 1e-9)
            step = np.asarray(vel, dtype=float) * dt
            C_new = self._C + step
            self._dr_dist_m += float(np.linalg.norm(step))
            if self._dr_dist_m > DEAD_RECKON_MAX_M:
                return None
        # Height from h/tof (direct measurements): z is the best-
        # measured axis, so take it from the prior rather than
        # holding the last visual z.
        h_m, _tof_m = self._state_prior.height(self._t_cur)
        if h_m is not None:
            C_new[1] = float(h_m)
        if not self._pose_plausible(R_new, C_new):
            return None
        diag["dead_reckon_age_s"] = age
        diag["dead_reckon_dist_m"] = self._dr_dist_m
        return R_new, C_new

    def _maybe_keyframe(self, gray, pts_cur, pts3d, ids, diag):
        """Tracking-keyframe rule + snapshot (report §1.2(f)).

        A new keyframe is set when the rotation since the last
        keyframe exceeds KEYFRAME_ROT_DEG, or KEYFRAME_MAX_GAP
        frames have passed, or the 3D map thins below
        KEYFRAME_MIN_MAP. (The report's negative-entropy rule is
        the ideal; this parallax + time rule fires on the same
        events — a turn, a long straight, or a thinning map — and
        is what the GRANITE ablation shows matters: keyframing on
        the right events, not every frame.) A keyframe snapshots
        the pose + the observed map points (by ID) for the windowed
        local BA and the relocalisation database, and triggers the
        BA over the keyframe window.
        """
        if self._kf_track_pose is None:
            self._kf_track_pose = (self._R.copy(), self._C.copy())
            self._kf_track_frame = self._n_frames
            return
        Rk, _Ck = self._kf_track_pose
        dR = self._R @ Rk.T
        rot_deg = float(np.degrees(np.arccos(np.clip(
            (np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
        gap = self._n_frames - (self._kf_track_frame or 0)
        n_map = int((~np.isnan(np.asarray(pts3d, dtype=float)).any(axis=1)).sum())
        if gap < KEYFRAME_MIN_GAP:
            return
        if not (rot_deg >= KEYFRAME_ROT_DEG or gap >= KEYFRAME_MAX_GAP
                or n_map < KEYFRAME_MIN_MAP):
            return
        # Snapshot this keyframe (descriptors for relocalisation).
        des, des_valid = self._orb_descriptors(gray, pts_cur)
        self._keyframes.append({
            "frame": self._n_frames, "t": self._t_cur,
            "R": self._R.copy(), "C": self._C.copy(),
            "ids": np.asarray(ids, dtype=int).copy(),
            "uv": np.asarray(pts_cur, dtype=float).copy(),
            "pts3d": np.asarray(pts3d, dtype=float).copy(),
            "des": des, "des_valid": des_valid})
        del self._keyframes[:-10]
        self._kf_track_pose = (self._R.copy(), self._C.copy())
        self._kf_track_frame = self._n_frames
        diag["keyframe"] = True
        self._local_ba(diag)

    def _local_ba(self, diag):
        """Windowed local BA over the keyframe window (report §4 fix 4).

        Refines the MAP POINT positions (not the poses — see below)
        to minimise the Huber reprojection error across all
        keyframes that observe each point. This is what fixes the
        ~11 px map reprojection error: without BA the map is never
        refined, so triangulation noise compounds monotonically
        through the turn and the PnP is dragged by it. Poses are
        held fixed here — refining points given the current poses
        makes the map self-consistent for PnP, which is the stated
        error; a full pose+point BA (which would also reduce pose
        drift) is a follow-up, and holding the poses fixed keeps
        this step from ever moving the reported pose. Runs only on
        keyframe insertion (not every frame) and is wrapped so a
        failure never breaks tracking.
        """
        if len(self._keyframes) < 2:
            return
        try:
            from scipy.optimize import least_squares
            from scipy.sparse import lil_matrix
        except ImportError:
            return
        window = self._keyframes[-BA_WINDOW_KEYFRAMES:]
        # Collect point IDs observed in >= 2 keyframes (a point seen
        # once has no BA constraint — its position is already its
        # triangulation, and refining it alone is meaningless).
        from collections import Counter
        counts = Counter()
        for kf in window:
            for pid in kf["ids"]:
                if pid >= 0:
                    counts[int(pid)] += 1
        pids = sorted(pid for pid, c in counts.items() if c >= 2)
        if len(pids) > BA_MAX_POINTS:
            pids = pids[-BA_MAX_POINTS:]
        if not pids:
            return
        pid_index = {pid: j for j, pid in enumerate(pids)}
        # Initial point positions from the newest keyframe that has them.
        P0 = np.full((len(pids), 3), np.nan)
        for kf in reversed(window):
            for pid, p in zip(kf["ids"], kf["pts3d"]):
                j = pid_index.get(int(pid))
                if j is not None and np.isnan(P0[j]).any() \
                        and not np.isnan(p).any():
                    P0[j] = p
        ok = ~np.isnan(P0).any(axis=1)
        if int(ok.sum()) < 5:
            return
        pids = [p for p, o in zip(pids, ok) if o]
        P0 = P0[ok]
        pid_index = {pid: j for j, pid in enumerate(pids)}
        # Observations: (keyframe, point, pixel).
        obs_kf, obs_pt, obs_uv = [], [], []
        for ki, kf in enumerate(window):
            for pid, uv in zip(kf["ids"], kf["uv"]):
                j = pid_index.get(int(pid))
                if j is not None:
                    obs_kf.append(ki)
                    obs_pt.append(j)
                    obs_uv.append(uv)
        if len(obs_uv) < 10:
            return
        obs_kf = np.array(obs_kf)
        obs_pt = np.array(obs_pt)
        obs_uv = np.array(obs_uv, dtype=float)
        Rs = [kf["R"] for kf in window]
        Cs = [kf["C"] for kf in window]
        Kmat = self._Kmat
        fx, fy, cx, cy = Kmat[0, 0], Kmat[1, 1], Kmat[0, 2], Kmat[1, 2]

        def residuals_v(x):
            P = x.reshape(-1, 3)
            Pc_all = np.empty((len(obs_uv), 3))
            for ki in range(len(window)):
                m = obs_kf == ki
                if m.any():
                    Pc_all[m] = (Rs[ki] @ (P[obs_pt[m]] - Cs[ki]).T).T
            z = np.maximum(Pc_all[:, 2], 0.05)
            uv = np.stack([fx * Pc_all[:, 0] / z + cx,
                           fy * Pc_all[:, 1] / z + cy], axis=1)
            return (uv - obs_uv).reshape(-1)
        # Sparsity: each residual pair depends only on its point.
        n_res = 2 * len(obs_uv)
        n_par = 3 * len(pids)
        sp = lil_matrix((n_res, n_par), dtype=int)
        for i, j in enumerate(obs_pt):
            sp[2 * i:2 * i + 2, 3 * j:3 * j + 3] = 1
        t0 = time.perf_counter()
        r0 = residuals_v(P0.reshape(-1))
        rms0 = float(np.sqrt(np.mean(r0 ** 2)))
        try:
            sol = least_squares(
                residuals_v, P0.reshape(-1), method="trf",
                loss="huber", f_scale=3.0, jac_sparsity=sp,
                x_scale="jac", max_nfev=30)
        except Exception:
            return
        rms1 = float(np.sqrt(np.mean(residuals_v(sol.x) ** 2)))
        # Accept only if the BA actually reduced the reprojection
        # error; otherwise keep the triangulated points as they were.
        if rms1 >= rms0:
            return
        P_new = sol.x.reshape(-1, 3)
        # Apply the refined positions to the live map (by ID) and
        # to the keyframe snapshots. The reported pose is NOT moved
        # (poses were held fixed in this BA).
        refined = {pid: P_new[j] for j, pid in enumerate(pids)}
        if self._prev_ids is not None:
            for i, pid in enumerate(self._prev_ids):
                if int(pid) in refined:
                    self._prev_pts3d[i] = refined[int(pid)]
        for kf in self._keyframes:
            for i, pid in enumerate(kf["ids"]):
                if int(pid) in refined:
                    kf["pts3d"][i] = refined[int(pid)]
        diag["ba_ms"] = (time.perf_counter() - t0) * 1000.0
        diag["ba_points"] = len(pids)
        diag["ba_obs"] = len(obs_uv)
        diag["ba_rms_before"] = rms0
        diag["ba_rms_after"] = rms1

    def _yaw_correct(self, R_new, diag):
        """Replace the yaw of a PnP rotation with the state prior's
        yaw (report §2/§6.4: the yaw prior "collapses PnP to 1-DoF
        yaw"). The visual PnP supplies pitch/roll + translation;
        the prior supplies the yaw — the degree of freedom that
        fails in the turn (measured yaw error 1.0-1.2 deg mean,
        4.0-4.5 deg max, vs the prior's ~0.5 deg quantisation
        error). The correction is a rotation about the world Y axis
        by the yaw delta (no Euler decomposition, so no gimbal
        ambiguity), applied in the external frame. The pitch/roll
        and the translation are untouched, so this is a yaw-only
        correction, not a prior pose.

        CURRENTLY UNUSED (2026-11-23, measured): applying it
        post-hoc in _track improved the translation drift a lot
        (s1 9.8/35.2 cm -> 4.1/7.7 cm mean/max) but WORSENED the
        joint error (s23 81.7 -> 111.2 cm, s4 55.2 -> 64.0 cm),
        because changing R without re-optimising C breaks the R-C
        consistency the depth triangulation relies on. A
        translation-only C re-optimisation after the correction
        diverged (the near-planar floor map has a yaw/translation
        degeneracy — the re-opt slid along it, drift 29.6/62.4 cm).
        The correct fix is to constrain the PnP itself (a 1-DoF
        yaw PnP, or a pose-graph with the prior as a factor), not a
        post-hoc correction — that is follow-up work, not done
        here. This method is kept (it is correct in isolation) for
        that follow-up.
        """
        if self._state_prior is None or self._t_cur is None:
            return R_new
        att = self._state_prior.attitude(self._t_cur)
        if att is None:
            return R_new
        yaw_prior = float(att[0])
        R_ext = _to_external_R(R_new)
        yaw_cur = float(yaw_pitch_roll_from_R(R_ext)[0])
        delta = (yaw_prior - yaw_cur + 180.0) % 360.0 - 180.0
        if abs(delta) < 0.5:
            return R_new
        th = math.radians(delta)
        c, s = math.cos(th), math.sin(th)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        R_corr = _to_internal_R(Ry @ R_ext)
        diag["yaw_corrected_deg"] = float(delta)
        return R_corr

    def _pose_plausible(self, R_new, C_new):
        """Jump guard + level-flight height guard (PnP failure signatures)."""
        if float(np.linalg.norm(C_new - self._C)) > MAX_JUMP_M:
            return False
        lo, hi = CAM_HEIGHT_RANGE_M
        if not (lo <= float(C_new[1]) <= hi):
            return False
        return True

    def _declare_lost(self, gray, diag, reason=""):
        """Mark tracking lost: keep the map + last good frame for
        relocalisation, report invalid, never return a guessed pose."""
        self._lost = True
        self._n_lost_frames += 1
        diag["lost_reason"] = reason
        diag["n_lost_frames"] = self._n_lost_frames
        # The current frame still becomes the "previous" frame for LK
        # purposes only if relocalisation can anchor on it — it cannot
        # (no pose), so keep _prev_* as they were and let
        # _try_relocalise anchor on the last GOOD frame instead.
        log.warning("VIO tracking lost at frame %d (%s): tracked=%d "
                    "inliers=%d map=%d; bridge will invalidate "
                    "(no guessed pose)", self._n_frames, reason,
                    diag.get("n_tracked", 0), diag.get("inliers", 0),
                    diag.get("n_map", 0))
        return self._external_pose(), False, diag

    # -- relocalisation --------------------------------------------------------

    def _try_relocalise(self, gray, diag):
        """LK-track the last good frame's map points into this frame + PnP.

        Succeeds only on a genuine re-observation of the mapped area.
        A re-initialisation from scratch is deliberately NOT attempted
        here (it would teleport the drone into a new world frame).
        Dead reckoning is deliberately NOT attempted here: once the
        VIO has declared lost the bridge is already invalidated, and
        a dead-reckoned pose would trigger a bridge reacquire (which
        restarts the uncertainty clock — wrong for an integrated
        pose, whose uncertainty only grows). Dead reckoning lives in
        _track, where it bridges brief visual gaps BEFORE the bridge
        ever invalidates (report §5.1).
        """
        if self._good_pts is None or self._good_pts3d is None:
            return self._external_pose(), False, diag
        tracked = self._track_lk(self._good_gray, self._good_pts, gray)
        if tracked[0] is None:
            diag["n_tracked"] = 0
            return self._external_pose(), False, diag
        pts_cur, _pts_prev, idx = tracked
        diag["n_tracked"] = len(pts_cur)
        pts3d = self._good_pts3d[idx]
        has3d = ~np.isnan(pts3d).any(axis=1)
        diag["n_map"] = int(has3d.sum())
        if int(has3d.sum()) < MIN_PNP_THIN_INLIERS:
            return self._external_pose(), False, diag
        Rg, Cg = self._good_pose
        sol = self._solve_pnp(pts3d[has3d], pts_cur[has3d],
                              Rg, Cg,
                              thresh_px=PNP_REPROJECTION_ERROR_PX,
                              diag=diag)
        if sol is None:
            rec = self._orb_recover(gray, diag)
            if rec is not None:
                R_new, C_new, n_inliers, m_pts, m_3d = rec
                lo, hi = CAM_HEIGHT_RANGE_M
                if lo <= float(C_new[1]) <= hi:
                    diag["inliers"] = n_inliers
                    diag["relocalised"] = "orb"
                    diag["pose_source"] = POSE_VISION
                    return self._adopt_recovered(
                        m_pts, m_3d, R_new, C_new, gray, diag)
            return self._external_pose(), False, diag
        R_new, C_new, n_inliers = sol
        ratio = n_inliers / max(1, int(has3d.sum()))
        diag["inliers"] = n_inliers
        diag["pnp_ratio"] = float(ratio)
        if not _accept_pnp(n_inliers, int(has3d.sum())):
            return self._external_pose(), False, diag
        # A relocalised pose may legitimately be far from the last
        # estimate (the drone kept moving while lost), so the jump
        # guard is relaxed to height only here — the PnP inlier gate
        # is the trust signal.
        lo, hi = CAM_HEIGHT_RANGE_M
        if not (lo <= float(C_new[1]) <= hi):
            return self._external_pose(), False, diag
        self._R, self._C = R_new, C_new
        self._lost = False
        self._n_lost_frames = 0
        self._lost_frames = 0
        self._dr_pose = None
        self._dr_start_t = None
        self._dr_dist_m = 0.0
        self._reset_pose_hist(R_new, C_new)
        diag["relocalised"] = True
        # Resume tracking from this frame.
        self._prev_gray = gray
        self._prev_pts = pts_cur
        self._prev_pts3d = pts3d
        has3d = ~np.isnan(pts3d).any(axis=1)
        ids = np.full(len(pts_cur), -1, dtype=int)
        n3d = int(has3d.sum())
        if n3d:
            ids[has3d] = np.arange(self._next_point_id,
                                   self._next_point_id + n3d, dtype=int)
            self._next_point_id += n3d
        self._prev_ids = ids
        self._prev_origin_pose = [
            (R_new.copy(), C_new.copy())] * len(pts_cur)
        self._prev_origin_uv = pts_cur.copy()
        self._good_gray, self._good_pts = gray, pts_cur.copy()
        self._good_pts3d = pts3d.copy()
        self._good_pose = (R_new.copy(), C_new.copy())
        log.info("VIO relocalised at frame %d: %d PnP inliers",
                 self._n_frames, n_inliers)
        return self._external_pose(), True, diag
