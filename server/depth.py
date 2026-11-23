"""Person depth from a drone view: bbox-height fallback and motion
triangulation.

Two methods (SPEC.md):

* ``bbox_height`` — ``Z = fx * H / h`` from the detection bbox height.
  Approximate: it assumes the person is H metres tall, upright, and
  fully visible, and that the bbox height equals the projected body
  height. Error is unvalidated in the literature; the tests measure it.
  sigma = 0.25 * depth.
* ``triangulation`` — the primary method whenever the drone has moved
  >= 0.3 m between views (>= 2 views): triangulate the feet ray from
  the drone poses. The ray is the bbox bottom-centre for a full person,
  or the visible-ankle ray for a partly occluded person (see
  ``partial_foot_uv`` — the visible part's bbox is not a stable world
  point). Same Z^2 error law as stereo, with the baseline being the
  drone's own motion. sigma = max(0.15, triangulation residual). A
  moving person breaks the static-scene assumption; the residual is the
  honest signal for that (it grows when the rays do not meet).

``estimate_person_depth`` never returns a depth without a method + sigma.

Third voter (2026-11-23, server/mono_depth.py): a monocular depth
model (Depth Anything V2 Small, relative depth anchored to the floor
plane at the known camera height — never a hard-coded scale) votes
in the cross-check via ``fuse_with_model_votes``. It arbitrates when
triangulation and bbox-height disagree, and stays out entirely when
they agree (its measured error is larger than triangulation's, so
fusing it there only adds noise — measured in s15_occluded). It
NEVER replaces triangulation as the primary method, and when no
model vote is available the behaviour is exactly the legacy
two-method one.

Joint 3D positions (``joints_3d``) use a planar approximation: every
joint's ray is intersected with the plane perpendicular to the drone
optical axis at ``depth_m``. That is honest only when the person is
roughly fronto-parallel to the drone; a person turned sideways, or
joints at very different depths (an outstretched arm), will be placed
on the wrong plane. The tests measure the resulting joint error.
Joints with confidence < 0.3 are NaN — the caller must not invent them.
"""

from __future__ import annotations

import numpy as np

from .geometry import DRONE_K, unproject_ray


def partial_foot_uv(keypoints, kpt_conf, min_missing=3):
    """Foot pixel for a PARTIALLY visible person, or None.

    When the detector is missing >= ``min_missing`` joints (conf < 0.3),
    the person is partly occluded and the detection bbox is the VISIBLE
    part's box, not the full person's — its bottom-centre is laterally
    offset and its height is short, so both depth methods (which derive
    from the same bbox) share a common-mode bias the cross-check cannot
    catch (measured in s15: triangulation 3.11 m and bbox-height 2.86 m
    agreed, yet the true depth was 2.51 m). The visible ANKLE keypoints
    are a stable world point that does not move with the occlusion, so
    return their mean pixel when at least one ankle is visible. When
    fewer than ``min_missing`` joints are missing the bbox is reliable
    and this returns None (the caller then uses the bbox bottom-centre,
    which is the behaviour the original 7 scenarios were measured with).
    Measured separation: full-person frames miss 0-1 joints, occluded
    frames (s15) miss 3-9, so the gate never fires on a full person.
    """
    kpt_conf = np.asarray(kpt_conf, dtype=float)
    keypoints = np.asarray(keypoints, dtype=float)
    n_missing = int((kpt_conf < 0.3).sum())
    if n_missing < min_missing:
        return None
    ankles = [i for i in (15, 16) if float(kpt_conf[i]) >= 0.3]
    if not ankles:
        return None
    uv = np.mean([keypoints[i] for i in ankles], axis=0)
    return (float(uv[0]), float(uv[1]))


def bbox_height_depth(fx, bbox_h_px, person_h_m=1.7):
    """Depth from bbox height: Z = fx * H / h. Approximate (see module doc)."""
    h = float(bbox_h_px)
    if h <= 0:
        raise ValueError("bbox height must be positive")
    return float(fx) * float(person_h_m) / h


def triangulate_point(origins, rays):
    """Least-squares intersection of >= 2 rays (DLT).

    ``origins``: list of (3,) ray origins (world). ``rays``: list of (3,)
    unit direction vectors (world). Returns ``(point (3,), residual_m)``
    where the residual is the RMS perpendicular distance from the point
    to the rays.
    """
    origins = [np.asarray(o, dtype=float) for o in origins]
    rays = [np.asarray(r, dtype=float) for r in rays]
    if len(origins) < 2 or len(origins) != len(rays):
        raise ValueError("triangulate_point needs >= 2 (origin, ray) pairs")
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for o, r in zip(origins, rays):
        r = r / np.linalg.norm(r)
        P = np.eye(3) - np.outer(r, r)  # projector onto plane perp. to ray
        A += P
        b += P @ o
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        point = np.linalg.lstsq(A, b, rcond=None)[0]
    sq = []
    for o, r in zip(origins, rays):
        r = r / np.linalg.norm(r)
        d = point - o
        perp = d - np.dot(d, r) * r
        sq.append(float(np.dot(perp, perp)))
    residual = float(np.sqrt(np.mean(sq)))
    return point, residual


def _rel_diff(a, b):
    """|a - b| / b (relative disagreement, b > 0)."""
    return abs(float(a) - float(b)) / float(b)


def fuse_with_model_votes(tri, bbox, model):
    """Third-voter fusion (monocular depth model).

    ``tri`` / ``bbox`` / ``model`` are ``(depth, sigma)`` or ``None``.
    Returns ``(depth, method, sigma, voter_changed)`` where ``method``
    is still ``"triangulation"`` or ``"bbox_height"`` — the model
    NEVER becomes the primary method; it votes (see the rules below).
    ``voter_changed`` is True only when the model changed the outcome
    relative to the two-method rule in :func:`estimate_person_depth`.

    Rules (principled + measured — the numbers are in
    tests/TEST-REPORT.md, "Depth-model tests"). The model is a PURE
    ARBITRATOR: it changes the CHOICE between triangulation and
    bbox-height, never the NUMBER. Fusing its depth into the winner
    was tried first and measured to hurt: in s15_occluded it took
    joint error 59.3 -> 64.5 cm (agree-case fusion) and 59.3 ->
    64.3 cm (disagree-case refinement), because the occluder biases
    the model vote the same way it biases the bbox, so "refining"
    the winner with the model just adds the model's error to it.
    With pure arbitration the mono runs reproduce the baseline
    exactly unless the voter flips a choice — and every flip is
    counted in ``voter_changed``.
    * Triangulation and bbox-height AGREE (<= 25% apart): the answer
      is triangulation alone. There is nothing to vote on, and the
      model's measured error is larger than triangulation's.
    * They DISAGREE (> 25%): the static-scene assumption is broken
      (moving person / occlusion). The model arbitrates: if it agrees
      (<= 25%) with triangulation only, triangulation wins (the legacy
      rule would have trusted bbox-height — outcome changed,
      ``voter_changed=True``); if it agrees with bbox-height only,
      bbox-height wins (same outcome as legacy); if it agrees with
      neither (or both — possible when it sits between them), the
      legacy bbox-height fallback stands. The winner's depth is
      returned EXACTLY — no fusion with the model.
    * No triangulation (baseline < 0.3 m): bbox-height alone. The
      model does not refine the only available method: in the
      occluded scene its vote shares the occlusion bias, and in
      clean scenes the first 1-3 detections barely affect the
      markers (a track needs 3 confirmations, by which time
      triangulation usually exists).
    """
    if model is None:
        raise ValueError("fuse_with_model_votes needs a model vote")
    m_depth, m_sigma = float(model[0]), float(model[1])
    if tri is not None and bbox is not None:
        t_depth, t_sigma = float(tri[0]), float(tri[1])
        b_depth, b_sigma = float(bbox[0]), float(bbox[1])
        if _rel_diff(t_depth, b_depth) <= 0.25:
            # Agree: nothing to vote on. The model stays out of the
            # fusion entirely (see the docstring: fusing its votes
            # here cost s15_occluded 59.3 -> 64.5 cm joint error
            # through small systematic shifts). Triangulation alone.
            return t_depth, "triangulation", t_sigma, False
        # Disagreement: the model arbitrates — pure arbitration (see
        # the docstring): the winner's depth is returned EXACTLY,
        # never fused with the model.
        close_tri = _rel_diff(m_depth, t_depth) <= 0.25
        close_bbox = _rel_diff(m_depth, b_depth) <= 0.25
        if close_tri and not close_bbox:
            return t_depth, "triangulation", t_sigma, True
        if close_bbox and not close_tri:
            return b_depth, "bbox_height", b_sigma, False
        return b_depth, "bbox_height", b_sigma, False
    # No triangulation: bbox-height alone (pure arbitration — see the
    # docstring). The model never refines the only available method.
    if bbox is None:
        raise ValueError("fuse_with_model_votes needs at least the "
                         "bbox-height vote — the model alone never stands")
    b_depth, b_sigma = float(bbox[0]), float(bbox[1])
    return b_depth, "bbox_height", b_sigma, False


# Detector bbox calibration (measured 2026-11-23, /tmp/bbox_probe.py).
#
# The YOLOv8n-pose bbox is systematically SHORTER than the person's
# full projected height: on the harness renderer (person 1.7 m tall,
# scenarios s1/s23/s20/s4, all in-view detected frames) the implied
# height H = true_depth * bbox_h_px / fx is 1.487-1.533 m
# (mean 1.51 m, within-scenario std 0.006-0.013 m) — the bbox spans
# ~0.89 of the person's height, not 1.00. The bbox itself is precise
# (std < 1%); the error is a stable BIAS, not noise, so temporal
# smoothing cannot fix it. Using H = 1.7 m in the bbox-height vote
# overestimates depth by ~13% (measured s4: +26.5 cm mean, std 1.7 cm
# — all 10 marker frames biased the same way, which is why s4 sat at
# 59.4 cm, 0.6 cm from the 60 cm budget). The keypoint-anchored
# alternatives were measured and REJECTED the same day: the ankle
# keypoint is noisier than the bbox bottom (implied-H std 0.017-0.022
# vs 0.006-0.013), so the anchored and nose-to-ankle heights give
# LARGER depth errors (0.35-0.53 m) than the plain bbox with the
# corrected H.
#
# BBOX_HEIGHT_FRACTION is the fraction of the person's height the
# detector's bbox spans. It is a DETECTOR calibration (YOLOv8n-pose's
# tight-box regression), not a harness-person tuning: the same
# detector on a real person insets the same way. Residual
# per-scenario variation is +/-0.02 m in implied H (~+/-1.3% in
# depth, ~3-4 cm at 2.5 m) — lighting/pose dependent, and inside the
# bbox vote's sigma (0.25 * depth). On a real person whose height is
# unknown the fraction is unvalidated; the sigma covers it, but the
# mean bias is corrected here. If the detector is swapped, re-measure
# the fraction (the probe is one script run).
BBOX_HEIGHT_FRACTION = 0.89


def estimate_person_depth_detailed(history, fx, K=None, person_h_m=1.7,
                                   model_depth=None, model_sigma=None):
    """Like :func:`estimate_person_depth` but with the model-vote detail.

    ``model_depth`` / ``model_sigma`` are the floor-anchored monocular
    model vote for the CURRENT frame (from server/mono_depth.py), or
    ``None`` when no vote is available — in which case this is exactly
    the legacy two-method behaviour. Returns a dict with ``depth_m``,
    ``method``, ``sigma_m``, ``voter_changed`` (bool: the model
    changed the outcome vs the two-method rule), ``tri_depth_m``,
    ``bbox_depth_m`` and ``model_depth_m`` (the three votes, or None).
    """
    if not history:
        raise ValueError("estimate_person_depth needs at least one view")
    K = dict(DRONE_K) if K is None else K
    R_last, C_last, bbox_last = history[-1][0][0], history[-1][0][1], history[-1][1]
    centres = [np.asarray(pose[1], dtype=float) for pose, *_ in history]
    baseline = 0.0
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            baseline = max(baseline, float(np.linalg.norm(centres[i] - centres[j])))

    def _bbox_depth(bbox):
        h_px = float(bbox[3]) - float(bbox[1])
        # Calibrated height (see BBOX_HEIGHT_FRACTION): the detector's
        # bbox spans ~0.89 of the person's height, so the raw
        # bbox_height_depth(H = person_h_m) overestimates depth by
        # ~13%. The effective height is person_h_m * the fraction.
        return bbox_height_depth(fx, h_px,
                                 person_h_m * BBOX_HEIGHT_FRACTION)

    def _foot_uv(entry):
        if len(entry) >= 4:
            return entry[3]
        return None

    bbox_depth = _bbox_depth(bbox_last)
    bbox_vote = (bbox_depth, 0.25 * bbox_depth)
    tri_vote = None
    if len(history) >= 2 and baseline >= 0.3:
        origins, rays = [], []
        for entry in history:
            (R, C), bbox, _t = entry[0], entry[1], entry[2]
            R = np.asarray(R, dtype=float)
            C = np.asarray(C, dtype=float)
            uv = _foot_uv(entry)
            if uv is not None:
                u, v = uv
            else:
                u = 0.5 * (float(bbox[0]) + float(bbox[2]))
                v = float(bbox[3])
            ray_cam = unproject_ray(K, u, v)
            ray_world = R.T @ ray_cam
            origins.append(C)
            rays.append(ray_world)
        point, residual = triangulate_point(origins, rays)
        p_cam = np.asarray(R_last, dtype=float) @ (
            point - np.asarray(C_last, dtype=float))
        depth = float(p_cam[2])
        if depth > 0:
            tri_vote = (depth, max(0.15, residual))

    model_vote = None
    if model_depth is not None and model_sigma is not None:
        md, ms = float(model_depth), float(model_sigma)
        if np.isfinite(md) and np.isfinite(ms) and md > 0 and ms > 0:
            model_vote = (md, ms)

    if model_vote is not None:
        depth, method, sigma, changed = fuse_with_model_votes(
            tri_vote, bbox_vote, model_vote)
        return {"depth_m": depth, "method": method, "sigma_m": sigma,
                "voter_changed": changed,
                "tri_depth_m": tri_vote[0] if tri_vote else None,
                "bbox_depth_m": bbox_depth,
                "model_depth_m": model_vote[0]}

    # Legacy two-method behaviour (unchanged — the regression
    # comparison in tests/run_tests.py depends on it reproducing the
    # recorded 2026-11-23 numbers exactly).
    if tri_vote is not None:
        depth, _sig = tri_vote
        if abs(depth - bbox_depth) / bbox_depth > 0.25:
            return {"depth_m": bbox_depth, "method": "bbox_height",
                    "sigma_m": 0.25 * bbox_depth, "voter_changed": False,
                    "tri_depth_m": depth, "bbox_depth_m": bbox_depth,
                    "model_depth_m": None}
        return {"depth_m": depth, "method": "triangulation",
                "sigma_m": _sig, "voter_changed": False,
                "tri_depth_m": depth, "bbox_depth_m": bbox_depth,
                "model_depth_m": None}
    return {"depth_m": bbox_depth, "method": "bbox_height",
            "sigma_m": 0.25 * bbox_depth, "voter_changed": False,
            "tri_depth_m": None, "bbox_depth_m": bbox_depth,
            "model_depth_m": None}


def estimate_person_depth(history, fx, K=None, person_h_m=1.7,
                          model_depth=None, model_sigma=None):
    """Estimate person depth from a history of drone views.

    ``history``: list of ``(drone_pose (R, C), bbox_xyxy, t, foot_uv)``
    with C the drone camera centre in world coordinates; ``foot_uv`` is
    the foot pixel from :func:`partial_foot_uv` (or None — older callers
    may pass 3-tuples, which are treated as ``foot_uv=None``). ``K``
    defaults to DRONE_K (only ``fx`` is required by the SPEC signature;
    cx/cy come from K).

    Returns ``(depth_m, method, sigma_m)``. ``depth_m`` is the
    z-coordinate of the person in the *latest* drone camera frame — the
    plane ``joints_3d`` intersects against. Triangulation is used when
    the drone baseline (max distance between drone centres in history)
    is >= 0.3 m and there are >= 2 views; otherwise bbox-height.
    ``model_depth`` / ``model_sigma`` (optional) are the floor-anchored
    monocular model vote for the current frame — see
    :func:`estimate_person_depth_detailed` and
    ``fuse_with_model_votes``; when given, the model votes, never
    replaces triangulation as the primary method. When omitted, this
    is exactly the legacy two-method behaviour.
    """
    d = estimate_person_depth_detailed(
        history, fx, K=K, person_h_m=person_h_m,
        model_depth=model_depth, model_sigma=model_sigma)
    return d["depth_m"], d["method"], d["sigma_m"]


def joints_3d(kpts_px, kpt_conf, K, drone_pose, depth_m):
    """Joint positions in the drone CAMERA frame, shape (17, 3).

    ``drone_pose`` is (R, C) with C the drone camera centre in world; it
    is accepted per the SPEC contract and validated, but the ray-plane
    intersection itself is in the camera frame, so the pose is not
    needed for the math — convert the result to world coordinates with
    ``bridge.world_from_drone(T_world_drone, ...)``.

    For each joint, the ray through (u, v) is intersected with the plane
    perpendicular to the drone optical axis at ``depth_m`` (planar
    approximation — see module docstring). Joints with ``kpt_conf < 0.3``
    (or rays pointing backwards) are NaN, never invented.
    """
    R, C = drone_pose
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    if R.shape != (3, 3) or C.shape != (3,):
        raise ValueError("drone_pose must be (R (3,3), C (3,))")
    kpts_px = np.asarray(kpts_px, dtype=float)
    kpt_conf = np.asarray(kpt_conf, dtype=float)
    if kpts_px.shape != (17, 2) or kpt_conf.shape != (17,):
        raise ValueError("expected 17 keypoints and 17 confidences")
    out = np.full((17, 3), np.nan, dtype=float)
    for i in range(17):
        if float(kpt_conf[i]) < 0.3:
            continue
        ray = unproject_ray(K, float(kpts_px[i, 0]), float(kpts_px[i, 1]))
        if ray[2] <= 1e-9:  # ray parallel to / behind the image plane
            continue
        out[i] = ray * (float(depth_m) / ray[2])
    return out


def foot_point(bbox_xyxy, K, drone_pose, depth_m, foot_uv=None):
    """World position of the bbox bottom-centre at ``depth_m``.

    The bbox bottom-centre ray is the feet ray (it is the same ray the
    triangulation in :func:`estimate_person_depth` uses), so placing it
    on the plane at ``depth_m`` reconstructs the person's ground-contact
    point. Use this as the track position: the median of the skeleton
    joints sits ~1 m above the floor and biases the displayed distance
    ~0.5 m short (measured in the virtual tests, 2026-11-23). Returns
    ``None`` when the ray is parallel to the image plane.

    ``foot_uv``: the foot pixel from :func:`partial_foot_uv` (visible
    ankles of a partly occluded person). The ankle ray points at the
    ankle, ~8 cm above the ground, so the result is projected straight
    down to y = 0 — the ground-contact point under the ankle, which is
    the same definition as the bbox bottom-centre for a full person.
    """
    R, C = drone_pose
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    if foot_uv is not None:
        u, v = float(foot_uv[0]), float(foot_uv[1])
    else:
        x1, _y1, x2, y2 = [float(v) for v in bbox_xyxy]
        u, v = (x1 + x2) / 2.0, y2
    ray = unproject_ray(K, u, v)
    if ray[2] <= 1e-9:
        return None
    point_cam = ray * (float(depth_m) / ray[2])
    point = R.T @ point_cam + C
    if foot_uv is not None:
        point = np.array([point[0], 0.0, point[2]], dtype=float)
    return point
