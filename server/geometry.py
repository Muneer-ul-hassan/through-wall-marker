"""Camera models, projection, transforms, and the COCO skeleton.

Implements the exact contract in SPEC.md (CORRECTED 2026-11-23).
Conventions (read before use):

* World/shared frame W: origin at the phone's camera centre at session
  start. +X = right from the phone's view, +Y = up, +Z = forward toward
  the wall. Phone position (0, 1.5, 0), looking +Z.
* A camera pose is a pair (R, C): R is the world-to-camera rotation and
  C is the camera centre in world coordinates. Projection is
  ``p_cam = R @ (P - C)``; ``u = fx * x/z + cx``, ``v = fy * y/z + cy``.
  (The standard translation vector is ``t = -R @ C``; see
  ``make_transform_from_rt``.)
* A 3D point projects to nothing when it is behind the camera (z <= 0).
* "Drone frame" below means the drone *camera* frame. The drone pose
  (R, C) is the drone camera's pose in the world; the world-from-drone
  transform ``T_world_drone`` is the inverse of the world-to-camera
  transform ``[R | -R@C]``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Camera models (SPEC.md; approximate models, documented as such)
# ---------------------------------------------------------------------------

# Tello camera, approximate.
DRONE_K = {"fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 360.0,
           "width": 960, "height": 720}

# Phone camera (virtual model), approximate.
PHONE_K = {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0,
           "width": 1280, "height": 720}

# Phone pose: rotation identity, camera centre (0, 1.5, 0) in world
# coordinates — exactly as SPEC.md (corrected) states it.
PHONE_POSE = (np.eye(3), np.array([0.0, 1.5, 0.0]))

# Phone camera centre in the world frame (used for distance).
PHONE_POSITION = np.array([0.0, 1.5, 0.0])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_point(K, R, C, P):
    """Project world point ``P`` into a camera with pose (R, C), intrinsics K.

    ``p_cam = R @ (P - C)``. Returns ``(u, v, depth)`` with ``depth`` the
    z-coordinate in camera frame, or ``None`` when the point is behind
    the camera (z <= 0).
    """
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    p_cam = R @ (P - C)
    z = float(p_cam[2])
    if z <= 0:
        return None
    u = float(K["fx"]) * float(p_cam[0]) / z + float(K["cx"])
    # Image rows are top-origin; camera +Y is up, so v = cy - fy*y/z.
    v = float(K["cy"]) - float(K["fy"]) * float(p_cam[1]) / z
    return (u, v, z)


def unproject_ray(K, u, v):
    """Unit ray in the camera frame through pixel (u, v).

    Camera frame is y-up, so the y component is (cy - v)/fy.
    """
    x = (float(u) - float(K["cx"])) / float(K["fx"])
    y = (float(K["cy"]) - float(v)) / float(K["fy"])
    ray = np.array([x, y, 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


# ---------------------------------------------------------------------------
# Transforms (4x4 homogeneous)
# ---------------------------------------------------------------------------

def make_transform(R, C):
    """World-to-camera 4x4 transform from rotation R and centre C: [R | -R@C]."""
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def make_transform_from_rt(R, t):
    """Standard 4x4 transform from rotation R and translation t: [R | t]."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def invert_transform(T):
    """Inverse of a 4x4 rigid transform."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def apply_transform(T, P):
    """Apply a 4x4 transform to a point (3,) or an array of points (N, 3)."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        return (T @ np.append(P, 1.0))[:3]
    ones = np.ones((P.shape[0], 1), dtype=float)
    return (T @ np.hstack([P, ones]).T).T[:, :3]


# ---------------------------------------------------------------------------
# COCO skeleton (17 keypoints, YOLO-Pose order)
# ---------------------------------------------------------------------------

COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

# Joint heights as fractions of person height H (SPEC.md).
_JOINT_HEIGHT_FRAC = {
    0: 0.94, 1: 0.93, 2: 0.93, 3: 0.93, 4: 0.93,
    5: 0.86, 6: 0.86, 7: 0.63, 8: 0.63, 9: 0.44, 10: 0.44,
    11: 0.53, 12: 0.53, 13: 0.29, 14: 0.29, 15: 0.05, 16: 0.05,
}

# Half-widths in metres at H = 1.7 m (SPEC.md). Scaled by H / 1.7.
_HALF_WIDTH_M_AT_1_7 = {
    5: 0.21, 6: 0.21, 7: 0.28, 8: 0.28, 9: 0.30, 10: 0.30,
    11: 0.10, 12: 0.10, 13: 0.09, 14: 0.09, 15: 0.09, 16: 0.09,
}

# Eyes / ears have no half-width in SPEC.md; small values scaled the same way.
_EYE_HALF_WIDTH_M_AT_1_7 = 0.035
_EAR_HALF_WIDTH_M_AT_1_7 = 0.075


def skeleton_local_joints(H):
    """Ground-truth skeleton in person-local coordinates, shape (17, 3).

    Person height H, standing, facing the drone. Local +x = person's left,
    +y = up, +z = facing direction. All joints lie in the z = 0 plane
    (SPEC.md gives heights and half-widths only, no forward offsets).
    """
    scale = float(H) / 1.7
    joints = np.zeros((17, 3), dtype=float)
    for i in range(17):
        joints[i, 1] = _JOINT_HEIGHT_FRAC[i] * float(H)
    # Left joints (odd indices) get +x, right joints (even) get -x.
    for i, w in _HALF_WIDTH_M_AT_1_7.items():
        joints[i, 0] = w * scale if i % 2 == 1 else -w * scale
    joints[1, 0] = _EYE_HALF_WIDTH_M_AT_1_7 * scale    # left eye
    joints[2, 0] = -_EYE_HALF_WIDTH_M_AT_1_7 * scale   # right eye
    joints[3, 0] = _EAR_HALF_WIDTH_M_AT_1_7 * scale    # left ear
    joints[4, 0] = -_EAR_HALF_WIDTH_M_AT_1_7 * scale   # right ear
    # nose (0): x = 0, z = 0.
    return joints
