"""Virtual test scenarios (SPEC.md + extensions, 2026-11-23).

Original 7 scenarios (60 frames @ 10 fps each) — UNCHANGED, with the
exact definitions from the first build:

* s1_basic: person 4.0 m from phone, centred, light 1.0.
* s2_offcentre: person 5.5 m, off to the side near the doorway.
* s3_close: person just behind the wall (z = 3.3 m).
* s4_dim: light 0.5, person 4.5 m.
* s5_tracking_loss: tracking_valid True until frame 35, False after.
  Person stays visible to the drone the whole time; the marker must
  vanish from frame 36 on and never return.
* s6_no_person: room B empty, same flight path. Zero markers, ever.
* s7_two_drones: two drones, different hover points, both see the
  person. Fused marker asserted; fused error must be <= best
  single-drone error.

Extended scenarios (added 2026-11-23, same 60 frames @ 10 fps):

NEW ROOMS / PLACES (different geometries, known ground truth):
* s8_wall_close: wall z = 2.0 m, doorway off-centre (left), person
  2.5 m (z) from the phone, just behind the wall.
* s9_wall_far: wall z = 4.0 m, doorway off-centre (right), person
  5.5 m (z) from the phone.
* s10_phone_offcentre: phone NOT at the origin — phone at
  (1.2, 1.5, 0.5); person 4.5 m (z) away, off to the other side.
* s11_large_room_far: larger room B (x in [-4, 4], z to 12 m),
  person 7.0 m (z) from the phone.
* s12_bright: brightness 1.2x, person 4.5 m.
* s24_empty_new_room: empty room B in a NEW geometry (wall z = 2.5 m,
  doorway off-centre right, room B to z = 9 m). Zero markers, ever.

ADVERSARIAL (hard cases — the suite is meant to catch mistakes):
* s13_dim_noise: light 0.4x with REAL per-pixel Gaussian noise
  (sigma 12), not just a brightness factor.
* s14_glare: bright window on the back wall directly behind the
  person + bloom — backlit person.
* s15_occluded: person partly hidden behind a wardrobe (occluder
  quad between drone and person, ~40% of the body covered).
* s16_sideways: person turned sideways (fixed yaw 90 deg to the
  viewing direction), billboard narrowed to match. Documented
  approximation: a real sideways body shows a profile; the harness
  narrows the frontal billboard, so this is a lower bound on real
  sideways difficulty.
* s17_edge: person at the very edge of the drone frame (billboard
  right edge ~30 px from the frame edge); the drone looks at a fixed
  point, not at the person.
* s18_moving: person walking slowly (0.25 m/s across, 0.12 m/s
  toward the wall) — triangulation assumes a static scene, so the
  numbers here are the honest measure of how bad it gets.
* s19_two_people: TWO people in room B, ~1.9 m apart. The system
  must mark a real person and never merge them into a midpoint
  (false) marker.
* s20_fast: fast drone — fast approach and fast, wide hover
  oscillation (amp 0.9 m, period 1.2 s).
* s21_loss_recovery: tracking loss frames 25-40, recovery from
  frame 41. The marker must vanish on loss and only return after
  the bridge is re-established + a fresh detection — never a
  guessed position.
* s22_clutter: cluttered room B (furniture boxes + extra posters).
* s23_blank_door: featureless blank wall at the doorway (no posters
  on the shared wall) — bridge stress. NOTE: the virtual bridge is
  fed true poses, so no VIO feature tracking is actually simulated;
  this scenario proves detection/rendering with a blank wall, and the
  real VIO bridge stress remains unproven (see TEST-REPORT.md).

MULTI-DRONE (4 and 5 drones, inverse-variance fusion as before):
* s25_four_drones: 4 drones, different hover points, all see the
  person; fused error must be <= best single-drone error.
* s26_five_drones: 5 drones, same check.
* s27_four_drones_partial: 4 drones, 2 of them lose sight of the
  person (they look at the back corners) — fusion must still work
  from the remaining 2 drones.

Drone trajectory (all scenarios): start beside the phone, fly to the
doorway, through it into room B, hover with small lateral motion so
triangulation has a baseline. Drone yaw faces along motion, then
faces the person (or a fixed look target) while hovering. The person
stands on the floor (y = 0), height 1.7 m; the ground-truth skeleton
is ``geometry.skeleton_local_joints`` rotated by the person's yaw
and translated to the person position.

Every frame carries ground truth in ``meta`` (computed from the
scenario definition — never from the detector output):
``person_world``, ``gt_skeleton_world`` (17,3), ``gt_distance_m``
(phone -> person, Euclidean), ``gt_skeleton_phone_px`` (17,2),
``in_view`` (whole person inside the drone frame and >= 80 px tall),
``height_px``, ``tracking_valid``, ``drone_id`` — plus, for the
extended scenarios, ``persons_world`` / ``persons_gt`` (per-person
lists), ``phone_position``, ``in_view_list``.

``get_frames(name, drone_id=...)`` returns the 60 frame tuples
``(frame_bgr, drone_pose (R, C), tracking_valid, meta)`` — the shape
``server.video_source.SyntheticSource`` and ``server.main`` expect.
``iter_frames`` is the same as a generator (lower memory for the runner).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from server.geometry import (DRONE_K, PHONE_K, PHONE_POSE, PHONE_POSITION,
                             project_point, skeleton_local_joints)
from tests.render import RoomRenderer, load_person_cutout, look_at_R

log = logging.getLogger(__name__)

FPS = 10.0
N_FRAMES = 60
PERSON_H = 1.7
# Smoothed-look VIO variants (2026-11-23 fix round): the harness look-at
# logic snaps the camera yaw/pitch in a single 0.1 s frame — measured:
# -31 deg at the doorway (f13->14) and +101 deg at the turn to face the
# person (f19->20), plus a ~38 deg pitch snap at the turn and a fast
# bearing sweep while the drone flies past the person's z. A real Tello
# yaws ~30-60 deg/s (1-2 deg/frame at 30 fps), so those snaps are a
# harness artifact, not a VIO failure. The VIO scenarios therefore run
# on SMOOTHED-LOOK variants (``smooth_look=True`` in ``iter_frames`` /
# ``get_frames``): the look direction's yaw and pitch follow the desired
# look-at direction through a critically-damped spring (SmoothDamp) —
# rate capped at VIO_MAX_YAW_RATE_DEG_S (6 deg/frame at 10 fps) with a
# smooth ramp, so the yaw/pitch trajectories are C1 (continuous rate),
# the way a real yaw controller flies them. A first-order (rate-only)
# limiter was tried first and REJECTED: it steps the yaw rate 0 -> 6
# deg/frame in a single frame when the turn starts, and the
# constant-velocity motion model in server/vio.py cannot predict a rate
# step — measured prediction error at f19->f20 was 19 cm + 6.7 deg
# (87 px median), and LK failed outright (3-5 tracks). A trapezoidal
# profiler was tried second and REJECTED too: its arrival snap
# (``rate = gap``) violates its own acceleration cap (measured rate
# changes up to 9.3 deg/frame^2). The spring ramps the rate smoothly
# and settles exactly, with no snap and no limit cycle. (Gaussian
# position smoothing was tried and REJECTED too: edge-replicated
# smoothing damps the takeoff so much that initialisation is delayed
# past the keyframe's usable window — E inliers 25-32 at f04-06 vs
# 69 at f03 unsmoothed — and VIO never initialises. Positions stay
# as the harness trajectory.) The 27 GT-pose scenarios are
# byte-identical — only the VIO runs (``tests/run_tests.py:
# run_vio_scenario``) and the ``--vio`` serve path use the smoothed
# variant.
VIO_MAX_YAW_RATE_DEG_S = 60.0
VIO_MAX_PITCH_RATE_DEG_S = 60.0
# Spring time constant: 0.4 s. The rate ramps 0 -> 60 deg/s in ~0.4 s
# (angular acceleration ~150 deg/s^2, a real Tello yaw acceleration),
# and a 102 deg turn completes in ~2.2 s (22 frames at 10 fps).
VIO_SMOOTH_TIME_S = 0.4
# VIO variants only: the frame at which the look target switches to
# the person (the turn to face them starts). The harness switches at
# frame 20; the VIO variant delays it to 25 (see person_frame in
# iter_frames) so the doorway slew settles before the turn starts.
VIO_TURN_FRAME = 25
# VIO variants only: re-timed flight phases (fly_end, enter_end,
# approach_end). The harness phases (14, 19, 24) make the approach
# leg a 2.24 m dash in 5 frames (4.5 m/s average, 6.7 m/s peak) whose
# per-frame displacement jumps 0.11 -> 0.23 m at the segment boundary
# — the coarse smoothstep sampling (5 frames per leg) turns the
# leg-length change into a velocity step no motion model can predict
# (measured: f21 PnP 18/43 inliers, ratio 0.42 — the gate fails).
# Stretching the legs samples the smoothsteps finely, so the
# per-frame displacement changes gradually and peak speeds drop to
# ~0.17-0.27 m/frame. Hover is then frames 45-59. The fly leg is
# KEPT at the harness 14 frames: the VIO initialisation needs the
# median LK flow to reach 15 px while enough keyframe points are
# still alive, and a 19-frame fly ramps so slowly that the blank-door
# room (s23: only 72 initial features, no posters) drains to 34
# tracked / 20 E-inliers by the time the flow gate is met (both
# gates fail, so s23 never initialises). Fly 14 + enter 10 +
# approach 20: the enter and approach legs are the ones that were
# too fast; the fly was never the problem. GT-pose runs keep the
# scenario's own phase_ends.
VIO_PHASE_ENDS = (14, 24, 44)
WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
WEIGHTS = str(WALL_MARKER_DIR / "yolov8n-pose.pt")

# person (x, z) per scenario; None = no person. z is the distance from
# the phone plane; the reported GT distance is the Euclidean
# phone -> person distance, which includes the 1.5 m phone height.
SCENARIOS = {
    "s1_basic": {
        "person": (0.0, 4.0), "brightness": 1.0,
        "hover": (0.9, 1.3, 6.0), "tracking_loss_frame": None,
        "desc": "person 4.0 m from phone, centred, light 1.0",
    },
    "s2_offcentre": {
        "person": (1.4, 5.5), "brightness": 1.0,
        "hover": (-0.7, 1.3, 4.6), "tracking_loss_frame": None,
        "desc": "person 5.5 m from phone, off to the side near the doorway",
    },
    "s3_close": {
        "person": (0.0, 3.3), "brightness": 1.0,
        "hover": (1.0, 1.3, 5.8), "tracking_loss_frame": None,
        "desc": "person just behind the wall (z = 3.3 m)",
    },
    "s4_dim": {
        "person": (0.0, 4.5), "brightness": 0.5,
        "hover": (1.0, 1.3, 6.5), "tracking_loss_frame": None,
        "desc": "light 0.5, person 4.5 m",
    },
    "s5_tracking_loss": {
        "person": (0.0, 4.0), "brightness": 1.0,
        "hover": (0.9, 1.3, 6.0), "tracking_loss_frame": 36,
        "desc": "tracking_valid True until frame 35, False after; "
                "person stays visible",
    },
    "s6_no_person": {
        "person": None, "brightness": 1.0,
        "hover": (0.9, 1.3, 6.0), "tracking_loss_frame": None,
        "desc": "room B empty, same flight path; zero markers, ever",
    },
    "s7_two_drones": {
        "person": (0.0, 4.5), "brightness": 1.0,
        "hover": (1.0, 1.3, 6.4), "hover2": (-1.0, 1.3, 6.4),
        "tracking_loss_frame": None,
        "desc": "two drones, different hover points, both see the person",
    },
}

# Wardrobe / doorframe-edge occluder for s15: a narrow vertical plane
# just in front of the person (person z = 4.5, occluder z = 4.9, drone
# hover z = 6.5), covering roughly 40% of the body width (measured:
# detected bbox ~100-110 px wide vs ~170 px unoccluded). Rendered AFTER
# the billboard (see render.py). The drone's approach path stays clear
# of it (x ~0.9 at z = 4.9).
_OCCLUDER_WARDROBE = (
    np.array([[0.22, 0.0, 4.9], [0.35, 0.0, 4.9],
              [0.35, 1.95, 4.9], [0.22, 1.95, 4.9]], dtype=float),
    (62, 52, 42),
)

# Extra posters for the cluttered room (s22): more features on the
# room-B walls.
_CLUTTER_POSTERS = [
    ("backB", -2.4, 1.6, 0.9, 0.7, (60, 110, 90)),
    ("backB", 2.4, 1.6, 0.9, 0.7, (110, 70, 100)),
    ("leftB", 6.8, 1.8, 1.0, 0.75, (130, 120, 60)),
    ("rightB", 6.8, 1.8, 1.0, 0.75, (60, 90, 130)),
    ("B", -2.4, 1.5, 0.8, 0.6, (90, 60, 110)),
    ("B", 2.4, 1.5, 0.8, 0.6, (110, 130, 80)),
]

# Furniture boxes for the cluttered room (s22): (x, y, z) ranges.
# Placed clear of the drone-person line of sight (real occlusion uses
# the occluder quads, not furniture — see render.py).
_CLUTTER_FURNITURE = [
    {"x": (1.8, 2.6), "y": (0.0, 0.7), "z": (5.5, 6.5),
     "color": (120, 95, 65)},
    {"x": (-2.6, -1.8), "y": (0.0, 1.8), "z": (5.0, 5.6),
     "color": (90, 70, 50)},
    {"x": (1.5, 2.2), "y": (0.0, 0.5), "z": (4.0, 4.6),
     "color": (140, 110, 80)},
    {"x": (-2.4, -1.6), "y": (0.0, 0.9), "z": (6.8, 7.4),
     "color": (100, 80, 60)},
]

NEW_SCENARIOS = {
    # --- new rooms / places -------------------------------------------
    "s8_wall_close": {
        "person": (-0.8, 2.5), "brightness": 1.0,
        "hover": (0.8, 1.3, 5.0), "tracking_loss": None,
        "room": {"wall_z": 2.0, "room_b_z": (2.0, 8.0),
                 "door_x": (-1.6, -0.4)},
        "desc": "wall z = 2.0 m, doorway off-centre (left), "
                "person 2.5 m (z) from phone",
    },
    "s9_wall_far": {
        "person": (0.3, 5.5), "brightness": 1.0,
        "hover": (0.5, 1.3, 8.0), "tracking_loss": None,
        "room": {"wall_z": 4.0, "room_b_z": (4.0, 9.0),
                 "door_x": (1.2, 2.4)},
        "desc": "wall z = 4.0 m, doorway off-centre (right), "
                "person 5.5 m (z) from phone",
    },
    "s10_phone_offcentre": {
        "person": (-0.6, 4.5), "brightness": 1.0,
        "hover": (0.6, 1.3, 6.2), "tracking_loss": None,
        "phone": (1.2, 1.5, 0.5),
        "desc": "phone NOT at the origin (1.2, 1.5, 0.5); "
                "person 4.5 m (z) away, other side",
    },
    "s11_large_room_far": {
        "person": (0.5, 7.0), "brightness": 1.0,
        "hover": (1.2, 1.3, 10.2), "tracking_loss": None,
        "room": {"room_x": (-4.0, 4.0), "room_b_z": (3.0, 12.0)},
        "desc": "larger room B (x +-4 m, z to 12 m), "
                "person 7.0 m (z) from phone",
    },
    "s12_bright": {
        "person": (0.0, 4.5), "brightness": 1.2,
        "hover": (1.0, 1.3, 6.4), "tracking_loss": None,
        "desc": "brightness 1.2x, person 4.5 m",
    },
    "s24_empty_new_room": {
        "person": None, "brightness": 1.0,
        "hover": (0.5, 1.3, 5.5), "tracking_loss": None,
        "room": {"wall_z": 2.5, "room_b_z": (2.5, 9.0),
                 "door_x": (0.8, 2.0)},
        "desc": "empty room B in a NEW geometry (wall 2.5 m, doorway "
                "off-centre right); zero markers, ever",
    },
    # --- adversarial ----------------------------------------------------
    "s13_dim_noise": {
        "person": (0.0, 4.5), "brightness": 0.4,
        "hover": (1.0, 1.3, 6.4), "tracking_loss": None,
        "noise_sigma": 12.0, "noise_seed": 20261123,
        "desc": "light 0.4x with REAL Gaussian noise (sigma 12), "
                "person 4.5 m",
    },
    "s14_glare": {
        "person": (0.0, 4.5), "brightness": 1.0,
        "hover": (1.0, 1.3, 6.4), "tracking_loss": None,
        "room": {"window": ("backB", 0.0, 1.7, 2.2, 1.6),
                 "glare": 0.35},
        "desc": "bright window directly behind the person + bloom "
                "(backlit), person 4.5 m",
    },
    "s15_occluded": {
        "person": (0.0, 4.5), "brightness": 1.0,
        "hover": (1.2, 1.3, 6.5), "tracking_loss": None,
        "room": {"occluders": [_OCCLUDER_WARDROBE]},
        "desc": "person partly hidden behind a wardrobe (~40% of the "
                "body covered), person 4.5 m",
    },
    "s16_sideways": {
        "persons": [{"pos": (0.0, 4.5), "yaw": math.pi / 2,
                     "width_scale": 0.55}],
        "brightness": 1.0, "hover": (1.0, 1.3, 6.4),
        "tracking_loss": None,
        "desc": "person turned sideways (facing +X, 90 deg to the "
                "viewing direction), narrowed billboard, person 4.5 m",
    },
    "s17_edge": {
        "person": (1.2, 6.5), "brightness": 1.0,
        "hover": (0.0, 1.3, 3.8), "tracking_loss": None,
        "hover_amp": (0.1, 0.05),
        "look_target": (0.0, 1.0, 7.5),
        "desc": "person at the very edge of the drone frame (billboard "
                "right edge ~30 px from the frame edge); drone looks at "
                "a fixed point, not at the person",
    },
    "s18_moving": {
        "persons": [{"pos": (-1.0, 5.2), "yaw": "face_drone",
                     "path": {"vx": 0.25, "vz": -0.12, "from_frame": 0}}],
        "brightness": 1.0, "hover": (1.2, 1.3, 7.0),
        "tracking_loss": None,
        "desc": "person walking slowly (0.25 m/s across, 0.12 m/s "
                "toward the wall) — triangulation assumes a static scene",
    },
    "s19_two_people": {
        "persons": [{"pos": (-0.9, 4.5), "yaw": "face_drone"},
                    {"pos": (0.9, 5.0), "yaw": "face_drone"}],
        "brightness": 1.0, "hover": (0.0, 1.3, 7.5),
        "tracking_loss": None,
        "look_target": "midpoint",
        "desc": "TWO people in room B (~1.9 m apart); the system must "
                "mark a real person, never a merged midpoint marker",
    },
    "s20_fast": {
        "person": (0.0, 4.5), "brightness": 1.0,
        "hover": (1.0, 1.3, 6.4), "tracking_loss": None,
        "hover_amp": (0.9, 0.3), "hover_period": (12.0, 10.0),
        "phase_ends": (8, 11, 14),
        "desc": "fast drone: fast approach + fast wide hover "
                "(amp 0.9 m, period 1.2 s)",
    },
    "s21_loss_recovery": {
        "person": (0.0, 4.5), "brightness": 1.0,
        "hover": (1.0, 1.3, 6.4), "tracking_loss": (25, 41),
        "desc": "tracking loss frames 25-40, recovery from frame 41; "
                "marker must vanish on loss and only return after the "
                "bridge is re-established",
    },
    "s22_clutter": {
        "person": (0.0, 4.5), "brightness": 1.0,
        "hover": (1.0, 1.3, 6.4), "tracking_loss": None,
        "room": {"furniture": _CLUTTER_FURNITURE,
                 "posters_extra": _CLUTTER_POSTERS},
        "desc": "cluttered room B (furniture boxes + extra posters), "
                "person 4.5 m",
    },
    "s23_blank_door": {
        "person": (-0.8, 3.6), "brightness": 1.0,
        "hover": (0.8, 1.3, 5.8), "tracking_loss": None,
        "room": {"wall_z": 2.5, "room_b_z": (2.5, 8.0),
                 "door_x": (-1.4, -0.2), "posters": "featureless_door"},
        "desc": "featureless blank wall at the doorway (no posters on "
                "the shared wall), wall 2.5 m, person 3.6 m (z)",
    },
    # --- multi-drone ------------------------------------------------------
    "s25_four_drones": {
        "person": (0.0, 4.5), "brightness": 1.0, "tracking_loss": None,
        "hovers": [(1.0, 1.3, 6.4), (-1.0, 1.3, 6.4),
                   (0.0, 1.3, 7.8), (2.2, 1.3, 6.8)],
        "desc": "4 drones, different hover points, all see the person",
    },
    "s26_five_drones": {
        "person": (0.0, 4.5), "brightness": 1.0, "tracking_loss": None,
        "hovers": [(1.0, 1.3, 6.4), (-1.0, 1.3, 6.4),
                   (0.0, 1.3, 7.8), (2.2, 1.3, 6.8),
                   (-2.2, 1.3, 6.8)],
        "desc": "5 drones, different hover points, all see the person",
    },
    "s27_four_drones_partial": {
        "person": (0.0, 4.5), "brightness": 1.0, "tracking_loss": None,
        "hovers": [(1.0, 1.3, 6.4), (-1.0, 1.3, 6.4),
                   (0.0, 1.3, 7.8), (2.2, 1.3, 6.8)],
        "blind_drones": [2, 3],
        "blind_look_target": (-2.5, 1.0, 7.5),
        "blind_look_target_3": (2.8, 1.0, 7.8),
        "desc": "4 drones, 2 lose sight of the person (they look at "
                "the back corners) — fusion must still work from the "
                "remaining 2",
    },
}

# All scenarios (original + extended), in run order.
ALL_SCENARIOS = dict(SCENARIOS)
ALL_SCENARIOS.update(NEW_SCENARIOS)

# Frame ranges of the flight phases (defaults; overridable per scenario).
PHASE_FLY_END = 14        # frames 0-14: start -> doorway
PHASE_ENTER_END = 19      # frames 15-19: doorway -> inside room B
PHASE_APPROACH_END = 24   # frames 20-24: approach -> hover point
HOVER_START = 25          # frames 25-59: hover with lateral motion


def _smooth(a, b, t):
    """Smoothstep interpolation between points a and b at t in [0, 1]."""
    s = t * t * (3.0 - 2.0 * t)
    return np.asarray(a, dtype=float) + (np.asarray(b, dtype=float)
                                         - np.asarray(a, dtype=float)) * s


def _trajectory(hover, start=None, doorway=None, inside=None,
                amp=(0.4, 0.15), periods=(30.0, 22.0),
                phase_ends=(PHASE_FLY_END, PHASE_ENTER_END,
                            PHASE_APPROACH_END),
                hover_ramp_frames=0):
    """Drone centre positions (60, 3) for a scenario with hover point.

    Defaults reproduce the original trajectory exactly: start beside
    the phone, doorway at the wall, inside room B, then hover with
    lateral motion (+/-amp) so triangulation has a baseline.
    ``hover_ramp_frames`` (VIO variants only): ramp the hover
    oscillation in over that many frames from the hover point
    (smoothstep 0 -> 1), so the approach -> hover transition is C1
    (offset and velocity both 0 at the handover). Without it the hover
    sine starts at a 0.84*amp_z offset (the +1.0 z-phase), i.e. the
    drone teleports ~0.13 m in one frame at the handover — invisible
    to the GT-pose runs, fatal to a VIO motion model.
    """
    hover = np.asarray(hover, dtype=float)
    fly_end, enter_end, approach_end = phase_ends
    if start is None:
        start = np.array([0.0, 1.3, 0.4])
    if doorway is None:
        doorway = np.array([0.0, 1.3, 2.9])
    if inside is None:
        inside = np.array([hover[0] * 0.6, 1.3, 3.8])
    pos = np.zeros((N_FRAMES, 3), dtype=float)
    for i in range(N_FRAMES):
        if i <= fly_end:
            t = i / fly_end
            pos[i] = _smooth(start, doorway, t)
        elif i <= enter_end:
            t = (i - fly_end) / (enter_end - fly_end)
            pos[i] = _smooth(doorway, inside, t)
        elif i <= approach_end:
            t = (i - enter_end) / (approach_end - enter_end)
            pos[i] = _smooth(inside, hover, t)
        else:
            k = i - (approach_end + 1)
            ramp = 1.0
            if hover_ramp_frames > 0:
                ramp = float(_smooth(0.0, 1.0,
                                     min(k / hover_ramp_frames, 1.0)))
            pos[i] = hover + ramp * np.array([
                amp[0] * np.sin(2.0 * np.pi * k / periods[0]),
                0.0,
                amp[1] * np.sin(2.0 * np.pi * k / periods[1] + 1.0),
            ])
    return pos


def _look_yaw_pitch(R):
    """Yaw/pitch of a look-at rotation's view direction (third row of R).

    yaw = atan2(z_x, z_z), pitch = -asin(z_y) — the same extraction
    ``tests/vio_ground_test.py: smooth_flight_poses`` uses.
    """
    z = np.asarray(R, dtype=float)[2]
    yaw = float(np.arctan2(z[0], z[2]))
    pitch = float(np.arcsin(np.clip(-z[1], -1.0, 1.0)))
    return yaw, pitch


def _look_R_from_yaw_pitch(eye, yaw, pitch):
    """Rebuild a look-at rotation from a yaw/pitch view direction."""
    direction = np.array([np.sin(yaw) * np.cos(pitch),
                          -np.sin(pitch),
                          np.cos(yaw) * np.cos(pitch)])
    return look_at_R(eye, np.asarray(eye, dtype=float) + direction)


def _smooth_damp(current, target, velocity, smooth_time, max_speed, dt):
    """One step of a critically-damped spring toward ``target``.

    The standard SmoothDamp (critically damped 2nd-order filter):
    the output approaches a (possibly moving) target exponentially
    with time constant ``smooth_time``, the velocity is a continuous
    state (no steps), and the speed is capped at ``max_speed`` — so
    the output is C1 with bounded acceleration and settles exactly
    on the target (no limit cycle, no arrival snap). All angles in
    radians, ``velocity`` in radians per second, ``max_speed`` in
    radians per second. Returns ``(new_value, new_velocity)``.
    """
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    orig_target = target
    change = current - target
    max_change = max_speed * smooth_time
    change = max(-max_change, min(max_change, change))
    target = current - change
    temp = (velocity + omega * change) * dt
    velocity = (velocity - omega * temp) * exp
    output = target + (change + temp) * exp
    # Prevent overshoot past the ORIGINAL target: if the output
    # crossed it, stop there with zero velocity.
    if ((orig_target - current > 0.0) == (output > orig_target)):
        output = orig_target
        velocity = 0.0
    return output, velocity


def _rot_y(yaw):
    """Rotation about +Y mapping local +z to (sin yaw, 0, cos yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _gt_skeleton(person_pos, yaw, H=PERSON_H):
    """Ground-truth skeleton in world coordinates (17, 3)."""
    local = skeleton_local_joints(H)
    world = (np.asarray(person_pos, dtype=float)
             + (_rot_y(yaw) @ local.T).T)
    return world


def _gt_phone_projection(gt_skeleton_world, phone_pose):
    """Ground-truth skeleton projected into the phone view (17, 2) or None."""
    R, C = phone_pose
    pts = []
    for joint in gt_skeleton_world:
        p = project_point(PHONE_K, R, C, joint)
        pts.append(None if p is None else (p[0], p[1]))
    return pts


def _get_detector(detector):
    if detector is not None:
        return detector
    from server.detector import PoseDetector
    return PoseDetector(weights=WEIGHTS)


def _person_specs(cfg):
    """Normalised person specs for a scenario config.

    Each spec: {"pos": (x, z), "yaw": "face_drone" | float,
    "width_scale": float, "path": None | {"vx","vz","from_frame"}}.
    """
    if "persons" in cfg:
        specs = []
        for s in cfg["persons"]:
            spec = {"pos": tuple(s["pos"]),
                    "yaw": s.get("yaw", "face_drone"),
                    "width_scale": float(s.get("width_scale", 1.0)),
                    "path": s.get("path")}
            specs.append(spec)
        return specs
    if cfg.get("person") is None:
        return []
    return [{"pos": tuple(cfg["person"]), "yaw": "face_drone",
             "width_scale": 1.0, "path": None}]


def _person_pos_at(spec, frame):
    """Foot position (3,) of a person spec at a frame (moving people)."""
    x, z = spec["pos"]
    path = spec.get("path")
    if path is not None:
        dt = max(0, frame - int(path.get("from_frame", 0))) / FPS
        x = x + float(path.get("vx", 0.0)) * dt
        z = z + float(path.get("vz", 0.0)) * dt
    return np.array([x, 0.0, z], dtype=float)


def _drone_index(drone_id):
    try:
        return int(str(drone_id).replace("drone", "")) - 1
    except ValueError:
        return 0


def _hover_for(cfg, name, drone_id):
    """Hover point for one drone of a scenario."""
    if "hovers" in cfg:
        idx = _drone_index(drone_id)
        return cfg["hovers"][idx % len(cfg["hovers"])]
    if name == "s7_two_drones" and drone_id == "drone2":
        return cfg["hover2"]
    return cfg["hover"]


def _tracking_valid(cfg, frame):
    """Tracking-valid flag for a frame.

    ``tracking_loss`` (new) or ``tracking_loss_frame`` (original):
    None = always valid; int = invalid from that frame on;
    (start, end) = invalid for start <= frame < end (recovery after).
    """
    loss = cfg.get("tracking_loss", cfg.get("tracking_loss_frame"))
    if loss is None:
        return True
    if isinstance(loss, int):
        return frame < loss
    start, end = loss
    return not (start <= frame < end)


def iter_frames(name, detector=None, drone_id="drone1", smooth_look=False):
    """Yield the 60 frame tuples for a scenario (generator).

    ``smooth_look=True`` runs the smoothed-look VIO variant: the look
    direction's yaw and pitch are slew-rate limited to
    ``VIO_MAX_YAW_RATE_DEG_S`` / ``VIO_MAX_PITCH_RATE_DEG_S`` (60 deg/s
    = 6 deg/frame at 10 fps, a real Tello yaw rate) instead of snapping
    at the doorway and the turn. Positions are unchanged; the default
    (``smooth_look=False``) reproduces the original trajectories
    exactly, so the 27 GT-pose scenarios are unaffected.
    """
    if name not in ALL_SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; "
                         f"known: {sorted(ALL_SCENARIOS)}")
    cfg = ALL_SCENARIOS[name]
    detector = _get_detector(detector)
    cutout, mask = load_person_cutout(detector)
    room = cfg.get("room")
    renderer = RoomRenderer(DRONE_K, cutout, mask,
                            person_height=PERSON_H,
                            brightness=cfg["brightness"],
                            room=room,
                            noise_sigma=cfg.get("noise_sigma", 0.0),
                            noise_seed=cfg.get("noise_seed", 0))

    phone_pos = np.asarray(cfg.get("phone", (0.0, 1.5, 0.0)), dtype=float)
    phone_pose = (np.eye(3), phone_pos)
    wall_z = (room or {}).get("wall_z", 3.0)
    room_b_z = (room or {}).get("room_b_z", (3.0, 8.0))
    door_x = (room or {}).get("door_x", (-0.6, 0.6))
    door_cx = 0.5 * (door_x[0] + door_x[1])

    hover = _hover_for(cfg, name, drone_id)
    start = np.array([phone_pos[0], 1.3, phone_pos[2] + 0.4])
    doorway = np.array([door_cx, 1.3, wall_z - 0.1])
    inside = np.array([hover[0] * 0.6, 1.3, wall_z + 0.8])
    # VIO variants use re-timed phases (see VIO_PHASE_ENDS); GT-pose
    # runs use the scenario's own phase_ends (unchanged).
    phase_ends_traj = (VIO_PHASE_ENDS if smooth_look else tuple(
        cfg.get("phase_ends",
                (PHASE_FLY_END, PHASE_ENTER_END, PHASE_APPROACH_END))))
    positions = _trajectory(
        hover, start=start, doorway=doorway, inside=inside,
        amp=tuple(cfg.get("hover_amp", (0.4, 0.15))),
        periods=tuple(cfg.get("hover_period", (30.0, 22.0))),
        phase_ends=phase_ends_traj,
        hover_ramp_frames=10 if smooth_look else 0)

    specs = _person_specs(cfg)
    is_blind = _drone_index(drone_id) in cfg.get("blind_drones", [])
    blind_target = cfg.get(
        f"blind_look_target_{_drone_index(drone_id)}",
        cfg.get("blind_look_target"))
    look_target = cfg.get("look_target", "person")
    phase_ends = tuple(cfg.get("phase_ends",
                               (PHASE_FLY_END, PHASE_ENTER_END,
                                PHASE_APPROACH_END)))
    approach_end = phase_ends[2]
    # Smoothed-look state (VIO variants only): previous frame's
    # spring-filtered yaw/pitch (unwrapped) and their velocities.
    prev_yaw = None
    prev_pitch = None
    yaw_vel = 0.0
    pitch_vel = 0.0
    dt = 1.0 / FPS
    max_yaw_speed = math.radians(VIO_MAX_YAW_RATE_DEG_S)
    max_pitch_speed = math.radians(VIO_MAX_PITCH_RATE_DEG_S)
    # VIO variants only: during the approach, look at a FIXED far
    # point (room centreline, back of room B) instead of the next
    # position. Looking at the next position yaws the camera with
    # every bend of the path — the doorway bend alone yaws it +31 deg
    # — and that doorway slew runs straight into the turn to face the
    # person, which goes the OPPOSITE way: the yaw rate reverses
    # (+3 deg/frame to -4 deg/frame at f25, measured s1) and the
    # motion model predicts "continue right" while the truth is "go
    # left" (LK keeps 7 tracks at f25). A fixed far point holds the
    # approach yaw nearly constant, so the turn is a single clean
    # slew in one direction with no reversal. The point is on the
    # ROOM CENTRELINE (x=0), not at the hover x: aiming at the hover
    # point biases the view toward one wall, and in the blank-door
    # room (s23) that side is the featureless wall — f00 features
    # drop 104 -> 72 and the initialisation starves. Centreline
    # keeps both poster walls in frame. A real pilot likewise looks
    # roughly forward through the doorway, not at the velocity
    # vector. GT-pose runs keep "next position".
    approach_look = np.array([0.0, 1.0, room_b_z[1] - 0.5])
    person_frame = VIO_TURN_FRAME if smooth_look else 20

    for i in range(N_FRAMES):
        t = i / FPS
        eye = positions[i]
        # Look direction: along motion early, then at the person (or a
        # fixed look target / midpoint), or away for blind drones.
        if i < 15:
            if smooth_look:
                target = approach_look
            else:
                nxt = positions[min(i + 1, N_FRAMES - 1)]
                target = nxt if np.linalg.norm(nxt - eye) > 1e-6 else eye + np.array([0, 0, 1.0])
        elif is_blind and blind_target is not None:
            target = np.asarray(blind_target, dtype=float)
        elif i < person_frame:
            if smooth_look:
                target = approach_look
            else:
                nxt = positions[min(i + 1, N_FRAMES - 1)]
                target = nxt if np.linalg.norm(nxt - eye) > 1e-6 else eye + np.array([0, 0, 1.0])
        else:
            if specs:
                if look_target == "midpoint":
                    mid = np.mean([_person_pos_at(s, i) for s in specs],
                                  axis=0)
                    target = mid + np.array([0.0, 0.85, 0.0])
                elif isinstance(look_target, (tuple, list)):
                    target = np.asarray(look_target, dtype=float)
                else:
                    p0 = _person_pos_at(specs[0], i)
                    target = p0 + np.array([0.0, 0.85, 0.0])
            else:
                target = np.array([hover[0] * 0.5, 1.0, room_b_z[1] - 0.5])
        R = look_at_R(eye, target)
        if smooth_look:
            # Critically-damped spring on the look direction: the
            # harness snaps yaw/pitch in one frame at the doorway and
            # the turn (see VIO_MAX_YAW_RATE_DEG_S); a real Tello
            # yaws <= 60 deg/s with a smooth ramp. The desired yaw is
            # unwrapped against the spring's current value so the
            # spring takes the short way round.
            yaw, pitch = _look_yaw_pitch(R)
            if prev_yaw is None:
                prev_yaw, prev_pitch = yaw, pitch
            else:
                gap_yaw = ((yaw - prev_yaw + math.pi)
                           % (2.0 * math.pi) - math.pi)
                prev_yaw, yaw_vel = _smooth_damp(
                    prev_yaw, prev_yaw + gap_yaw, yaw_vel,
                    VIO_SMOOTH_TIME_S, max_yaw_speed, dt)
                prev_pitch, pitch_vel = _smooth_damp(
                    prev_pitch, max(-1.4, min(1.4, pitch)), pitch_vel,
                    VIO_SMOOTH_TIME_S, max_pitch_speed, dt)
                R = _look_R_from_yaw_pitch(eye, prev_yaw, prev_pitch)
        pose = (R, eye)

        tracking_valid = _tracking_valid(cfg, i)

        # Per-person ground truth at this frame.
        persons_render = []
        persons_world = []
        persons_gt = []
        persons_yaw = []
        for spec in specs:
            pos = _person_pos_at(spec, i)
            if spec["yaw"] == "face_drone":
                d = eye - pos
                yaw = float(np.arctan2(d[0], d[2]))
            else:
                yaw = float(spec["yaw"])
            persons_render.append({"pos": pos, "yaw": yaw,
                                   "width_scale": spec["width_scale"]})
            persons_world.append(pos)
            persons_gt.append(_gt_skeleton(pos, yaw))
            persons_yaw.append(yaw)

        frame, info = renderer.render(pose, persons=persons_render)
        in_view_list = [bool(p["in_view"]) for p in info["persons"]]

        if persons_world:
            gt_skel = persons_gt[0]
            gt_dist = float(np.linalg.norm(persons_world[0] - phone_pos))
            gt_phone = _gt_phone_projection(gt_skel, phone_pose)
            person_world = persons_world[0].tolist()
        else:
            gt_skel = None
            gt_dist = None
            gt_phone = None
            person_world = None

        meta = {
            "scenario": name,
            "drone_id": drone_id,
            "frame": i,
            "t": t,
            "person_world": person_world,
            "gt_skeleton_world": (None if gt_skel is None
                                  else gt_skel.tolist()),
            "gt_distance_m": gt_dist,
            "gt_skeleton_phone_px": gt_phone,
            "person_yaw": persons_yaw[0] if persons_yaw else 0.0,
            "in_view": bool(any(in_view_list)),
            "in_view_list": in_view_list,
            "height_px": float(info["height_px"]),
            "billboard_box": info["billboard_box"],
            "billboard_boxes": [p["billboard_box"]
                                for p in info["persons"]],
            "tracking_valid": bool(tracking_valid),
            "brightness": cfg["brightness"],
            "phone_position": phone_pos.tolist(),
            "persons_world": [p.tolist() for p in persons_world],
            "persons_gt": [g.tolist() for g in persons_gt],
            "persons_yaw": persons_yaw,
            "n_persons": len(persons_world),
        }
        yield frame, pose, tracking_valid, meta


def get_frames(name, detector=None, drone_id="drone1", smooth_look=False):
    """The 60 frame tuples for a scenario, as a list (main.py compat).

    ``smooth_look=True`` selects the smoothed-look VIO variant (see
    ``iter_frames``); the default reproduces the original trajectories.
    """
    return list(iter_frames(name, detector=detector, drone_id=drone_id,
                            smooth_look=smooth_look))


def get_frames_s7(detector=None):
    """s7: (drone1 frames, drone2 frames), each 60 tuples, same timestamps."""
    return (get_frames("s7_two_drones", detector=detector, drone_id="drone1"),
            get_frames("s7_two_drones", detector=detector, drone_id="drone2"))


def iter_frames_multi(name, detector=None):
    """Yield per-frame tuples for ALL drones of a multi-drone scenario.

    Each yielded item is a tuple of frame tuples, one per drone, all at
    the same timestamp: ((f1, pose1, tv1, meta1), (f2, ...), ...).
    """
    cfg = ALL_SCENARIOS[name]
    n = len(cfg["hovers"])
    gens = [iter_frames(name, detector=detector, drone_id=f"drone{i + 1}")
            for i in range(n)]
    yield from zip(*gens)


def get_frames_multi(name, detector=None):
    """Multi-drone frames as a list of per-drone frame lists."""
    cfg = ALL_SCENARIOS[name]
    n = len(cfg["hovers"])
    return [get_frames(name, detector=detector, drone_id=f"drone{i + 1}")
            for i in range(n)]


def list_scenarios():
    """Original 7 scenario names in SPEC order (unchanged)."""
    return ["s1_basic", "s2_offcentre", "s3_close", "s4_dim",
            "s5_tracking_loss", "s6_no_person", "s7_two_drones"]


def list_new_scenarios():
    """Extended scenario names (new rooms, adversarial, multi-drone)."""
    return list(NEW_SCENARIOS)


def list_all_scenarios():
    """All scenarios: original 7 first, then the extended set."""
    return list_scenarios() + list_new_scenarios()
