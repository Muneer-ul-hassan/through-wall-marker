"""Bridge state machine: the shared phone<->drone frame.

The bridge is the one link between the phone frame and the drone frame:
the drone starts beside the phone and flies through the doorway while
keeping visual tracking, so one continuous map chain links the two
frames. The bridge stores the latest drone pose (world-to-camera (R, C),
as fed by the harness / SLAM) as the transform ``T_world_drone``.

Rules (SPEC.md, updated 2026-11-23 for loss+recovery):
* While ``tracking_valid`` is True, the latest pose is stored.
* When ``tracking_valid`` is False, the bridge is invalidated.
  The link stays gone until ``reacquire()`` restores it: a fresh
  valid pose after a loss is treated as a relocalisation event and
  starts a NEW chain (uncertainty clock restarts at 0.10 m). In the
  VIRTUAL harness the pose fed in is the true pose, so that stands
  in for relocalisation; in the REAL system reacquire must be gated
  on a genuine relocalisation (re-observing a mapped area / a doorway
  loop closure), never on a bare "tracking is valid again" flag.
  The harness proves the state-machine logic (marker hidden during
  loss, returns only after re-establishment + a fresh detection),
  not the real relocalisation itself.
* ``uncertainty_m = 0.10 + 0.02 * seconds_since_start``, capped at 1.0 m.
  This is a MODEL, not a measurement — the tests check whether real
  errors stay inside the displayed uncertainty.

``state()`` uses the time of the latest ``update()`` call as "now"
(the virtual clock is scenario time, not wall time), so it stays
deterministic and testable: ``age_s`` is the seconds since the last
valid pose.
"""

from __future__ import annotations

import numpy as np

from .geometry import apply_transform, invert_transform, make_transform


class Bridge:
    """Valid/invalid shared-frame bridge with a stored transform."""

    def __init__(self):
        self._valid = False
        self._lost = False  # set on tracking loss; only reacquire() clears it
        self._T_world_drone = None
        self._t_start = None
        self._last_valid_t = None
        self._last_update_t = None

    def update(self, drone_pose_world, tracking_valid, t):
        """Feed a drone pose (R, C) at scenario time ``t``.

        While tracking is valid the pose is stored as the transform.
        A ``tracking_valid=False`` update invalidates the bridge; the
        next valid pose after a loss re-establishes it via
        :meth:`reacquire` (a new chain — see the module docstring).
        """
        t = float(t)
        if self._t_start is None:
            self._t_start = t
        self._last_update_t = t
        if not tracking_valid:
            self.invalidate()
            return
        if self._lost:
            # Tracking was lost earlier: re-establish the link from
            # this fresh pose (virtual stand-in for relocalisation).
            self.reacquire(drone_pose_world, t)
            return
        R, C = drone_pose_world
        R = np.asarray(R, dtype=float)
        C = np.asarray(C, dtype=float)
        if R.shape != (3, 3) or C.shape != (3,):
            raise ValueError("drone_pose_world must be (R (3,3), C (3,))")
        T_cam_world = make_transform(R, C)  # world -> drone camera
        self._T_world_drone = invert_transform(T_cam_world)
        self._valid = True
        self._last_valid_t = t

    def invalidate(self):
        """Invalidate the bridge. No marker until reacquire."""
        self._valid = False
        self._lost = True
        self._T_world_drone = None

    def reacquire(self, drone_pose_world, t):
        """Re-establish the bridge from a fresh pose after tracking loss.

        Starts a NEW chain: the pose is stored as the transform, the
        lost flag is cleared, and the uncertainty clock restarts
        (``_t_start = t``). In the virtual harness the pose is the
        true pose, so this is the relocalisation stand-in; in the real
        system the caller must only invoke this after a genuine
        relocalisation event, never merely because tracking resumed.
        """
        t = float(t)
        R, C = drone_pose_world
        R = np.asarray(R, dtype=float)
        C = np.asarray(C, dtype=float)
        if R.shape != (3, 3) or C.shape != (3,):
            raise ValueError("drone_pose_world must be (R (3,3), C (3,))")
        T_cam_world = make_transform(R, C)  # world -> drone camera
        self._T_world_drone = invert_transform(T_cam_world)
        self._valid = True
        self._lost = False
        self._t_start = t
        self._last_valid_t = t
        self._last_update_t = t

    def state(self):
        """``{"valid", "T_world_drone", "uncertainty_m", "age_s"}``.

        ``uncertainty_m`` is evaluated at the latest update time:
        ``min(1.0, 0.10 + 0.02 * (t_last - t_start))``. ``age_s`` is the
        seconds since the last valid pose (0 while tracking is valid).
        """
        if self._t_start is None or self._last_update_t is None:
            return {"valid": False, "T_world_drone": None,
                    "uncertainty_m": 0.10, "age_s": 0.0}
        seconds = self._last_update_t - self._t_start
        uncertainty = min(1.0, 0.10 + 0.02 * seconds)
        if self._last_valid_t is not None:
            age = max(0.0, self._last_update_t - self._last_valid_t)
        else:
            age = 0.0
        return {"valid": bool(self._valid),
                "T_world_drone": self._T_world_drone,
                "uncertainty_m": float(uncertainty),
                "age_s": float(age)}


def world_from_drone(T, P):
    """Transform point(s) P from the drone camera frame to the world frame."""
    return apply_transform(T, P)
