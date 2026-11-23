"""Multi-drone fusion and marker emission.

* :class:`PersonTrack` — one person in the world frame.
* :class:`Fusion` — associates per-drone 3D detections to tracks
  (< 1.0 m), fuses by inverse-variance weighting, and emits the marker
  JSON. A track is "confirmed" after 3 updates.
* HOLD RULE: if the latest update had no usable pose but the track is
  < 0.5 s old, the last good skeleton is returned with ``"stale": true``.
  After 0.5 s, None. If the bridge is invalid, ALWAYS None.
* The marker is a POSE SKELETON (COCO 17 joints), never a dot. Joints
  that are unknown are ``null`` in the JSON (``[null, null, null]``) —
  the viewer must skip them, never invent them.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import PHONE_POSITION

ASSOCIATE_DIST_M = 1.0
CONFIRM_UPDATES = 3
TRACK_MAX_AGE_S = 1.0
HOLD_MAX_AGE_S = 0.5
MIN_JOINT_CONF = 0.3


class PersonTrack:
    """One tracked person in the world frame."""

    def __init__(self, track_id, pos, sigma, skeleton, kpt_conf, t, drone_id):
        self.id = int(track_id)
        self.pos = np.asarray(pos, dtype=float)          # (3,) world
        self.sigma = float(sigma)
        self.skeleton = np.asarray(skeleton, dtype=float)  # (17,3) world, NaN ok
        self.kpt_conf = np.asarray(kpt_conf, dtype=float)  # (17,)
        self.last_seen_t = float(t)
        self.confirm_count = 1
        self.sources = {drone_id}

    @property
    def confirmed(self):
        return self.confirm_count >= CONFIRM_UPDATES


class Fusion:
    """Fuse per-drone 3D person detections into tracks; emit markers."""

    def __init__(self, bridge, phone_position=None):
        self.bridge = bridge
        # Phone camera centre in the world frame. Defaults to the SPEC
        # phone (0, 1.5, 0); scenarios may pass a different phone
        # position (phone NOT at the origin) — the marker distance is
        # then measured from that phone.
        self.phone_position = np.asarray(
            PHONE_POSITION if phone_position is None else phone_position,
            dtype=float)
        self.tracks = []
        self._next_id = 1
        self._last_call_t = None
        self._last_call_had_detection = False

    # -- updates -----------------------------------------------------------

    def update(self, drone_id, detections_3d, t):
        """Fuse one drone's detections at time ``t``.

        Each detection is ``{"pos": (3,), "skeleton": (17,3),
        "sigma": float, "conf": float}`` with pos/skeleton in the world
        frame. Call every frame, with ``[]`` when the drone saw no
        usable pose — the empty call is what drives the HOLD rule.
        """
        t = float(t)
        self._last_call_t = t
        self._last_call_had_detection = len(detections_3d) > 0
        for det in detections_3d:
            pos = np.asarray(det["pos"], dtype=float)
            skel = np.asarray(det["skeleton"], dtype=float)
            sigma = float(det["sigma"])
            conf = float(det.get("conf", 1.0))
            kpt_conf = np.asarray(
                det.get("kpt_conf", np.ones(17)), dtype=float)
            track = self._associate(pos)
            if track is None:
                track = PersonTrack(self._next_id, pos, sigma, skel,
                                    kpt_conf, t, drone_id)
                self._next_id += 1
                self.tracks.append(track)
            else:
                self._fuse(track, pos, sigma, skel, kpt_conf, t, drone_id)

    def _associate(self, pos):
        best, best_d = None, ASSOCIATE_DIST_M
        for track in self.tracks:
            d = float(np.linalg.norm(track.pos - pos))
            if d < best_d:
                best, best_d = track, d
        return best

    @staticmethod
    def _fuse(track, pos, sigma, skel, kpt_conf, t, drone_id):
        w_old = 1.0 / max(track.sigma, 1e-6) ** 2
        w_new = 1.0 / max(sigma, 1e-6) ** 2
        track.pos = (w_old * track.pos + w_new * pos) / (w_old + w_new)
        track.sigma = 1.0 / math.sqrt(w_old + w_new)
        for j in range(17):
            old_ok = (track.kpt_conf[j] >= MIN_JOINT_CONF
                      and not np.isnan(track.skeleton[j]).any())
            new_ok = (kpt_conf[j] >= MIN_JOINT_CONF
                      and not np.isnan(skel[j]).any())
            if old_ok and new_ok:
                track.skeleton[j] = (w_old * track.skeleton[j]
                                     + w_new * skel[j]) / (w_old + w_new)
                track.kpt_conf[j] = (w_old * track.kpt_conf[j]
                                     + w_new * kpt_conf[j]) / (w_old + w_new)
            elif new_ok:
                track.skeleton[j] = skel[j]
                track.kpt_conf[j] = kpt_conf[j]
            # else: keep the old joint (or NaN if there never was one)
        track.last_seen_t = t
        track.confirm_count += 1
        track.sources.add(drone_id)

    # -- marker ------------------------------------------------------------

    def _live_tracks(self, t):
        """Confirmed tracks the bridge currently vouches for.

        Returns [] when the bridge is invalid — a lost bridge never
        vouches for any track, so no marker is ever emitted from it.
        """
        state = self.bridge.state()
        if not state["valid"]:
            return []
        return [tr for tr in self.tracks
                if tr.confirmed and (t - tr.last_seen_t) < TRACK_MAX_AGE_S]

    def _marker_for(self, track, t, bridge_uncertainty):
        """Marker dict for one track, or None if it is too stale.

        HOLD: if the latest update had no usable pose for this track
        but it is < 0.5 s old, the last good skeleton is returned with
        ``"stale": true``. After 0.5 s, None. The 0.05 s tolerance is
        half a 10 fps frame interval, so a marker asked for a hair
        after its update still counts as fresh.
        """
        age = t - track.last_seen_t
        stale = age > 0.05
        if stale and age >= HOLD_MAX_AGE_S:
            return None
        distance = float(np.linalg.norm(track.pos - self.phone_position))
        uncertainty = math.sqrt(track.sigma ** 2 + bridge_uncertainty ** 2)
        skeleton = []
        for j in range(17):
            if np.isnan(track.skeleton[j]).any():
                skeleton.append([None, None, None])
            else:
                skeleton.append([float(x) for x in track.skeleton[j]])
        return {
            "type": "marker",
            "person_id": track.id,
            "skeleton": skeleton,
            "distance_m": distance,
            "uncertainty_m": float(uncertainty),
            "timestamp": t,
            "stale": bool(stale),
            "drones": sorted(track.sources),
        }

    def markers(self, t):
        """Marker list for time ``t``: up to 3, nearest first.

        Same rules as :meth:`marker` per track (bridge valid,
        confirmed, age < 1.0 s, stale hold <= 0.5 s). The viewer
        colours by position in this list, so nearest-first ordering
        gives each person a stable colour while the order holds.
        Returns [] when the bridge is invalid — never a stale marker
        presented as live.
        """
        t = float(t)
        state = self.bridge.state()
        if not state["valid"]:
            return []
        out = []
        for track in self._live_tracks(t):
            m = self._marker_for(track, t, state["uncertainty_m"])
            if m is not None:
                out.append(m)
        out.sort(key=lambda m: m["distance_m"])
        return out[:3]

    def marker(self, t):
        """Marker JSON for time ``t``, or None.

        Only when the bridge is valid AND a confirmed track exists with
        age < 1.0 s. HOLD: if the latest update had no usable pose for
        this track but it is < 0.5 s old, the last good skeleton is
        returned with ``"stale": true``. Bridge invalid => ALWAYS None.
        Kept for backward compatibility (single-marker payload); the
        SSE stream now uses :meth:`markers`.
        """
        t = float(t)
        candidates = self._live_tracks(t)
        if not candidates:
            return None
        # Freshest track first; lowest id breaks ties.
        track = min(candidates,
                    key=lambda tr: (t - tr.last_seen_t, tr.id))
        state = self.bridge.state()
        return self._marker_for(track, t, state["uncertainty_m"])
