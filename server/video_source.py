"""Frame sources for the wall-marker pipeline.

* :class:`FileVideoSource` — a recorded video file + a JSON pose list.
* :class:`SyntheticSource` — in-memory frames from the test renderer.
* :class:`TelloVideoSource` — documented stub. Ground/virtual only: it
  raises a clear error and never fakes a drone stream. The real Tello
  path is out of scope until the user gives the go-ahead for a live run.

Every source yields ``(frame_bgr, drone_pose, tracking_valid, meta)``
where ``drone_pose`` is ``(R, C)`` — R the world-to-camera rotation, C
the drone camera centre in world coordinates — and ``meta`` is a dict
of per-frame extras (may be empty). ``read()`` returns ``None`` at the
end of the stream.
"""

from __future__ import annotations

import json

import numpy as np


class FrameSource:
    """Base class: ``read()`` yields one frame tuple or ``None``."""

    def read(self):
        raise NotImplementedError


class FileVideoSource(FrameSource):
    """Recorded video + poses from JSON.

    ``poses_json`` is a JSON file holding a list with one entry per
    video frame (in order), each ``{"R": [[...]x3]x3, "C": [x, y, z],
    "tracking_valid": bool, "meta": {...}}`` (``meta`` optional).
    Reading stops when the video or the pose list runs out.
    """

    def __init__(self, path, poses_json):
        import cv2
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"cannot open video file: {path}")
        with open(poses_json, "r", encoding="utf-8") as fh:
            self._poses = json.load(fh)
        self._index = 0

    def read(self):
        if self._index >= len(self._poses):
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        entry = self._poses[self._index]
        self._index += 1
        R = np.asarray(entry["R"], dtype=float)
        C = np.asarray(entry["C"], dtype=float)
        tracking_valid = bool(entry.get("tracking_valid", True))
        meta = dict(entry.get("meta", {}))
        return frame, (R, C), tracking_valid, meta


class SyntheticSource(FrameSource):
    """In-memory frames: list of (frame_bgr, drone_pose, tracking_valid, meta)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def read(self):
        if self._index >= len(self._frames):
            return None
        item = self._frames[self._index]
        self._index += 1
        return item

    def reset(self):
        """Rewind to the first frame (for replaying a scenario)."""
        self._index = 0


class TelloVideoSource(FrameSource):
    """Real-Tello source — STUB. Ground/virtual only.

    Instantiating (or reading) raises: a live run needs the real drone,
    djitellopy, and the user's go-ahead. Nothing here fakes a stream.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Tello source needs the real drone; use --mode virtual "
            "(ground/virtual only — no live drone in this build).")

    def read(self):
        raise RuntimeError(
            "Tello source needs the real drone; use --mode virtual "
            "(ground/virtual only — no live drone in this build).")
