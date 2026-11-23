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
    """Real-Tello source via djitellopy.

    Connects to the Tello drone over Wi-Fi, starts video streaming,
    and yields (frame_bgr, start_pose, tracking_valid, meta).

    The pose returned is ONLY the start pose (takeoff beside the
    phone) for VIO initialisation, and ``tracking_valid`` is ALWAYS
    False from this source: the source has no pose estimate, only the
    VIO (server/vio.py) can vouch for tracking. Returning True here
    used to validate the bridge on a frozen pose and emit a garbage
    marker (2026-11-23 live run). The live loop passes the VIO
    verdict straight through instead (see server/main.py).
    """

    def __init__(self, start_height_m=1.3):
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError(
                "djitellopy is not installed. Install with: pip install djitellopy"
            ) from exc

        self.tello = Tello()
        try:
            self.tello.connect()
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Tello drone ({exc}). "
                "Ensure your laptop Wi-Fi is connected to the drone's network (TELLO-XXXXXX)."
            ) from exc

        try:
            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read(with_queue=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not start Tello video stream: {exc}"
            ) from exc

        self.start_height_m = float(start_height_m)
        self.initial_pose = (np.eye(3, dtype=float), np.array([0.0, self.start_height_m, 0.0], dtype=float))

    def read(self):
        if not hasattr(self, "frame_reader") or self.frame_reader is None:
            return None
        frame = self.frame_reader.frame
        if frame is None:
            return None
        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        meta = {}
        try:
            state = self.tello.get_current_state()
            if state and "bat" in state:
                meta["battery"] = int(state["bat"])
        except Exception:
            pass
        # tracking_valid=False: this source measures no pose. Only the
        # VIO may report tracking valid (server/main.py passes the VIO
        # verdict through; the frozen start pose must never validate
        # the bridge on its own).
        return frame_bgr, self.initial_pose, False, meta

    def close(self):
        try:
            if hasattr(self, "tello") and self.tello:
                self.tello.streamoff()
        except Exception:
            pass
