"""Frame sources for the wall-marker pipeline.

* :class:`FileVideoSource` — a recorded video file + a JSON pose list.
* :class:`SyntheticSource` — in-memory frames from the test renderer.
* :class:`TelloVideoSource` — real-Tello source via djitellopy
  (live mode). Detects a dead video stream (frozen frame / stopped
  reader) and restarts it, up to 3 attempts per run, then stays
  honestly dead instead of serving a frozen frame.

Every source yields ``(frame_bgr, drone_pose, tracking_valid, meta)``
where ``drone_pose`` is ``(R, C)`` — R the world-to-camera rotation, C
the drone camera centre in world coordinates — and ``meta`` is a dict
of per-frame extras (may be empty). ``read()`` returns ``None`` at the
end of the stream.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Video-death detection (TelloVideoSource): djitellopy's decode thread
# can die — 2026-11-25 live run: ``av.error.OSError`` on
# ``udp://@0.0.0.0:11111`` — and leave the last frame sitting in the
# reader forever. A frame object unchanged for this long means the
# stream is dead, not the scene standing still.
VIDEO_FROZEN_AFTER_S = 3.0
# Restart attempts per run before giving up honestly (never serve a
# frozen frame as if it were live).
VIDEO_RESTART_MAX = 3


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

    Video death: if the decode thread dies, the reader keeps the
    last frame forever. ``read()`` treats a frame object unchanged
    for more than ``VIDEO_FROZEN_AFTER_S`` (or a stopped reader) as
    a dead stream and restarts it (``streamoff`` / ``streamon`` /
    fresh ``get_frame_read``), up to ``VIDEO_RESTART_MAX`` attempts
    per run; after that the source stays dead and ``read()`` returns
    ``None`` — an honest pause the pipeline loop already handles —
    instead of a frozen frame.
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
        # Video-death tracking (see _restart_video): the frame object
        # last served, and the wall-clock time it first appeared.
        self._last_frame = None
        self._last_frame_at = time.monotonic()
        self._video_restarts = 0
        self._video_dead = False

    def read(self):
        if not hasattr(self, "frame_reader") or self.frame_reader is None:
            return None
        if getattr(self, "_video_dead", False):
            return None
        frame = self.frame_reader.frame
        if frame is None:
            return None
        # Dead-video detection: djitellopy's decode thread can die
        # (2026-11-25 live run: av.error.OSError on udp://@:11111)
        # and leave the last frame in the reader. The reader's
        # ``stopped`` flag may never be set — the exception kills the
        # thread — so a frame object unchanged for > 3 s is the
        # primary signal. Never serve a frozen frame as live.
        # NOTE: the timestamp marks when the frame object FIRST
        # appeared, so it is only updated when a *new* object
        # arrives; refreshing it on every read of the same object
        # would stop a frozen frame ever ageing out.
        now = time.monotonic()
        last_frame = getattr(self, "_last_frame", None)
        last_frame_at = getattr(self, "_last_frame_at", now)
        stopped = getattr(self.frame_reader, "stopped", False)
        if stopped or (frame is last_frame
                       and now - last_frame_at > VIDEO_FROZEN_AFTER_S):
            self._restart_video()
            return None
        if frame is not last_frame:
            self._last_frame = frame
            self._last_frame_at = now
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

    def _restart_video(self):
        """Restart the drone video stream after the decode thread died.

        The 2026-11-25 live run died with ``av.error.OSError:
        [Errno 5] I/O error: 'udp://@0.0.0.0:11111'`` in djitellopy's
        ``update_frame`` thread, leaving a stale frame in the reader
        while ``read()`` kept serving it as if live. Most likely
        trigger: the Tello app on the phone grabbed the video stream
        (the drone sends video to one device).

        At most VIDEO_RESTART_MAX attempts per run; after that the
        source stays dead and ``read()`` returns None until the
        server is restarted. Returns True if a fresh reader is in
        place, False otherwise.
        """
        restarts = getattr(self, "_video_restarts", 0)
        if restarts >= VIDEO_RESTART_MAX:
            if not getattr(self, "_video_dead", False):
                self._video_dead = True
                logger.error(
                    "VIDEO LOST: the drone stopped sending video after "
                    "%d restart attempts. The Tello app on the phone is "
                    "probably stealing the video stream - force-close "
                    "the Tello app, reboot the drone and restart the "
                    "server.", VIDEO_RESTART_MAX)
            return False
        self._video_restarts = restarts + 1
        logger.warning(
            "VIDEO LOST: restarting the drone video stream "
            "(attempt %d/%d)", self._video_restarts, VIDEO_RESTART_MAX)
        try:
            self.tello.streamoff()
        except Exception:
            pass
        try:
            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read(with_queue=True)
        except Exception as exc:
            logger.warning(
                "VIDEO LOST: restart attempt %d failed: %s",
                self._video_restarts, exc)
            return False
        # Fresh reader: forget the frozen frame so its first frame is
        # accepted as new.
        self._last_frame = None
        self._last_frame_at = time.monotonic()
        return True

    def close(self):
        try:
            if hasattr(self, "tello") and self.tello:
                self.tello.streamoff()
        except Exception:
            pass
