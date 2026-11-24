"""Flight recorder for the through-wall person marker (live runs).

Auto-saves each run's annotated drone video plus the console log into a
timestamped run folder, so a failure can be diagnosed from what the camera
actually saw.

Cheap and off the critical path by design: :meth:`RunRecorder.write_frame`
only does a light validity check and a non-blocking ``put_nowait`` into a
BOUNDED queue; a background writer thread owns the ``cv2.VideoWriter``.
If the queue is full the frame is dropped — the 4-5 fps pipeline loop is
never made to wait on disk I/O.

Wiring (done by the caller, e.g. ``server/main.py`` — this module does not
touch it)::

    run_dir = new_run_dir()                       # <repo>/runs/<timestamp>/
    handler = attach_file_log(run_dir / "server.log")
    recorder = RunRecorder(run_dir, fps=5)
    recorder.start()
    ...
    recorder.write_frame(vis)                     # inside pipeline_loop
    ...
    recorder.stop()
    logging.getLogger().removeHandler(handler)
    handler.close()

Dependency-free beyond cv2 / numpy / logging. No network, no ML.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

__all__ = ["new_run_dir", "RunRecorder", "attach_file_log"]

# Repo root (``server/``'s parent); the default base for run folders.
REPO_ROOT = Path(__file__).resolve().parent.parent

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?$")

# Format shared with server/main.py's basicConfig so server.log reads the
# same as the console.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

_DEFAULT_MAX_QUEUE = 30


def new_run_dir(base_dir=None):
    """Create and return a timestamped run directory.

    ``<base_dir>/runs/<YYYY-MM-DD_HH-MM-SS>/``. ``base_dir`` defaults to
    the repo root. Two calls in the same second get different directories
    (``-1``, ``-2``, ... suffix on collision).
    """
    base = Path(base_dir) if base_dir is not None else REPO_ROOT
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / "runs" / stamp
    suffix = 0
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = base / "runs" / f"{stamp}-{suffix}"


class RunRecorder:
    """Non-blocking recorder for annotated drone frames.

    ``write_frame`` never blocks the caller and never raises: bad frames
    are skipped, a full queue drops the frame, and a frame of a different
    size than the first accepted frame is skipped (the video size is
    fixed by the first frame). The background writer thread owns the
    ``cv2.VideoWriter`` and writes ``video.mp4`` (mp4v) into ``run_dir``.
    """

    def __init__(self, run_dir, fps=5, max_queue=_DEFAULT_MAX_QUEUE):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.run_dir / "video.mp4"
        self.fps = fps
        self.max_queue = max_queue
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False
        self._size = None  # (height, width) of first accepted frame
        self.frames_written = 0
        self.frames_dropped = 0

    def start(self):
        """Start the background writer thread (idempotent)."""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop, name="run-recorder", daemon=True)
        self._thread.start()

    def write_frame(self, frame):
        """Queue ``frame`` for writing. Never blocks, never raises.

        Returns True if the frame was queued, False if it was dropped
        (bad frame, recorder not started/stopped, wrong size, or a full
        queue).
        """
        try:
            if not self._started or self._stop_event.is_set():
                self.frames_dropped += 1
                return False
            if not isinstance(frame, np.ndarray):
                self.frames_dropped += 1
                return False
            if frame.ndim != 3 or frame.size == 0:
                self.frames_dropped += 1
                return False
            if frame.dtype != np.uint8:
                # VideoWriter wants uint8 BGR; never convert here (that
                # would cost the caller) — drop non-uint8 frames.
                self.frames_dropped += 1
                return False
            shape = (frame.shape[0], frame.shape[1])
            size = self._size
            if size is not None and shape != size:
                # Wrong-size frame: the video size is fixed by the first
                # accepted frame; skip, never resize mid-stream.
                self.frames_dropped += 1
                return False
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self.frames_dropped += 1
                return False
            if self._size is None:
                # First accepted frame fixes the video size (single
                # caller in practice — the pipeline loop — so a race
                # here can only queue one extra frame, which the
                # writer thread's own size check still skips).
                self._size = shape
            return True
        except Exception:
            # The recorder must never take the pipeline loop down.
            try:
                self.frames_dropped += 1
            except Exception:
                pass
            return False

    def stop(self, timeout=10.0):
        """Flush the queue, close the video file, stop the thread.

        Idempotent; safe to call even if ``start()`` was never called.
        """
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=timeout)

    # -- internals (writer thread only) ---------------------------------

    def _writer_loop(self):
        writer = None
        size = None  # (width, height) fixed by the first accepted frame
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    frame = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                        continue
                    h, w = frame.shape[0], frame.shape[1]
                    if size is None:
                        size = (w, h)
                        writer = cv2.VideoWriter(
                            str(self.video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            self.fps, size)
                        if not writer.isOpened():
                            writer = None
                            size = None
                            continue
                    elif (w, h) != size:
                        # Wrong-size frame: skip, never resize mid-stream.
                        continue
                    if writer is None:
                        continue
                    writer.write(frame)
                    self.frames_written += 1
                except Exception:
                    # A bad frame must never kill the writer thread.
                    continue
                finally:
                    self._queue.task_done()
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass


def attach_file_log(path):
    """Attach a logging.FileHandler (INFO+) for ``path`` to the root logger.

    Returns the handler so the caller can remove it on shutdown. Calling
    twice with the same path returns the existing handler instead of
    adding a duplicate (which would double-write every line).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                existing = Path(handler.baseFilename).resolve()
            except Exception:
                continue
            if existing == resolved:
                return handler
    handler = logging.FileHandler(str(path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    root.addHandler(handler)
    # INFO records must actually reach the handler: if the root level is
    # above INFO (or unset, i.e. effectively WARNING), lower it to INFO.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler
