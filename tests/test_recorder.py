"""Tests for server/recorder.py (flight recorder).

Standalone: run from the project root with the project venv::

    ~/workspace/tello-search/.venv/bin/python -m pytest tests/test_recorder.py -q

No drone, no flight, no network — frames are synthetic numpy arrays and
all outputs go to tmp dirs (tempfile), never the real runs/ directory.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server import recorder  # noqa: E402
from server.recorder import RunRecorder, attach_file_log, new_run_dir  # noqa: E402

W, H = 320, 240


def make_frame(i, w=W, h=H):
    """Synthetic BGR frame; the moving bar keeps the encoder honest."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 12) % 256
    frame[:, :, 1] = 120
    frame[:, :, 2] = 200
    x = (i * 10) % (w - 40)
    frame[50:150, x:x + 40] = (255, 255, 255)
    return frame


def count_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot open {path}"
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def test_video_written_and_readable(tmp_path):
    run_dir = tmp_path / "run1"
    rec = RunRecorder(run_dir, fps=5)
    rec.start()
    for i in range(20):
        assert rec.write_frame(make_frame(i)) is True
    rec.stop()
    video = run_dir / "video.mp4"
    assert video.exists(), "video.mp4 was not created"
    assert video.stat().st_size > 0, "video.mp4 is empty"
    n = count_video_frames(video)
    # All 20 queued frames must be flushed on stop (allow a frame of
    # encoder slack either way).
    assert 18 <= n <= 20, f"expected ~20 frames, got {n}"


def test_queue_full_does_not_block(tmp_path):
    """With a tiny queue and a slow writer, write_frame returns fast
    and drops frames instead of hanging the caller."""
    run_dir = tmp_path / "run2"

    class SlowWriter:
        def __init__(self, *args, **kwargs):
            self._opened = True

        def isOpened(self):
            return True

        def write(self, frame):
            time.sleep(0.2)  # disk/encoder stall, simulated

        def release(self):
            pass

    rec = RunRecorder(run_dir, fps=5, max_queue=2)
    with mock.patch("cv2.VideoWriter", SlowWriter):
        rec.start()
        assert rec.write_frame(make_frame(0)) is True
        t0 = time.monotonic()
        results = [rec.write_frame(make_frame(i)) for i in range(1, 11)]
        elapsed = time.monotonic() - t0
        rec.stop()
    # 10 queued-while-slow calls must take ~0 s, not ~2 s of writer time.
    assert elapsed < 1.0, f"write_frame blocked: {elapsed:.2f}s for 10 frames"
    assert any(r is False for r in results), "full queue should drop frames"
    assert rec.frames_dropped > 0


def test_write_before_start_drops_and_returns_fast(tmp_path):
    """Frames written before start() (or after stop()) are dropped, not
    queued and not blocking — the recorder must be safe to call from the
    pipeline loop at any lifecycle point."""
    rec = RunRecorder(tmp_path / "run3", fps=5, max_queue=2)
    t0 = time.monotonic()
    results = [rec.write_frame(make_frame(i)) for i in range(10)]
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert all(r is False for r in results), "unstarted recorder drops all"
    assert rec.frames_dropped == 10
    rec.stop()  # safe even though never started
    assert not (tmp_path / "run3" / "video.mp4").exists()


def test_new_run_dir_timestamp_and_unique(tmp_path):
    d1 = new_run_dir(tmp_path)
    d2 = new_run_dir(tmp_path)
    assert d1.is_dir() and d2.is_dir()
    assert d1 != d2, "two calls must give different dirs"
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?$")
    assert pat.match(d1.name), f"bad timestamp name: {d1.name}"
    assert pat.match(d2.name), f"bad timestamp name: {d2.name}"
    assert d1.parent == tmp_path / "runs"
    assert d2.parent == tmp_path / "runs"


def test_new_run_dir_default_base_is_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "REPO_ROOT", tmp_path)
    d = new_run_dir()
    assert d.is_dir()
    assert d.parent == tmp_path / "runs"


def test_attach_file_log_writes_line(tmp_path):
    log_path = tmp_path / "run" / "server.log"
    logger = logging.getLogger("test_recorder_probe")
    handler = attach_file_log(log_path)
    try:
        logger.info("probe line %s", "hello-recorder")
        handler.flush()
        assert log_path.exists()
        assert "probe line hello-recorder" in log_path.read_text()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_attach_file_log_no_duplicate_handler(tmp_path):
    log_path = tmp_path / "server.log"
    h1 = attach_file_log(log_path)
    h2 = attach_file_log(log_path)
    try:
        assert h1 is h2, "same path must return the existing handler"
        root = logging.getLogger()
        same = [h for h in root.handlers
                if isinstance(h, logging.FileHandler)
                and Path(h.baseFilename).resolve() == log_path.resolve()]
        assert len(same) == 1, f"expected 1 handler, found {len(same)}"
    finally:
        logging.getLogger().removeHandler(h1)
        h1.close()


def test_bad_frames_never_raise(tmp_path):
    run_dir = tmp_path / "run4"
    rec = RunRecorder(run_dir, fps=5)
    rec.start()
    assert rec.write_frame(make_frame(0)) is True
    for bad in (None, "not an array", 42, np.zeros((10,), dtype=np.uint8),
                np.zeros((0, 0, 3), dtype=np.uint8),
                np.zeros((H, W, 3), dtype=np.float32),
                make_frame(1, w=160, h=120)):  # wrong size: skipped
        assert rec.write_frame(bad) is False
    assert rec.write_frame(make_frame(2)) is True
    rec.stop()
    n = count_video_frames(run_dir / "video.mp4")
    assert n == 2, f"only the 2 good same-size frames should be kept, got {n}"


def test_stop_is_idempotent_and_write_after_stop_is_safe(tmp_path):
    rec = RunRecorder(tmp_path / "run5", fps=5)
    rec.start()
    assert rec.write_frame(make_frame(0)) is True
    rec.stop()
    rec.stop()  # second stop must not raise or hang
    assert rec.write_frame(make_frame(1)) is False
    assert (tmp_path / "run5" / "video.mp4").exists()
