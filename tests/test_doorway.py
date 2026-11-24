"""Tests for server.doorway — DoorwayDetector.

Synthetic frames only (numpy/cv2 drawing on a textured background),
no live drone, no network. Run with:
    python -m pytest tests/test_doorway.py -q
"""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from server.doorway import Doorway, DoorwayDetector

W, H = 960, 720


def _textured_background(seed=0):
    """A wall-like textured background: smooth noise + faint streaks."""
    rng = np.random.default_rng(seed)
    base = rng.integers(150, 200, size=(H, W), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (9, 9), 0)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # Faint horizontal wall streaks (must NOT be mistaken for a door).
    for y in range(60, H, 110):
        cv2.line(img, (0, y), (W, y), (120, 120, 120), 1)
    return img


def _draw_door(img, left_x, right_x, y0=40, y1=700, thickness=6):
    """Draw a doorway: two dark jambs + a lintel across the top."""
    cv2.line(img, (left_x, y0), (left_x, y1), (30, 30, 30), thickness)
    cv2.line(img, (right_x, y0), (right_x, y1), (30, 30, 30), thickness)
    cv2.line(img, (left_x, y0), (right_x, y0), (30, 30, 30), thickness)
    return img


@pytest.fixture()
def detector():
    return DoorwayDetector()


def test_synthetic_door_found_within_tolerance(detector):
    img = _draw_door(_textured_background(), 300, 620)
    result = detector.detect(img)
    assert result is not None, "doorway not detected on synthetic door"
    assert isinstance(result, Doorway)
    # Centre within ~10% of frame width of truth (truth centre 460).
    assert abs(result.center_x - 460) <= 0.10 * W
    # Width within ~25% of truth (truth width 320).
    assert abs(result.width - 320) <= 0.25 * 320
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.3
    assert 0 <= result.left_x < result.right_x <= W
    assert abs((result.left_x + result.right_x) / 2 - result.center_x) < 1.0
    assert abs(result.right_x - result.left_x - result.width) < 1.0


def test_synthetic_door_other_position_and_size(detector):
    img = _draw_door(_textured_background(seed=3), 500, 800, y0=20, y1=710)
    result = detector.detect(img)
    assert result is not None
    assert abs(result.center_x - 650) <= 0.10 * W
    assert abs(result.width - 300) <= 0.25 * 300


def test_plain_wall_returns_none(detector):
    img = _textured_background(seed=1)
    assert detector.detect(img) is None


def test_blank_frame_returns_none(detector):
    blank = np.full((H, W, 3), 180, dtype=np.uint8)
    assert detector.detect(blank) is None


def test_small_poster_rectangle_rejected(detector):
    """Poster-sized rectangle must be rejected (None) or very low conf."""
    img = _textured_background(seed=2)
    cv2.rectangle(img, (350, 200), (550, 480), (30, 30, 30), 5)
    result = detector.detect(img)
    assert result is None or result.confidence < 0.3


def test_door_partially_at_frame_edge_no_crash(detector):
    """Right jamb near the frame edge: plausible centre or None, never
    a crash and never a wild centre outside the frame."""
    img = _draw_door(_textured_background(seed=4), 700, 920)
    result = detector.detect(img)
    if result is not None:
        assert 0 <= result.center_x <= W
        assert 0 <= result.left_x < result.right_x <= W
        assert abs(result.center_x - 810) <= 0.15 * W


def test_single_jamb_only_returns_none(detector):
    """One jamb alone is not a doorway."""
    img = _textured_background(seed=5)
    cv2.line(img, (400, 40), (400, 700), (30, 30, 30), 6)
    assert detector.detect(img) is None


def test_tiny_and_degenerate_frames_do_not_crash(detector):
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    assert detector.detect(np.zeros((H, W), dtype=np.uint8)) is None
    assert detector.detect(None) is None


def test_module_does_not_import_server_main():
    """Importing server.doorway alone must not pull in server.main
    (checked in a fresh subprocess, since this session's other test
    files may already have loaded it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "server", "doorway.py")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    assert "server.main" not in source
    assert "import main" not in source
    assert "from server" not in source
    assert "import server" not in source
    repo_root = os.path.dirname(here)
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import server.doorway; "
        "assert 'server.main' not in sys.modules, 'doorway pulled in server.main'"
        % repo_root
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
