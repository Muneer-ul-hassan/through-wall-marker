"""Live-path regression tests (BUG 1 + BUG 2 fixes, 2026-11-24).

Standalone: run from the project root with the project venv::

    ~/workspace/tello-search/.venv/bin/python tests/test_live_fixes.py

No drone, no flight, no network — the Tello is a fake, the VIO runs on
synthetic frames, the pipeline runs on a stub detector. NOT wired into
tests/run_tests.py (that runner is the 27-scenario virtual suite; these
are live-path unit tests).

What each test proves:
* SafeFlightController (BUG 1, the other AI's controller from commit
  d89be655, kept as the ONE flight controller): one serialized sender
  (lock, non-blocking acquire, busy commands skipped); movement gated
  on a successful takeoff (status HOVERING); land sent exactly once
  (busy-skip + the LANDED idempotency guard); "Not joystick" logs the
  force-close-the-app guidance and leaves the drone LANDED, so a
  retry after fixing the app is accepted (no session block). Added
  2026-11-24 after the live run: a move failing with "Not joystick"
  or "No valid imu" also resets to LANDED (the drone is down or no
  longer under SDK control, so [T] must work again), the rejection
  is labelled MOVE not TAKEOFF, and a phone browser disconnect
  (WinError 10053) is swallowed instead of dumped as a traceback.
* live_pose_step (BUG 2): the VIO verdict passes through honestly — a
  failed VIO is never papered over with the frozen start pose.
* Pipeline gate (BUG 2): with tracking invalid, a detected person
  still produces NO marker (bridge invalid). The control shows the
  hazard the fix removes: the same frozen pose fed as tracking VALID
  *does* emit a marker — from a fiction frame.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.main import Pipeline, SafeFlightController, live_pose_step  # noqa: E402
from server.vio import MonocularVO  # noqa: E402

START_POSE = (np.eye(3), np.array([0.0, 1.3, 0.0]))


def wait_until(pred, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


# ---------------------------------------------------------------------------
# Fake Tello (BUG 1)
# ---------------------------------------------------------------------------

class FakeTello:
    """Records every call in order; tracks concurrent execution."""

    def __init__(self, takeoff_error=None, move_error=None, move_delay=0.0):
        self.takeoff_error = takeoff_error
        self.move_error = move_error
        self.move_delay = move_delay
        self.calls = []
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0

    def _record(self, name, *args):
        if self.move_error is not None and (
                name.startswith("move_") or name.startswith("rotate_")):
            raise Exception(self.move_error)
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            if self.move_delay:
                time.sleep(self.move_delay)
        finally:
            with self._lock:
                self._inflight -= 1
        with self._lock:
            self.calls.append((name, args))

    def takeoff(self):
        if self.takeoff_error is not None:
            raise Exception(self.takeoff_error)
        self._record("takeoff")

    def land(self):
        self._record("land")

    def move_forward(self, cm):
        self._record("move_forward", cm)

    def move_back(self, cm):
        self._record("move_back", cm)

    def move_left(self, cm):
        self._record("move_left", cm)

    def move_right(self, cm):
        self._record("move_right", cm)

    def move_up(self, cm):
        self._record("move_up", cm)

    def move_down(self, cm):
        self._record("move_down", cm)

    def rotate_clockwise(self, deg):
        self._record("rotate_clockwise", deg)

    def rotate_counter_clockwise(self, deg):
        self._record("rotate_counter_clockwise", deg)

    def names(self):
        with self._lock:
            return [c[0] for c in self.calls]


import logging


def wait_idle(ctrl, timeout=3.0):
    """Wait until the controller's lock is free (no command in flight)."""
    return wait_until(lambda: not ctrl.lock.locked(), timeout=timeout)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def test_takeoff_not_joystick_blocks_moves():
    """Takeoff answered 'Not joystick' -> no move ever reaches the drone,
    and the log tells the user to force-close the Tello app."""
    fake = FakeTello(takeoff_error=(
        "Command 'takeoff' was unsuccessful for 4 tries. "
        "Latest response: 'error Not joystick'"))
    ctrl = SafeFlightController(fake)
    cap = LogCapture()
    logging.getLogger("server.main").addHandler(cap)
    try:
        ctrl.send_command("takeoff", fake.takeoff)
        assert wait_until(lambda: ctrl.status == "LANDED", timeout=3.0), \
            f"status after failed takeoff: {ctrl.status}"
        assert wait_idle(ctrl), "controller lock still held after failure"
        ctrl.send_command("forward 30", fake.move_forward, 30)
        ctrl.send_command("back 30", fake.move_back, 30)
        ctrl.send_command("turn_right 30", fake.rotate_clockwise, 30)
        ctrl.send_command("land", fake.land)
        time.sleep(0.4)
        assert fake.names() == [], \
            f"commands reached the drone after a failed takeoff: {fake.names()}"
        assert any("Force-close" in m for m in cap.records), \
            "no 'force-close the Tello app' guidance logged"
    finally:
        logging.getLogger("server.main").removeHandler(cap)
    print("PASS takeoff_not_joystick_blocks_moves")


def test_takeoff_retry_after_not_joystick():
    """After a 'Not joystick' failure, fixing the app and pressing [T]
    again is accepted -- no session block, just the failed attempt."""
    fake = FakeTello(takeoff_error="error Not joystick")
    ctrl = SafeFlightController(fake)
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "LANDED", timeout=3.0)
    assert wait_idle(ctrl)
    assert fake.names() == [], "failed takeoff reached the drone"
    fake.takeoff_error = None  # user force-closed the app, rebooted
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0), \
        f"retry takeoff did not reach HOVERING: {ctrl.status}"
    assert fake.names() == ["takeoff"]
    print("PASS takeoff_retry_after_not_joystick")


def test_movement_gated_on_takeoff():
    """Moves before takeoff are dropped; after takeoff they execute in order."""
    fake = FakeTello()
    ctrl = SafeFlightController(fake)
    ctrl.send_command("forward 30", fake.move_forward, 30)
    ctrl.send_command("turn_right 30", fake.rotate_clockwise, 30)
    time.sleep(0.3)
    assert fake.names() == [], \
        f"movement sent while grounded: {fake.names()}"
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0)
    assert wait_idle(ctrl)
    ctrl.send_command("forward 30", fake.move_forward, 30)
    assert wait_until(lambda: "move_forward" in fake.names(), timeout=3.0)
    assert wait_idle(ctrl)
    ctrl.send_command("turn_right 30", fake.rotate_clockwise, 30)
    assert wait_until(
        lambda: fake.names() == ["takeoff", "move_forward",
                                 "rotate_clockwise"], timeout=3.0), \
        f"unexpected call order: {fake.names()}"
    print("PASS movement_gated_on_takeoff")


def test_land_exactly_once():
    """Five rapid lands + one more after landing still sends one land."""
    fake = FakeTello()
    ctrl = SafeFlightController(fake)
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0)
    assert wait_idle(ctrl)
    for _ in range(5):
        ctrl.send_command("land", fake.land)
    assert wait_until(lambda: fake.names().count("land") == 1, timeout=3.0), \
        f"land calls after burst: {fake.names()}"
    assert wait_idle(ctrl)
    ctrl.send_command("land", fake.land)  # already LANDED: guard must drop it
    time.sleep(0.3)
    assert fake.names().count("land") == 1, \
        f"land sent more than once: {fake.names()}"
    print("PASS land_exactly_once")


def test_move_not_joystick_resets_to_landed():
    """A move failing with 'Not joystick' mid-flight resets the
    controller to LANDED, so [T] is accepted again instead of being
    ignored as 'already HOVERING' — and the log names the MOVE, not
    a takeoff. (2026-11-24 live run: forward 30 failed this way
    while the drone was on the ground and five [T] presses did
    nothing.)"""
    fake = FakeTello()
    ctrl = SafeFlightController(fake)
    cap = LogCapture()
    logging.getLogger("server.main").addHandler(cap)
    try:
        ctrl.send_command("takeoff", fake.takeoff)
        assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0)
        assert wait_idle(ctrl)
        fake.move_error = (
            "Command 'forward 30' was unsuccessful for 4 tries. "
            "Latest response: 'error Not joystick'")
        ctrl.send_command("forward 30", fake.move_forward, 30)
        assert wait_until(lambda: ctrl.status == "LANDED", timeout=3.0), \
            f"status after Not-joystick move: {ctrl.status}"
        assert wait_idle(ctrl)
        assert any("MOVE REJECTED" in m for m in cap.records), \
            "move failure was not labelled MOVE"
        assert not any("TAKEOFF REJECTED" in m for m in cap.records), \
            "move failure was mislabelled as a takeoff rejection"
        assert any("Force-close" in m for m in cap.records), \
            "no force-close guidance logged"
        fake.move_error = None  # app closed, drone rebooted
        ctrl.send_command("takeoff", fake.takeoff)
        assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0), \
            f"[T] after a failed move did not reach HOVERING: {ctrl.status}"
    finally:
        logging.getLogger("server.main").removeHandler(cap)
    print("PASS move_not_joystick_resets_to_landed")


def test_move_no_valid_imu_resets_to_landed():
    """'No valid imu' on a move means the drone is down: same reset,
    so [T] works again."""
    fake = FakeTello()
    ctrl = SafeFlightController(fake)
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0)
    assert wait_idle(ctrl)
    fake.move_error = (
        "Command 'forward 30' was unsuccessful for 4 tries. "
        "Latest response: 'error No valid imu'")
    ctrl.send_command("forward 30", fake.move_forward, 30)
    assert wait_until(lambda: ctrl.status == "LANDED", timeout=3.0), \
        f"status after No-valid-imu move: {ctrl.status}"
    assert wait_idle(ctrl)
    fake.move_error = None
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0), \
        f"[T] after a No-valid-imu move did not reach HOVERING: {ctrl.status}"
    print("PASS move_no_valid_imu_resets_to_landed")


def test_handle_error_swallows_disconnect():
    """A phone browser closing the viewer mid-request (WinError 10053)
    is swallowed, not dumped as a traceback; a real error still is."""
    import contextlib
    import io

    from server.main import RobustThreadingHTTPServer
    srv = RobustThreadingHTTPServer.__new__(RobustThreadingHTTPServer)
    try:
        raise ConnectionAbortedError(10053, "test disconnect")
    except ConnectionAbortedError:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            srv.handle_error(None, ("192.168.10.2", 58497))
        assert buf.getvalue() == "", \
            f"disconnect traceback was printed: {buf.getvalue()[:200]}"
    try:
        raise ValueError("real server fault")
    except ValueError:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            srv.handle_error(None, ("192.168.10.2", 58497))
        assert "ValueError" in buf.getvalue(), \
            "a real error was swallowed along with disconnects"
    print("PASS handle_error_swallows_disconnect")


def test_commands_serialized_and_coalesced():
    """Concurrent move calls never overlap on the drone and never backlog."""
    fake = FakeTello(move_delay=0.05)
    ctrl = SafeFlightController(fake)
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0)
    assert wait_idle(ctrl)
    barrier = threading.Barrier(10)
    threads = []
    for _ in range(10):
        def _fire():
            barrier.wait(timeout=5)
            ctrl.send_command("forward 30", fake.move_forward, 30)
        threads.append(threading.Thread(target=_fire))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert wait_idle(ctrl, timeout=5.0)
    time.sleep(0.3)
    assert fake.max_inflight == 1, \
        f"drone commands overlapped: max_inflight={fake.max_inflight}"
    assert fake.names().count("move_forward") == 1, \
        "busy commands were queued instead of skipped: " \
        f"{fake.names().count('move_forward')}"
    print("PASS commands_serialized_and_coalesced")


# ---------------------------------------------------------------------------
# live_pose_step (BUG 2)
# ---------------------------------------------------------------------------

class StubVIO:
    def __init__(self, valid, pose=None):
        self.valid = valid
        self.pose = pose if pose is not None else START_POSE

    def update(self, frame, t):
        return self.pose, self.valid, {}


class RaisingVIO:
    def update(self, frame, t):
        raise RuntimeError("simulated VIO crash")


def test_pose_step_passes_verdict_through():
    """A stub VIO's verdict is passed through unchanged, never upgraded."""
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    _, _, valid = live_pose_step(
        StubVIO(False), frame, START_POSE, 0.0)
    assert valid is False, "invalid VIO was upgraded to valid"
    _, _, valid = live_pose_step(
        StubVIO(True), frame, START_POSE, 0.0)
    assert valid is True, "valid VIO verdict was lost"
    _, pose, valid = live_pose_step(
        RaisingVIO(), frame, START_POSE, 0.0)
    assert valid is False, "raising VIO was reported valid"
    assert pose is START_POSE, "raising VIO must return the start pose"
    print("PASS pose_step_passes_verdict_through")


def test_vio_honest_on_static_camera():
    """A real VIO fed identical static frames never reports valid.

    No motion means no parallax, so initialisation cannot succeed.
    (This is the live desk-camera case: the drone sitting still.)"""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 960, 3), dtype=np.uint8)
    vio = MonocularVO(start_pose=START_POSE)
    for i in range(10):
        _, valid, _diag = vio.update(frame, t=i * 0.1)
        assert valid is False, \
            f"VIO reported valid on static frame {i} (no motion observed)"
    print("PASS vio_honest_on_static_camera")


# ---------------------------------------------------------------------------
# Pipeline gate (BUG 2): no bridge -> no marker, even with a detection
# ---------------------------------------------------------------------------

def _stub_detection():
    kpts = np.array([
        [480, 230], [470, 225], [490, 225], [465, 228], [495, 228],
        [440, 280], [520, 280], [430, 350], [530, 350], [425, 420],
        [535, 420], [445, 430], [515, 430], [445, 470], [515, 470],
        [445, 500], [515, 500],
    ], dtype=float)
    return {
        "keypoints": kpts,
        "kpt_conf": np.full(17, 0.9),
        "bbox_xyxy": np.array([330.0, 200.0, 630.0, 502.0]),
        "conf": 0.95,
    }


class StubDetector:
    def detect(self, frame):
        return [_stub_detection()]


def test_no_marker_without_bridge():
    """Detection present + tracking invalid -> no marker, ever.

    This is the fixed live behaviour: the frozen start pose with an
    invalid VIO must not validate the bridge."""
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    pipe = Pipeline(StubDetector(), drone_id="drone1")
    for i in range(10):
        marker, info = pipe.process_frame(
            frame, START_POSE, False, t=i * 0.1)
        assert marker is None, f"marker emitted with invalid tracking (frame {i})"
        assert info["bridge_valid"] is False
        assert pipe.fusion.markers(i * 0.1) == []
    assert pipe.markers_emitted == 0
    print("PASS no_marker_without_bridge")


def test_forced_valid_would_emit_marker():
    """Control: the SAME frozen pose fed as tracking VALID *does* emit
    a marker — from a fiction frame. This is the hazard the live-loop
    fix removes; the pipeline trusts its inputs, so the loop must
    never feed it forced validity."""
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    pipe = Pipeline(StubDetector(), drone_id="drone1")
    seen = None
    for i in range(6):
        marker, _info = pipe.process_frame(
            frame, START_POSE, True, t=i * 0.1)
        if marker is not None:
            seen = marker
    assert seen is not None, \
        "control failed: forced-valid frozen pose emitted no marker"
    print("PASS forced_valid_would_emit_marker (hazard documented)")


def main():
    tests = [
        test_takeoff_not_joystick_blocks_moves,
        test_takeoff_retry_after_not_joystick,
        test_movement_gated_on_takeoff,
        test_land_exactly_once,
        test_move_not_joystick_resets_to_landed,
        test_move_no_valid_imu_resets_to_landed,
        test_handle_error_swallows_disconnect,
        test_commands_serialized_and_coalesced,
        test_pose_step_passes_verdict_through,
        test_vio_honest_on_static_camera,
        test_no_marker_without_bridge,
        test_forced_valid_would_emit_marker,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} live-fix tests PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
