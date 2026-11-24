"""Autonomous-mode wiring tests (server/main.py, 2026-11-25).

Standalone: run from the project root with the project venv::

    ~/workspace/tello-search/.venv/bin/python -m pytest tests/test_wiring.py -q

No drone, no flight, no network — the Tello is a fake, the autopilot
runs against a stub controller, the operator endpoint runs on a
local ephemeral port. NOT wired into tests/run_tests.py (that runner
is the 27-scenario virtual suite; these are live-path unit tests).

What each test proves:
* U key STARTS the autopilot (start-only, 2026-11-25 user's order:
  a second U while it runs does NOTHING — the old toggle-off made
  him kill the autopilot right after starting it by pressing U
  again), start WITH the current ctx, so the start gate enforces
  HOVERING + tracking_valid; refused otherwise, and U with no
  autopilot wired does nothing.
* Any manual key stops a running autopilot FIRST, then runs the
  manual command (manual takeover) — including after a U start.
* 20 cm steps: dispatch sends "forward 20" / move_forward(20);
  turns stay 30 deg.
* Not-joystick: SafeFlightController.last_reject is "not-joystick"
  after a rejection, None after a success, the lowercased error
  otherwise; the loop-side check stops a running autopilot and
  surfaces the factory-reset message (the app is ruled out —
  DJITelloPy issue #201 — so the reset is primary, force-close is
  only a secondary check).
* Hover-stable gate: no tick until 5 s of continuous HOVERING; the
  timer resets whenever the status leaves HOVERING; the gate lifts
  exactly once per hover episode.
* build_auto_context: max-confidence detection's bbox centre-x,
  battery "" / None -> None, "72" -> 72.0.
* format_status_line: old format unchanged without the new args;
  auto=<state> and height=<cm>cm appended when given.
* Map wiring via fakes: save/load round-trip, empty map saves
  nothing, missing file raises (startup treats it as "no saved
  map"), try_relocalise on an empty map is False and never raises.
* /operator endpoint: /api/status carries auto_state / auto_message
  / height_cm / not_joystick, and POST /api/autopilot/stop stops a
  running autopilot.
"""

from __future__ import annotations

import http.client
import json
import logging
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.autopilot import AutoContext, Autopilot  # noqa: E402
from server.main import (HoverGate, NOT_JOYSTICK_MESSAGE, ServerState,  # noqa: E402
                         SafeFlightController, build_auto_context,
                         check_not_joystick_stop, dispatch_drone_key,
                         format_status_line, make_handler)
from server.map_store import MapData, load_map, save_map, try_relocalise  # noqa: E402


def wait_until(pred, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def wait_idle(ctrl, timeout=3.0):
    return wait_until(lambda: not ctrl.lock.locked(), timeout=timeout)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTello:
    """Records every call; moves can be made to fail."""

    def __init__(self, move_error=None):
        self.move_error = move_error
        self.calls = []
        self._lock = threading.Lock()

    def _record(self, name, *args):
        if self.move_error is not None and (
                name.startswith("move_") or name.startswith("rotate_")):
            raise Exception(self.move_error)
        with self._lock:
            self.calls.append((name, args))

    def takeoff(self):
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


class StubController:
    """Minimal autopilot controller: HOVERING, records send_command,
    executes the move on its fake tello at once (synchronous)."""

    def __init__(self, status="HOVERING"):
        self.status = status
        self.tello = FakeTello()
        self.sent = []

    def send_command(self, name, func, *args):
        self.sent.append((name, args))
        func(*args)


class StubAutopilot:
    """Autopilot stand-in for the loop-side stop check."""

    def __init__(self, state="SCAN"):
        self.state = state
        self.message = "stub"
        self.stops = 0

    def stop(self):
        self.stops += 1
        self.state = "OFF"
        self.message = "autopilot stopped"


def _hovering_ctrl(fake):
    """A SafeFlightController forced to HOVERING (no real takeoff)."""
    ctrl = SafeFlightController(fake)
    ctrl.status = "HOVERING"
    return ctrl


def _ctx(tracking_valid=True):
    return AutoContext(tracking_valid=tracking_valid, bridge_valid=False,
                       battery_pct=72.0, doorway=None,
                       person_visible=False, person_center_x=None, time=0.0)


# ---------------------------------------------------------------------------
# U key start-only + start gate (2026-11-25, user's order)
# ---------------------------------------------------------------------------

def test_u_is_start_only_second_press_ignored():
    """U starts the autopilot (with ctx, so the gate applies) and a
    second, third, fourth U while it runs does NOTHING — U is
    start-only, not a toggle. The old toggle-off made the user kill
    the autopilot right after starting it by pressing U again."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    auto = Autopilot(ctrl)
    assert dispatch_drone_key(ord("u"), ctrl, auto, _ctx()) is True
    assert auto.state == "SCAN", f"U did not start the autopilot: {auto.state}"
    for _ in range(3):
        assert dispatch_drone_key(ord("u"), ctrl, auto, _ctx()) is True
        assert auto.state == "SCAN", \
            f"repeat U stopped the autopilot: {auto.state}"
    print("PASS u_is_start_only_second_press_ignored")


def test_manual_key_still_takes_over_after_u_start():
    """Start via U, then W: the manual key stops the autopilot FIRST
    and the move still reaches the drone — U being start-only did
    not remove manual takeover."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    auto = Autopilot(ctrl)
    assert dispatch_drone_key(ord("u"), ctrl, auto, _ctx()) is True
    assert auto.state == "SCAN"
    assert dispatch_drone_key(ord("w"), ctrl, auto, _ctx()) is True
    assert auto.state == "OFF", "manual key did not stop the autopilot"
    assert wait_until(lambda: ("move_forward", (20,)) in fake.calls,
                      timeout=3.0), f"move not sent: {fake.calls}"
    assert wait_idle(ctrl)
    print("PASS manual_key_still_takes_over_after_u_start")


def test_u_start_refused_when_not_hovering_or_no_tracking():
    """The U start passes the current ctx, so the gate refuses when
    grounded or untracked — the autopilot stays OFF either way."""
    fake = FakeTello()
    ctrl = SafeFlightController(fake)  # LANDED, never took off
    auto = Autopilot(ctrl)
    assert dispatch_drone_key(ord("u"), ctrl, auto, _ctx()) is True
    assert auto.state == "OFF", "U started the autopilot while LANDED"
    assert "not hovering" in auto.message
    ctrl2 = _hovering_ctrl(FakeTello())
    auto2 = Autopilot(ctrl2)
    assert dispatch_drone_key(ord("u"), ctrl2, auto2,
                              _ctx(tracking_valid=False)) is True
    assert auto2.state == "OFF", "U started with no tracking"
    assert "no tracking" in auto2.message
    print("PASS u_start_refused_when_not_hovering_or_no_tracking")


def test_u_without_autopilot_does_nothing():
    """No autopilot wired (old two-argument callers): U is an
    unknown key, nothing happens."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    assert dispatch_drone_key(ord("u"), ctrl) is False
    assert dispatch_drone_key(ord("u"), ctrl, None, _ctx()) is False
    assert wait_idle(ctrl)
    assert fake.names() == []
    print("PASS u_without_autopilot_does_nothing")


# ---------------------------------------------------------------------------
# Manual takeover + 20 cm steps
# ---------------------------------------------------------------------------

def test_manual_key_stops_autopilot_before_executing():
    """W while the autopilot runs: the autopilot is OFF before the
    move reaches the drone — manual takeover, not a fight."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    auto = Autopilot(ctrl)
    assert auto.start(_ctx()) is True
    assert auto.state == "SCAN"
    assert dispatch_drone_key(ord("w"), ctrl, auto, _ctx()) is True
    assert auto.state == "OFF", "manual key did not stop the autopilot"
    assert wait_until(lambda: ("move_forward", (20,)) in fake.calls,
                      timeout=3.0), f"move not sent: {fake.calls}"
    assert wait_idle(ctrl)
    print("PASS manual_key_stops_autopilot_before_executing")


def test_dispatch_sends_20cm_steps_and_30deg_turns():
    """W/S/A/D/I/K send 20 cm; J/L still turn 30 deg."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    cap = LogCapture()
    logger = logging.getLogger("server.main")
    logger.addHandler(cap)
    # INFO is filtered by default (root level WARNING); the live
    # server configures INFO via basicConfig, so set it here too.
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        for key, name in ((ord("w"), "move_forward"),
                          (ord("s"), "move_back"),
                          (ord("a"), "move_left"),
                          (ord("d"), "move_right"),
                          (ord("i"), "move_up"),
                          (ord("k"), "move_down")):
            assert dispatch_drone_key(key, ctrl) is True
            assert wait_until(lambda n=name: (n, (20,)) in fake.calls,
                              timeout=3.0), \
                f"{name} did not get a 20 cm step: {fake.calls}"
            assert wait_idle(ctrl)
        assert any("forward 20" in m for m in cap.records), \
            "command name was not 'forward 20'"
        assert not any("forward 30" in m for m in cap.records), \
            "a 30 cm step is still being sent"
        assert dispatch_drone_key(ord("j"), ctrl) is True
        assert wait_until(
            lambda: ("rotate_counter_clockwise", (30,)) in fake.calls,
            timeout=3.0)
        assert dispatch_drone_key(ord("l"), ctrl) is True
        assert wait_until(
            lambda: ("rotate_clockwise", (30,)) in fake.calls,
            timeout=3.0)
    finally:
        logger.removeHandler(cap)
        logger.setLevel(old_level)
    print("PASS dispatch_sends_20cm_steps_and_30deg_turns")


# ---------------------------------------------------------------------------
# Not-joystick: last_reject + loop-side hard stop
# ---------------------------------------------------------------------------

def test_last_reject_set_on_not_joystick_and_cleared_on_success():
    """A move rejected with 'error Not joystick' sets last_reject;
    the next successful command clears it again."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    assert ctrl.last_reject is None
    fake.move_error = "error Not joystick"
    ctrl.send_command("forward 20", fake.move_forward, 20)
    assert wait_until(lambda: ctrl.status == "LANDED", timeout=3.0)
    assert wait_idle(ctrl)
    assert ctrl.last_reject == "not-joystick", \
        f"last_reject after rejection: {ctrl.last_reject!r}"
    fake.move_error = None
    ctrl.send_command("takeoff", fake.takeoff)
    assert wait_until(lambda: ctrl.status == "HOVERING", timeout=3.0)
    assert wait_idle(ctrl)
    assert ctrl.last_reject is None, \
        f"last_reject not cleared by a success: {ctrl.last_reject!r}"
    print("PASS last_reject_set_on_not_joystick_and_cleared_on_success")


def test_last_reject_other_error_is_lowercased_error():
    """A different rejection records the lowercased error text —
    not 'not-joystick', so the hard stop does not fire on it."""
    fake = FakeTello()
    ctrl = _hovering_ctrl(fake)
    fake.move_error = "Command 'forward 20' failed: error No valid imu"
    ctrl.send_command("forward 20", fake.move_forward, 20)
    assert wait_until(lambda: ctrl.status == "LANDED", timeout=3.0)
    assert wait_idle(ctrl)
    assert ctrl.last_reject is not None
    assert ctrl.last_reject != "not-joystick"
    assert "no valid imu" in ctrl.last_reject
    assert ctrl.last_reject == ctrl.last_reject.lower()
    print("PASS last_reject_other_error_is_lowercased_error")


def test_not_joystick_stop_halts_running_autopilot():
    """Loop-side check: running autopilot + last_reject
    'not-joystick' -> stopped, and the factory-reset message is
    recorded on the operator state."""
    state = ServerState()
    auto = StubAutopilot(state="SCAN")
    flight_ctrl = SimpleNamespace(last_reject="not-joystick")
    assert check_not_joystick_stop(flight_ctrl, auto, state) is True
    assert auto.state == "OFF"
    assert auto.stops == 1
    payload = state.status_payload()
    assert payload["not_joystick"] == NOT_JOYSTICK_MESSAGE
    assert "reset the drone (hold power 5 s until yellow flash)" \
        in NOT_JOYSTICK_MESSAGE
    assert "force-close the Tello app completely" in NOT_JOYSTICK_MESSAGE
    print("PASS not_joystick_stop_halts_running_autopilot")


def test_not_joystick_stop_ignores_off_autopilot_and_clean_link():
    """No stop when the autopilot is already OFF, or when the last
    command was not a Not-joystick rejection."""
    state = ServerState()
    auto = StubAutopilot(state="OFF")
    assert check_not_joystick_stop(
        SimpleNamespace(last_reject="not-joystick"), auto, state) is False
    assert auto.stops == 0
    auto2 = StubAutopilot(state="SCAN")
    assert check_not_joystick_stop(
        SimpleNamespace(last_reject=None), auto2, state) is False
    assert auto2.stops == 0
    assert check_not_joystick_stop(
        SimpleNamespace(last_reject="error no valid imu"), auto2,
        state) is False
    assert auto2.stops == 0
    print("PASS not_joystick_stop_ignores_off_autopilot_and_clean_link")


# ---------------------------------------------------------------------------
# Hover-stable gate
# ---------------------------------------------------------------------------

def test_hover_gate_needs_5s_continuous_hover():
    """Not stable before 5 s, stable at 5 s, and the lift is
    reported exactly once per hover episode."""
    gate = HoverGate()
    assert gate.is_stable(0.0) is False
    assert gate.update("HOVERING", 0.0) is False
    assert gate.is_stable(4.9) is False
    assert gate.update("HOVERING", 4.9) is False
    assert gate.is_stable(5.0) is True
    assert gate.update("HOVERING", 5.0) is True  # lifts once
    assert gate.update("HOVERING", 6.0) is False  # never twice
    assert gate.is_stable(6.0) is True
    print("PASS hover_gate_needs_5s_continuous_hover")


def test_hover_gate_resets_when_status_leaves_hovering():
    """Any interruption restarts the 5 s clock from zero."""
    gate = HoverGate()
    gate.update("HOVERING", 0.0)
    assert gate.update("HOVERING", 5.0) is True
    assert gate.is_stable(5.0) is True
    assert gate.update("MOVING", 5.5) is False
    assert gate.is_stable(5.5) is False
    assert gate.update("HOVERING", 6.0) is False
    assert gate.is_stable(10.9) is False
    assert gate.is_stable(11.0) is True
    assert gate.update("HOVERING", 11.0) is True
    print("PASS hover_gate_resets_when_status_leaves_hovering")


# ---------------------------------------------------------------------------
# build_auto_context + status line
# ---------------------------------------------------------------------------

def _det(conf, x1, x2):
    return {"conf": conf,
            "bbox_xyxy": np.array([float(x1), 0.0, float(x2), 100.0])}


def test_build_auto_context_person_is_max_conf_centre_x():
    """The person is the max-confidence detection; centre-x is the
    bbox centre-x. No detections -> not visible, centre None."""
    ctx = build_auto_context(True, True, "72",
                             doorway=None,
                             detections=[_det(0.5, 0, 100),
                                         _det(0.9, 100, 300)],
                             t=1.5)
    assert ctx.person_visible is True
    assert ctx.person_center_x == 200.0, ctx.person_center_x
    assert ctx.battery_pct == 72.0
    assert ctx.tracking_valid is True and ctx.bridge_valid is True
    assert ctx.time == 1.5
    ctx = build_auto_context(False, False, "", None, [], 0.0)
    assert ctx.person_visible is False
    assert ctx.person_center_x is None
    assert ctx.battery_pct is None
    ctx = build_auto_context(True, False, None, None, None, 0.0)
    assert ctx.battery_pct is None
    print("PASS build_auto_context_person_is_max_conf_centre_x")


def test_status_line_additive_auto_and_height():
    """Old format is byte-identical without the new args; with them
    auto=<state> and height=<cm>cm are appended."""
    diag = {"n_tracked": 10, "inliers": 5, "n_map": 3,
            "pose_source": "vision"}
    old = format_status_line(50, "HOVERING", True, diag, True, 12)
    assert "auto=" not in old and "height=" not in old, old
    new = format_status_line(50, "HOVERING", True, diag, True, 12,
                             auto_state="SCAN", height_cm=120)
    assert new.startswith(old), f"old format not preserved: {new!r}"
    assert "auto=SCAN" in new, new
    assert "height=120cm" in new, new
    print("PASS status_line_additive_auto_and_height")


# ---------------------------------------------------------------------------
# Map wiring via fakes
# ---------------------------------------------------------------------------

def _sample_map_data():
    return MapData(
        points=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0],
                         [0.0, 1.0, 3.0]]),
        descriptors=np.zeros((3, 32), dtype=np.uint8),
        des_valid=np.array([True, True, True]),
        ids=np.array([0, 1, 2]),
        K={"fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 360.0,
           "width": 960, "height": 720},
        ref_R=np.eye(3), ref_C=np.zeros(3), ref_frame=2,
        keyframe_R=np.zeros((0, 3, 3)), keyframe_C=np.zeros((0, 3)),
        keyframe_frame=np.zeros(0, dtype=int),
        n_frames=3, n_keyframes=0)


def test_map_save_load_roundtrip_and_empty_saves_nothing():
    """save_map -> load_map returns the same points; an empty map
    saves nothing and writes no file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "latest.npz"
        assert save_map(path, _sample_map_data()) == 3
        assert path.exists()
        loaded = load_map(path)
        assert len(loaded) == 3
        assert np.allclose(loaded.points,
                           _sample_map_data().points)
        empty_path = Path(tmp) / "empty.npz"
        assert save_map(empty_path, MapData()) == 0
        assert not empty_path.exists()
    print("PASS map_save_load_roundtrip_and_empty_saves_nothing")


def test_map_load_missing_raises_and_relocalise_empty_is_false():
    """A missing map file raises (startup treats it as 'no saved
    map'); relocalising against an empty map is False, never a
    raise."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_map(Path(tmp) / "nope.npz")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("load_map on a missing file did "
                                 "not raise FileNotFoundError")
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    assert try_relocalise(object(), frame, MapData()) is False
    assert try_relocalise(object(), frame, None) is False
    print("PASS map_load_missing_raises_and_relocalise_empty_is_false")


# ---------------------------------------------------------------------------
# Operator endpoint
# ---------------------------------------------------------------------------

def test_operator_status_and_autopilot_stop_endpoint():
    """GET /api/status carries the auto fields; POST
    /api/autopilot/stop stops a running autopilot."""
    state = ServerState()
    state.update_auto("SCAN", "scanning for doorway", 120)
    state.set_not_joystick(NOT_JOYSTICK_MESSAGE)
    auto = StubAutopilot(state="SCAN")
    state.autopilot = auto
    handler = make_handler(state, Path("/nonexistent/viewer.html"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]

        def _req(method, path):
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                             timeout=5)
            conn.request(method, path)
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            return resp.status, body

        status, body = _req("GET", "/api/status")
        assert status == 200, status
        payload = json.loads(body.decode("utf-8"))
        assert payload["auto_state"] == "SCAN", payload
        assert payload["auto_message"] == "scanning for doorway"
        assert payload["height_cm"] == 120
        assert payload["not_joystick"] == NOT_JOYSTICK_MESSAGE
        status, body = _req("POST", "/api/autopilot/stop")
        assert status == 200, status
        assert json.loads(body.decode("utf-8"))["auto_state"] == "OFF"
        assert auto.state == "OFF"
        assert auto.stops == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("PASS operator_status_and_autopilot_stop_endpoint")


def main():
    tests = [
        test_u_is_start_only_second_press_ignored,
        test_manual_key_still_takes_over_after_u_start,
        test_u_start_refused_when_not_hovering_or_no_tracking,
        test_u_without_autopilot_does_nothing,
        test_manual_key_stops_autopilot_before_executing,
        test_dispatch_sends_20cm_steps_and_30deg_turns,
        test_last_reject_set_on_not_joystick_and_cleared_on_success,
        test_last_reject_other_error_is_lowercased_error,
        test_not_joystick_stop_halts_running_autopilot,
        test_not_joystick_stop_ignores_off_autopilot_and_clean_link,
        test_hover_gate_needs_5s_continuous_hover,
        test_hover_gate_resets_when_status_leaves_hovering,
        test_build_auto_context_person_is_max_conf_centre_x,
        test_status_line_additive_auto_and_height,
        test_map_save_load_roundtrip_and_empty_saves_nothing,
        test_map_load_missing_raises_and_relocalise_empty_is_false,
        test_operator_status_and_autopilot_stop_endpoint,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} wiring tests PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
