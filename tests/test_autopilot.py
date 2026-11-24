"""Autopilot state-machine tests (2026-11-25).

Standalone: run from the project root with the project venv::

    ~/workspace/tello-search/.venv/bin/python -m pytest tests/test_autopilot.py -q

No drone, no flight, no network — the controller and the Tello are
fakes, the doorway/person inputs are scripted. NOT wired into
tests/run_tests.py (that runner is the 27-scenario virtual suite).

What each test proves:
* Full mission: SCAN -> AIM -> APPROACH -> PASS -> SEARCH -> TRACK with
  scripted doorway/person inputs, 20 cm forward steps, 15 deg blind
  scan turns (SCAN / SEARCH), 30 deg correction turns (AIM / TRACK).
* AIM centres an off-centre doorway before approaching; APPROACH
  re-centres when the doorway drifts.
* PASS keeps stepping its bounded count when the doorway leaves view
  mid-pass (it fills the frame when close) and never chases it.
* Safety: tracking loss -> HOLD with no moves issued, resume of the
  pre-hold state on recovery; battery < 25 lands exactly once and ends
  DONE; a missing battery reading never lands; a busy controller gets
  nothing; stop() halts with no further moves.
* No doorway after a full 360 deg scan -> NODOOR, hovering, no forward
  moves (no wandering).
* tick() never blocks (returns quickly, thousands of ticks in < 2 s).
* The mover-mapping constructor path works for a controller with no
  ``.tello`` (move functions supplied explicitly).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.autopilot import AutoContext, Autopilot  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTello:
    """Records every drone call in order."""

    def __init__(self):
        self.calls = []

    def land(self):
        self.calls.append(("land", ()))

    def move_forward(self, cm):
        self.calls.append(("move_forward", (cm,)))

    def move_back(self, cm):
        self.calls.append(("move_back", (cm,)))

    def move_left(self, cm):
        self.calls.append(("move_left", (cm,)))

    def move_right(self, cm):
        self.calls.append(("move_right", (cm,)))

    def rotate_clockwise(self, deg):
        self.calls.append(("rotate_clockwise", (deg,)))

    def rotate_counter_clockwise(self, deg):
        self.calls.append(("rotate_counter_clockwise", (deg,)))


class FakeController:
    """Duck-typed SafeFlightController: .status, .tello, .send_command.

    send_command records the command and runs the move function at once
    (pretending success); a land also flips status to LANDED, as the
    real controller does when the drone is down.
    """

    def __init__(self):
        self.status = "HOVERING"
        self.tello = FakeTello()
        self.commands = []  # (name, args) in issue order

    def send_command(self, name, func, *args):
        self.commands.append((name, args))
        func(*args)
        if name == "land":
            self.status = "LANDED"

    def command_names(self):
        return [name for name, _ in self.commands]


class FakeControllerNoTello:
    """Controller with no .tello: moves come from a mover mapping."""

    def __init__(self, mover):
        self.status = "HOVERING"
        self.mover = mover
        self.commands = []

    def send_command(self, name, func, *args):
        self.commands.append((name, args))
        func(*args)


def doorway(center_x=480, width=100, confidence=0.9):
    return SimpleNamespace(center_x=center_x, width=width,
                           confidence=confidence)


def ctx(**kwargs):
    """AutoContext with tracking valid and no battery reading by default."""
    base = dict(tracking_valid=True, bridge_valid=False, battery_pct=None,
                doorway=None, person_visible=False, person_center_x=None,
                time=0.0)
    base.update(kwargs)
    return AutoContext(**base)


def run_until(ap, c, state, max_ticks=400):
    """Tick until ap.state == state (or max_ticks); returns ap.state."""
    for i in range(max_ticks):
        c.time = float(i)
        ap.tick(c)
        if ap.state == state:
            return ap.state
    return ap.state


def run_ticks(ap, c, n, start_time=0.0):
    for i in range(n):
        c.time = start_time + float(i)
        ap.tick(c)


# ---------------------------------------------------------------------------
# Full mission
# ---------------------------------------------------------------------------

def test_full_mission_scan_to_track():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    assert ap.state == "OFF"
    assert ap.start() is True
    assert ap.state == "SCAN"

    c = ctx()
    run_ticks(ap, c, 5)  # scanning, no doorway yet
    assert ap.state == "SCAN"
    # One 15-deg turn per (1 issue + 3 pause) ticks: 2 turns in 5 ticks.
    assert ctrl.command_names() == ["turn_right 15", "turn_right 15"]

    # Doorway appears, centred -> AIM -> APPROACH.
    c.doorway = doorway(center_x=480, width=100)
    assert run_until(ap, c, "AIM") == "AIM"
    assert run_until(ap, c, "APPROACH") == "APPROACH"

    # Doorway close (wide) -> PASS; it leaves view mid-pass, steps continue.
    c.doorway = doorway(center_x=480, width=450)
    assert run_until(ap, c, "PASS") == "PASS"
    c.doorway = None
    assert run_until(ap, c, "SEARCH") == "SEARCH"
    forwards = [n for n, _ in ctrl.commands if n == "forward 20"]
    assert len(forwards) == 3, f"PASS must take exactly 3 steps, got {forwards}"
    assert all(args == (20,) for _, args in ctrl.commands
               if _ == "forward 20")

    # Person appears in the other room -> TRACK, hovering centred.
    c.person_visible = True
    c.person_center_x = 480
    assert run_until(ap, c, "TRACK") == "TRACK"
    n_before = len(ctrl.commands)
    run_ticks(ap, c, 10)
    assert ap.state == "TRACK"
    assert len(ctrl.commands) == n_before  # centred person: hover, no moves


def test_aim_turns_toward_offcentre_doorway():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=200, width=100))  # left of centre
    assert run_until(ap, c, "AIM") == "AIM"
    run_ticks(ap, c, 6)
    assert "turn_left 30" in ctrl.command_names()
    assert "turn_right 30" not in ctrl.command_names()[1:]  # never turns right here
    # Doorway now centred -> APPROACH.
    c.doorway = doorway(center_x=480, width=100)
    assert run_until(ap, c, "APPROACH") == "APPROACH"


def test_approach_recentres_when_doorway_drifts():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100))
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    # Doorway drifts right of centre: re-centre turn before any forward.
    c.doorway = doorway(center_x=700, width=120)
    run_ticks(ap, c, 6)
    assert "turn_right 30" in ctrl.command_names()
    n_forward = sum(1 for n, _ in ctrl.commands if n == "forward 20")
    assert n_forward == 0
    # Back to centre -> forward steps resume.
    c.doorway = doorway(center_x=480, width=120)
    run_ticks(ap, c, 10)
    assert sum(1 for n, _ in ctrl.commands if n == "forward 20") >= 1


def test_pass_keeps_stepping_when_doorway_lost():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100))
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    c.doorway = doorway(center_x=480, width=500)  # close -> PASS
    assert run_until(ap, c, "PASS") == "PASS"
    c.doorway = None  # fills frame / leaves view mid-pass
    assert run_until(ap, c, "SEARCH") == "SEARCH"
    forwards = [n for n, _ in ctrl.commands if n == "forward 20"]
    assert len(forwards) == 3


# ---------------------------------------------------------------------------
# Tracking / safety
# ---------------------------------------------------------------------------

def test_tracking_loss_holds_and_resumes():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100))
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    run_ticks(ap, c, 6)  # at least one forward step issued
    assert any(n == "forward 20" for n, _ in ctrl.commands)

    # Tracking lost mid-approach: HOLD, no moves at all.
    c.tracking_valid = False
    n_before = len(ctrl.commands)
    run_ticks(ap, c, 20)
    assert ap.state == "HOLD"
    assert "tracking lost" in ap.message
    assert len(ctrl.commands) == n_before

    # Tracking recovers: pre-hold state (APPROACH) resumes, moves restart.
    c.tracking_valid = True
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    run_ticks(ap, c, 10)
    assert len(ctrl.commands) > n_before


def test_tracking_loss_in_scan_holds_no_moves():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(tracking_valid=False)
    run_ticks(ap, c, 10)
    assert ap.state == "HOLD"
    assert ctrl.commands == []
    c.tracking_valid = True
    assert run_until(ap, c, "SCAN") == "SCAN"


def test_low_battery_lands_once_and_done():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(battery_pct=24.0)
    ap.tick(c)
    assert ap.state == "LANDING"
    assert ctrl.command_names().count("land") == 1
    run_ticks(ap, c, 5)  # fake controller is LANDED after the land
    assert ap.state == "DONE"
    assert ctrl.command_names().count("land") == 1  # never re-issued


def test_low_battery_lands_from_hold_too():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(tracking_valid=False, battery_pct=10.0)
    ap.tick(c)
    assert ap.state == "LANDING"  # battery beats the hold
    assert ctrl.command_names().count("land") == 1


def test_battery_boundary_and_missing_never_land():
    # Exactly 25% is not below the threshold: keep flying.
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(battery_pct=25.0)
    run_ticks(ap, c, 10)
    assert ap.state == "SCAN"
    assert "land" not in ctrl.command_names()
    # No reading at all: never land, however long the scan runs (the
    # scan legitimately ends in NODOOR after a full 360 with no doorway).
    c.battery_pct = None
    run_ticks(ap, c, 60)
    assert ap.state in ("SCAN", "NODOOR")
    assert "land" not in ctrl.command_names()


def test_busy_controller_issues_nothing():
    ctrl = FakeController()
    ctrl.status = "MOVING"  # a move is in progress
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx()
    run_ticks(ap, c, 10)
    assert ap.state == "SCAN"
    assert ctrl.commands == []
    ctrl.status = "HOVERING"
    run_ticks(ap, c, 3)
    assert ctrl.commands  # free again: the scan turn goes out


def test_stop_halts_no_further_moves():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx()
    run_ticks(ap, c, 5)
    assert ctrl.commands
    ap.stop()
    assert ap.state == "OFF"
    n_before = len(ctrl.commands)
    run_ticks(ap, c, 20)
    assert ap.state == "OFF"
    assert len(ctrl.commands) == n_before


def test_start_is_idempotent_while_running():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    assert ap.start() is True
    assert ap.start() is False  # already running: no restart, no reset


# ---------------------------------------------------------------------------
# No doorway / TRACK edges
# ---------------------------------------------------------------------------

def test_no_doorway_full_scan_goes_nodoor_no_wandering():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx()  # doorway never appears
    assert run_until(ap, c, "NODOOR", max_ticks=300) == "NODOOR"
    assert "door not found" in ap.message
    names = ctrl.command_names()
    assert names.count("turn_right 15") == 24, f"full 360 = 24 turns, got {names}"
    assert not any(n.startswith("forward") for n in names)
    assert not any(n.startswith("move_") for n, _ in ctrl.tello.calls)
    # NODOOR hovers: further ticks issue nothing.
    n_before = len(ctrl.commands)
    run_ticks(ap, c, 20)
    assert ap.state == "NODOOR"
    assert len(ctrl.commands) == n_before


def test_track_person_at_edge_turns_toward_them():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(person_visible=True, person_center_x=480)
    # Drive to TRACK via SEARCH (no doorway needed for this unit path:
    # force the state through the mission shortcut below).
    c.doorway = doorway(center_x=480, width=100)
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    c.doorway = doorway(center_x=480, width=500)
    assert run_until(ap, c, "SEARCH") == "SEARCH"
    c.doorway = None
    assert run_until(ap, c, "TRACK") == "TRACK"
    # Person drifts to the left edge -> turn left; to the right -> right.
    c.person_center_x = 50
    run_ticks(ap, c, 6)
    assert "turn_left 30" in ctrl.command_names()
    c.person_center_x = 910
    run_ticks(ap, c, 6)
    assert "turn_right 30" in ctrl.command_names()


def test_track_person_lost_returns_to_search():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100),
            person_visible=True, person_center_x=480)
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    c.doorway = doorway(center_x=480, width=500)
    assert run_until(ap, c, "SEARCH") == "SEARCH"
    assert run_until(ap, c, "TRACK") == "TRACK"
    c.person_visible = False
    c.person_center_x = None
    assert run_until(ap, c, "SEARCH") == "SEARCH"


def test_scan_and_search_turns_are_15_deg_corrections_stay_30():
    """2026-11-25 tracker fix: blind-scan turns (SCAN / SEARCH) are
    15 deg — a 30-deg turn sheds the VIO map to 0 in 3 frames —
    while targeted correction turns (AIM re-centre, TRACK edge)
    stay 30 deg. Manual keys stay 30 deg (test_wiring)."""
    # SCAN: 15 deg.
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx()
    run_ticks(ap, c, 5)
    assert ap.state == "SCAN"
    assert ctrl.command_names() == ["turn_right 15", "turn_right 15"]
    # SEARCH: 15 deg (drive to SEARCH, person never appears).
    ctrl2 = FakeController()
    ap2 = Autopilot(ctrl2)
    ap2.start()
    c2 = ctx(doorway=doorway(center_x=480, width=100))
    assert run_until(ap2, c2, "APPROACH") == "APPROACH"
    c2.doorway = doorway(center_x=480, width=500)
    assert run_until(ap2, c2, "SEARCH") == "SEARCH"
    n_before = len(ctrl2.commands)
    run_ticks(ap2, c2, 5)
    search_turns = [n for n, _ in ctrl2.commands[n_before:]
                    if n.startswith("turn_")]
    assert search_turns, "SEARCH issued no turn"
    assert all(n == "turn_right 15" for n in search_turns), \
        f"SEARCH turns are not 15 deg: {search_turns}"
    # AIM correction: still 30 deg (off-centre doorway, left).
    ctrl3 = FakeController()
    ap3 = Autopilot(ctrl3)
    ap3.start()
    c3 = ctx(doorway=doorway(center_x=200, width=100))
    assert run_until(ap3, c3, "AIM") == "AIM"
    run_ticks(ap3, c3, 6)
    assert "turn_left 30" in ctrl3.command_names()
    assert "turn_left 15" not in ctrl3.command_names()
    print("PASS scan_and_search_turns_are_15_deg_corrections_stay_30")


# ---------------------------------------------------------------------------
# Mover mapping + non-blocking
# ---------------------------------------------------------------------------

def test_mover_mapping_path_without_tello():
    calls = []

    def forward(cm):
        calls.append(("forward", cm))

    def cw(deg):
        calls.append(("cw", deg))

    def ccw(deg):
        calls.append(("ccw", deg))

    def land():
        calls.append(("land",))

    mover = {"forward": forward, "cw": cw, "ccw": ccw, "land": land}
    ctrl = FakeControllerNoTello(mover)
    assert not hasattr(ctrl, "tello")
    ap = Autopilot(ctrl, mover=mover)
    ap.start()
    c = ctx()
    run_ticks(ap, c, 5)
    assert ("cw", 15) in calls  # SCAN turn went through the mapping
    c.doorway = doorway(center_x=480, width=100)
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    run_ticks(ap, c, 10)
    assert ("forward", 20) in calls  # 20 cm steps, not 30


def test_tick_never_blocks():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100))
    start = time.monotonic()
    for i in range(2000):
        c.time = float(i)
        ap.tick(c)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"2000 ticks took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Self-landing stop + start gate (2026-11-25 04:03 live run: the drone
# landed itself at 77% battery — not a battery landing — after drifting
# while a hand-held blanket moved in front of it in dim light)
# ---------------------------------------------------------------------------

def test_self_landing_in_search_stops_autopilot():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100))
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    c.doorway = doorway(center_x=480, width=500)
    assert run_until(ap, c, "SEARCH") == "SEARCH"
    c.doorway = None

    # Drone lands itself mid-search (tracking/bridge state is stale).
    ctrl.status = "LANDED"
    n_before = len(ctrl.commands)
    ap.tick(c)
    assert ap.state == "DONE"
    assert "landed" in ap.message
    assert "stopped" in ap.message
    assert len(ctrl.commands) == n_before  # no land/move re-issued
    assert "land" not in ctrl.command_names()

    # Subsequent ticks never resume or re-takeoff: still DONE, no moves.
    run_ticks(ap, c, 20)
    assert ap.state == "DONE"
    assert len(ctrl.commands) == n_before


def test_self_landing_in_scan_stops_on_landing_status():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx()
    run_ticks(ap, c, 2)
    assert ap.state == "SCAN"
    n_before = len(ctrl.commands)

    # Controller mid-landing (not yet LANDED) also ends the mission.
    ctrl.status = "LANDING"
    ap.tick(c)
    assert ap.state == "DONE"
    assert "landed" in ap.message
    assert "stopped" in ap.message
    assert len(ctrl.commands) == n_before
    run_ticks(ap, c, 10)
    assert ap.state == "DONE"
    assert len(ctrl.commands) == n_before


def test_self_landing_from_hold_stops_not_resumes():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    ap.start()
    c = ctx(doorway=doorway(center_x=480, width=100))
    assert run_until(ap, c, "APPROACH") == "APPROACH"
    # Tracking lost -> HOLD.
    c.tracking_valid = False
    ap.tick(c)
    assert ap.state == "HOLD"
    # Drone lands itself while held; tracking then recovers, but the
    # autopilot must NOT resume the pre-hold state.
    ctrl.status = "LANDED"
    c.tracking_valid = True
    n_before = len(ctrl.commands)
    ap.tick(c)
    assert ap.state == "DONE"
    assert "landed" in ap.message
    assert "stopped" in ap.message
    assert len(ctrl.commands) == n_before
    run_ticks(ap, c, 10)
    assert ap.state == "DONE"
    assert len(ctrl.commands) == n_before


def test_start_gate_refuses_when_not_hovering():
    # Drone on the ground: refuse, stay OFF, no moves.
    ctrl = FakeController()
    ctrl.status = "LANDED"
    ap = Autopilot(ctrl)
    assert ap.start(ctx()) is False
    assert ap.state == "OFF"
    assert ap.message == "cannot start: not hovering"
    assert ctrl.commands == []
    # A move in progress is not airborne either.
    ctrl2 = FakeController()
    ctrl2.status = "MOVING"
    ap2 = Autopilot(ctrl2)
    assert ap2.start(ctx()) is False
    assert ap2.state == "OFF"
    assert ap2.message == "cannot start: not hovering"
    assert ctrl2.commands == []


def test_start_gate_refuses_when_no_tracking():
    ctrl = FakeController()  # HOVERING, but no tracking yet
    ap = Autopilot(ctrl)
    assert ap.start(ctx(tracking_valid=False)) is False
    assert ap.state == "OFF"
    assert ap.message == "cannot start: no tracking"
    assert ctrl.commands == []
    # Not hovering takes precedence when both gate conditions fail.
    ctrl.status = "LANDED"
    ap2 = Autopilot(ctrl)
    assert ap2.start(ctx(tracking_valid=False)) is False
    assert ap2.state == "OFF"
    assert ap2.message == "cannot start: not hovering"


def test_start_gate_accepts_hovering_with_tracking():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    assert ap.start(ctx(tracking_valid=True)) is True
    assert ap.state == "SCAN"
    assert ap.message == "scanning for doorway"


def test_start_without_ctx_keeps_old_behaviour():
    # No context: old behaviour — start to SCAN even with no gate check.
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    assert ap.start() is True
    assert ap.state == "SCAN"
    # ... including when the drone is on the ground: SCAN is entered,
    # but the first tick then ends the mission via the self-landing
    # rule, with no move ever issued.
    ctrl2 = FakeController()
    ctrl2.status = "LANDED"
    ap2 = Autopilot(ctrl2)
    assert ap2.start() is True
    assert ap2.state == "SCAN"
    ap2.tick(ctx())
    assert ap2.state == "DONE"
    assert "landed" in ap2.message
    assert ctrl2.commands == []


def test_first_tick_without_tracking_holds_no_moves():
    ctrl = FakeController()
    ap = Autopilot(ctrl)
    assert ap.start(ctx()) is True  # gate passed: hovering + tracking
    assert ap.state == "SCAN"
    # First tick: tracking now False -> HOLD immediately, no move first.
    c = ctx(tracking_valid=False)
    ap.tick(c)
    assert ap.state == "HOLD"
    assert ctrl.commands == []
    # Recovery resumes SCAN.
    c.tracking_valid = True
    assert run_until(ap, c, "SCAN") == "SCAN"
