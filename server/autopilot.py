"""Autopilot state machine for the through-wall person marker (2026-11-25).

The drone flies itself through a doorway into the next room and finds the
person there, so the phone can show the marker. Tick-driven and
non-blocking: the live loop calls :meth:`Autopilot.tick` once per
iteration; a tick issues at most one command and never sleeps.

Flight goes through the ONE flight controller (``SafeFlightController`` in
``server/main.py``) via its ``send_command(name, func, *args)`` — this
module never talks to the drone directly and never imports ``server.main``
or ``server.doorway``. The move functions are resolved, in order, from:

1. an explicit ``mover`` mapping passed to the constructor, with keys
   ``forward`` / ``back`` / ``left`` / ``right`` / ``cw`` / ``ccw`` /
   ``land`` (each a callable taking the same args the Tello methods take:
   cm for moves, degrees for turns, nothing for land); or
2. the controller's underlying drone object, ``controller.tello``, whose
   ``move_forward`` / ``move_back`` / ``move_left`` / ``move_right`` /
   ``rotate_clockwise`` / ``rotate_counter_clockwise`` / ``land`` methods
   are used (this is the real ``SafeFlightController`` path — its
   ``.tello`` attribute is the djitellopy Tello, and ``dispatch_drone_key``
   in ``server/main.py`` passes exactly these methods to ``send_command``).

Move step is 20 cm for every forward/sideways move. Blind-scan turns
(SCAN / SEARCH) are 15 deg: a 30-deg turn sheds LK bearings faster
than the map can be rebuilt — measured on the synthetic room, one
30-deg turn dropped the 3D map from 144 points to 0 in 3 frames,
and the 04:31 live run's map collapsed 482 -> 15 -> 0 the same way
(2026-11-25 tracker fix). Targeted correction turns (AIM re-centre,
TRACK edge) and manual keys stay 30 deg: they are single corrections,
not a sustained spin, so the map survives them.

Safety layer (hard rules, no exceptions):
* Tracking lost (``ctx.tracking_valid`` False) during autonomous flight
  -> HOLD: no move commands at all, hover in place. The pre-hold state
  is remembered and resumed when tracking recovers. Never fly blind.
* Battery < 25 % (when a reading exists) -> land once via the
  controller's land path, then DONE. A missing battery reading never
  triggers a landing.
* A busy controller (``controller.status`` != ``"HOVERING"``) -> the tick
  issues nothing; ``send_command`` itself also skips when busy.
* Self-landing: if the controller reports LANDED (or LANDING) while the
  autopilot is in a flying state (SCAN, AIM, APPROACH, PASS, SEARCH,
  TRACK, HOLD) -> DONE at once, "landed — autopilot stopped", no
  further moves. The autopilot never resumes or re-takes off on its
  own after the drone has landed itself (2026-11-25 live run: the
  drone landed itself at 77 % battery, i.e. not a battery landing, so
  a stale tracking/bridge state must not restart flight).
* Start gate: ``start(ctx)`` with a context refuses unless the drone
  is actually airborne and tracking (``controller.status`` ==
  ``"HOVERING"`` and ``ctx.tracking_valid``); otherwise it stays OFF
  and returns False. ``start()`` with no context keeps the old
  behaviour. The first tick after a start with ``tracking_valid``
  False goes to HOLD immediately (Safety 2 above) — no move is ever
  issued before tracking is confirmed.
* ``stop()`` halts immediately: state OFF, no further moves.

States (``.state``): OFF, SCAN, AIM, APPROACH, PASS, SEARCH, TRACK,
HOLD, NODOOR, LANDING, DONE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

#: Frame width of the Tello forward camera (960x720), in pixels.
FRAME_WIDTH_PX = 960
#: Frame centre-x, in pixels.
FRAME_CENTRE_X = FRAME_WIDTH_PX // 2


@dataclass
class AutoContext:
    """One tick's worth of perception, handed to :meth:`Autopilot.tick`.

    ``doorway`` is any object with ``.center_x`` (pixels), ``.width``
    (pixels) and ``.confidence`` (0-1), or None when no doorway is
    detected. ``person_center_x`` is the detected person's centre-x in
    pixels, or None. ``time`` is seconds (monotonic); it is recorded for
    diagnostics only — pacing is in ticks, so behaviour is identical in
    tests and on the live loop.
    """

    tracking_valid: bool = True
    bridge_valid: bool = False
    battery_pct: Optional[float] = None
    doorway: Any = None
    person_visible: bool = False
    person_center_x: Optional[float] = None
    time: float = 0.0


# ---------------------------------------------------------------------------
# Autopilot
# ---------------------------------------------------------------------------

#: States in which the drone is (or should be) airborne. A self-landing
#: (controller LANDED/LANDING) in any of these ends the mission at once.
_FLYING_STATES = frozenset(
    {"SCAN", "AIM", "APPROACH", "PASS", "SEARCH", "TRACK", "HOLD"})

#: Tello method name on ``controller.tello`` for each mover key.
_MOVER_ATTRS = {
    "forward": "move_forward",
    "back": "move_back",
    "left": "move_left",
    "right": "move_right",
    "cw": "rotate_clockwise",
    "ccw": "rotate_counter_clockwise",
    "land": "land",
}

class Autopilot:
    """Tick-driven doorway-pass + person-search autopilot.

    Constructed with a controller object (duck-typed: ``.status`` and
    ``.send_command(name, func, *args)``) and an optional ``mover``
    mapping (see module docstring). Tunables are keyword arguments.
    """

    def __init__(self, controller, mover: Optional[dict] = None,
                 turn_deg: int = 30, scan_turn_deg: int = 15,
                 step_cm: int = 20,
                 centre_tol_px: float = 60,
                 frame_width_px: int = FRAME_WIDTH_PX,
                 doorway_close_width_px: float = 400,
                 max_approach_steps: int = 10, pass_steps: int = 3,
                 turn_pause_ticks: int = 3, move_pause_ticks: int = 2,
                 scan_turns: int = 24, battery_min_pct: float = 25,
                 person_edge_px: float = 120,
                 doorway_min_confidence: float = 0.5):
        self.controller = controller
        self._mover = dict(mover) if mover else {}
        # Tunables.
        self.turn_deg = int(turn_deg)
        # Blind-scan turns (SCAN / SEARCH): 15 deg, not 30 — see the
        # module docstring (2026-11-25 tracker fix). Correction
        # turns (AIM / TRACK) use turn_deg (30). scan_turns is 24
        # so a full SCAN still covers 360 deg (24 x 15).
        self.scan_turn_deg = int(scan_turn_deg)
        self.step_cm = int(step_cm)
        self.centre_tol_px = float(centre_tol_px)
        self.frame_width_px = int(frame_width_px)
        self.doorway_close_width_px = float(doorway_close_width_px)
        self.max_approach_steps = int(max_approach_steps)
        self.pass_steps = int(pass_steps)
        self.turn_pause_ticks = int(turn_pause_ticks)
        self.move_pause_ticks = int(move_pause_ticks)
        self.scan_turns = int(scan_turns)
        self.battery_min_pct = float(battery_min_pct)
        self.person_edge_px = float(person_edge_px)
        self.doorway_min_confidence = float(doorway_min_confidence)
        # State.
        self.state: str = "OFF"
        self.message: str = "autopilot off"
        self._pre_hold: Optional[str] = None
        self._pause: int = 0          # ticks left before the next command
        self._scan_turns: int = 0    # 15-deg turns issued in this SCAN
        self._search_turns: int = 0  # 15-deg turns issued in this SEARCH
        self._approach_steps: int = 0  # forward steps issued in APPROACH
        self._pass_count: int = 0     # forward steps issued in PASS
        self._land_issued: bool = False
        self.last_time: float = 0.0

    # -- public control ----------------------------------------------------

    def start(self, ctx: Optional[AutoContext] = None) -> bool:
        """Begin (or restart) the mission: SCAN for a doorway.

        Only starts from OFF, DONE or NODOOR; returns False when already
        running (state unchanged).

        Start gate: when a context is given, refuse unless the drone is
        actually airborne and tracking — ``controller.status`` must be
        ``"HOVERING"`` and ``ctx.tracking_valid`` must be True.
        On refusal the state is unchanged (stays OFF from OFF) and the
        message says why. With no context the old behaviour is kept:
        start to SCAN unconditionally.
        """
        if self.state not in ("OFF", "DONE", "NODOOR"):
            return False
        if ctx is not None:
            if getattr(self.controller, "status", "HOVERING") != "HOVERING":
                self.message = "cannot start: not hovering"
                return False
            if not ctx.tracking_valid:
                self.message = "cannot start: no tracking"
                return False
        self._pre_hold = None
        self._pause = 0
        self._scan_turns = 0
        self._search_turns = 0
        self._approach_steps = 0
        self._pass_count = 0
        self._land_issued = False
        self._set_state("SCAN", "scanning for doorway")
        return True

    def stop(self) -> None:
        """Halt immediately: state OFF, no further moves are issued."""
        self._pre_hold = None
        self._pause = 0
        self._set_state("OFF", "autopilot stopped")

    # -- main tick ---------------------------------------------------------

    def tick(self, ctx: AutoContext) -> str:
        """Advance one loop iteration. Issues at most one command, never
        sleeps, never blocks. Returns the current state."""
        self.last_time = ctx.time
        if self.state in ("OFF", "DONE"):
            return self.state

        # Safety 0 — self-landing: the drone is down (it landed itself,
        # e.g. after drifting into fabric — not a battery landing), so
        # the mission ends at once. Checked before battery/tracking so
        # a stale tracking/bridge state can never resume flight or
        # re-issue a land/move, and no further moves are ever issued.
        if self.state in _FLYING_STATES and getattr(
                self.controller, "status", "HOVERING") in ("LANDED", "LANDING"):
            self._pre_hold = None
            self._pause = 0
            self._set_state("DONE", "landed — autopilot stopped")
            return self.state

        # Safety 1 — battery: land once, whatever else is happening.
        # A missing reading (None) never lands.
        if self.state != "LANDING":
            if (ctx.battery_pct is not None
                    and ctx.battery_pct < self.battery_min_pct):
                self._begin_landing("low battery — landing")
                return self.state

        # Landing in progress: watch the controller, issue nothing else.
        if self.state == "LANDING":
            self._tick_landing()
            return self.state

        # Safety 2 — tracking: never fly blind. HOLD issues no moves and
        # remembers where to resume; recovery restores that state.
        if not ctx.tracking_valid:
            if self.state != "HOLD":
                self._pre_hold = self.state
                self._set_state("HOLD", "tracking lost — holding")
            return self.state
        if self.state == "HOLD":
            resumed = self._pre_hold or "SCAN"
            self._pre_hold = None
            self._set_state(resumed, "tracking recovered — resuming")
            return self.state

        # Per-state behaviour (each handler paces itself via _pause).
        handler = getattr(self, f"_tick_{self.state.lower()}", None)
        if handler is not None:
            handler(ctx)
        return self.state

    # -- state handlers ----------------------------------------------------

    def _tick_scan(self, ctx: AutoContext) -> None:
        # Doorway visible at any moment (even mid-pause) -> AIM at once.
        if self._doorway(ctx) is not None:
            self._scan_turns = 0
            self._set_state("AIM", "doorway found — aiming")
            return
        if self._scan_turns >= self.scan_turns:
            self._set_state("NODOOR", "door not found")
            return
        if not self._ready():
            return
        # Yaw in 15-deg steps (2026-11-25 tracker fix), pausing
        # between turns so the tracker and the detector can keep up.
        # Rotation gives no parallax, so no sideways drift is needed
        # here — looking around is the job. 30-deg turns shed the
        # VIO map to 0 in one turn, so the scan itself killed
        # tracking; 24 x 15 deg still covers a full circle.
        if self._issue(f"turn_right {self.scan_turn_deg}", "cw",
                       self.scan_turn_deg):
            self._scan_turns += 1
            self._pause = self.turn_pause_ticks
            self.message = "scanning for doorway"

    def _tick_aim(self, ctx: AutoContext) -> None:
        dw = self._doorway(ctx)
        if dw is None:
            self._scan_turns = 0
            self._set_state("SCAN", "doorway lost — rescanning")
            return
        if self._centred(dw.center_x):
            self._approach_steps = 0
            self._set_state("APPROACH", "doorway centred — approaching")
            return
        if not self._ready():
            return
        if self._issue_turn_toward(dw.center_x):
            self._pause = self.turn_pause_ticks
            self.message = "aiming at doorway"

    def _tick_approach(self, ctx: AutoContext) -> None:
        dw = self._doorway(ctx)
        if dw is None:
            self._scan_turns = 0
            self._set_state("SCAN", "doorway lost — rescanning")
            return
        # Close enough (doorway fills the frame) or step budget spent:
        # commit to the pass instead of inching forever.
        if (dw.width >= self.doorway_close_width_px
                or self._approach_steps >= self.max_approach_steps):
            self._pass_count = 0
            self._set_state("PASS", "passing through doorway")
            return
        if not self._ready():
            return
        if not self._centred(dw.center_x):
            if self._issue_turn_toward(dw.center_x):
                self._pause = self.turn_pause_ticks
                self.message = "re-centring on doorway"
            return
        if self._issue("forward 20", "forward", self.step_cm):
            self._approach_steps += 1
            self._pause = self.move_pause_ticks
            self.message = "approaching doorway"

    def _tick_pass(self, ctx: AutoContext) -> None:
        # The doorway fills the frame (or leaves view) when close: keep
        # stepping the bounded count, never chase a lost detection.
        if self._pass_count >= self.pass_steps:
            self._search_turns = 0
            self._set_state("SEARCH", "through — searching for person")
            return
        if not self._ready():
            return
        if self._issue("forward 20", "forward", self.step_cm):
            self._pass_count += 1
            self._pause = self.move_pause_ticks
            self.message = "passing through doorway"
            if self._pass_count >= self.pass_steps:
                self._search_turns = 0
                self._set_state("SEARCH", "through — searching for person")

    def _tick_search(self, ctx: AutoContext) -> None:
        # Person visible at any moment (even mid-pause) -> TRACK at once.
        if ctx.person_visible:
            self._set_state("TRACK", "person found — tracking")
            return
        if not self._ready():
            return
        # 15-deg steps here too (2026-11-25 tracker fix): SEARCH is
        # the same blind spin as SCAN, so 30-deg turns would drain
        # the VIO map the same way.
        if self._issue(f"turn_right {self.scan_turn_deg}", "cw",
                       self.scan_turn_deg):
            self._search_turns += 1
            self._pause = self.turn_pause_ticks
            self.message = "searching for person"

    def _tick_track(self, ctx: AutoContext) -> None:
        if not ctx.person_visible:
            self._set_state("SEARCH", "person lost — searching")
            return
        cx = ctx.person_center_x
        if cx is None:
            self.message = "tracking person"
            return
        if not self._ready():
            return
        # Hover centred; only yaw when the person drifts to a frame edge.
        if cx < self.person_edge_px:
            if self._issue_turn_toward(cx):
                self._pause = self.turn_pause_ticks
                self.message = "person at left edge — turning left"
        elif cx > self.frame_width_px - self.person_edge_px:
            if self._issue_turn_toward(cx):
                self._pause = self.turn_pause_ticks
                self.message = "person at right edge — turning right"
        else:
            self.message = "tracking person"

    def _tick_nodoor(self, ctx: AutoContext) -> None:
        # Terminal hover: report, do NOT wander. start() restarts.
        self.message = "door not found"

    # -- landing -----------------------------------------------------------

    def _begin_landing(self, message: str) -> None:
        """Enter LANDING and, if the drone is hovering, send land now."""
        self._land_issued = False
        self._set_state("LANDING", message)
        self._maybe_land()

    def _tick_landing(self) -> None:
        if getattr(self.controller, "status", "HOVERING") == "LANDED":
            self._set_state("DONE", "landed")
            return
        self._maybe_land()

    def _maybe_land(self) -> None:
        """Send the land command once, only while HOVERING (a move in
        progress is waited out, never fought)."""
        if self._land_issued:
            return
        if getattr(self.controller, "status", "HOVERING") != "HOVERING":
            return
        if self._issue("land", "land"):
            self._land_issued = True

    # -- helpers -----------------------------------------------------------

    def _set_state(self, state: str, message: str) -> None:
        self.state = state
        self.message = message

    def _ready(self) -> bool:
        """Pause countdown: True when a new command may be issued."""
        if self._pause > 0:
            self._pause -= 1
            return False
        return True

    def _doorway(self, ctx: AutoContext):
        """The ctx doorway, or None when absent / too low-confidence."""
        dw = ctx.doorway
        if dw is None:
            return None
        conf = getattr(dw, "confidence", 1.0)
        if conf is not None and conf < self.doorway_min_confidence:
            return None
        return dw

    def _centred(self, center_x: float) -> bool:
        return abs(center_x - self.frame_width_px / 2) <= self.centre_tol_px

    def _issue_turn_toward(self, target_x: float) -> bool:
        """One 30-deg yaw step toward a target x: left (ccw) when the
        target is left of frame centre, right (cw) when right."""
        if target_x < self.frame_width_px / 2:
            return self._issue("turn_left 30", "ccw", self.turn_deg)
        return self._issue("turn_right 30", "cw", self.turn_deg)

    def _resolve(self, key: str) -> Optional[Callable]:
        """Move function for a mover key: explicit ``mover`` mapping
        first, else ``controller.tello``'s method of the same motion."""
        fn = self._mover.get(key)
        if fn is not None:
            return fn
        tello = getattr(self.controller, "tello", None)
        if tello is not None:
            return getattr(tello, _MOVER_ATTRS[key], None)
        return None

    def _issue(self, name: str, key: str, *args) -> bool:
        """Send one command via the controller. False when nothing was
        sent (controller busy / not hovering, or no mover available)."""
        if getattr(self.controller, "status", "HOVERING") != "HOVERING":
            return False
        fn = self._resolve(key)
        if fn is None:
            self.message = "no mover available"
            return False
        self.controller.send_command(name, fn, *args)
        return True
