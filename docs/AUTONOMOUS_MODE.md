# Autonomous mode — through-wall marker

Plain English. Auto mode flies the drone itself: it scans the room
for a doorway, flies through it, then looks for the person in the
next room so the phone can show the marker.

## How to start and stop it

- **Start:** take off with **T** (or **Q**), wait for a steady hover,
  then press **U**. Auto mode only starts if the drone is hovering
  AND tracking is valid (the camera can see the room well enough to
  track). If either is missing, U does nothing and the log says why.
- **U is start-only (2026-11-25, user's order):** pressing **U**
  while auto mode is already running does NOTHING — it does not stop
  it. (The old toggle-off made him kill the autopilot right after
  starting it by pressing U again.) There is no stop key.
- **Stop:** press the **Stop autopilot** button on the laptop
  operator page (`/operator`), or press **any manual key** — W / S /
  A / D / I / K / J / L / Q / T / E. A manual key always takes over:
  it stops auto mode first, then flies the drone yourself. The human
  always wins.
- **Land:** **E** lands (and stops auto mode first, like any manual key).

## What each state means

| State | Meaning |
|---|---|
| OFF | Auto mode is off. |
| SCAN | Turning slowly, looking for a doorway. |
| AIM | Doorway found — turning to face its centre. |
| APPROACH | Flying toward the doorway in 20 cm steps. |
| PASS | Going through the doorway (a few 20 cm steps). |
| SEARCH | Through — turning slowly, looking for the person. |
| TRACK | Person found — hovering and keeping them in view. |
| HOLD | Tracking lost — hovering in place, moving nothing, until tracking comes back. |
| NODOOR | Turned a full circle and found no doorway. Press U to try again. |
| LANDING | Landing (low battery). |
| DONE | Finished — the drone landed, or landed itself. Press U to start a new run. |

The operator page shows the current state and what it is doing, plus
the drone's height. A LANDED status with height above 0 means the
drone is still in the air — check the height, not just the status.

## Safety rules

- **Never flies blind:** if tracking is lost, it holds still (HOLD)
  and moves nothing until tracking comes back.
- **Low battery:** under 25% it lands once, then stops (DONE).
- **Self-landing:** if the drone lands itself for any reason, auto
  mode stops at once (DONE) and never takes off again on its own.
- **Not-joystick hard stop:** if a command is rejected with
  `error Not joystick`, auto mode stops immediately and no more
  moves are sent. See below — this is the #1 blocker.
- **Hover-stable gate:** no auto move is sent until the drone has
  hovered steadily for 5 seconds after takeoff. The log says
  `hover stable 5 s — autopilot live` when the gate lifts. Never
  send the first move until the hover is stable.
- **No retries:** djitellopy already retries each command 4 times.
  Nothing here retries on top of that. Auto mode sends at most one
  command per tick.

## The Not-joystick problem — read this first

**The command path is the gating dependency: the autopilot cannot
fly at all until the Not-joystick rejections are resolved.** Every
move being rejected means no auto flight, no doorway pass, no
marker — manual flight is blocked the same way.

If commands are rejected with `error Not joystick`, the fix to try
is:

> **Reset the drone (hold power 5 s until yellow flash) and try again.**

Long-press the power button for 5 seconds while the drone is on,
until the light turns off and then flashes yellow. That is a
factory reset (WiFi name/password go back to factory). This is a
known Tello/djitellopy issue that also happens in pure SDK scripts
with no app at all (DJITelloPy issue #201: commands work for a
while, then suddenly `error Not joystick`). Reported causes:
commands sent too fast while the drone is not ready; starting the
program too soon after connecting; low battery / overheating; old
djitellopy.

**The Tello app is ruled out.** The app was not opened at all on
the day this happened, so the app cannot be the cause of these
rejections. Secondary check only: force-close the Tello app
completely — swipe it away from the app switcher; locking the phone
is not enough (a background app can still hold the controls). But
do the factory reset first.

## Setup rules (before every run)

1. **Factory-reset the drone first** if you have seen a
   Not-joystick rejection (hold power 5 s until yellow flash).
2. **Force-close the Tello app completely** — swipe it away from
   the app switcher. Locking the phone is not enough. (Secondary
   check — the app is ruled out for today's runs, but a background
   app can still steal control.)
3. **Take off facing the room**, with the floor visible in the
   lower part of the camera frame, and good light. The tracker
   needs floor texture to initialise — if the camera faces a plain
   wall, tracking never starts and U will refuse to start auto mode.
   The camera is fixed; there is no tilt. Wait for the steady hover
   (5 seconds) before pressing U.
4. **Pin the blanket still** — never hand-held — and keep the drone
   clear of fabric. A moving blanket confuses the drone's downward
   sensor and the props can catch it.
5. **The saved map is only valid from the same takeoff spot.** The
   server saves the room map at the end of a run
   (`maps/latest.npz`) and loads it at the next start so tracking
   can recover after a loss. If you take off from a different spot,
   that map is wrong for the new run — delete `maps/latest.npz`
   before starting.
6. **The doorway pass cannot be live-tested until there is a real
   doorway.** SCAN / AIM / APPROACH / PASS have only been tested
   against unit-test fakes, not a real doorway flight.

## What is recorded

Every run saves its annotated drone video and the server log in a
timestamped folder under `runs/` (for example
`runs/2026-11-25_04-30-00/video.mp4` and `server.log`), so a
failure can be diagnosed from what the camera actually saw.
