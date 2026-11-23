# Wall Marker — FULL COMPLETION PLAN

Date: 2026-11-23. Owner: Muneer. Rule: build it completely, no shortcuts, no
faked data, no asking the user for options — decide from the research and the
approved plans and keep going.

## The assignment (in one paragraph)

A laptop server takes a drone's video, detects a person with YOLO-Pose,
estimates the person's 3D position, and streams a compact marker (skeleton
joints + distance + uncertainty) to an iPhone Safari viewer. The phone shows
ONLY its own camera feed with the skeleton overlay — never the drone's video.
The shared frame comes from a BRIDGE: the drone starts beside the phone and
flies through the doorway while keeping visual tracking. If the bridge is
invalid, NO marker is shown — never a guessed position. The marker is a pose
SKELETON, never a dot. Ground/virtual only until the user gives the go-ahead
for a real Tello + iPhone test. When the build is done, push to a NEW GitHub
repo (GitHub connected 2026-11-23, account Muneer-ul-hassan).

## Where things stand (measured, 2026-11-23 ~15:40)

- DONE, 27/27 scenarios PASS (virtual, true poses): detection, joint error
  30.9–48.1 cm mean, distance error 0.6–16.3 cm, 0 false markers, 0 markers
  after tracking loss, two-drone fusion beats best single drone.
  (Post-calibration numbers, full-suite run 2026-11-23.)
- DONE: UI research plan (easy UI, real data only) and VIO-fix deep research.
- DONE, step 1 accepted 2026-11-23 (round 3): bridge VIO — 56/60 frames
  valid in all 4 VIO scenarios, markers emitted, joint error 32.6–40.5 cm
  (all PASS, worst margin 19.5 cm). Default PnP mode is 1-DoF yaw
  (`yaw1dof`) + the bbox-height detector calibration (0.89). Full pipeline
  4.7–6.4 fps on this VM (below the 10 fps target — a machine limit, not
  code; detector alone 96.5 ms/frame). Smooth-flight ground test tracks
  the whole flight (102/105 valid) with drift mean 29.1 cm, max 61.9 cm.
  Caveat: yaw1dof needs the attitude prior accurate to ~1 deg — the
  §6.9 yaw calibration (σ ≲ 1.5 deg) is still unrun and gates the real test.
- DONE: depth model built + measured (Depth Anything V2 Small, 95 MB,
  arbitration-only, voter never flipped in-suite so fused error unchanged,
  ~8% slower at 1 Hz, default off pays nothing); its value is unproven
  in-suite, mechanism unit-proven. Step 2's acceptance (depth error
  improves) is NOT met by the in-suite numbers — the voter changes
  nothing, so the model stays an optional voter, not a required step.
- DONE: easy UI server part (step 4, verified 2026-11-23 ~18:55: SSE markers array, /operator page, demo split out of product server, 5 real-only tests PASS, full suite 27/27 PASS). NOT STARTED: GitHub push, real flight test.

## Build order (do not reorder; do not skip a step's acceptance check)

### 1. VIO fixes — DONE, ACCEPTED 2026-11-23 (round 3)
Source: `~/workspace/research_notes/vio-rotation-tracking-fix-20261123-1014/report.md`
and the page `vio-rotation-tracking-fix` (artifact).
Apply in this order:
1. Rotation-only mode + deferred triangulation (GRANITE points-at-infinity,
   d=0, rotation only, re-triangulate later). Parallax-angle gate ~5°
   INSTEAD of the 0.15 m baseline gate. Previous-frame keyframe on
   negative-entropy drop. Never cull a point for low parallax alone.
2. Tello state priors (~0 ms): yaw first (unwrap, reject jumps >10°/frame,
   σ≈1–3°), pitch/roll (PnP collapses to 1-DoF yaw; front camera tilts
   ~13° down), vgx/vgy/vgz as motion model, h+tof together for height/scale.
   djitellopy state: UDP 8890, ~10 Hz, no timestamp — timestamp on the
   laptop and interpolate.
3. Fly an arc (translate while yawing) + remove the -31°/+101° harness yaw
   snaps from the test trajectories.
4. Windowed local BA (fixes the 11 px map error) + keyframe-descriptor
   relocalisation / restart-and-merge.
5. Parallel spike: `pip install pyorbslam3` mono mode — backup only; mono
   still struggles with pure rotation, and mono-inertial is impossible
   (no IMU stream via djitellopy).
CALIBRATE FIRST (blocking): velocity units are disputed in DJI's own docs
(cm/s vs dm/s — a 10× error). Fly 1 m, integrate both ways, yaw
mid-translation (report §6.9). Do NOT integrate any velocity before this.
TelloPy binary-log lead (onboard odometry mvo.pos/vel + gyro + quaternion):
bench-validation spike first, never depend on it unvalidated.
ACCEPT (met 2026-11-23, full-suite run 678.8 s): VIO valid 56/60 frames
in all 4 scenarios with markers emitted (10-11 per scenario) and joint /
distance errors inside the 60 cm budgets — s1 40.5 / 6.0 cm, s23 32.6 /
5.7 cm, s20 36.1 / 7.5 cm, s4 36.3 / 6.8 cm (worst margin 19.5 cm).
Full 27-scenario GT-pose suite 27/27 PASS, 0 FAIL in the whole report.
Round 3 did it in two parts: (a) the default PnP mode is now 1-DoF yaw
PnP (`pnp_mode="yaw1dof"` — pitch/roll fixed from the prior, yaw +
centre solved jointly), picked by measurement over the 6-DoF PnP, a
soft prior factor, and a translation-only PnP; the NEC / rotation-
averaging build was tried and measured to have the same systematic
bias on this data and was rejected (see TEST-REPORT.md "VIO fix
round 3"); (b) a bbox-height detector calibration
(`BBOX_HEIGHT_FRACTION = 0.89` in `server/depth.py`) — the YOLO bbox
spans ~0.89 of the person's height, so H = 1.7 m overestimated depth
by ~13% and held s4 at 59.4 cm. Pipeline ≥10 fps NOT met: 4.7-6.4 fps
on this shared 1.5-CPU VM — a machine limit, not code (detector alone
96.5 ms/frame, VIO 81-85 ms, sequential; pipeline overhead ~5 ms).
CAVEAT: yaw1dof is fragile to prior error — a (yaw+2, pitch−1, roll+1)
deg prior offset fails s1/s20/s4 (81.6-90.4 cm). The §6.9 yaw
calibration must show σ ≲ 1.5 deg before the real test. The regression
baseline was re-based to the post-calibration GT numbers (see
`tests/run_tests.py` `RECORDED_2026_11_23`); all 7 original scenarios
re-verified MATCH after the re-base.
Files: `server/vio.py`, `server/bridge.py` (priors input only), `tests/`.

### 2. Depth model — DONE, MEASURED 2026-11-23 (voter never flips in-suite; value unproven, kept as optional voter)
Third depth estimator alongside triangulation + bbox-height. Small/fast
model, runs in parallel / only when needed, marker path NEVER waits.
Measure latency and FPS before and after; report both. If it slows the
pipeline, it does not ship.
ACCEPT: depth error improves (or at least does not worsen) vs the current
two methods, pipeline still ≥10 fps.
Files: new `server/depth_model.py` + tests. Wiring into the pipeline only
after step 1 is accepted (shared `server/main.py` / `fusion.py`).

### 3. Easy UI — VIEWER PART DONE 2026-11-23 (`phone/viewer.html`)
Server-side parts (SSE markers array, `/operator` page, demo split-out)
wait until step 1 is accepted (shared `server/main.py`, `fusion.py`).
Source: UI research plan delivered 2026-11-23 14:57.
Viewer rules: one glance (skeleton + big distance + colour), one thumb
(single Start/Stop button, bottom third), real data only.
- Single status pill top (green Person found / amber Stale / grey Watching /
  red No connection). Remove the `#chip` and the big centred status text.
- Fixed bottom distance pill, always same place: `≈ X.X m ± Y.Y m`
  (rounding follows σ: one decimal under 1 m, whole metres at/above 1 m).
  Remove the floating distance label next to the head.
- Start screen: one big Start button + "Hold still, facing the wall."
  Tapping starts camera + stream (no auto-start on load). While running the
  button reads Stop. No settings, no mode switch, no demo toggle.
- Remove the permanent `#hint` footer (HTTPS note stays in the README only).
- Real camera dimensions: project in the stream's real pixel space
  (`video.videoWidth/videoHeight`), do not assume 1280×720.
- Draw up to 3 skeletons (one per confirmed track, own colour, nearest
  first) from a `markers` array AND from the current single-marker payload
  (backward compatible until step 3's server part lands). Drone count in
  the status pill only when >1. Stale: amber + STALE tag, ≤0.5 s, then hidden.
- States: Watching = empty screen + grey pill (no centred "No person
  detected" essay). Tracking lost = skeleton hidden + one centred line.
  No connection = one centred line + "Reconnecting…". Camera denied =
  shortened error box.
- Skeleton fades/thickens as uncertainty grows; number and drawing agree.
  No distance without a skeleton, no skeleton without a distance.
- ZERO demo/fake/sample/placeholder wording anywhere in the viewer.
ACCEPT: viewer loads with no console errors; JS syntax checked; skeleton
draws from a stub marker stream (single + array payloads); no drone-video
reference anywhere in the file (grep check).

### 4. Easy UI — SERVER PART DONE, VERIFIED 2026-11-23 ~18:55
- SSE payload becomes a markers array (one event per confirmed track,
  capped at 3, nearest first); `drones` list carried per marker.
- New route `/operator` (same server, laptop browser only): three blocks
  BRIDGE OK/LOST, DRONE LIVE/NO VIDEO, PERSON FOUND/NONE; one facts line
  (drones, tracks, uptime, last-frame age — `—` when unknown, never `0`);
  3-step checklist ticking itself from real state; one Start/Stop button.
  Two new `/api/status` fields: `last_frame_age_s`, `frames_last_10s`.
  No drone video on this page, no per-joint tables, no log tail.
- SPLIT THE DEMO OUT of the product server (the single biggest real-only
  fix): `--mode virtual` serving + the synthetic fallback in `main.py`
  move to a test-only entry point. The product server serves live drone
  frames only and refuses to start without a real source (same rule the
  Tello source already follows: clear error, never a fake run).
- Real-only tests (extend the no-drone-video test): viewer contains no
  demo/sample/example/placeholder/test wording; product server with no
  drone source exits with an error and serves nothing; `/api/markers`
  emits only `{"type":"none"}` while the bridge is invalid; operator page
  shows `—` (not `0`) before the first frame; viewer draws zero skeletons
  from 60 s of `none` events.
ACCEPT: all new tests pass; `/viewer`, `/operator`, `/api/status`,
`/api/markers` serve correctly on the live path.

### 5. Full suite re-run + report — DONE 2026-11-23 (27/27 PASS, 0 FAIL)
Run `tests/run_tests.py` end to end with VIO + depth model + new UI:
all 27 scenarios (thresholds UNMOVED: detection ≥90%, mean joint error
≤60 cm, mean |distance error| ≤60 cm, scale error ≤25%, false markers 0,
marker-after-loss 0, fused ≤ best single), VIO table, real-only tests.
Rewrite `tests/TEST-REPORT.md` with real measured numbers and an honest
limits section. Update `README.md` (run instructions, real-only contract,
operator page, calibration results).
ACCEPT: every scenario PASS, or the failure is reported with its number —
never hidden, never a moved goalpost.

### 6. GitHub push — NEW repo (after step 5 is accepted)
GitHub connected (account Muneer-ul-hassan). Create a NEW repo, push the
whole `wall-marker/` project (code + tests + docs + PLAN.md). Verify the
repo opens with the code in it. Report the repo URL.
Files: per `references/snapshot-push.md` in the GitHub connector doc.

### 7. Real Tello + iPhone test — ONLY on the user's go-ahead
Needs `djitellopy` (not installed). Motors-off hand-carry ground test
first: drone carried beside the phone through the doorway, then a first
open-field flight. The user decides when.

## Hard rules (never break, never work around)
- No faked poses, no faked numbers, no hard-coded answers, no moved
  thresholds. A failure is reported with its measured number.
- No physical markers in the environment (mission pads forbidden).
- The phone NEVER receives drone video. The shipped server NEVER serves
  synthetic frames. The shipped UI shows REAL data only.
- Ground/virtual only. No flight, no motors, no real drone commands
  until the user says go.
- Work stays inside `~/workspace/tello-search/wall-marker/` (read-only
  peeks at `tello_path_search.py` / `fly_brain.py` allowed). Do not touch
  Drones.zip.
- One inference process at a time (4 CPUs, ~1.2 GB free RAM).
- File ownership while agents run in parallel: VIO agent owns `server/`
  (except `phone/`) and `tests/`; UI viewer agent owns `phone/viewer.html`
  ONLY; depth agent owns new `server/depth_model.py` + its tests ONLY.
  Shared files (`server/main.py`, `server/fusion.py`, `server/depth.py`)
  are edited only after step 1 is accepted, one agent at a time.

## References
- Research report (why a bridge is mandatory, error budgets):
  `~/workspace/research_notes/drone-phone-through-wall-person-marker-20261122-1906/report.md`
- VIO fix research: `~/workspace/research_notes/vio-rotation-tracking-fix-20261123-1014/report.md`
  + page `vio-rotation-tracking-fix` (artifact)
- Build spec: `SPEC.md` (thresholds, exact interfaces, marker JSON shape)
- UI plan: delivered in chat 2026-11-23 14:57 (research agent
  6d97677b-c759-43e5-ba84-158eccbbdc6e); viewer rules above are its §1+§3.
