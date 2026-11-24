# Wall Marker — through-wall person marker (drone + phone)

A person holds a phone whose camera shows a wall in an unknown indoor
room. Another person stands behind that wall in the next room, invisible
to the phone. A drone flies in the other room, sees the person, and the
phone displays a **pose skeleton** of the hidden person in its live camera
view, with distance and uncertainty. Part 2: several bigger drones feed
the same view while the user points the phone anywhere.

This is a **full working system**, not a demo shell: real YOLO-Pose
detection, real depth estimation, a real bridge transform, real
multi-drone fusion, a real phone viewer, and a virtual test harness that
measures the whole pipeline against known ground truth. It has been
tested **on the ground / virtually only** — no flight has happened yet.
The real Tello + phone test happens later, on the user's go-ahead.

## How it works

```
drone video ──> laptop server ── SSE JSON ──> iPhone (Safari)
  (Tello /        YOLO-Pose        marker only:      own camera +
   synthetic)     depth (triang.)  skeleton 3D +     skeleton overlay
                  bridge           distance + σ     + "≈4.2 m ±0.5 m"
                  fusion
```

1. **Detection** — YOLOv8n-pose finds the person and 17 body joints in
   the drone view (server/detector.py).
2. **Depth** — the person's depth comes from motion triangulation
   (the drone's own movement is the stereo baseline; primary method
   whenever the baseline is ≥ 0.3 m) with a bbox-height fallback
   (server/depth.py). A monocular depth model (Depth Anything V2
   Small, floor-plane anchored) votes as a third estimator: it only
   arbitrates when triangulation and bbox-height disagree, never
   replaces triangulation, and never refines a depth number
   (server/mono_depth.py; `--mono-depth`). Joints are placed in 3D
   by intersecting each joint's ray with the plane at the person's
   depth (planar approximation — honest limit, see below).
3. **Bridge** — the shared frame comes from the drone taking off beside
   the phone and flying through the doorway while keeping visual
   tracking: one continuous map chain links the phone frame to the drone
   frame (server/bridge.py). **Hard rule: if the bridge is invalid
   (no bridge, tracking lost), NO marker is shown — never a guessed
   position.**
4. **Fusion** — detections from one or more drones are fused into a
   person track (inverse-variance weighting). A track needs 3
   consecutive detections before a marker appears (server/fusion.py).
5. **Phone viewer** — a web page (phone/viewer.html) showing ONLY the
   phone's own camera with the skeleton overlay, distance and
   uncertainty. The phone **never receives the drone's video** — only
   compact JSON marker data. Verified by test (see below).

## The marker is a skeleton, not a dot

The marker is the COCO 17-joint pose skeleton, drawn as a stick figure
and scaled by distance through real 3D projection. If pose estimation
fails on a frame, the last good skeleton is held for **at most 0.5 s**
(flagged STALE, amber), then hidden. A dot is never substituted.

## iPhone notes (read before the real run)

- The viewer is a web page for **Safari on iPhone**: camera via
  `getUserMedia`, **no WebXR** (Safari does not support it), **no
  device-orientation tracking at all** — the phone is held still and its
  pose comes from the bridge setup.
- Safari needs a **secure context** for the camera: plain `http://`
  works on localhost only. On the laptop's LAN IP you must use HTTPS:
  `tools/make_cert.sh` generates a self-signed cert; open
  `https://<laptop-ip>:8080/viewer` and accept the certificate warning.
- **Honest limit:** full 6-DoF phone tracking on iPhone would need a
  native ARKit app, which is out of scope for this build. The web
  viewer is the working version; if the user moves the phone after the
  bridge setup, the marker goes stale (it is anchored to the setup pose).

## Measured results (virtual tests, 2026-11-23)

Full table + method: `tests/TEST-REPORT.md`. All 7 original scenarios
PASS, and the extended 27-scenario suite (2026-11-23) also passes in
full — the re-run reproduces the recorded numbers exactly (see the
regression table in TEST-REPORT.md). Thresholds were set in SPEC.md
before the runs and were not moved.

| Scenario | Detection | Mean joint err | Mean |dist err| | Scale err | False markers |
|---|---|---|---|---|---|
| s1_basic (4.0 m) | 100% | 38.6 cm | 11.0 cm | 11.9% | 0 |
| s2_offcentre (5.5 m) | 100% | 35.6 cm | 0.6 cm | 14.0% | 0 |
| s3_close (3.0 m) | 100% | 37.8 cm | 13.5 cm | 11.2% | 0 |
| s4_dim (0.5× light) | 100% | 38.9 cm | 10.3 cm | 12.5% | 0 |
| s5_tracking_loss | 100% | 43.5 cm | 16.1 cm | 9.7% | 0 after loss |
| s6_no_person | — | — | — | — | 0 |
| s7_two_drones | 100% | 37.1 cm | 9.4 cm | 12.1% | 0 |

- Two-drone fusion: fused joint error 37.1 cm ≤ best single drone
  (39.6 cm). Triangulation depth bias −3 to +2 cm; bbox-height bias
  +20 to +34 cm.
- **No-drone-video test: PASS** — the server's phone-facing routes
  serve only HTML + JSON; 12 probed video-like routes all 404.
- Two bugs were found and fixed during the build, both in the record:
  the projection `v` sign (it flipped skeletons upside-down; fixed in
  SPEC.md, server/geometry.py and phone/viewer.html together) and the
  person position definition (median-of-joints biased the distance
  ~0.5 m short; fixed by using the foot point — see
  server/depth.py: foot_point).

## Third depth estimator: monocular depth model (2026-11-23)

A third depth estimator votes in the triangulation / bbox-height
cross-check (`server/mono_depth.py`, `server/depth.py:
fuse_with_model_votes`, wired in `server/main.py: Pipeline`).
Enable it with `--mono-depth` on the server CLI; the test harness
runs it on 13 scenarios (see `tests/TEST-REPORT.md`, "Depth-model
tests").

**Model choice (measured): Depth Anything V2 Small**
(`depth-anything/Depth-Anything-V2-Small-hf`, transformers, fp32,
CPU) — 24.8M params, 95 MB weights, Apache-2.0 (the only DA-V2
checkpoint under a permissive licence). The metric alternative,
ZoeDepth (`Intel/zoedepth-nyu-kitti`, DPT-Large backbone), is
~1.37 GB fp32 — ~14x larger and larger than the ~1.2 GB free RAM
on this machine — so it was excluded on measured download size /
RAM before any inference run. CPU inference on this machine
(single frame, process-default threads): 256x192 ≈ 380-470 ms,
336x252 ≈ 650-750 ms, 518x388 ≈ 1.6-2.3 s. The pipeline uses
256x192.

**Metric scale — the floor-plane anchor (measured per frame, never
a hard-coded constant):** DA-V2-Small outputs relative depth only.
Its scale comes from a robust linear fit `true_z = a * model + b`
over floor pixels in the lower frame, where the true depth follows
from the ray / y = 0 plane intersection using the known drone pose
(camera height above the floor — the same known-height fact the
VIO scale method uses). The fit is accepted only with >= 20 floor
samples, residual RMS <= 0.35 m and |Pearson r| >= 0.8, otherwise
there is no model vote that frame. The person depth is the fitted
value at the median model value over the visible keypoint pixels
(keypoints are on the person; a bbox median would mix in the wall
behind them). The fit slope is negative on this model (its output
grows as things get closer — disparity-like); the fit measures the
sign, it is not assumed. The vote's sigma carries a floor of
0.15 * depth: the floor-fit residual alone understates the model's
real error (see below).

**Voting rule — pure arbitration:** the model NEVER replaces
triangulation as the primary method and NEVER refines a depth
number — it only breaks ties between the two established methods.
When triangulation and bbox-height agree (<= 25% apart), the
answer is triangulation alone. When they disagree (> 25%), the
model arbitrates: it backs triangulation (outcome changes vs the
legacy bbox-height fallback — counted as `voter_changed`) or
bbox-height (same outcome), and if it agrees with neither the
legacy fallback stands. Without triangulation the model does
nothing (bbox-height alone). Fusing the model depth into the
winner was tried first and measured to hurt: in s15_occluded it
took joint error 59.3 -> 64.5 cm (agree-case fusion) and 59.3 ->
64.3 cm (disagree-case refinement), because the occluder biases
the model vote the same way it biases the bbox. With pure
arbitration the mono runs reproduce the baseline exactly unless
the voter flips a choice.

**Latency (the user's condition):** the model NEVER runs on the
marker path. It runs in a background thread, at most every 10th
detection frame (1 Hz at the 10 fps harness rate), on a 256x192
copy; the pipeline reads only the latest completed vote, age-gated
to 2.0 s with the sigma inflated 0.2 m/s of age. If the worker is
busy the frame is skipped, never queued. Every 5th frame was tried
first and kept the worker busy ~100% of the time, costing 7-28%
of pipeline fps to CPU contention on this 1.5-CPU VM. At 1 Hz the
measured cost is smaller but NOT zero on this machine: across the
12 single-drone voter scenarios the mean pipeline fps is ~9.6 with
the voter vs ~10.5 without it (about 8% slower; per-scenario
-33% to +4%, noisy — the baseline itself moves run to run on this
shared VM). The cost is pure CPU contention on the shared VM; on a
real multi-core laptop the background thread costs ~nothing. And
the voter is opt-in (`--mono-depth`): the default pipeline is
unchanged and pays nothing.

**Measured results (virtual tests, 2026-11-23)** — full tables in
`tests/TEST-REPORT.md` under "Depth-model tests". Model depth error
vs GT foot depth (probe, 3 frames x 7 scenarios, 256x192):

| Scenario | Model err mean | Model err mean |abs|| Model err worst |
|---|---|---|---|
| s1_basic | +11 cm | 15 cm | 20 cm |
| s4_dim | +8 cm | 8 cm | 16 cm |
| s13_dim_noise | -14 cm | 21 cm | 49 cm |
| s14_glare | +5 cm | 5 cm | 10 cm |
| s15_occluded | -20 cm | 71 cm | 137 cm |
| s18_moving | +22 cm | 22 cm | 47 cm |
| s11_large_room_far | +47 cm | 47 cm | 58 cm |

The model is per-frame (no static-scene assumption) and beats
bbox-height in clean scenes, but its error is larger than
triangulation's in static scenes and it is worst exactly where
the established methods are already stressed (occluded, far) —
which is why it arbitrates instead of refining. In the full
13-scenario voter suite the voter flipped the choice on 0 frames:
every mono run reproduces its baseline joint / distance error
exactly (see TEST-REPORT.md). The flip logic itself is proven by
direct checks of `fuse_with_model_votes`. A case the virtual
suite does not contain — triangulation right and bbox-height
badly wrong on the same frame (e.g. a crouching person whose bbox
height collapses) — is where the voter would change the outcome.

## Bridge pose source: monocular VIO (2026-11-23)

The Tello reports no pose, so the bridge's pose comes from laptop-side
monocular visual odometry on the drone video (`server/vio.py`,
`MonocularVO`). GoodFeaturesToTrack + Lucas-Kanade optical flow,
essential matrix (`cv2.findEssentialMat` with `DRONE_K`) +
`recoverPose` for the initial relative motion, then per-frame PnP
against a triangulated map (a custom Huber-LM PnP — `cv2.solvePnP`
and `solvePnPRansac` are unreliable on this near-planar map; see the
code). Poses are composed over time; there is no global loop
closure, so drift grows with distance travelled. A windowed local
BA (point positions only, poses held fixed) refines the map on
keyframes, and a keyframe-descriptor database supports
relocalisation after a loss.

**Scale** (monocular VO has none): the drone takes off beside the
phone from a known start pose at a known constant height above a flat
floor (`camera_height_m`, default 1.3 m). At initialisation the floor
points in the lower part of the keyframe image are triangulated with
a unit baseline; their true world height is y = 0, so the true
baseline is `s = C_y / (C_y - median_y_unit)` using the nearer half
of the floor points (far floor points bias the scale — measured:
+3.4% error with all floor points vs +1.2% with the nearer half).
After initialisation the scale is carried by the map: new frames are
localised by PnP against absolute-scale map points. Scale drift is
measured, not assumed (see below).

**Validity gate:** a pose is valid only if the PnP has >= 20 inliers
at >= 25% ratio, or (thin-map tier) >= 12 inliers at >= 50% ratio —
healthy frames measure 80-250 inliers; genuinely lost frames collapse
to < 10 at < 0.1 ratio. `tracking_valid=False` invalidates the bridge
(no marker until reacquire), never a guessed pose. Recovery wires into
`Bridge.reacquire()`; in the real system reacquire must be gated on
genuine relocalisation (re-observing a mapped area / doorway loop
closure) — the harness proves the state-machine logic only (same
caveat as the existing s21 scenario).

**Rotation-dominant motion** (the turn) is the hard case: pure
rotation has no parallax, so no new map point can be triangulated
and the map drains. The VIO therefore keeps untriangulated tracks
as 2D bearings (inverse depth d=0), triangulates only when the
parallax angle reaches 5° across keyframes (deferred triangulation),
and estimates rotation only from the bearings (homography) when the
3D map is too thin for PnP. An optional Tello state prior
(`server/tello_state.py`) supplies yaw/pitch/roll + h/tof, and a
calibrated velocity as a motion model; every pose is flagged
(vision / attitude-aided / dead-reckoned / invalid) so the bridge
never silently presents an attitude-only pose as visual tracking.
The velocity unit/frame is UNCALIBRATED by default (velocity
priors OFF) until `tools/calibrate_tello.py` has been run on a real
recording — see `docs/TELLO_CALIBRATION.md`. The harness feeds
synthetic state packets derived from the GT trajectory, so the
prior code path is exercised (with the matching calibration).

**Measured results (virtual tests, 2026-11-23, fix round 3)** —
full table in `tests/TEST-REPORT.md` under `## VIO tests`, and the
round-3 narrative under `## VIO fix round 3` in the same report.
The harness trajectories are now smoothed (yaw slew-limited to
60°/s, the real Tello rate) and the VIO tracks through the doorway
and the turn. Markers ARE emitted (10–11 per scenario) — the
end-to-end VIO→marker path is proven, and **all 4 VIO scenarios
now pass the 60 cm joint-error budget**.

| Scenario (VIO) | VIO valid | Drift mean | Drift max | Joint mean | Markers | Result |
|---|---|---|---|---|---|---|
| s1_basic | 56/60 | 6.7 cm | 21.1 cm | 40.5 cm | 10 | PASS (round 2: 71.8 FAIL) |
| s23_blank_door | 56/60 | 5.4 cm | 14.7 cm | 32.6 cm | 11 | PASS (round 2: 81.7 FAIL) |
| s20_fast | 56/60 | 8.3 cm | 71.5 cm | 36.1 cm | 11 | PASS (round 2: 38.6) |
| s4_dim | 56/60 | 3.8 cm | 15.0 cm | 36.3 cm | 10 | PASS (round 2: 55.2) |

The default PnP mode is now **1-DoF yaw PnP** (`pnp_mode="yaw1dof"`:
pitch/roll fixed from the Tello attitude prior, yaw + centre
solved jointly). It was picked by measurement over the 6-DoF PnP,
a soft prior factor, and a translation-only PnP — see the report
for the full mode table. The other half of the fix is a
**bbox-height detector calibration** (`server/depth.py`
`BBOX_HEIGHT_FRACTION = 0.89`): the YOLO bbox spans ~0.89 of the
person's height, so H = 1.7 m overestimated depth by ~13% — in s4
that was a −26.5 cm systematic distance bias on all 10 marker
frames, which is why s4 sat at 59.4 cm before the calibration.
(VIO step 81–85 ms/frame mean on this shared 1.5-CPU VM; pipeline
fps 4.7–6.4, below the 10 fps target — a machine limit, not code:
the detector alone is 96.5 ms/frame on this VM, and the VIO +
detector run sequentially; see the fps profile in the report. The
pose-source split is mostly vision with 5–11 attitude-aided
(rotation-only) frames per scenario and 5 invalid frames (the
initialisation frames); bridge-invalid markers = 0 everywhere.)

(GT-pose results for the same scenarios, post-calibration: s1 joint
36.1 cm, dist 8.7 cm; s23 joint 36.0 cm, dist 11.8 cm; s20 joint
35.2 cm, dist 7.4 cm; s4 joint 35.6 cm, dist 8.0 cm — all PASS.
The VIO now adds only 0–4.4 cm of joint error on top of the GT
baseline. The remaining gap is the rotation error during the turn
(yaw 0.7–1.0° mean, 3.2–7.6° max), which is a slowly-varying bias
after the turn, not frame noise — a yaw smoother cannot remove it,
and the NEC / rotation-averaging build was tried and measured to
have the same systematic bias on this data (see the report). The
post-hoc yaw correction from round 2 stays reverted: changing R
without re-optimising C breaks consistency.)

**Caveat — prior accuracy is the binding constraint.** The 1-DoF
yaw PnP is fragile to prior error: a (yaw+2, pitch−1, roll+1) deg
offset on the synthetic state packets fails s1/s20/s4 (81.6–
90.4 cm). The harness prior is accurate to ~1 deg (quantised GT),
so the harness pass holds; on a real Tello the magnetometer yaw
accuracy decides it, and the §6.9 yaw calibration
(`tools/calibrate_tello.py`) must show σ ≲ 1.5 deg with no doorway
bias before the real test.

**Two follow-up pieces (2026-11-23, measured — see
`tests/TEST-REPORT.md` under `## VIO improvement — calibration +
third yaw`):**
- **Yaw calibration gate (piece 1).** `tools/calibrate_tello.py`
  now prints a plain `YAW GATE: PASS/FAIL` verdict: pass needs yaw
  sigma ≤ 1.5 deg AND |doorway bias| ≤ 1.5 deg, else the real test
  must not proceed. Verified end-to-end on synthetic data (PASS at
  sigma 1.0 deg, FAIL at sigma 2.5 deg and at +3.0 deg doorway
  bias). The REAL calibration still awaits a real drone — no real
  calibration is claimed.
- **Third yaw measurement (piece 2).** `server/vp_yaw.py` estimates
  absolute yaw per frame from vanishing points (LSD lines,
  Manhattan compass) — unbiased, sigma 1.07 deg, available on
  60–82% of frames (s1 44/60, s23 49/60, s20 42/60, s4 36/60),
  ~60 ms/frame full-res or ~37 ms half-res with the same accuracy.
  It needs the magnetometer yaw as a hint (the X/Z assignment is
  90 deg ambiguous without one). **Wiring decision: NOT wired into
  `server/vio.py`.** Fusing it per frame into the VIO prior
  (1/sigma²-weighted, gated at 10 deg) does NOT help on a clean
  prior — joint error worse on average (+6.6 cm; s1 40.5 → 56.0,
  s23 32.6 → 46.6 cm) because the VIO's output yaw is
  vision-dominated and the fusion only perturbs the PnP. It DOES
  rescue a bad prior (offset +2/−1/+1 deg): s23 93.9 → 36.5, s20
  76.4 → 54.9, s4 79.8 → 47.6 cm — but s1 stays at 60.2 cm, a
  0.2 cm FAIL, because a yaw-only measurement cannot fix the
  pitch/roll offsets. Recommended next step (unmeasured): a slow
  bias tracker that corrects the prior yaw only when the
  VP-vs-prior disagreement persists over ~30 frames.

**Ground test** (`tests/vio_ground_test.py`): renders a smooth
doorway fly-through (no yaw snaps — the way a real flight is flown)
to a lossless video + poses JSON, then runs the VIO on it and reports
drift. Measured 2026-11-23 with the yaw1dof default: 102/105 frames
valid (97%), drift mean 29.1 cm, max 61.9 cm, final 39.9 cm
(approach 7.0 cm, pass 14.7 cm, turn 27.8 cm, hover 47.7 cm), PnP
inliers mean 9, VIO step 58–64 ms/frame. Round 2 lost tracking
mid-turn (55/105 valid, drift 9.2 cm over the valid frames only) —
round 3 tracks the whole flight, but the hover drift (47.7 cm
mean) is the remaining weakness: the turn's yaw bias persists as
a position drift while hovering. The bridge invalidates correctly
on a genuine loss (see the harness s21 scenario).
Blank-wall variant (`--blank-door`, the s23 geometry with no posters
on the shared wall): 102/105 frames valid, drift mean 29.1 cm, max
61.9 cm — IDENTICAL to the poster version to 0.1 cm in every phase.
The door posters contribute nothing the VIO uses on a smooth flight
(the floor and the other walls carry the features), so the blank
wall does not degrade the ground test at all. On the harness's snap
trajectory the blank-wall scenario is worse (s23 VIO drift 5.4 cm
mean in round 3, vs 13.3 cm in round 2) because the snap already
leaves few points and the blank wall removes the doorway anchors —
but with the yaw1dof default s23 now passes at 32.6 cm joint error.

```bash
# generate the smooth flight + run the VIO ground test
~/workspace/tello-search/.venv/bin/python -m tests.vio_ground_test \
  --generate --video /tmp/smooth_flight.avi --poses /tmp/smooth_flight.json
# run the VIO on a recorded video + poses JSON (FileVideoSource format)
~/workspace/tello-search/.venv/bin/python -m tests.vio_ground_test \
  --video /tmp/smooth_flight.avi --poses /tmp/smooth_flight.json
# serve one virtual scenario with VIO poses instead of ground truth
~/workspace/tello-search/.venv/bin/python -m tests.virtual_server \
  --scenario s1_basic --no-serve --vio
```

**Real ground-test procedure:** carry the Tello by hand (motors OFF)
through a doorway while it records, measure the ground-truth path
(tape-measured waypoints + a level flight at a known height), then
run `tests/vio_ground_test.py --video <file> --poses <file>` against
the recording + the measured path. Fly and turn smoothly (yaw
~30-60 deg/s, not the harness's 300-1000 deg/s snaps) and keep the
floor visible and textured at takeoff — the scale method needs it.
A real H.264 recording will add compression noise the lossless
ground-test video does not have; that is a separate measurement on
real footage, not something to assume.

## Honest limits (what is NOT proven yet)

- **Virtual poses are exact.** The harness feeds the bridge the true
  drone pose, so no SLAM drift is simulated. The bridge uncertainty
  (0.10 + 0.02 m/s, capped at 1.0 m) is a model, not a measurement. The
  research report's real-world budget (±20–60 cm at 3–6 m with good
  depth, ±0.5–2 m monocular/Tello) still stands for the real system.
- **The VIO bridge is built and measured, but not flight-proven.**
  The Tello has no pose output and no depth sensor, so on the real run
  the drone's pose comes from laptop-side monocular VIO
  (`server/vio.py`) — implemented, wired into the pipeline
  (`--vio` in `tests/virtual_server.py`, VIO scenarios in `tests/run_tests.py`)
  and measured on synthetic video. What is NOT proven: real Tello
  video (H.264 compression, motion blur, real lighting, real texture),
  the real yaw rates and floor visibility at takeoff, and genuine
  relocalisation after a loss (the harness proves the state-machine
  logic only). The ORIGINAL harness trajectories' yaw snaps (-31 deg
  at the doorway, +101 deg at the turn, each in one 0.1 s frame) are
  un-trackable by any frame-to-frame method at 10 fps — LK, ORB
  descriptor matching and a photometric yaw search were all tried and
  measured to fail on them — so the real run must be flown smoothly,
  and the VIO scenarios now use slew-limited (60°/s) trajectories.
  Fix round 2 (2026-11-23) tracks 56/60 frames and emits 10–11
  markers per VIO scenario, but the joint error still exceeds the
  60 cm budget in s1 (71.8 cm) and s23 (81.7 cm) — the turn's yaw
  error (1.0–1.2° mean, 4.0–4.5° max) corrupts the depth
  triangulation. The remaining fix is to constrain the PnP itself
  (1-DoF yaw PnP or a pose-graph with the yaw prior as a factor).
  Until the ground test is run on real footage, the real run is a
  controlled procedure: drone takes off beside the phone, flies the
  doorway slowly in good light, laptop on the Tello Wi-Fi.
- The person in the tests is a photo cutout (fronto-parallel by
  construction), so the planar joint approximation is flattered; a real
  person turned sideways or with outstretched arms will have larger
  joint errors. The cutout person also has crossed arms vs the
  ground-truth skeleton's hanging arms (wrists up to ~64 cm off on
  single frames).
- One static person, one room pair, no occlusion, no motion blur, no
  moving person (triangulation assumes a static scene), no network
  latency, no live-phone rendering check.
- "Exact" is not achievable: the system reports the best estimate with
  a stated uncertainty (e.g. "≈4.2 m ±0.5 m"), never false centimetre
  precision. See the research report for the full reasoning:
  `~/workspace/research_notes/drone-phone-through-wall-person-marker-20261122-1906/report.md`.

## Run it

Environment: `~/workspace/tello-search/.venv/` (Python 3.12.3,
opencv 5.0.0, numpy 2.5.4, ultralytics 8.4.216, torch 2.15.0).
First YOLO run downloads `yolov8n-pose.pt` (~6.5 MB).

```bash
cd ~/workspace/tello-search/wall-marker
# install note: pip used /tmp for the ~2 GB torch download and /tmp is a
# 512 MB tmpfs here — if the install fails with "No space left on
# device", re-run with TMPDIR and PIP_CACHE_DIR under home:
#   TMPDIR=$HOME/.cache/pip-tmp PIP_CACHE_DIR=$HOME/.cache/pip \
#   ~/workspace/tello-search/.venv/bin/pip install ultralytics

# 1. full virtual suite (27 GT-pose scenarios + VIO + mono voter +
#    no-drone-video test + 5 real-only UI tests, ~12 min)
~/workspace/tello-search/.venv/bin/python tests/run_tests.py

# 2. serve one virtual scenario and open the viewer (TEST-ONLY entry
#    point — the product server serves live drone frames only and
#    refuses virtual mode; see step 3)
~/workspace/tello-search/.venv/bin/python -m tests.virtual_server \
  --scenario s1_basic --port 8080
# open http://localhost:8080/viewer (laptop) or
# https://<laptop-ip>:8080/viewer (iPhone, with --https --cert/--key)
# operator page (laptop only): http://localhost:8080/operator

# 3. real run (LATER, only on the user's go-ahead):
#    laptop on the Tello Wi-Fi, drone on the ground beside the phone.
#    Requires djitellopy (real-run only; not installed for virtual tests).
~/workspace/tello-search/.venv/bin/python -m server.main --mode live --port 8080
```

## Operator page (laptop only)

`GET /operator` on the same server (product or virtual): three blocks
BRIDGE OK/LOST, DRONE LIVE/NO VIDEO, PERSON FOUND/NONE; one facts
line (drones, tracks, markers, uptime, last-frame age, frames in the
last 10 s — "—" when unknown, never a made-up 0); a 3-step checklist
that ticks itself from real state (bridge linked, drone frames
flowing, person found); one Start/Stop button that pauses/resumes
the pipeline via `POST /api/control` (while paused the marker stream
honestly reports no marker, reason "paused"). No drone video on this
page, no per-joint tables, no log tail. Two new `/api/status` fields
feed it: `last_frame_age_s` and `frames_last_10s` (both null until
the first frame).

## Layout

```
SPEC.md             the build contract (interfaces, thresholds, camera models)
README.md           this file
requirements.txt    ultralytics + transformers/timm (Depth Anything V2
                    Small, third depth voter); djitellopy commented
                    (real-run only)
server/             geometry, detector (YOLO-Pose), depth, mono_depth
                    (Depth Anything V2 Small voter), bridge,
                    fusion, video_source, vio (monocular VO pose source),
                    nec (NEC/ROBA rotation, reference only),
                    tello_state (Tello state priors),
                    main (PRODUCT server: live frames only, refuses
                    virtual mode; Pipeline + HTTP/SSE server + /operator)
phone/viewer.html   iPhone Safari viewer (own camera + skeleton overlay)
tests/              render.py (two-room renderer), scenarios.py (7 scenarios),
                    run_tests.py (runner), vio_ground_test.py (VIO ground
                    test + smooth-flight generator),
                    virtual_server.py (TEST-ONLY virtual server entry
                    point: scenario frames + VIO/mono flags),
                    test_real_only.py (5 real-only UI tests),
                    TEST-REPORT.md (measured results)
tools/make_cert.sh  self-signed cert for iPhone HTTPS on LAN
```

## Build history (2026-11-23)

- Spec written from the research report; server, viewer and harness
  built by three parallel agents and verified by the coordinator.
- Two real bugs found and fixed before delivery: the `v`-sign flip
  (upside-down skeletons) and the median-of-joints distance bias.
- VIO bridge pose source built (2026-11-23): `server/vio.py`
  (monocular VO, floor-scale initialisation, two-tier PnP gate),
  `tests/vio_ground_test.py` (smooth-flight generator + drift report),
  VIO scenarios in `tests/run_tests.py`, `--vio` in `tests/virtual_server.py`.
  Measured: approach + pass track at 2-6 cm drift; the harness
  trajectories' yaw snaps defeat frame-to-frame tracking (reported
  honestly with measurements, not hidden); the smooth-flight ground
  test tracks 55/105 frames at 9.2 cm mean drift.
- Monocular depth model, third voter (2026-11-23):
  `server/mono_depth.py` (Depth Anything V2 Small, floor-plane
  scale anchor), voting in `server/depth.py`
  (`fuse_with_model_votes`, pure arbitration), background worker in
  `server/main.py` (`Pipeline`, `--mono-depth` on the virtual server
  CLI, 1 Hz, marker path never waits),
  `tests/mono_depth_probe.py` (standalone model measurement),
  13-scenario voter suite in `tests/run_tests.py`. Measured: model
  depth error mean |err| 5-47 cm clean, 71 cm occluded (worst
  137 cm); the voter flipped no choice in the suite, so every mono
  run reproduces its baseline exactly — the fusion variant was
  tried first, measured to hurt s15_occluded (59.3 -> 64.5 cm),
  and replaced by pure arbitration. Pipeline fps with the voter is
  ~8% lower on this shared 1.5-CPU VM (mean 9.6 vs 10.5 fps) and
  the voter is opt-in, so the default pipeline pays nothing.
- Easy UI server part (2026-11-23, PLAN.md step 4): SSE now streams
  a markers array (`{"type":"markers","markers":[...]}` — up to 3
  people, nearest first, drone count when >1; the single-marker
  payload still works, `Fusion.marker()` unchanged); new laptop
  `/operator` page (BRIDGE / DRONE / PERSON blocks, self-ticking
  3-step checklist, Start/Stop via `POST /api/control`, two new
  `/api/status` fields `last_frame_age_s` and `frames_last_10s`,
  null until the first frame); the virtual serving path is split
  out of the product server into the test-only entry point
  `tests/virtual_server.py` — `server/main.py` serves live drone
  frames only and refuses `--mode virtual` (exit 2, points at the
  test entry point) and `--mode live` without djitellopy/Tello.
  5 real-only tests in `tests/test_real_only.py`, wired into the
  full suite: all 5 PASS (measured — see TEST-REPORT.md).
- Nothing outside `wall-marker/` was modified; the existing
  `tello_path_search.py`, `fly_brain.py` and `test_videos/` are untouched.

- Live-path fixes, reconciled with d89be655 (2026-11-24): the other AI's `SafeFlightController` (commit d89be655)
  is kept as the ONE flight controller — it serializes commands
  (single lock, busy commands skipped) and gates movement on a
  successful takeoff. Two completions added to it: land is now
  idempotent (repeat `[L]` after landing sends nothing) and a
  "Not joystick" takeoff failure logs the force-close-the-Tello-app
  guidance; the `finally` block lands only while holding the
  controller lock and only if airborne. Marker-format fix (the part
  d89be655 did not touch): the live loop no longer overrides the VIO
  verdict with the frozen start pose — new `live_pose_step()` passes
  it through honestly, so no marker is emitted without a bridge
  (the 2026-11-23 scrambled skeleton, 2.2 m for a person 0.5 m
  away, came from feeding that fiction frame as tracking-valid);
  scenario time is now wall-clock. `_serve_sse` now catches
  `OSError` (WinError 10053 on browser disconnect). Verified:
  `tests/test_live_fixes.py` 9/9 PASS; full virtual suite re-run
  27/27 PASS with VIO numbers unchanged (s1 40.5, s23 32.6, s20
  36.1, s4 36.3 cm). Pushed as commit 531ba8ba.

- Follow-up fixes from the 2026-11-24 live run (same day): a move
  failing with "Not joystick" or "No valid imu" now resets the
  controller to LANDED so [T] works again (the live log showed the
  drone on the ground while the controller still said HOVERING, so
  five [T] presses were ignored); the rejection is labelled by the
  command that failed (MOVE vs TAKEOFF); the HTTP server swallows
  phone-browser disconnects (WinError 10053) instead of dumping a
  traceback. `tests/test_live_fixes.py` extended to 12 tests, all
  PASS; real-only UI server tests re-run: PASS.

- Video-restart fix from the 2026-11-25 live run: djitellopy's
  decode thread died (`av.error.OSError` on `udp://@:11111`) and
  `TelloVideoSource.read()` kept serving the last frame forever.
  `read()` now detects a dead stream (same frame object > 3 s, or
  a stopped reader) and restarts it (`streamoff` / `streamon` /
  fresh reader), up to 3 attempts per run; after that it stays
  honestly dead and returns None, with the log naming the Tello
  app as the likely cause. `tests/test_live_fixes.py` extended to
  16 tests, all PASS.
- Key remap from the 2026-11-25 live run: Q used to close the camera
  window (the user read that as the program crashing — it only hid
  the window). Keys are now the user's own map: Q or T takeoff, E
  land, L turn right (L is deliberately NOT land any more), J turn
  left; W/S/A/D/I/K unchanged. Dispatch is a tested function
  (`dispatch_drone_key`); the window closes with its X button.
  `tests/test_live_fixes.py` extended to 21 tests, all PASS.
- VIO diagnostics + takeoff reset + ground watchdog from the
  2026-11-25 live runs: the bridge never became valid because the
  VIO keyframe was a pre-takeoff ground frame (~2.5 min stale);
  takeoff now resets an uninitialised VIO so the keyframe is
  airborne. Init failures log the reason with numbers; a 5 s
  console status line carries battery % over time, bridge and VIO
  state. The controller now watches the height while HOVERING and
  resets to LANDED if the drone lands itself (low-battery failsafe,
  blinking red), so [T] works again. `tests/test_live_fixes.py`
  extended to 30 tests, all PASS. Battery note: ~2 min flights on
  a full charge vs ~13 min healthy — the battery is worn; the %
  trace will show any in-flight sag.
