# Wall-Marker Virtual Test Report

Generated: 2026-11-22 22:53:46 (UTC)

## Environment

- Python 3.12.3
- opencv 5.1.0, numpy 2.5.4, ultralytics 8.4.216
- Weights: `yolov8n-pose.pt` (in `wall-marker/`), detector conf 0.5, imgsz 640, CPU
- Total runtime: 90.4 s (1.5 min), one inference process
- Person billboard: largest person detection from ultralytics' bundled `bus.jpg`, cut out with grabCut (mask eroded 2x), pasted at the TRUE projected position (bottom-centre at the projected foot pixel, height = true projected height). Standalone-detection gate passed (conf 0.90 on a plain background).
- Drone/room model: SPEC.md (wall z = 3.0 m, doorway x in [-0.6, 0.6] m, phone fixed at (0, 1.5, 0)). Projection `v = cy - fy*y/z` (corrected 2026-11-23).

## Results

| Scenario | Detection rate | Mean joint err | Median joint err | Mean |dist err| | Scale err | False markers | Marker after loss | Methods (tri / bbox-h) | Result |
|---|---|---|---|---|---|---|---|---|---|
| s1_basic | 100% (36/36) | 38.6 cm | 34.0 cm | 11.0 cm | 11.9% | 0 | 0 | 31 / 5 | PASS |
| s2_offcentre | 100% (34/34) | 35.6 cm | 30.0 cm | 0.6 cm | 14.0% | 0 | 0 | 32 / 2 | PASS |
| s3_close | 100% (38/38) | 37.8 cm | 34.5 cm | 13.5 cm | 11.2% | 0 | 0 | 37 / 1 | PASS |
| s4_dim | 100% (36/36) | 38.9 cm | 35.6 cm | 10.3 cm | 12.5% | 0 | 0 | 31 / 5 | PASS |
| s5_tracking_loss | 100% (36/36) | 43.5 cm | 39.4 cm | 16.1 cm | 9.7% | 0 | 0 | 31 / 5 | PASS |
| s6_no_person | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 / 0 | PASS |
| s7_two_drones | 100% (36/36) | 37.1 cm | 32.8 cm | 9.4 cm | 12.1% | 0 | 0 | 31 / 5 | PASS |

Thresholds (SPEC.md, not moved): detection rate >= 90%, mean joint error <= 60 cm, mean |distance error| <= 60 cm, skeleton scale error <= 25%, false markers = 0 (s6), marker-after-loss = 0 (s5), s7 fused error <= best single-drone error.

### s1_basic — person 4.0 m from phone, centred, light 1.0

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 576; distance-error samples: 34.
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +31 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -11 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- **Result: PASS**

### s2_offcentre — person 5.5 m from phone, off to the side near the doorway

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 34; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 26.
- Markers emitted: 36 (4 stale); joint-error samples: 544; distance-error samples: 32.
- Depth methods: {'bbox_height': 2, 'triangulation': 32}; first triangulation frame: 22.
- Measured depth bias (estimated - true foot depth): bbox-height +33 cm (n=2); triangulation -3 cm (n=32).
- Mean signed distance error (marker - GT): -1 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- **Result: PASS**

### s3_close — person just behind the wall (z = 3.3 m)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 36 (0 stale); joint-error samples: 612; distance-error samples: 36.
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height +20 cm (n=1); triangulation +2 cm (n=37).
- Mean signed distance error (marker - GT): -14 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- **Result: PASS**

### s4_dim — light 0.5, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +34 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -10 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- **Result: PASS**

### s5_tracking_loss — tracking_valid True until frame 35, False after; person stays visible

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 10 (0 stale); joint-error samples: 168; distance-error samples: 10.
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +31 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -16 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_after_loss`: PASS
- **Result: PASS**

### s6_no_person — room B empty, same flight path; zero markers, ever

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 0; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 60.
- Markers emitted: 0 (0 stale); joint-error samples: 0; distance-error samples: 0.
- Depth methods: {}; first triangulation frame: None.
- Measured depth bias (estimated - true foot depth): bbox-height n/a; triangulation n/a.
- Mean signed distance error (marker - GT): n/a (negative = marker distance shorter than the true phone->person distance).
- check `false_markers`: PASS
- **Result: PASS**

### s7_two_drones — two drones, different hover points, both see the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +26 cm (n=5); triangulation -2 cm (n=23).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Fused mean joint error, hover frames 25-59: 37.1 cm; single-drone mean joint error over the same frames: drone1 39.6 cm, drone2 41.6 cm; best single: 39.6 cm. Single-drone detection (hover frames): {'drone1': {'detected': 35, 'in_view': 35}, 'drone2': {'detected': 27, 'in_view': 27}}.
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `fused_le_best_single`: PASS
- **Result: PASS**

## No-drone-video test

- Server started on 127.0.0.1:18765 (real `server.main` handler).
- `/viewer`: 1 `<video>` tag(s); getUserMedia present: True; video-endpoint hints found: none.
- `/api/markers`: 7 SSE events read in 3 s; all parse as JSON with type marker/none: True; content-type text/event-stream, no video content-type anywhere: True.
- Unknown routes probed (12): all 404: True ({'/video': 404, '/stream': 404, '/frames': 404, '/api/video': 404, '/api/frames': 404, '/drone': 404, '/api/drone': 404, '/snapshot': 404, '/mjpeg': 404, '/api/markers/1': 404, '/admin': 404, '/api/status/extra': 404}).
- check `root_redirects_to_viewer`: PASS
- check `viewer_200_html`: PASS
- check `exactly_one_video_tag`: PASS
- check `no_img_tag`: PASS
- check `no_fetch_or_xhr`: PASS
- check `no_drone_stream_reference`: PASS
- check `uses_getusermedia`: PASS
- check `no_video_content_type`: PASS
- check `status_json_shape`: PASS
- check `sse_content_type`: PASS
- check `sse_no_video_content_type`: PASS
- check `sse_events_received`: PASS
- check `sse_all_json`: PASS
- check `sse_types_json_only`: PASS
- check `unknown_routes_all_404`: PASS
- **Result: PASS**

## Honest limits (what this harness does NOT prove)

- Virtual poses are exact: the bridge is fed the true drone pose, so no SLAM drift is simulated. The bridge uncertainty (0.10 + 0.02 s/s, capped at 1.0 m) is a model, not a measurement; the tests check whether real errors stay inside it, but a real VIO/SLAM chain will add drift the harness cannot see. The research report's end-to-end budget (±20-60 cm at 3-6 m with good depth, ±0.5-2 m monocular/Tello) still stands for the real system.
- The person is a photo cutout, not a 3D body: it is fronto-parallel to the drone by construction, so the planar joint approximation in `depth.joints_3d` is flattered. A real person turned sideways, or with outstretched arms, will have larger joint errors.
- The planar joint approximation is the largest single joint-error source, and it is measured, not assumed: the drone looks down at the person (camera 1.3 m, chest target 0.85 m), so the person's joints span ~0.3 m of depth along the optical axis (ankles farthest, head nearest), but `joints_3d` places every joint on the foot-depth plane. Upper-body joints are therefore pushed ~0.2-0.3 m too far from the drone; head-joint errors of ~30 cm occur even though their projected pixels are only ~17 px off. A per-joint depth (or a real depth sensor) would remove most of this.
- The cutout person stands with arms crossed, while the ground-truth skeleton assumes arms hanging at the sides; elbow/wrist joint errors therefore include a real pose mismatch of a few tens of cm (wrists up to ~64 cm on single frames). The person also carries a blue bag that is part of the cutout, and one leg is crossed in front of the other, so the detected ankles sit ~0.2 m above the ground-truth ankle height.
- The detector's bbox is ~15% shorter than the true billboard height (tight box around the visible body, shoes/hair excluded); the measured depth bias per method is in each scenario section above. Triangulation uses the bbox bottom-centre ray, which sits ~0.15 m above the true foot point.
- The pipeline's person position is the FOOT point (ground contact), reconstructed from the bbox bottom-centre ray at the estimated depth (server/depth.py: foot_point). An earlier build used the median of the detected joints (~1.07 m above the floor), which biased the displayed distance ~0.5 m short; the foot-point fix (2026-11-23) cut the mean distance error from 44-54 cm to 9-16 cm across the scenarios. The remaining error is dominated by the planar joint approximation and the bbox bottom edge sitting ~0.15 m above the true foot point. The mean signed distance error per scenario is in each scenario section above.
- One static person, one room pair, no occlusion, no motion blur, no moving person (triangulation assumes a static scene), no network latency, no phone-side rendering check (the viewer projection is verified by code inspection + the GT phone projection in the scenarios, not by a live phone).
- Detection at 640 px on CPU; dim scenario is a uniform 0.5x brightness factor, not real low-light noise.

## Method used per frame

Per scenario, frames using triangulation vs bbox-height are counted in the table above. In every person scenario the pipeline switches to triangulation as soon as the drone's motion baseline reaches 0.3 m (a few frames after the person first appears) and stays there; bbox-height is used only for the first 1-3 detections. Exact per-frame method sequences:

- s1_basic: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s2_offcentre: f0-19:-, f20-21:bbox_height, f22-28:triangulation, f29-34:-, f35-59:triangulation
- s3_close: f0-21:-, f22-22:bbox_height, f23-59:triangulation
- s4_dim: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s5_tracking_loss: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s6_no_person: f0-59:-
- s7_two_drones: f0-23:-, f24-28:bbox_height, f29-59:triangulation

## VIO fix round 3 — constrained PnP + bbox calibration (2026-11-23)

Round 2 left the VIO joint error over the 60 cm budget in s1_basic
(71.8 cm) and s23_blank_door (81.7 cm); s20_fast (38.6 cm) and s4_dim
(55.2 cm) passed. The diagnosed cause was turn yaw error (1.0-1.2 deg
mean, 4.0-4.5 deg max) corrupting the depth triangulation. Round 3
was ordered repeat-until-perfect: search for a better method first,
then test → fix → implement → re-test until all 4 VIO scenarios pass.
This section records what was tried, what won, and what was rejected
— with numbers, so nothing is re-tried blindly. The per-scenario
round 2 vs round 3 table is in the extended section below
("### VIO fix round 3 — constrained PnP"); the numbers there are the
regenerated full-suite run (2026-11-23, 678.8 s).

### Winner: 1-DoF yaw PnP (default `pnp_mode="yaw1dof"`)

`server/vio.py` gained four PnP modes (`pnp_mode`: "full" = the
round-2 6-DoF PnP; "yaw1dof" = pitch/roll fixed from the Tello
attitude prior, yaw + centre solved jointly; "attitude" =
translation-only PnP with the full attitude from the prior;
"prior" = the 6-DoF PnP with the prior as a soft factor,
sigmas (2, 1, 1) deg). Each mode falls back down the chain
(attitude → yaw1dof → full, etc.) when its candidate fails the
`_accept_pnp` gate, so frames stay valid. Measured on the 4 VIO
scenarios (harness prior = quantised GT attitudes, accurate to
~1 deg):

| Mode | s1 | s23 | s20 | s4 | Verdict |
|---|---|---|---|---|---|
| full (round 2) | 71.8 FAIL | 81.7 FAIL | 38.6 PASS | 55.2 PASS | baseline |
| yaw1dof | 47.9 PASS | 45.9 PASS | 42.2 PASS | 59.4 PASS | best before the bbox fix — but s4 by only 0.6 cm |
| prior σ=2 | 57.3 PASS | 81.3 FAIL | 38.9 PASS | 69.5 FAIL | soft prior helps s1, hurts s4/s23 |
| prior σ=1 / σ=3 | 63.5 / 56.5 | 81.1 / 77.9 | 40.3 / 38.2 | 69.8 / 69.6 | no sigma passes all 4 |
| attitude (exact prior) | 37.9 PASS | 76.3 FAIL | 46.6 PASS | 56.7 PASS | best s1 ever, but s23 fails even with an exact prior |
| attitude (yaw ±2 deg) | 51.7 / 59.8 | 77.5 / 58.6 | 42.1 / 42.9 | 43.5 / 43.2 | non-monotonic, scenario-dependent |
| attitude (y+2/p−1/r+1) | 80.0 FAIL | 93.9 FAIL | 76.4 FAIL | 79.8 FAIL | fragile to realistic ±1 deg pitch/roll prior error |

The yaw-vs-joint correlation is NEGATIVE (r = −0.48 for "full"):
marker frames with yaw error < 0.5 deg had joint error 78.6 cm,
frames with yaw > 2 deg had 49.6 cm — so the round-2 diagnosis
(yaw error corrupts triangulation) was incomplete; depth quality
matters as much as yaw. s20 passes despite the largest marker-frame
yaw (4.2 deg) because fast motion gives a large triangulation
baseline. **"yaw1dof" is the default** (`MonocularVO(pnp_mode=...)`,
default changed from "full" to "yaw1dof" in round 3).

### The s4 fix was not the yaw at all: the bbox-height bias

With yaw1dof, s4 sat at 59.4 cm — 0.6 cm from the budget. Its
marker frames use bbox-height exclusively (all 10), and the
distance error there is a SYSTEMATIC bias, not noise: mean −26.5 cm,
std 1.7 cm (estimated depth 2.86 m vs true 2.53 m, every frame the
same way). The triangulation in s4 is corrupted and correctly
rejected by the 25% cross-check (baseline only 0.36 m — the person
is first detected at f48, so the history is short — and a 2.2 deg
marker-frame yaw error displaces the triangulated point ~0.7 m
shallow by the stereo error law depth²·Δθ/baseline; measured tri
depths 1.68-1.93 m vs true 2.36-2.58 m). The bbox bias is a wrong
height: the YOLOv8n-pose bbox spans ~0.89 of the person's height,
not 1.00. Measured on the harness renderer (person 1.7 m, s1/s23/
s20/s4, all in-view detected frames): implied H = true_depth ·
bbox_h_px / fx is 1.487-1.533 m (mean 1.51 m, within-scenario std
0.006-0.013 m). The bbox itself is precise (std < 1%); the error
is a stable bias. Fix: `server/depth.py` `BBOX_HEIGHT_FRACTION =
0.89` — the bbox vote now uses H = person_h_m × 0.89. The
keypoint-anchored alternatives were measured and REJECTED: the
ankle keypoint is noisier than the bbox bottom (implied-H std
0.017-0.022 vs 0.006-0.013), so the anchored and nose-to-ankle
heights give larger depth errors (0.35-0.53 m) than the plain bbox
with the corrected H. After the calibration, all 4 VIO scenarios
pass with margin (full-suite run): s1 40.5 cm, s23 32.6 cm, s20
36.1 cm, s4 36.3 cm — worst margin 19.5 cm. The calibration also
improved the GT-pose runs (s1 38.6 → 36.1, s4 38.9 → 35.6, s5
43.5 → 35.7, s7 37.1 → 34.1 cm), so the regression baseline
(`tests/run_tests.py` `RECORDED_2026_11_23`) was re-based to the
post-calibration numbers — the pre-calibration numbers are
preserved in the original section above and in this narrative.
The scale error rose slightly on the same runs (s1 11.9 → 13.1%,
s5 9.7 → 13.5%) — a side effect of the corrected depth on the
cross-check; still PASS, and the joint error improved everywhere.
The full 27-scenario GT-pose suite is 27/27 PASS (0 FAIL in the
whole report), and the VIO+mono runs reproduce the VIO-only
numbers exactly (voter changed 0 frames).

### Tried and rejected: NEC / ROBA rotation + rotation averaging

Per the parallel research report
(`~/workspace/research_notes/vio-yaw-error-better-methods-20261123/report.md`),
the NEC (Kneip & Lynen — rotation from bearings only, translation
dropped out) plus ROBA/Shonan rotation averaging was the top pick:
it breaks the coupling (yaw estimated from the same map that the
yaw error corrupts, chained frame-to-frame). Built in
`server/nec.py` (GTSAM 4.3.0 ShonanAveraging3 + a SciPy fallback)
and measured offline on the scenario frames. It does NOT beat
yaw1dof on this data, and was stopped with numbers:

- NEC per-pair (warm-started from the prior, LK tracks):
  0.73-0.87 deg mean, 2.6-3.8 deg max (s1/s23) — comparable to the
  PnP, not better. Cold-start NEC: 1.21 deg mean, 7.45 deg max.
  With GT correspondences (perfect tracks): 1.21 deg mean — so it
  is NOT track error and NOT planarity; the NEC is ill-conditioned
  at the 2.7 cm baseline (parallax ~0.4 deg).
- The per-pair error has a SYSTEMATIC bias: signed mean −0.37 deg
  (s1), −0.26 deg (s23); −0.75 deg/frame during the turn. The
  λ_min formulation (Kneip's original objective, R only) shows the
  same bias (−0.30/−0.26 deg), so it is in the NEC objective / the
  near-planar two-plane scene, not the solver. Chaining the bias
  gives 27.8 deg mean absolute yaw error over 60 frames.
- Long-range pairs are WORSE, not better: lag 5 → 1.16/1.67 deg,
  lag 10 → 1.55/3.51 deg (s1/s23). Averaging cannot fix a
  systematic bias, and the pairs are all warm-started from the
  same prior, so their errors are correlated.
- Two real bugs were found and fixed while building it: the
  Shonan gauge is RIGHT multiplication (the measurements
  R_i @ R_j.T are invariant under R_k → R_k @ Q), and the GTSAM
  4.3.0 `ShonanAveraging3.run()` signature is `run(min_p, max_p)`
  returning `(Values, cost)`. `server/nec.py` is kept as reference
  (documented as rejected) so the NEC is not re-tried blindly.
- Next ranked candidates NOT built (the loop stopped because
  yaw1dof + the bbox calibration already passes all 4 with
  margin): vanishing-point / Manhattan compass (a full new
  subsystem; the NEC result suggests vision-only rotation is
  weakly constrained on this data), and the magnetometer-as-rotation
  gate (the "attitude" mode above is its harness equivalent — it
  fails s23 even with an exact prior, and the §6.9 real-Tello yaw
  calibration is still unrun, so no real magnetometer accuracy is
  claimed).

### Tried and rejected: the cheap add-ons

- **Yaw-trajectory smoothing** (rate-limited): the yaw1dof signed
  yaw error has lag-1 autocorrelation 0.87-0.94 (s1/s23/s4) — it
  is a slowly-varying bias, not frame noise (f1-19: ±0.2-0.5 deg;
  f20-39 turn: a steady +0.65/−1.01/−0.59 deg; f40+: a persistent
  1-2 deg offset). A causal smoother tracks a smooth bias; it
  cannot remove it. Not built.
- **Insurance #5 (weight bbox-height more when yaw uncertainty is
  high)**: measured backwards — on marker frames, triangulation
  is MORE accurate than bbox-height in all 3 scenarios where it
  fires (s1: tri 44.1 vs bbox 57.3 cm; s23: 28.1 vs 67.3 cm; s20:
  39.0 vs 44.5 cm). Preferring bbox would hurt. Not built. The
  25% cross-check already rejects gross triangulation failures.
- **Mono depth voter on the VIO path**: still never flips
  (voter_changed = 0 on all 4 VIO scenarios, and on all 12 GT
  mono scenarios). Its value remains unproven in-suite.

### Sensitivity — the honest caveat on the winner

yaw1dof is FRAGILE to prior error. With a (yaw+2, pitch−1, roll+1)
deg offset applied to the synthetic state packets (simulating a
real Tello attitude off by that much — the video and GT are
untouched), yaw1dof fails s1 (90.4 cm), s20 (81.6 cm) and s4
(89.5 cm); only s23 passes (58.1 cm). Drift jumps to 21-22 cm
(vs 3.8-8.3 cm unperturbed). The "full" mode is robust to the same
offset (71.8/64.0/61.5/64.7 cm — barely worse than unperturbed,
because it solves all 6 DoF from vision) but fails the 60 cm
budget even unperturbed. So the harness pass holds only while the
attitude prior is accurate to ~1 deg — which the quantised-GT
harness prior is. On a real Tello the magnetometer yaw accuracy
is the binding constraint, and the §6.9 yaw calibration
(`tools/calibrate_tello.py`) must show σ ≲ 1.5 deg with no doorway
bias before the real test. This is the single most important
unmeasured number for the real system.

### Latent bug fixed: dead-reckoning NameError

`server/vio.py` `_dead_reckon` referenced `DEAD_RECKON_MAX_S` /
`DEAD_RECKON_MAX_M`, which were defined in `server/tello_state.py`
(2.0 s, 1.0 m) but never imported into `vio.py` — a NameError
that fires only when vision fails badly enough to reach dead
reckoning (hit during the attitude-mode sensitivity run on
s20_fast). Fixed by importing both names; the dead-reckoning path
was effectively untested/broken in round 2.

### FPS profile — machine limit, not code

Measured 2026-11-23 (`/tmp/vio_profile.py`, s1_basic, 60 frames,
shared 1.5-CPU VM): detector alone 96.5 ms/frame (10.4 fps
ceiling with the detector alone); VIO alone (yaw1dof) 81.0 ms;
`process_frame` with GT poses 101.8 ms; full pipeline with VIO
159.5 ms/frame (6.3 fps). The VIO and the detector run
sequentially, so their costs add; the pipeline's own overhead
(depth + bridge + fusion) is ~5 ms/frame. The 10 fps target is
not reachable on THIS VM with CPU-only YOLO + VIO — the detector
alone is 96.5 ms. A torch thread-count probe (evidence only,
shipped threading unchanged) found 2 threads faster than 4
(81.8 vs 96.5 ms/frame; a later 4-thread run measured 216.5 ms,
so the numbers are noisy on a shared VM). No code fix is claimed
for a machine limit, and the marker path was not slowed to buy
fps. On a real laptop the detector and VIO should each be
30-50 ms and the pipeline should fit under 100 ms — unmeasured.

### Ground test (smooth flight, new default)

`tests/vio_ground_test.py` with the yaw1dof default: 102/105
frames valid (97%), drift mean 29.1 cm, max 61.9 cm, final
39.9 cm; approach 7.0 cm, pass 14.7 cm, turn 27.8 cm, hover
47.7 cm; VIO step 58-64 ms/frame. Round 2 lost tracking mid-turn
(55/105 valid, drift 9.2 cm over the valid frames only) — round 3
tracks the whole flight, but the hover drift (47.7 cm mean) is
the remaining weakness: the turn's yaw bias persists as a
position drift while hovering. The ground-test drift is not the
60 cm marker budget, but the hover phase is where the real test
will live or die.

## Extended tests — 2026-11-23 19:59:55 (UTC)

New rooms/places, multi-drone (4 and 5 drones), and adversarial scenarios on top of the original 7. Thresholds unchanged from SPEC.md (detection >= 90%, mean joint error <= 60 cm, mean |distance error| <= 60 cm, skeleton scale error <= 25%, false markers = 0, marker-after-loss = 0, fused error <= best single-drone error).

Scenarios run: s1_basic, s2_offcentre, s3_close, s4_dim, s5_tracking_loss, s6_no_person, s7_two_drones, s8_wall_close, s9_wall_far, s10_phone_offcentre, s11_large_room_far, s12_bright, s24_empty_new_room, s13_dim_noise, s14_glare, s15_occluded, s16_sideways, s17_edge, s18_moving, s19_two_people, s20_fast, s21_loss_recovery, s22_clutter, s23_blank_door, s25_four_drones, s26_five_drones, s27_four_drones_partial

## Environment (extended run)

- Python 3.12.3
- opencv 5.1.0, numpy 2.5.4, ultralytics 8.4.216
- Weights: `yolov8n-pose.pt` (in `wall-marker/`), detector conf 0.5, imgsz 640, CPU
- Total runtime: 724.0 s (12.1 min), one inference process
- Person billboard: largest person detection from ultralytics' bundled `bus.jpg`, cut out with grabCut (mask eroded 2x), pasted at the TRUE projected position. Two-person scenarios paste the same cutout twice at different positions (identical twins — documented).
- Room model: SPEC.md default (wall z = 3.0 m, doorway x in [-0.6, 0.6] m, phone fixed at (0, 1.5, 0)) unless a scenario overrides it (see its description). Projection `v = cy - fy*y/z`.

## Extended results

| Scenario | Detection rate | Mean joint err | Worst joint err | Mean |dist err| | Worst dist err | Scale err | False markers | Bridge-invalid markers | Result |
|---|---|---|---|---|---|---|---|---|---|
| s1_basic | 100% (36/36) | 36.1 cm | 64.2 cm | 8.7 cm | 9.9 cm | 13.1% | 0 | 0 | PASS |
| s2_offcentre | 100% (34/34) | 35.5 cm | 63.6 cm | 0.9 cm | 2.4 cm | 14.1% | 0 | 0 | PASS |
| s3_close | 100% (38/38) | 37.6 cm | 62.2 cm | 13.4 cm | 15.2 cm | 11.3% | 0 | 0 | PASS |
| s4_dim | 100% (36/36) | 35.6 cm | 61.4 cm | 8.0 cm | 10.9 cm | 13.7% | 0 | 0 | PASS |
| s5_tracking_loss | 100% (36/36) | 35.7 cm | 60.5 cm | 9.1 cm | 9.9 cm | 13.5% | 0 | 0 | PASS |
| s6_no_person | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | PASS |
| s7_two_drones | 100% (36/36) | 34.1 cm | 65.2 cm | 6.6 cm | 7.4 cm | 13.7% | 0 | 0 | PASS |
| s8_wall_close | 100% (38/38) | 33.9 cm | 70.2 cm | 12.1 cm | 12.3 cm | 14.4% | 0 | 0 | PASS |
| s9_wall_far | 100% (37/37) | 37.8 cm | 60.9 cm | 10.4 cm | 15.0 cm | 10.8% | 0 | 0 | PASS |
| s10_phone_offcentre | 100% (30/30) | 36.6 cm | 67.5 cm | 10.0 cm | 10.2 cm | 12.1% | 0 | 0 | PASS |
| s11_large_room_far | 100% (39/39) | 31.9 cm | 71.4 cm | 6.7 cm | 6.9 cm | 15.0% | 0 | 0 | PASS |
| s12_bright | 100% (36/36) | 36.2 cm | 61.1 cm | 8.5 cm | 8.9 cm | 13.4% | 0 | 0 | PASS |
| s24_empty_new_room | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | PASS |
| s13_dim_noise | 100% (36/36) | 36.8 cm | 64.3 cm | 8.9 cm | 9.8 cm | 17.2% | 0 | 0 | PASS |
| s14_glare | 100% (36/36) | 36.4 cm | 64.1 cm | 8.4 cm | 9.0 cm | 13.5% | 0 | 0 | PASS |
| s15_occluded | 100% (37/37) | 42.1 cm | 79.5 cm | 16.3 cm | 25.8 cm | 10.1% | 0 | 0 | PASS |
| s16_sideways | 100% (36/36) | 33.3 cm | 68.6 cm | 8.2 cm | 12.1 cm | 18.2% | 0 | 0 | PASS |
| s17_edge | 100% (42/42) | 32.6 cm | 74.8 cm | 4.5 cm | 7.3 cm | 12.4% | 0 | 0 | PASS |
| s18_moving | 100% (37/37) | 48.1 cm | 122.9 cm | 12.6 cm | 22.6 cm | 17.5% | 0 | 0 | PASS |
| s19_two_people | 100% (37/37) | 36.1 cm | 73.0 cm | 8.6 cm | 15.9 cm | 12.9% | 0 | 0 | PASS |
| s20_fast | 100% (33/33) | 35.2 cm | 65.7 cm | 7.4 cm | 7.6 cm | 13.6% | 0 | 0 | PASS |
| s21_loss_recovery | 100% (36/36) | 38.2 cm | 64.8 cm | 8.3 cm | 9.0 cm | 13.4% | 0 | 0 | PASS |
| s22_clutter | 100% (36/36) | 36.4 cm | 64.1 cm | 8.4 cm | 9.0 cm | 13.5% | 0 | 0 | PASS |
| s23_blank_door | 100% (38/38) | 36.0 cm | 66.3 cm | 11.8 cm | 12.2 cm | 13.7% | 0 | 0 | PASS |
| s25_four_drones | 100% (38/38) | 32.7 cm | 69.5 cm | 7.5 cm | 9.0 cm | 14.5% | 0 | 0 | PASS |
| s26_five_drones | 100% (38/38) | 30.9 cm | 63.9 cm | 6.1 cm | 6.6 cm | 14.5% | 0 | 0 | PASS |
| s27_four_drones_partial | 100% (36/36) | 34.1 cm | 65.2 cm | 6.6 cm | 7.4 cm | 13.7% | 0 | 0 | PASS |

Thresholds (SPEC.md, not moved): detection rate >= 90%, mean joint error <= 60 cm, mean |distance error| <= 60 cm, skeleton scale error <= 25%, false markers = 0 (empty rooms), marker-after-loss = 0 (s5), marker-during-loss = 0 (s21), no merged markers (s19), marker-while-bridge-invalid = 0 (all scenarios), fused error <= best single-drone error (multi-drone). Worst-frame errors are reported, not thresholded — they are the adversarial signal, not the pass criterion.

### Regression: original 7 vs the recorded 2026-11-23 run

The original 7 scenarios were re-run in this extended run. Rendering and inference are deterministic, so the numbers must reproduce the recorded run (tolerance 0.5 cm on means, identical detection counts and PASS/FAIL). Any drift means the refactor changed the original rendering — that would be a bug, not noise.

| Scenario | Detection (recorded / re-run) | Mean joint (recorded / re-run) | Mean |dist| (recorded / re-run) | Scale (recorded / re-run) | Match |
|---|---|---|---|---|---|
| s1_basic | 100% / 100% | 36.1 cm / 36.1 cm | 8.7 cm / 8.7 cm | 13.1% / 13.1% | MATCH |
| s2_offcentre | 100% / 100% | 35.5 cm / 35.5 cm | 0.9 cm / 0.9 cm | 14.1% / 14.1% | MATCH |
| s3_close | 100% / 100% | 37.6 cm / 37.6 cm | 13.4 cm / 13.4 cm | 11.3% / 11.3% | MATCH |
| s4_dim | 100% / 100% | 35.6 cm / 35.6 cm | 8.0 cm / 8.0 cm | 13.7% / 13.7% | MATCH |
| s5_tracking_loss | 100% / 100% | 35.7 cm / 35.7 cm | 9.1 cm / 9.1 cm | 13.5% / 13.5% | MATCH |
| s6_no_person | n/a / n/a | n/a / n/a | n/a / n/a | n/a / n/a | MATCH |
| s7_two_drones | 100% / 100% | 34.1 cm / 34.1 cm | 6.6 cm / 6.6 cm | 13.7% / 13.7% | MATCH |

### s1_basic — person 4.0 m from phone, centred, light 1.0

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 576; distance-error samples: 34.
- Worst single-frame joint error: 64.2 cm; worst single-frame distance error: 9.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -0 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 89.1 ms/frame mean (11.2 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s2_offcentre — person 5.5 m from phone, off to the side near the doorway

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 34; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 26.
- Markers emitted: 36 (4 stale); joint-error samples: 544; distance-error samples: 32.
- Worst single-frame joint error: 63.6 cm; worst single-frame distance error: 2.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 2, 'triangulation': 32}; first triangulation frame: 22.
- Measured depth bias (estimated - true foot depth): bbox-height +1 cm (n=2); triangulation -3 cm (n=32).
- Mean signed distance error (marker - GT): -1 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 100.0 ms/frame mean (10.0 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s3_close — person just behind the wall (z = 3.3 m)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 36 (0 stale); joint-error samples: 612; distance-error samples: 36.
- Worst single-frame joint error: 62.2 cm; worst single-frame distance error: 15.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -7 cm (n=1); triangulation +2 cm (n=37).
- Mean signed distance error (marker - GT): -13 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 90.4 ms/frame mean (11.1 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s4_dim — light 0.5, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 61.4 cm; worst single-frame distance error: 10.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +2 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 112.5 ms/frame mean (8.9 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s5_tracking_loss — tracking_valid True until frame 35, False after; person stays visible

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 10 (0 stale); joint-error samples: 168; distance-error samples: 10.
- Worst single-frame joint error: 60.5 cm; worst single-frame distance error: 9.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -0 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 88.8 ms/frame mean (11.3 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_after_loss`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s6_no_person — room B empty, same flight path; zero markers, ever

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 0; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 60.
- Markers emitted: 0 (0 stale); joint-error samples: 0; distance-error samples: 0.
- Worst single-frame joint error: n/a; worst single-frame distance error: n/a (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- False markers (empty room): 0.
- Depth methods: {}; first triangulation frame: None.
- Measured depth bias (estimated - true foot depth): bbox-height n/a; triangulation n/a.
- Mean signed distance error (marker - GT): n/a (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 97.9 ms/frame mean (10.2 fps).
- check `false_markers`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s7_two_drones — two drones, different hover points, both see the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Worst single-frame joint error: 65.2 cm; worst single-frame distance error: 7.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -4 cm (n=5); triangulation -2 cm (n=23).
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 205.1 ms/frame mean (4.9 fps).
- Fused mean joint error, hover frames 25-59: 34.1 cm; single-drone mean joint error over the same frames: drone1 36.4 cm, drone2 37.6 cm; best single: 36.4 cm. Single-drone detection (hover frames): {'drone1': {'detected': 35, 'in_view': 35}, 'drone2': {'detected': 27, 'in_view': 27}}.
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- check `fused_le_best_single`: PASS
- **Result: PASS**

### s8_wall_close — wall z = 2.0 m, doorway off-centre (left), person 2.5 m (z) from phone

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 36 (0 stale); joint-error samples: 612; distance-error samples: 36.
- Worst single-frame joint error: 70.2 cm; worst single-frame distance error: 12.3 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -4 cm (n=1); triangulation -0 cm (n=37).
- Mean signed distance error (marker - GT): -12 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 88.7 ms/frame mean (11.3 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s9_wall_far — wall z = 4.0 m, doorway off-centre (right), person 5.5 m (z) from phone

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 37; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 23.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Worst single-frame joint error: 60.9 cm; worst single-frame distance error: 15.0 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 36}; first triangulation frame: 24.
- Measured depth bias (estimated - true foot depth): bbox-height -3 cm (n=1); triangulation +1 cm (n=36).
- Mean signed distance error (marker - GT): -10 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 97.3 ms/frame mean (10.3 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s10_phone_offcentre — phone NOT at the origin (1.2, 1.5, 0.5); person 4.5 m (z) away, other side

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 30; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 30.
- Markers emitted: 32 (4 stale); joint-error samples: 476; distance-error samples: 28.
- Worst single-frame joint error: 67.5 cm; worst single-frame distance error: 10.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 25}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -2 cm (n=5); triangulation -3 cm (n=25).
- Mean signed distance error (marker - GT): -10 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 98.1 ms/frame mean (10.2 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s11_large_room_far — larger room B (x +-4 m, z to 12 m), person 7.0 m (z) from phone

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 39; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 21.
- Markers emitted: 37 (0 stale); joint-error samples: 592; distance-error samples: 37.
- Worst single-frame joint error: 71.4 cm; worst single-frame distance error: 6.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 38}; first triangulation frame: 20.
- Measured depth bias (estimated - true foot depth): bbox-height +21 cm (n=1); triangulation +1 cm (n=38).
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 87.6 ms/frame mean (11.4 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s12_bright — brightness 1.2x, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 61.1 cm; worst single-frame distance error: 8.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -2 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 97.3 ms/frame mean (10.3 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s24_empty_new_room — empty room B in a NEW geometry (wall 2.5 m, doorway off-centre right); zero markers, ever

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 0; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 60.
- Markers emitted: 0 (0 stale); joint-error samples: 0; distance-error samples: 0.
- Worst single-frame joint error: n/a; worst single-frame distance error: n/a (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- False markers (empty room): 0.
- Depth methods: {}; first triangulation frame: None.
- Measured depth bias (estimated - true foot depth): bbox-height n/a; triangulation n/a.
- Mean signed distance error (marker - GT): n/a (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 84.9 ms/frame mean (11.8 fps).
- check `false_markers`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s13_dim_noise — light 0.4x with REAL Gaussian noise (sigma 12), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 556; distance-error samples: 34.
- Worst single-frame joint error: 64.3 cm; worst single-frame distance error: 9.8 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +0 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 72.0 ms/frame mean (13.9 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s14_glare — bright window directly behind the person + bloom (backlit), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 64.1 cm; worst single-frame distance error: 9.0 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -1 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 79.2 ms/frame mean (12.6 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s15_occluded — person partly hidden behind a wardrobe (~40% of the body covered), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 37; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 23.
- Markers emitted: 35 (0 stale); joint-error samples: 548; distance-error samples: 35.
- Worst single-frame joint error: 79.5 cm; worst single-frame distance error: 25.8 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 21, 'triangulation': 16}; first triangulation frame: 44.
- Measured depth bias (estimated - true foot depth): bbox-height +7 cm (n=21); triangulation +27 cm (n=16).
- Mean signed distance error (marker - GT): -16 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 92.4 ms/frame mean (10.8 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s16_sideways — person turned sideways (facing +X, 90 deg to the viewing direction), narrowed billboard, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 544; distance-error samples: 34.
- Worst single-frame joint error: 68.6 cm; worst single-frame distance error: 12.1 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +4 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 98.2 ms/frame mean (10.2 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s17_edge — person at the very edge of the drone frame (billboard right edge ~30 px from the frame edge); drone looks at a fixed point, not at the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 42; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 18.
- Markers emitted: 40 (0 stale); joint-error samples: 680; distance-error samples: 40.
- Worst single-frame joint error: 74.8 cm; worst single-frame distance error: 7.3 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 13, 'triangulation': 29}; first triangulation frame: 20.
- Measured depth bias (estimated - true foot depth): bbox-height +11 cm (n=13); triangulation +5 cm (n=29).
- Mean signed distance error (marker - GT): +5 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 88.2 ms/frame mean (11.3 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s18_moving — person walking slowly (0.25 m/s across, 0.12 m/s toward the wall) — triangulation assumes a static scene

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 37; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 23.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Worst single-frame joint error: 122.9 cm; worst single-frame distance error: 22.6 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 23, 'triangulation': 14}; first triangulation frame: 24.
- Measured depth bias (estimated - true foot depth): bbox-height +2 cm (n=23); triangulation -5 cm (n=14).
- Mean signed distance error (marker - GT): +13 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 87.2 ms/frame mean (11.5 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s19_two_people — TWO people in room B (~1.9 m apart); the system must mark a real person, never a merged midpoint marker

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 37; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 23.
- Markers emitted: 34 (1 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 73.0 cm; worst single-frame distance error: 15.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Merged markers (marker > 1.0 m from BOTH people): 0 (must be 0).
- Depth methods: {'bbox_height': 3, 'triangulation': 34}; first triangulation frame: 25.
- Measured depth bias (estimated - true foot depth): bbox-height -32 cm (n=3); triangulation -25 cm (n=34).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 100.8 ms/frame mean (9.9 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `no_merged_markers`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s20_fast — fast drone: fast approach + fast wide hover (amp 0.9 m, period 1.2 s)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 33; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 27.
- Markers emitted: 35 (4 stale); joint-error samples: 527; distance-error samples: 31.
- Worst single-frame joint error: 65.7 cm; worst single-frame distance error: 7.6 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 32}; first triangulation frame: 24.
- Measured depth bias (estimated - true foot depth): bbox-height -1 cm (n=1); triangulation -4 cm (n=32).
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 102.6 ms/frame mean (9.7 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s21_loss_recovery — tracking loss frames 25-40, recovery from frame 41; marker must vanish on loss and only return after the bridge is re-established

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 18 (0 stale); joint-error samples: 300; distance-error samples: 18.
- Worst single-frame joint error: 64.8 cm; worst single-frame distance error: 9.0 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Markers during the loss window: 0 (must be 0); first marker after recovery: frame 42.
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -1 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 100.9 ms/frame mean (9.9 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_during_loss`: PASS
- check `marker_resumes_after_recovery`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s22_clutter — cluttered room B (furniture boxes + extra posters), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 64.1 cm; worst single-frame distance error: 9.0 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -1 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 102.4 ms/frame mean (9.8 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s23_blank_door — featureless blank wall at the doorway (no posters on the shared wall), wall 2.5 m, person 3.6 m (z)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 36 (0 stale); joint-error samples: 612; distance-error samples: 36.
- Worst single-frame joint error: 66.3 cm; worst single-frame distance error: 12.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -6 cm (n=1); triangulation +0 cm (n=37).
- Mean signed distance error (marker - GT): -12 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 95.9 ms/frame mean (10.4 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s25_four_drones — 4 drones, different hover points, all see the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 37 (0 stale); joint-error samples: 629; distance-error samples: 37.
- Worst single-frame joint error: 69.5 cm; worst single-frame distance error: 9.0 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -13 cm (n=1); triangulation -0 cm (n=37).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Fused mean joint error, hover frames 25-59: 32.8 cm; single-drone mean joint error over the same frames: drone1 36.4 cm, drone2 37.6 cm, drone3 34.4 cm, drone4 33.1 cm; best single: 33.1 cm. Single-drone detection (hover frames): {'drone1': {'detected': 35, 'in_view': 35}, 'drone2': {'detected': 27, 'in_view': 27}, 'drone3': {'detected': 35, 'in_view': 35}, 'drone4': {'detected': 35, 'in_view': 35}}.
- Per-drone detection in the fused run (all frames / hover frames): {'drone1': {'in_view': 36, 'detected': 36, 'in_view_hover': 35, 'detected_hover': 35}, 'drone2': {'in_view': 28, 'detected': 28, 'in_view_hover': 27, 'detected_hover': 27}, 'drone3': {'in_view': 37, 'detected': 37, 'in_view_hover': 35, 'detected_hover': 35}, 'drone4': {'in_view': 38, 'detected': 38, 'in_view_hover': 35, 'detected_hover': 35}}.
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- check `fused_le_best_single`: PASS
- **Result: PASS**

### s26_five_drones — 5 drones, different hover points, all see the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 37 (0 stale); joint-error samples: 629; distance-error samples: 37.
- Worst single-frame joint error: 63.9 cm; worst single-frame distance error: 6.6 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -2 cm (n=1); triangulation -1 cm (n=37).
- Mean signed distance error (marker - GT): -6 cm (negative = marker distance shorter than the true phone->person distance).
- Fused mean joint error, hover frames 25-59: 31.0 cm; single-drone mean joint error over the same frames: drone1 36.4 cm, drone2 37.6 cm, drone3 34.4 cm, drone4 33.1 cm, drone5 34.5 cm; best single: 33.1 cm. Single-drone detection (hover frames): {'drone1': {'detected': 35, 'in_view': 35}, 'drone2': {'detected': 27, 'in_view': 27}, 'drone3': {'detected': 35, 'in_view': 35}, 'drone4': {'detected': 35, 'in_view': 35}, 'drone5': {'detected': 35, 'in_view': 35}}.
- Per-drone detection in the fused run (all frames / hover frames): {'drone1': {'in_view': 36, 'detected': 36, 'in_view_hover': 35, 'detected_hover': 35}, 'drone2': {'in_view': 28, 'detected': 28, 'in_view_hover': 27, 'detected_hover': 27}, 'drone3': {'in_view': 37, 'detected': 37, 'in_view_hover': 35, 'detected_hover': 35}, 'drone4': {'in_view': 38, 'detected': 38, 'in_view_hover': 35, 'detected_hover': 35}, 'drone5': {'in_view': 38, 'detected': 38, 'in_view_hover': 35, 'detected_hover': 35}}.
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- check `fused_le_best_single`: PASS
- **Result: PASS**

### s27_four_drones_partial — 4 drones, 2 lose sight of the person (they look at the back corners) — fusion must still work from the remaining 2

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Worst single-frame joint error: 65.2 cm; worst single-frame distance error: 7.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height n/a; triangulation n/a.
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Fused mean joint error, hover frames 25-59: 34.1 cm; single-drone mean joint error over the same frames: drone1 36.4 cm, drone2 37.6 cm, drone3 n/a, drone4 n/a; best single: 36.4 cm. Single-drone detection (hover frames): {'drone1': {'detected': 35, 'in_view': 35}, 'drone2': {'detected': 27, 'in_view': 27}, 'drone3': {'detected': 0, 'in_view': 0}, 'drone4': {'detected': 0, 'in_view': 0}}.
- Per-drone detection in the fused run (all frames / hover frames): {'drone1': {'in_view': 36, 'detected': 36, 'in_view_hover': 35, 'detected_hover': 35}, 'drone2': {'in_view': 28, 'detected': 28, 'in_view_hover': 27, 'detected_hover': 27}, 'drone3': {'in_view': 0, 'detected': 0, 'in_view_hover': 0, 'detected_hover': 0}, 'drone4': {'in_view': 0, 'detected': 0, 'in_view_hover': 0, 'detected_hover': 0}}.
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- check `fused_le_best_single`: PASS
- **Result: PASS**

## No-drone-video test (extended run)

- Server started on 127.0.0.1:18765 (real `server.main` handler).
- `/viewer`: 1 `<video>` tag(s); getUserMedia present: True; video-endpoint hints found: none.
- `/api/markers`: 7 SSE events read in 3 s; all parse as JSON with type marker/none: True; content-type text/event-stream, no video content-type anywhere: True.
- Unknown routes probed (12): all 404: True ({'/video': 404, '/stream': 404, '/frames': 404, '/api/video': 404, '/api/frames': 404, '/drone': 404, '/api/drone': 404, '/snapshot': 404, '/mjpeg': 404, '/api/markers/1': 404, '/admin': 404, '/api/status/extra': 404}).
- check `root_redirects_to_viewer`: PASS
- check `viewer_200_html`: PASS
- check `exactly_one_video_tag`: PASS
- check `no_img_tag`: PASS
- check `no_fetch_or_xhr`: PASS
- check `no_drone_stream_reference`: PASS
- check `uses_getusermedia`: PASS
- check `no_video_content_type`: PASS
- check `status_json_shape`: PASS
- check `status_new_fields_null_before_first_frame`: PASS
- check `operator_200_html`: PASS
- check `operator_no_video_tag`: PASS
- check `operator_no_img_tag`: PASS
- check `operator_no_video_content_type`: PASS
- check `operator_has_three_blocks`: PASS
- check `operator_facts_dash_before_first_frame`: PASS
- check `sse_content_type`: PASS
- check `sse_no_video_content_type`: PASS
- check `sse_events_received`: PASS
- check `sse_all_json`: PASS
- check `sse_types_json_only`: PASS
- check `unknown_routes_all_404`: PASS
- **Result: PASS**

## UI server part — real-only tests (extended run)

- **wording**: PASS
  - check `viewer_no_banned_words`: PASS
  - check `operator_no_banned_words`: PASS
  - check `operator_no_video_tag`: PASS
  - check `operator_no_img_tag`: PASS
- **product-server-refuses**: PASS
  - check `live_exits_nonzero`: PASS
  - check `live_serves_nothing`: PASS
  - check `virtual_exits_nonzero`: PASS
  - check `virtual_serves_nothing`: PASS
  - check `live_error_names_djitellopy_or_tello`: PASS
  - check `virtual_error_points_at_test_entry`: PASS
  - product server --mode live: exit 2
  - product server --mode virtual: exit 2
- **sse-none-while-bridge-invalid**: PASS
  - check `bridge_invalid_seen`: PASS
  - check `sse_content_type`: PASS
  - check `sse_events_received`: PASS
  - check `sse_all_json`: PASS
  - check `sse_only_none_while_bridge_invalid`: PASS
  - SSE events read: 4; types seen: ['none']
- **operator-dash-before-first-frame**: PASS
  - check `operator_200`: PASS
  - check `facts_fDrones_is_dash`: PASS
  - check `facts_fTracks_is_dash`: PASS
  - check `facts_fMarkers_is_dash`: PASS
  - check `facts_fUptime_is_dash`: PASS
  - check `facts_fLastFrame_is_dash`: PASS
  - check `facts_fFrames10_is_dash`: PASS
  - check `status_last_frame_age_null`: PASS
  - check `status_frames_last_10s_null`: PASS
  - check `status_tracks_zero_is_real_not_shown`: PASS
- **viewer-zero-skeletons**: PASS
  - check `chrome_available`: PASS
  - check `driver_returned_json`: PASS
  - check `driver_ok`: PASS
  - check `stream_was_live`: PASS
  - check `no_markers_held`: PASS
  - check `no_distance_pills`: PASS
  - check `zero_skeleton_pixels`: PASS
  - check `no_console_errors`: PASS
  - viewer canvas 800x600: 0 non-transparent pixels after 60 s of none events; markers held: 0; distance pills: 0; last SSE event 489 ms before the read; console errors: none.
- **Result: PASS**

## Found and fixed (extended suite)

- Bridge.reacquire() implemented (2026-11-23): tracking loss used to invalidate the bridge permanently (reacquire() raised NotImplementedError), so a loss+recovery scenario could never pass. A fresh valid pose after a loss now starts a NEW chain (uncertainty clock restarts at 0.10 m) via Bridge.reacquire(), wired into Bridge.update(). The virtual harness proves the state-machine logic (marker hidden during loss, returns only after re-establishment + a fresh detection); real relocalisation is still unproven and must gate reacquire in the real system. Required by s21_loss_recovery.
- Triangulation cross-check (2026-11-23): estimate_person_depth now compares the triangulated depth against the per-frame bbox-height depth and trusts the bbox-height estimate when they disagree by >25% (relative). Measured: static scenes agree within ~14% max (mean 11-13% across s1/s2/s3/s4/s7/s11/s12/s25), while a moving person (s18: up to 59%), an occluded person (s15: up to 93%) and mixed two-person rays (s19: up to 100%) disagree far beyond it. The triangulation residual does NOT catch this — a moving person's rays can genuinely nearly meet at a wrong point (measured residual <= 0.15 while the depth was 1.3 m off) — so the cross-check is the signal, not the residual. The bbox-height estimate is per-frame (no static-scene assumption) with a measured +20-34 cm bias, so it is the safe fallback.
- Per-person depth histories in Pipeline (2026-11-23): the depth history must contain views of ONE person only. The max-conf detection can alternate between two people, and mixing their rays triangulated a ghost point between them (measured in s19: a phantom track at (0, 7.4 m) and triangulated depths of 0.01-0.22 m). Each person now keeps a separate history, associated by a rough foot point within 1.0 m of that person's last foot (the 1.0 m gate matches the fusion's association distance). A single person keeps exactly one history, so single-person behaviour is unchanged. A moving person moves ~0.03 m/frame, so the association never fires on motion alone.
- Ankle ray for partial detections (2026-11-23): when the detector misses >= 3 joints, the person is partly occluded and the bbox is the visible part's box — its bottom-centre is laterally offset and its height is short, so both depth methods share a common-mode bias the cross-check cannot catch (measured in s15: triangulation 3.11 m and bbox-height 2.86 m agreed, yet the true depth was 2.51 m). The triangulation and foot point then use the visible-ankle ray (partial_foot_uv), a stable world point that does not move with the occlusion; the ankle-derived foot is projected straight down to y = 0 (ground contact). Measured separation: full-person frames miss 0-1 joints, occluded frames (s15) miss 3-9, so the gate never fires on a full person and the original 7 scenarios are untouched.

## Honest limits — extended scenarios (what this still does NOT prove)

- Virtual poses are exact: the bridge is fed the true drone pose, so no SLAM drift is simulated. The bridge uncertainty (0.10 + 0.02 s/s, capped at 1.0 m) is a model, not a measurement. The research report's end-to-end budget (±20-60 cm at 3-6 m with good depth, ±0.5-2 m monocular/Tello) still stands for the real system.
- s23_blank_door stresses a featureless wall at the doorway, but the virtual bridge does not do visual feature tracking at all (it is fed true poses), so the real VIO bridge stress of a blank wall is NOT measured here — the scenario only proves detection/rendering still works with a blank wall. A real VIO bridge could still lose tracking at a blank doorway.
- s21_loss_recovery proves the state-machine logic (marker hidden during loss, returns only after re-establishment + a fresh detection). The virtual reacquire stands in for a real relocalisation step with the true pose; a real relocalisation (re-observing a mapped area / doorway loop closure) is not built or tested here.
- s16_sideways is an approximation: a sideways body is rendered as a narrowed frontal billboard, not a true profile (one shoulder hidden, different arm pose). The measured joint errors are a lower bound on real sideways difficulty.
- s18_moving: the person walks slowly (0.28 m/s) and triangulation assumes a static scene. See the scenario section for the measured numbers; the residual is the honest signal that the static assumption broke.
- s15_occluded: the occluder is a flat wardrobe plane drawn after the billboard; a real occluder (doorframe edge, furniture) would also change the person's visible shape in 3D. Detection of a half-hidden person is measured, not assumed.
- s13_dim_noise uses synthetic Gaussian noise (sigma 12 on 0-255 after a 0.4x brightness factor) — real low-light noise is signal-dependent and coloured; this is a harder test than brightness alone, but not a real night shot.
- s14_glare uses a bright window quad + a Gaussian bloom (radius 200 px, strength 0.35) — a synthetic glare model, not a real sunlit window with lens flare.
- Two-person scenarios paste the SAME cutout twice (identical twins). Real second people differ in height, clothing and pose; the merge check (no midpoint marker) is the point of the scenario, not the realism of the second person.
- One static room pair per scenario, no occlusion beyond s15, no motion blur, no network latency, no live-phone rendering check (the viewer projection is verified by code inspection + the GT phone projection in the scenarios, not by a live phone).
- Detection at 640 px on CPU.

## Method used per frame (extended scenarios)

Per scenario, frames using triangulation vs bbox-height are counted in the table above. In every person scenario the pipeline switches to triangulation as soon as the drone's motion baseline reaches 0.3 m (a few frames after the person first appears) and stays there; bbox-height is used only for the first 1-3 detections. Exact per-frame method sequences:

- s1_basic: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s2_offcentre: f0-19:-, f20-21:bbox_height, f22-28:triangulation, f29-34:-, f35-59:triangulation
- s3_close: f0-21:-, f22-22:bbox_height, f23-59:triangulation
- s4_dim: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s5_tracking_loss: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s6_no_person: f0-59:-
- s7_two_drones: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s8_wall_close: f0-21:-, f22-22:bbox_height, f23-59:triangulation
- s9_wall_far: f0-22:-, f23-23:bbox_height, f24-59:triangulation
- s10_phone_offcentre: f0-23:-, f24-28:bbox_height, f29-40:triangulation, f41-46:-, f47-59:triangulation
- s11_large_room_far: f0-18:-, f19-19:bbox_height, f20-20:triangulation, f21-22:-, f23-59:triangulation
- s12_bright: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s24_empty_new_room: f0-59:-
- s13_dim_noise: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s14_glare: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s15_occluded: f0-22:-, f23-43:bbox_height, f44-59:triangulation
- s16_sideways: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s17_edge: f0-15:-, f16-17:bbox_height, f18-19:-, f20-48:triangulation, f49-59:bbox_height
- s18_moving: f0-22:-, f23-23:bbox_height, f24-37:triangulation, f38-59:bbox_height
- s19_two_people: f0-22:-, f23-24:bbox_height, f25-26:triangulation, f27-27:bbox_height, f28-59:triangulation
- s20_fast: f0-19:-, f20-20:bbox_height, f21-23:-, f24-48:triangulation, f49-51:-, f52-58:triangulation, f59-59:-
- s21_loss_recovery: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s22_clutter: f0-23:-, f24-28:bbox_height, f29-59:triangulation
- s23_blank_door: f0-21:-, f22-22:bbox_height, f23-59:triangulation
- s25_four_drones: f0-21:-, f22-22:bbox_height, f23-59:triangulation
- s26_five_drones: f0-21:-, f22-22:bbox_height, f23-59:triangulation
- s27_four_drones_partial: f0-23:-, f24-28:bbox_height, f29-59:triangulation

## VIO tests

The same Pipeline, but the bridge is fed VIO-estimated poses (``server/vio.py``) instead of ground truth. The VIO is initialised with the first frame's GT pose as the known start pose (the drone takes off beside the phone); from then on only the video frames drive it. The GT-pose runs above are untouched — the VIO results are a separate table so the regression comparison still reads the GT runs.

VIO pose drift is reported against the 60 cm joint-error budget (SPEC.md) — there is no separate VIO threshold, and none was invented. Frame rate is the VIO step alone and the full pipeline with VIO (VIO + detector + depth + bridge + fusion) on CPU; the 10 fps harness needs < 100 ms/frame.

| Scenario (VIO) | VIO valid | Drift mean | Drift max | Yaw err mean | Yaw err max | Yaw err, marker frames | PnP inliers mean/min | VIO fps | Pipeline fps | Detection | Mean joint err | Mean |dist err| | Markers | Bridge-invalid markers | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_basic | 56/60 | 6.7 cm | 21.1 cm | 1.0 deg | 7.6 deg | 3.3 deg | 64/0 | 8.4 | 4.9 | 100% (12/12) | 40.5 cm | 6.0 cm | 10 | 0 | PASS |
| s23_blank_door | 56/60 | 5.4 cm | 14.7 cm | 0.9 deg | 3.2 deg | 1.1 deg | 57/0 | 12.3 | 6.7 | 100% (13/13) | 32.6 cm | 5.7 cm | 11 | 0 | PASS |
| s20_fast | 56/60 | 8.3 cm | 71.5 cm | 0.7 deg | 7.2 deg | 2.7 deg | 30/0 | 13.5 | 6.7 | 100% (10/10) | 36.1 cm | 7.5 cm | 11 | 0 | PASS |
| s4_dim | 56/60 | 3.8 cm | 15.0 cm | 0.7 deg | 5.1 deg | 2.2 deg | 55/0 | 12.9 | 6.5 | 100% (12/12) | 36.3 cm | 6.8 cm | 10 | 0 | PASS |

### VIO fix round 3 — constrained PnP (round 2 vs round 3)

Round 2 (2026-11-23) left joint error over the 60 cm budget in s1_basic (71.8 cm) and s23_blank_door (81.7 cm) with turn yaw error 1.0-1.2 deg mean, 4.0-4.5 deg max corrupting depth triangulation. Round 3 constrains the PnP itself with the Tello attitude prior — see ``server/vio.py`` ``pnp_mode`` — instead of correcting the yaw after the fact (tried and reverted in round 2: drift improved, joint error worsened to 111 cm, because changing R without re-optimising C breaks consistency).

| Scenario | Joint err, round 2 / round 3 | |Dist err|, round 2 / round 3 | Drift mean, round 2 / round 3 | Markers, round 2 / round 3 |
|---|---|---|---|---|
| s1_basic | 71.8 cm / 40.5 cm | 35.9 cm / 6.0 cm | 9.8 cm / 6.7 cm | 10 / 10 |
| s23_blank_door | 81.7 cm / 32.6 cm | 14.0 cm / 5.7 cm | 13.3 cm / 5.4 cm | 11 / 11 |
| s20_fast | 38.6 cm / 36.1 cm | 18.2 cm / 7.5 cm | 7.9 cm / 8.3 cm | 11 / 11 |
| s4_dim | 55.2 cm / 36.3 cm | 25.7 cm / 6.8 cm | 4.7 cm / 3.8 cm | 10 / 10 |

### VIO vs ground-truth poses (same scenarios)

| Scenario | Joint err, GT poses | Joint err, VIO poses | |Dist err|, GT poses | |Dist err|, VIO poses | Markers, GT / VIO |
|---|---|---|---|---|---|
| s1_basic | 36.1 cm | 40.5 cm | 8.7 cm | 6.0 cm | 34 / 10 |
| s23_blank_door | 36.0 cm | 32.6 cm | 11.8 cm | 5.7 cm | 36 / 11 |
| s20_fast | 35.2 cm | 36.1 cm | 7.4 cm | 7.5 cm | 35 / 11 |
| s4_dim | 35.6 cm | 36.3 cm | 8.0 cm | 6.8 cm | 34 / 10 |

### s1_basic (VIO) — person 4.0 m from phone, centred, light 1.0

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 12; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 48.
- Markers emitted: 10 (0 stale); joint-error samples: 167; distance-error samples: 10.
- Worst single-frame joint error: 74.1 cm; worst single-frame distance error: 8.7 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 7}; first triangulation frame: 53.
- Measured depth bias (estimated - true foot depth): bbox-height -2 cm (n=5); triangulation +8 cm (n=7).
- Mean signed distance error (marker - GT): -6 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

- VIO: initialised at frame 4; first loss at frame None; valid 56/60 frames; drift mean 6.7 cm, max 21.1 cm, final 21.1 cm; yaw error mean 1.0 deg, max 7.6 deg, on marker frames 3.3 deg; PnP inliers mean 64, min 0; VIO step 119.0 ms/frame (8.4 fps); full pipeline with VIO 202.5 ms/frame (4.9 fps).

### s23_blank_door (VIO) — featureless blank wall at the doorway (no posters on the shared wall), wall 2.5 m, person 3.6 m (z)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 13; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 47.
- Markers emitted: 11 (0 stale); joint-error samples: 187; distance-error samples: 11.
- Worst single-frame joint error: 69.8 cm; worst single-frame distance error: 12.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 6, 'triangulation': 7}; first triangulation frame: 53.
- Measured depth bias (estimated - true foot depth): bbox-height +5 cm (n=6); triangulation -21 cm (n=7).
- Mean signed distance error (marker - GT): -3 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

- VIO: initialised at frame 4; first loss at frame None; valid 56/60 frames; drift mean 5.4 cm, max 14.7 cm, final 14.7 cm; yaw error mean 0.9 deg, max 3.2 deg, on marker frames 1.1 deg; PnP inliers mean 57, min 0; VIO step 81.3 ms/frame (12.3 fps); full pipeline with VIO 149.5 ms/frame (6.7 fps).

### s20_fast (VIO) — fast drone: fast approach + fast wide hover (amp 0.9 m, period 1.2 s)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 10; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 50.
- Markers emitted: 11 (5 stale); joint-error samples: 134; distance-error samples: 8.
- Worst single-frame joint error: 80.3 cm; worst single-frame distance error: 10.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 6, 'triangulation': 4}; first triangulation frame: 51.
- Measured depth bias (estimated - true foot depth): bbox-height -4 cm (n=6); triangulation -22 cm (n=4).
- Mean signed distance error (marker - GT): +5 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

- VIO: initialised at frame 4; first loss at frame None; valid 56/60 frames; drift mean 8.3 cm, max 71.5 cm, final 64.2 cm; yaw error mean 0.7 deg, max 7.2 deg, on marker frames 2.7 deg; PnP inliers mean 30, min 0; VIO step 74.2 ms/frame (13.5 fps); full pipeline with VIO 149.9 ms/frame (6.7 fps).

### s4_dim (VIO) — light 0.5, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 12; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 48.
- Markers emitted: 10 (0 stale); joint-error samples: 169; distance-error samples: 10.
- Worst single-frame joint error: 67.2 cm; worst single-frame distance error: 8.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 12}; first triangulation frame: None.
- Measured depth bias (estimated - true foot depth): bbox-height +1 cm (n=12); triangulation n/a.
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

- VIO: initialised at frame 4; first loss at frame None; valid 56/60 frames; drift mean 3.8 cm, max 15.0 cm, final 15.0 cm; yaw error mean 0.7 deg, max 5.1 deg, on marker frames 2.2 deg; PnP inliers mean 55, min 0; VIO step 77.3 ms/frame (12.9 fps); full pipeline with VIO 154.5 ms/frame (6.5 fps).

### Honest limits — VIO (what this still does NOT prove)

- The harness trajectories have yaw snaps the VIO cannot track: the look-at logic snaps the camera -31 deg at the doorway (frame 13->14) and +101 deg at the turn to face the person (frame 19->20) in a single 0.1 s frame. Measured 2026-11-23: LK carried only 36 of 86 points across the doorway (19 of them hallucinated onto points that had left the frame); ORB descriptor matching found 0 geometrically correct matches across the doorway snap on this repetitive checker floor, and a photometric yaw search found the wrong yaw (-54 deg vs -31 deg GT; +33 deg vs +101 deg GT) because near-wall parallax dominates the photometric error. A real Tello flight yaws at ~30-60 deg/s (1-2 deg/frame at 30 fps), not 300-1000 deg/s — the snaps are a harness artifact, and the real-run procedure is to fly and turn smoothly (see README.md).
- The smooth-flight ground test (``tests/vio_ground_test.py``) is the end-to-end VIO validation on a realistic flight (no snaps): approach + pass + most of the turn track with 9.2 cm mean drift, then tracking is lost mid-turn when the map drains in rotation-dominant motion (the camera translates only ~2.7 cm/frame while yawing 3.4 deg/frame, so new points cannot reach the 0.15 m triangulation baseline before leaving the frame). The bridge invalidates correctly there — no guessed pose, no marker.
- VIO reacquire in this harness proves the state-machine logic only (same caveat as s21): a fresh VIO pose after loss re-establishes the bridge only when VIO reports solid re-tracking. In the real system reacquire must be gated on genuine relocalisation (re-observing a mapped area / doorway loop closure), which is not built here.
- No loop closure and no bundle adjustment: odometry drift grows with distance travelled, and the map distorts non-rigidly (measured: the map's reprojection error at the GT pose grows to ~11 px by the turn, which the PnP absorbs as pose drift). The ground-test drift (9.2 cm mean, 33.6 cm max) is the honest measure of that.
- The VIO step runs on CPU at ~40-90 fps alone; the full pipeline with VIO is dominated by the YOLO detector (see the pipeline fps column).

## Depth-model tests (monocular third voter)

Model: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`, transformers, fp32, CPU) — 24.8M params, 95 MB weights (Apache-2.0). The alternative, ZoeDepth (`Intel/zoedepth-nyu-kitti`, DPT-Large backbone, metric output), is ~1.37 GB fp32 — ~14x larger than DA-V2-Small and larger than the ~1.2 GB free RAM on this machine — so it was excluded on measured download size / RAM before any inference run, and its CPU inference time was not measured. DA-V2-Small outputs RELATIVE depth only; its metric scale comes from a MEASURED per-frame anchor — a robust linear fit true_z = a * model + b over floor pixels, where the true depth follows from the ray / y = 0 plane intersection using the known drone pose (camera height above the floor). No hard-coded scale constant anywhere: the fit is accepted only with >= 20 floor samples, residual RMS <= 0.35 m and |Pearson r| >= 0.8, otherwise there is no model vote that frame. The person depth is the fitted value at the median model value over the visible keypoint pixels.

Voting rule (server/depth.py: ``fuse_with_model_votes``): the model NEVER replaces triangulation as the primary method. When triangulation and bbox-height agree (<= 25%), the model stays OUT — there is nothing to vote on, and its measured error is larger than triangulation's. When they disagree (> 25%), the model arbitrates: it backs triangulation (outcome changes vs the legacy bbox-height fallback) or bbox-height (same outcome), and if it agrees with neither the legacy fallback stands. Pure arbitration: the winner's depth is returned EXACTLY, never fused with the model. Fusing the model depth into the winner was tried first and measured to hurt — in s15_occluded it took joint error 59.3 -> 64.5 cm (agree-case fusion) and 59.3 -> 64.3 cm (disagree-case refinement), because the occluder biases the model vote the same way it biases the bbox. Without triangulation the model does nothing (bbox-height alone). The model sigma carries a floor of 0.15 * depth (server/mono_depth.py): the floor-fit residual alone understates the model's real error (probe mean |err| 5-47 cm clean, 57-71 cm occluded, vs a ~0.2 m fit sigma) — the floor keeps the vote's stated uncertainty honest. The model runs in a background thread, at most every 10th detection frame (1 Hz at the 10 fps harness rate — every 5th frame kept the worker busy ~100% of the time and cost 7-28% of pipeline fps to CPU contention on this 1.5-CPU VM, while the disagreements it arbitrates persist for many frames, so 1 Hz loses nothing), on a 256x192 copy; the pipeline reads only the latest completed vote, age-gated to 2.0 s with the sigma inflated 0.2 m/s of age. The marker path never waits for it. The model's torch thread count is left at the process default so the YOLO detector's threading is identical with the voter on or off.

### Model choice — measured size / CPU latency

| Model | Params | Weights | Licence | CPU inference (this machine) |
|---|---|---|---|---|
| Depth Anything V2 Small | 24.8M | 95 MB | Apache-2.0 | 256x192: 425 ms; 336x252: 761 ms; 518x388: 1661 ms (single frame, process-default thread count; the pipeline uses 256x192) |
| ZoeDepth (nyu-kitti) | ~345M (DPT-Large) | ~1.37 GB fp32 | MIT | not measured — excluded on size / RAM (see above) |

Model calls / votes over the whole depth-model suite: {'n_calls': 61, 'n_votes': 56, 'ms_mean': 455.73383989284366, 'ms_min': 239.5500229999925, 'ms_max': 1167.5120889999562}.

### Baseline (two methods) vs model voter — same scenarios, GT poses

| Scenario | Joint err base / +model | |Dist err| base / +model | Model depth err mean |worst| (n) | Voter changed (frames) | Pipeline fps base / +model | Result base / +model |
|---|---|---|---|---|---|---|
| s1_basic | 36.1 cm / 36.1 cm | 8.7 cm / 8.7 cm | +14 cm | 55 cm (n=27) | 0 | 11.2 / 7.8 | PASS / PASS |
| s2_offcentre | 35.5 cm / 35.5 cm | 0.9 cm / 0.9 cm | +8 cm | 68 cm (n=18) | 0 | 10.0 / 9.9 | PASS / PASS |
| s3_close | 37.6 cm / 37.6 cm | 13.4 cm / 13.4 cm | +22 cm | 57 cm (n=28) | 0 | 11.1 / 10.0 | PASS / PASS |
| s4_dim | 35.6 cm / 35.6 cm | 8.0 cm / 8.0 cm | +10 cm | 47 cm (n=28) | 0 | 8.9 / 9.8 | PASS / PASS |
| s13_dim_noise | 36.8 cm / 36.8 cm | 8.9 cm / 8.9 cm | -2 cm | 42 cm (n=29) | 0 | 13.9 / 11.6 | PASS / PASS |
| s14_glare | 36.4 cm / 36.4 cm | 8.4 cm / 8.4 cm | +14 cm | 50 cm (n=29) | 0 | 12.6 / 10.2 | PASS / PASS |
| s15_occluded | 42.1 cm / 42.1 cm | 16.3 cm / 16.3 cm | +3 cm | 132 cm (n=27) | 0 | 10.8 / 7.5 | PASS / PASS |
| s18_moving | 48.1 cm / 48.1 cm | 12.6 cm / 12.6 cm | +30 cm | 78 cm (n=28) | 0 | 11.5 / 8.7 | PASS / PASS |
| s11_large_room_far | 31.9 cm / 31.9 cm | 6.7 cm / 6.7 cm | +48 cm | 71 cm (n=28) | 0 | 11.4 / 9.2 | PASS / PASS |
| s20_fast | 35.2 cm / 35.2 cm | 7.4 cm / 7.4 cm | +11 cm | 78 cm (n=24) | 0 | 9.7 / 9.1 | PASS / PASS |
| s23_blank_door | 36.0 cm / 36.0 cm | 11.8 cm / 11.8 cm | +35 cm | 81 cm (n=28) | 0 | 10.4 / 9.6 | PASS / PASS |
| s17_edge | 32.6 cm / 32.6 cm | 4.5 cm / 4.5 cm | n/a | 0 | 11.3 / 9.0 | PASS / PASS |
| s7_two_drones | 34.1 cm / 34.1 cm | 6.6 cm / 6.6 cm | +9 cm | 67 cm (n=19) | 0 | 4.9 / 3.2 | PASS / PASS |

Per-scenario detail (model runs) follows; the baseline detail is in the extended-results section above. The model-vote error is the floor-anchored model depth minus the GT foot depth, pooled over frames where a vote was available — it is the model's own error, not the fused pipeline error.

### s1_basic — person 4.0 m from phone, centred, light 1.0

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 576; distance-error samples: 34.
- Worst single-frame joint error: 64.2 cm; worst single-frame distance error: 9.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -0 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 127.5 ms/frame mean (7.8 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +14 cm, mean |err| 23 cm, worst 55 cm (n=27 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s2_offcentre — person 5.5 m from phone, off to the side near the doorway

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 34; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 26.
- Markers emitted: 36 (4 stale); joint-error samples: 544; distance-error samples: 32.
- Worst single-frame joint error: 63.6 cm; worst single-frame distance error: 2.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 2, 'triangulation': 32}; first triangulation frame: 22.
- Measured depth bias (estimated - true foot depth): bbox-height +1 cm (n=2); triangulation -3 cm (n=32).
- Mean signed distance error (marker - GT): -1 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 101.0 ms/frame mean (9.9 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +8 cm, mean |err| 40 cm, worst 68 cm (n=18 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s3_close — person just behind the wall (z = 3.3 m)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 36 (0 stale); joint-error samples: 612; distance-error samples: 36.
- Worst single-frame joint error: 62.2 cm; worst single-frame distance error: 15.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -7 cm (n=1); triangulation +2 cm (n=37).
- Mean signed distance error (marker - GT): -13 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 100.1 ms/frame mean (10.0 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +22 cm, mean |err| 22 cm, worst 57 cm (n=28 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s4_dim — light 0.5, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 61.4 cm; worst single-frame distance error: 10.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +2 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 102.0 ms/frame mean (9.8 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +10 cm, mean |err| 15 cm, worst 47 cm (n=28 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s13_dim_noise — light 0.4x with REAL Gaussian noise (sigma 12), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 556; distance-error samples: 34.
- Worst single-frame joint error: 64.3 cm; worst single-frame distance error: 9.8 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height +0 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -9 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 86.0 ms/frame mean (11.6 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean -2 cm, mean |err| 21 cm, worst 42 cm (n=29 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s14_glare — bright window directly behind the person + bloom (backlit), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 34 (0 stale); joint-error samples: 578; distance-error samples: 34.
- Worst single-frame joint error: 64.1 cm; worst single-frame distance error: 9.0 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -1 cm (n=5); triangulation -3 cm (n=31).
- Mean signed distance error (marker - GT): -8 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 98.4 ms/frame mean (10.2 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +14 cm, mean |err| 15 cm, worst 50 cm (n=29 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s15_occluded — person partly hidden behind a wardrobe (~40% of the body covered), person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 37; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 23.
- Markers emitted: 35 (0 stale); joint-error samples: 548; distance-error samples: 35.
- Worst single-frame joint error: 79.5 cm; worst single-frame distance error: 25.8 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 21, 'triangulation': 16}; first triangulation frame: 44.
- Measured depth bias (estimated - true foot depth): bbox-height +7 cm (n=21); triangulation +27 cm (n=16).
- Mean signed distance error (marker - GT): -16 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 132.5 ms/frame mean (7.5 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +3 cm, mean |err| 60 cm, worst 132 cm (n=27 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s18_moving — person walking slowly (0.25 m/s across, 0.12 m/s toward the wall) — triangulation assumes a static scene

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 37; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 23.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Worst single-frame joint error: 122.9 cm; worst single-frame distance error: 22.6 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 23, 'triangulation': 14}; first triangulation frame: 24.
- Measured depth bias (estimated - true foot depth): bbox-height +2 cm (n=23); triangulation -5 cm (n=14).
- Mean signed distance error (marker - GT): +13 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 115.0 ms/frame mean (8.7 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +30 cm, mean |err| 30 cm, worst 78 cm (n=28 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s11_large_room_far — larger room B (x +-4 m, z to 12 m), person 7.0 m (z) from phone

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 39; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 21.
- Markers emitted: 37 (0 stale); joint-error samples: 592; distance-error samples: 37.
- Worst single-frame joint error: 71.4 cm; worst single-frame distance error: 6.9 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 38}; first triangulation frame: 20.
- Measured depth bias (estimated - true foot depth): bbox-height +21 cm (n=1); triangulation +1 cm (n=38).
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 108.8 ms/frame mean (9.2 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +48 cm, mean |err| 48 cm, worst 71 cm (n=28 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s20_fast — fast drone: fast approach + fast wide hover (amp 0.9 m, period 1.2 s)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 33; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 27.
- Markers emitted: 35 (4 stale); joint-error samples: 527; distance-error samples: 31.
- Worst single-frame joint error: 65.7 cm; worst single-frame distance error: 7.6 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 32}; first triangulation frame: 24.
- Measured depth bias (estimated - true foot depth): bbox-height -1 cm (n=1); triangulation -4 cm (n=32).
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 110.4 ms/frame mean (9.1 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +11 cm, mean |err| 44 cm, worst 78 cm (n=24 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s23_blank_door — featureless blank wall at the doorway (no posters on the shared wall), wall 2.5 m, person 3.6 m (z)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 38; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 22.
- Markers emitted: 36 (0 stale); joint-error samples: 612; distance-error samples: 36.
- Worst single-frame joint error: 66.3 cm; worst single-frame distance error: 12.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 1, 'triangulation': 37}; first triangulation frame: 23.
- Measured depth bias (estimated - true foot depth): bbox-height -6 cm (n=1); triangulation +0 cm (n=37).
- Mean signed distance error (marker - GT): -12 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 103.6 ms/frame mean (9.6 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +35 cm, mean |err| 35 cm, worst 81 cm (n=28 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s17_edge — person at the very edge of the drone frame (billboard right edge ~30 px from the frame edge); drone looks at a fixed point, not at the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 42; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 18.
- Markers emitted: 40 (0 stale); joint-error samples: 680; distance-error samples: 40.
- Worst single-frame joint error: 74.8 cm; worst single-frame distance error: 7.3 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 13, 'triangulation': 29}; first triangulation frame: 20.
- Measured depth bias (estimated - true foot depth): bbox-height +11 cm (n=13); triangulation +5 cm (n=29).
- Mean signed distance error (marker - GT): +5 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 110.7 ms/frame mean (9.0 fps).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s7_two_drones — two drones, different hover points, both see the person

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 36; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 24.
- Markers emitted: 35 (0 stale); joint-error samples: 595; distance-error samples: 35.
- Worst single-frame joint error: 65.2 cm; worst single-frame distance error: 7.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 31}; first triangulation frame: 29.
- Measured depth bias (estimated - true foot depth): bbox-height -4 cm (n=5); triangulation -2 cm (n=23).
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Pipeline latency (process_frame, incl. detector): 313.1 ms/frame mean (3.2 fps).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +9 cm, mean |err| 31 cm, worst 67 cm (n=19 frames with a vote); voter changed the outcome on 0 frame(s).
- Fused mean joint error, hover frames 25-59: 34.1 cm; single-drone mean joint error over the same frames: drone1 36.4 cm, drone2 37.6 cm; best single: 36.4 cm. Single-drone detection (hover frames): {'drone1': {'detected': 35, 'in_view': 35}, 'drone2': {'detected': 27, 'in_view': 27}}.
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- check `fused_le_best_single`: PASS
- **Result: PASS**

### Model voter on the VIO path (same 4 VIO scenarios)

| Scenario (VIO) | VIO valid | Joint err VIO-only / VIO+model | |Dist err| VIO-only / VIO+model | Model err mean |worst| | Voter changed | Pipeline fps VIO-only / VIO+model | Result VIO-only / VIO+model |
|---|---|---|---|---|---|---|---|
| s1_basic | 56/60 | 40.5 cm / 40.5 cm | 6.0 cm / 6.0 cm | -3 cm | 13 cm (n=9) | 0 | 4.9 / 6.0 | PASS / PASS |
| s23_blank_door | 56/60 | 32.6 cm / 32.6 cm | 5.7 cm / 5.7 cm | +15 cm | 31 cm (n=9) | 0 | 6.7 / 6.5 | PASS / PASS |
| s20_fast | 56/60 | 36.1 cm / 36.1 cm | 7.5 cm / 7.5 cm | +9 cm | 35 cm (n=6) | 0 | 6.7 / 6.8 | PASS / PASS |
| s4_dim | 56/60 | 36.3 cm / 36.3 cm | 6.8 cm / 6.8 cm | +1 cm | 16 cm (n=9) | 0 | 6.5 / 6.2 | PASS / PASS |

### s1_basic (VIO) — person 4.0 m from phone, centred, light 1.0

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 12; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 48.
- Markers emitted: 10 (0 stale); joint-error samples: 167; distance-error samples: 10.
- Worst single-frame joint error: 74.1 cm; worst single-frame distance error: 8.7 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 5, 'triangulation': 7}; first triangulation frame: 53.
- Measured depth bias (estimated - true foot depth): bbox-height -2 cm (n=5); triangulation +8 cm (n=7).
- Mean signed distance error (marker - GT): -6 cm (negative = marker distance shorter than the true phone->person distance).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean -3 cm, mean |err| 8 cm, worst 13 cm (n=9 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s23_blank_door (VIO) — featureless blank wall at the doorway (no posters on the shared wall), wall 2.5 m, person 3.6 m (z)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 13; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 47.
- Markers emitted: 11 (0 stale); joint-error samples: 187; distance-error samples: 11.
- Worst single-frame joint error: 69.8 cm; worst single-frame distance error: 12.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 6, 'triangulation': 7}; first triangulation frame: 53.
- Measured depth bias (estimated - true foot depth): bbox-height +5 cm (n=6); triangulation -21 cm (n=7).
- Mean signed distance error (marker - GT): -3 cm (negative = marker distance shorter than the true phone->person distance).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +15 cm, mean |err| 15 cm, worst 31 cm (n=9 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s20_fast (VIO) — fast drone: fast approach + fast wide hover (amp 0.9 m, period 1.2 s)

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 10; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 50.
- Markers emitted: 11 (5 stale); joint-error samples: 134; distance-error samples: 8.
- Worst single-frame joint error: 80.3 cm; worst single-frame distance error: 10.2 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 6, 'triangulation': 4}; first triangulation frame: 51.
- Measured depth bias (estimated - true foot depth): bbox-height -4 cm (n=6); triangulation -22 cm (n=4).
- Mean signed distance error (marker - GT): +5 cm (negative = marker distance shorter than the true phone->person distance).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +9 cm, mean |err| 11 cm, worst 35 cm (n=6 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### s4_dim (VIO) — light 0.5, person 4.5 m

- Frames: 60 @ 10 fps; in-view frames (whole person in frame, >= 80 px tall): 12; frames excluded from the metrics (person partly out of frame, e.g. drone flying past them): 48.
- Markers emitted: 10 (0 stale); joint-error samples: 169; distance-error samples: 10.
- Worst single-frame joint error: 67.2 cm; worst single-frame distance error: 8.4 cm (means hide bad frames — these do not).
- Markers emitted while the bridge was invalid: 0 (must be 0).
- Depth methods: {'bbox_height': 12}; first triangulation frame: None.
- Measured depth bias (estimated - true foot depth): bbox-height +1 cm (n=12); triangulation n/a.
- Mean signed distance error (marker - GT): -7 cm (negative = marker distance shorter than the true phone->person distance).
- Monocular model vote (Depth Anything V2 Small, floor-anchored): depth error vs GT foot depth mean +1 cm, mean |err| 8 cm, worst 16 cm (n=9 frames with a vote); voter changed the outcome on 0 frame(s).
- check `detection_rate`: PASS
- check `joint_error`: PASS
- check `distance_error`: PASS
- check `scale_error`: PASS
- check `marker_while_bridge_invalid`: PASS
- **Result: PASS**

### Honest limits — depth model (what this still does NOT prove)

- The floor anchor needs visible floor in the lower frame: a drone looking level or up, or a floor with no texture / a mirror floor, gives no fit and therefore no vote — the pipeline then falls back to the two-method behaviour silently (by design: no vote is better than a guessed scale). The virtual floor is a high-contrast checker; a real plain floor may fit worse — the fit RMS / correlation gates are the honest check, and their pass rate on real footage is unmeasured.
- s17_edge produced NO model vote at all (0 of 4 calls): the floor-fit residual RMS there is 0.74-0.75 m, above the 0.35 m gate (|r| = 0.81 passes, barely). The drone sits deep in room B (z ~ 3.8 m) looking at a fixed point, so the floor samples span 2.5-6.9 m of true depth and the model's output is not affine in depth over that range — the gate working as designed, not a bug: a bad fit must not vote. The s17 mono run therefore reproduces its baseline exactly. Restricting the fit to nearer floor (as the VIO scale does) might recover votes there; it was NOT done — it would change the fit for every scenario and needs its own measurement round.
- The model vote is sampled at the keypoint pixels: an occluder in front of the person can contaminate it (measured in s15_occluded: worst model error 137 cm on one frame). The outlier drop in the voting rule is what keeps a bad model frame out of the fused depth; the raw model error is still reported above, not hidden.
- The model is per-frame (no static-scene assumption), which is why it can arbitrate a moving person — but its own error (~15-47 cm mean |err| in the scenarios above) is larger than triangulation's in static scenes, so it is a pure arbitrator, not a refiner: it only ever flips the choice between triangulation and bbox-height, and it flipped that choice on 0 frames in this suite (see the voter-changed column — the mono runs reproduce the baseline exactly). Its value is for the case the virtual suite does not contain: triangulation right and bbox-height badly wrong on the same frame (e.g. a crouching or half-out-of-frame person whose bbox height collapses). The flip logic itself is proven by direct checks of ``fuse_with_model_votes`` (disagree-backs-tri returns triangulation with voter_changed=True).
- The background worker makes the model runs timing-dependent: which frames carry a vote depends on worker speed on the day, so the model-run numbers reproduce approximately, not exactly, unlike the deterministic baseline. The regression comparison above still reads the baseline (model-off) runs.
- Latency is measured on this shared 1.5-CPU VM with the YOLO detector running concurrently; a real laptop (more cores, no shared-CPU throttle) will differ. The pipeline fps numbers are process_frame only (rendering excluded).

## Live-path fixes — reconciliation with d89be655 (2026-11-24)

Pushed 2026-11-24 ~01:13 as commit 531ba8ba (verified live). The laptop live code (commit 0e8d507b) was merged
into the local copy first, then the other AI's commit d89be655
("Add SafeFlightController with thread-safe locking, debouncing, and
flight HUD status") landed on top of it and changed only
`server/main.py`. The local `FlightController` was dropped — there is
ONE flight controller now, theirs, reviewed on its merits and kept.

### SafeFlightController (d89be655) — kept, plus two completions

- Serializes: one lock, non-blocking acquire, a busy command is
  skipped with a warning — no concurrent movement. Verified.
- Gates movement: moves run only when status is HOVERING, set only
  after a successful takeoff; a failed takeoff resets status to
  LANDED, so later moves are refused. Verified.
- Completion 1 — land idempotency: the land branch returned no
  early-exit when already LANDED, so a repeat `[L]` after landing
  re-sent `land`. Now returns early; land is sent exactly once.
- Completion 2 — "Not joystick" guidance: on a takeoff error
  containing "not joystick" the log now says plainly to force-close
  the Tello app on the phone, reboot the drone, restart the server
  and press `[T]` again. A retry after fixing the app is accepted
  (no session block).
- Moved to module level so tests can import it (was nested inside
  `main()`); behaviour unchanged. The `finally` block no longer
  calls `tello.land()` directly — it lands only while holding the
  controller lock and only if airborne, so it cannot double-send
  with a landing already in flight.

### Marker-format fix (BUG 2) — the part d89be655 did not touch

- Root cause: the live loop overrode the VIO verdict —
  `if not tracking_valid or pose is None: pose = item[1];
  tracking_valid = True` — and `TelloVideoSource.read()` returned
  the frozen start pose with `tracking_valid=True` on every frame.
  The bridge validated on that fiction frame and the detected
  person's joints were transformed into it, so the viewer (which
  expects 17 3D joints in the phone frame, COCO order, +Y up)
  exploded the projection — the scrambled skeleton of the 2026-11-23
  live run (2.2 m shown for a person 0.5 m away). The same scrambled
  skeleton reproduces on a local run; same sender bug, same fix.
- Fix: new `live_pose_step()` in `server/main.py` — the VIO verdict
  passes through honestly, mirroring `tests/virtual_server.py`; the
  start pose is only the VIO initialisation pose, never returned as
  valid. `server/video_source.py` now returns `tracking_valid=False`
  (the source measures no pose). With tracking invalid the bridge
  stays invalid and the viewer honestly shows "Tracking lost" until
  VIO initialises on real motion. No marker without a bridge, ever.
- Secondary fix in the same loop: scenario time is now wall-clock
  (`time.monotonic()`); the frame counter assumed 10 fps while the
  real loop runs ~4-5 fps, so tracks stayed "live" ~2x too long.

### SSE disconnect (BUG 3)

- `_serve_sse` caught `(BrokenPipeError, ConnectionResetError)` but
  not `ConnectionAbortedError` (WinError 10053) — it escaped and
  spammed a traceback. Now catches `OSError` and closes quietly.

### Measured verification (2026-11-24, ground/virtual only, no flight)

- `tests/test_live_fixes.py` (new, standalone, not wired into
  `run_tests.py`): **12/12 PASS** (re-run 2026-11-24 after the
  follow-up fixes below) — Not-joystick blocks moves and logs
  the force-close guidance; retry after fixing the app works;
  movement gated on takeoff; land exactly once (5 rapid lands + 1
  after landing = 1 send); 10 concurrent moves never overlap
  (max_inflight 1) and coalesce to 1; `live_pose_step` passes stub
  verdicts through and returns valid=False on a raising VIO; real
  VIO on identical static frames stays invalid for 10 frames;
  pipeline with a stub detector + tracking invalid emits no marker
  for 10 frames; control proves the hazard — the same frozen pose
  fed as tracking VALID does emit a marker; a move failing with
  "Not joystick" or "No valid imu" resets to LANDED so [T] works
  again (and is labelled MOVE, not TAKEOFF); a phone-browser
  disconnect (WinError 10053) is swallowed, a real error still
  prints.
- Full virtual suite re-run on the reconciled code
  (`full_suite_2026-11-24_reconciled.log`, 724.0 s, exit 0):
  **27/27 main scenarios PASS, 0 FAIL**; VIO numbers unchanged
  (s1 40.5, s23 32.6, s20 36.1, s4 36.3 cm); real-only UI server
  tests PASS; no-drone-video test PASS.
- Follow-up (not part of these fixes): live VIO runs vision-only —
  the real Tello state stream (UDP 8890, `server/tello_state.py`
  StatePrior) is not fed to it. And `connect_tello_to_home_wifi.py`
  puts the drone in station mode, after which `Tello()` (default
  host 192.168.10.1) would not find it.

### Follow-up fixes from the 2026-11-24 live run (same day, ~03:51)

- What the log showed: takeoff worked twice (HOVERING both times).
  Every A/D press landed while takeoff/landing was still running
  (~9 s window) and was dropped by the lock as "Drone busy" — so
  from the user's side A/S/D never moved the drone; only W executed
  once (forward 30 ok), then failed with "error Not joystick" (the
  Tello app on the phone grabbed control mid-flight) and then
  "error No valid imu" (drone down). The controller still said
  HOVERING, so five [T] presses were ignored as "already HOVERING".
  The 10053 traceback at the end came from the HTTP request handler
  (phone browser disconnecting mid-request), not from `_serve_sse`.
- Fixes in `server/main.py`: (1) a move failing with "Not joystick"
  or "No valid imu" now resets status to LANDED, so [T] is accepted
  again; (2) the rejection is labelled by the command that failed
  (MOVE vs TAKEOFF) — the old message wrongly said "TAKEOFF
  REJECTED" for a failed forward move; (3)
  `RobustThreadingHTTPServer.handle_error` swallows
  `ConnectionAbortedError` / `ConnectionResetError` /
  `BrokenPipeError` (client disconnects) and still prints real
  errors. `tests/test_live_fixes.py` extended to 12 tests, all PASS
  (re-run by the author, 2026-11-24); real-only UI server tests
  re-run standalone via `run_all(port=18865)`: PASS (all 5
  sub-tests). Full 27-scenario suite not re-run for this change:
  the edits touch only the flight controller's error path and the
  HTTP error handler, both covered by the 12 live-fix tests and
  the real-only tests.
- Still open: the status can only desync again if the drone's real
  state changes without the controller knowing (app takeover,
  forced landing). The proper fix is feeding the Tello state
  stream (UDP 8890 height) into the controller — same follow-up as
  above. Until then: close the Tello app fully, reboot the drone,
  wait for green HOVERING before touching A/S/D, one key at a time.

### Video-restart fix from the 2026-11-25 live run (same day, ~02:13)

- What the log showed: `command` ok, `streamon` ok, then
  djitellopy's decode thread died with `av.error.OSError: [Errno 5]
  I/O error: 'udp://@0.0.0.0:11111'` — after that the laptop got no
  video and `TelloVideoSource.read()` kept serving the last frame
  forever, so the pipeline ran on a frozen frame with nothing
  telling the user. Most likely trigger: the Tello app on the phone
  grabbed the video stream (the drone sends video to one device).
- Fix in `server/video_source.py` (`TelloVideoSource` only): `read()`
  treats a frame object unchanged for > 3.0 s (wall clock) or a
  stopped reader as a dead stream and restarts it (`streamoff` /
  `streamon` / fresh `get_frame_read`), logging "VIDEO LOST:
  restarting the drone video stream (attempt N/3)" and returning
  None for that read (the pipeline loop already sleeps and
  continues on None). After 3 failed restarts the source stays dead
  and `read()` returns None; the log names the Tello app and tells
  the user to force-close it, reboot the drone and restart the
  server. `close()` still works after restarts; `tracking_valid`
  stays False and the start-pose return is unchanged.
- `tests/test_live_fixes.py` extended to 16 tests, all PASS
  (2026-11-25): healthy reads never touch the stream; a frozen
  frame restarts exactly once and the next read serves a fresh
  frame; a stopped reader restarts; after 3 failed restarts no
  further stream calls happen and the log names the Tello app.
  Full 27-scenario suite not re-run: live-path only, the suite
  uses SyntheticSource/FileVideoSource, not TelloVideoSource.

### Key remap from the 2026-11-25 live run (same night, ~02:45)

- What the user saw: pressing Q made "the program go off". Q only
  hid the camera window by design (the server kept running) — it
  never crashed or exited the program. Keys are now the user's own
  map: Q or T takeoff, E land, L turn right (L is deliberately NOT
  land any more — an accidental L press mid-flight used to drop the
  drone), J turn left; W/S/A/D/I/K moves unchanged. The key dispatch
  is now a tested function (`dispatch_drone_key` in `server/main.py`);
  the camera window is closed with its X button, and Ctrl+C in the
  console is the full exit and lands the drone.
- `tests/test_live_fixes.py` extended to 21 tests, all PASS
  (2026-11-25): Q and T take off, E lands, L turns right and never
  lands, J turns left, all six moves unchanged, unknown keys send
  nothing.

### VIO diagnostics + takeoff reset + ground watchdog (BUG 5, 2026-11-25)

- The finding: the bridge NEVER became valid in any live run.
  Root cause, from the code + the 03:01 log: the VIO keyframe is the
  FIRST frame the server processes — a pre-takeoff ground frame.
  Server started serving 02:58:49, takeoff 03:01:23 (~155 s later).
  `_try_initialise` LK-tracks from that keyframe on every frame,
  with no prediction and no refresh; LK cannot survive the 1.3 m
  rise plus turns, so init can never fire. The thresholds are NOT
  the cause (40 tracked / 30 inliers / 15 px are harness-measured)
  and were not moved. Secondary gates that can also block init in
  this scene: the floor-scale step needs >= 8 floor points in the
  lower 40% of the keyframe (a blanket filling the view fails it),
  and pure rotation gives no parallax for the essential matrix.
- Fix in `server/main.py`: `maybe_reset_vio_at_takeoff` — on the
  LANDED→HOVERING transition, an uninitialised VIO is reset so the
  next frame becomes an airborne keyframe at the known start pose.
  A working bridge is never wiped; one reset per takeoff.
- Diagnostics in `server/vio.py`: every init-failure path now sets
  `diag["init_fail"]` with numbers (tracked, median flow, inliers,
  floor points, keyframe age) and `_log_init_wait` logs one INFO
  line per reason change (repeat at most every 5 s) with a plain
  hint, e.g. 'VIO init waiting: few-tracked (tracked=18/40 ...) --
  point the camera at a textured scene'. Init success logs the
  numbers; `_declare_lost` logs tracked/inliers/map. Console status
  line in the live loop (~1 per 5 s): battery %, flight status,
  bridge, VIO state + numbers, frames processed — the battery %
  over time proves (or rules out) a sag.
- Ground watchdog in `SafeFlightController`: a daemon thread polls
  the height every 0.5 s; HOVERING + height <= 10 cm for ~2 s
  resets status to LANDED and logs that [T] works again. Never
  fires while TAKING_OFF / MOVING / LANDING, never while the drone
  is actually airborne, and failed height reads are skipped. (Why:
  the drone lands itself on the low-battery failsafe — blinking red
  — without telling us, so status stayed HOVERING and [T] was
  ignored: 'Takeoff ignored: drone already HOVERING'.)
- Battery note (2026-11-25): on a full charge the drone flies only
  ~2 min before landing itself; a healthy Tello flies ~13 min. The
  battery is worn/degraded — the likely cause of the self-landings.
  The red light + shutdown is the drone's own fault/critical-battery
  signal; this code cannot switch the drone off or change its light.
  'No valid imu' appeared once in the logs — an IMU/system fault
  remains a secondary candidate, but the 2-minute flights point at
  the battery. The new battery % trace in the log will show the %
  collapsing in flight if so.
- `tests/test_live_fixes.py` extended to 30 tests, all PASS
  (2026-11-25): status line carries battery/bridge/VIO numbers;
  init-failure reasons (low-displacement on static frames, lk-empty
  on blank frames) with one log line per transition; takeoff reset
  fires once, never on a working bridge, and clears the stale
  keyframe; watchdog resets a grounded drone and the next takeoff
  is accepted, keeps an airborne drone hovering, never fires without
  a height reading or while taking off. Full 27-scenario suite not
  re-run: live-path only.

### process_frame None-foot crash (BUG 6, 2026-11-25)

- The live run: the BUG 5 fix worked — 'VIO reset at takeoff'
  fired, 'VIO initialised at frame 7' (E inliers 648, tracked 673),
  bridge=valid at 03:37:43 and 03:37:51, the first time ever live.
  Then at 03:37:51 VIO tracking was lost (lk-empty, turn toward
  the blanket) and the pipeline thread crashed:
  `TypeError: unsupported operand type(s) for -: 'float' and
  'NoneType'` at `server/main.py` line 226, `rough - st["foot"]`.
  The thread died, so no more frames were processed and the
  keyboard (including land) stopped working; Ctrl+C in the console
  is the way to land after such a crash.
- Root cause: a new person track is created with `"foot": None`
  (`server/main.py`, the `person_id is None` branch) and the foot
  is only filled in later, when a depth is estimated AND the
  bridge is valid. After tracking is lost the bridge stays invalid,
  so the foot is never set — but `_last_depth` / `_last_foot` are
  never cleared either, so the next frame's association still runs
  and dereferences the None foot. Reproduced exactly in isolation
  before fixing (same line, same TypeError): valid frames localise
  a person, a lost frame at a moved pose creates the footless
  track, the next lost frame crashes.
- Fix in `server/main.py` (one guard, no restructure): the
  association loop skips tracks whose foot is None — a footless
  track has nothing to compare, so it can never match. The other
  Nones on that path were already guarded (`rough` checked, and
  `_last_depth` / `_last_foot` checked before the block). The VIO
  reset, the STATUS line and the ground watchdog are untouched.
- `tests/test_live_fixes.py` extended to 32 tests, all PASS
  (pytest and the standalone runner, 2026-11-25): a seeded
  None-foot track is skipped and `process_frame` returns normally
  with tracking invalid (twice in a row); the end-to-end replay of
  the 03:37:51 crash — localise, lose tracking at a moved pose,
  keep feeding lost frames — returns normally on every frame with
  no marker and bridge invalid. Full 27-scenario suite not
  re-run: live-path only.

## Autonomous mode (2026-11-25, Tello only, laptop CPU only)

Built on the user's order ("no bigger drone, make THIS one the best,
whatever is possible"): the drone flies itself — take off, scan the
room, find the doorway, fly through it into the other room, find the
person, and the phone shows the skeleton marker through the wall.
Five components plus one wiring pass; `server/main.py` had exactly one
writer for the wiring. Nothing was pushed yet (report only).

### Doorway detection (`server/doorway.py`, new)

OpenCV only, no ML: downscale to 480 px wide, grayscale, Gaussian
blur, Canny (50/150), HoughLinesP; keep near-vertical segments spanning
>= 50% of frame height, merge collinear segments into jambs (15 px
x-tolerance), pair jambs whose width is 8-55% of frame width with >= 60%
vertical overlap and similar height. Returns a `Doorway` (centre-x,
width, left/right x in original-frame pixels, confidence 0..1) or None
honestly. Confidence = 0.5*jamb-length + 0.25*length-symmetry +
0.25*width/height plausibility (ideal ratio ~0.45). Degenerate inputs
return None, never crash. ~4.2 ms per `detect()` on a realistic
textured 960x720 frame (~95 ms worst case on pure noise). `tests/
test_doorway.py`: 9 tests — synthetic door found (centre 460.3 vs truth
460, width 320.6 vs 320, confidence 0.90), second door position/size,
plain textured wall -> None, blank frame -> None, poster-sized
rectangle rejected, door half at the frame edge -> plausible centre, no
crash, single jamb -> None, degenerate frames -> None, standalone
import (does not load `server.main`). Limits: edge-only, no semantics —
a floor-to-ceiling window or a tall bookshelf/cabinet edge pair can
fool it; low-contrast jambs (same-colour trim, poor light, motion
blur) return None rather than a guess; width is pixels only, no
depth/scale check (the autopilot must combine it with distance); one
jamb visible = None.

### Autopilot (`server/autopilot.py`, new)

Tick-driven, non-blocking state machine; duck-typed controller
(`.status`, `.send_command(name, func, *args)`); mover resolved from
an explicit mapping first, else `controller.tello` methods. States:
OFF, SCAN, AIM, APPROACH, PASS, SEARCH, TRACK, HOLD, NODOOR, LANDING,
DONE. SCAN yaws 30 deg with pauses (doorway checked before the pause
expires); AIM centres the doorway (60 px tolerance); APPROACH takes
20 cm forward steps, re-centring on drift, commits to PASS at doorway
width >= 400 px or after 10 steps; PASS takes exactly 3 x 20 cm steps
ignoring a lost doorway, then SEARCH; SEARCH slow yaw scan, person ->
TRACK; TRACK hovers, yawing only when the person nears a frame edge
(120 px), person lost -> SEARCH. `tests/test_autopilot.py`: 17 tests
— all state transitions with scripted inputs, safety stop on tracking
loss (no move while lost, resume after recovery), battery < 25 lands
once, no doorway after a full 360 scan -> NODOOR with no wandering,
stop() halts with no further moves, tick never blocks.

### Autopilot safety patch (same files, 2026-11-25, from the 04:03 video run)

The 04:03 run: takeoff OK at 80%, drone hovered ~12-15 s swinging
between the user and the blanket on its own, then landed itself at 77%
— NOT a battery landing. The blanket was hand-held and moving, the
room dim; a moving surface confuses the downward vision sensor. Two
rules added: (1) self-landing is a stop condition — controller LANDED
or LANDING while in any flying state (including HOLD) -> DONE with
"landed — autopilot stopped", no further moves, never resumes or
re-takeoffs on its own; (2) start gate — `start(ctx=None)`: with a ctx,
refuses unless HOVERING ("cannot start: not hovering") and tracking
valid ("cannot start: no tracking"); without a ctx, old behaviour, and
the first tick without tracking holds immediately with no move. No
oscillation detector or fabric sensor was invented (the Tello has no
forward depth sensing; keeping clear of fabric and keeping the scene
still are setup rules). `tests/test_autopilot.py` extended to 25 tests
(8 new).

### Saved-map relocalisation (`server/map_store.py`, new; `server/vio.py` +~150 lines, two methods only)

`save_map(path, vio) -> int` (points saved; empty map writes nothing),
`load_map(path=None) -> MapData` (default `<repo>/maps/latest.npz`;
raises FileNotFoundError/ValueError), `try_relocalise(vio, frame,
map_data, diag=None) -> bool` (never raises; ORB + RANSAC PnP ~100 ms;
call only on tracking loss). `vio.py` gained `export_map_data()`
(consolidates the distributed map, deduplicated by persistent point
ID, newest/BA-refined observation wins; fills missing live-point
descriptors from the keyframe gray at save time only) and
`adopt_saved_map_pose(...)` (marks the VIO initialised from a saved
map and delegates to the existing `_adopt_recovered`). No tracking,
threshold, or init-path changes. Matching uses the VIO's own
`_solve_pnp` (yaw1dof chain + full fallback, same 3 px threshold,
same accept gate), seeded from the reference keyframe pose then up to
2 older keyframes; a `cv2.solvePnPRansac` fallback with cheirality
check exists only for duck-typed VIO stand-ins. `tests/
test_relocalisation.py`: 10 tests with a self-contained synthetic-room
renderer — save/load round-trip bit-for-bit on points/descriptors/IDs/
K/poses (driven map: 660 points, 79 with descriptors, 2 keyframes);
same-view relocalisation True, 30 inliers from 37 matches, recovered
pose 0.015 m from the VIO's own map-frame estimate (asserted at
0.35 m), tracking resumes on the next frame; shifted view (+0.3 m,
+8 deg yaw) True, 17 inliers, 0.053 m error; different scene False, 8
matches, VIO state byte-identical; fresh VIO relocalising against
another VIO's saved map True, 0.015 m error (server-restart path
works). Viewpoint sweep (probe): translation OK to +0.5 m, fails at
+0.8 m; yaw OK to +25 deg, fails at +40 deg; combined +0.5 m / +15 deg
OK. Limits: pose is recovered in the map frame, not ground truth (the
VIO's init scale is stretched ~1.4x in x on the synthetic flight, so
the map frame sits ~0.67 m from truth at frame 10 — VIO scale drift,
not relocalisation error); descriptor coverage is thin (79/660 points
carry a usable ORB descriptor); one map per takeoff spot, no spot key
(`maps/latest.npz` is overwritten on every save — a stale map from
another spot would place the drone in the wrong world frame with a
passing PnP gate).

### Flight recorder (`server/recorder.py`, new)

`new_run_dir(base_dir=None)` -> `<base>/runs/<YYYY-MM-DD_HH-MM-SS>/`
(same-second collisions get -1, -2, ... suffixes);
`RunRecorder(run_dir, fps=5, max_queue=30)` — background writer thread
owns the `cv2.VideoWriter` (mp4v, `video.mp4`); `write_frame(frame)`
is non-blocking (bounded queue, drops when full, never raises, never
blocks the caller); `stop(timeout=10.0)` drains and closes, idempotent;
`frames_written` / `frames_dropped` counters. `attach_file_log(path)`
attaches an INFO+ FileHandler (same format as the console) to the root
logger, deduplicated per path. `tests/test_recorder.py`: 9 tests — 20
frames -> video exists, non-empty, ~20 frames readable; slow-writer
test proves `write_frame` returns in < 1 s total with frames dropped;
write-before-start drops fast with no video; run-dir timestamp format
and uniqueness; file-log writes a line and returns the same handler on
a second call; bad frames (None, str, int, 1-D, empty, float32, wrong
size) never raise. Limits: no video file if no frames ever arrive;
video size is fixed by the first frame (resolution change mid-run
drops later frames); queue bound is frames, not bytes (~80 MB worst
case at 720p if the writer stalls completely).

### Wiring (`server/main.py` only; `tests/test_live_fixes.py` step expectations; new `tests/test_wiring.py`; new `docs/AUTONOMOUS_MODE.md`)

- 20 cm steps: `dispatch_drone_key` sends `"forward 20"` etc. with
  distance 20; turns stay 30 deg. HUD reads `Move 20cm … [U] Auto`.
- Startup: one `DoorwayDetector()`, one `Autopilot(flight_ctrl)`,
  `new_run_dir()` + `attach_file_log(run_dir/"server.log")` +
  `RunRecorder(run_dir, fps=5)` started; saved map loaded at startup
  (missing/corrupt -> log `no saved map`, continue with None).
- Per-iteration tick after `live_pose_step`/`process_frame`:
  `build_auto_context` from live values (tracking, bridge, battery,
  throttled doorway, max-confidence detection's bbox centre-x or
  None); doorway detector runs only in SCAN/AIM/APPROACH/PASS and at
  most every ~0.4 s, last detection held between runs.
- U key toggles (`dispatch_drone_key(key, flight_ctrl, autopilot=None,
  ctx=None)`): OFF -> `autopilot.start(ctx)` so the gate enforces
  HOVERING + tracking_valid; running -> `stop()`. Any manual key
  (W/S/A/D/I/K/J/L/Q/T/E plus arrows) while running calls
  `autopilot.stop()` FIRST, then executes. Old two-arg callers still
  work; U with no autopilot returns False.
- Hover-stable gate (`HoverGate`): `tick()` is not called until the
  controller has been continuously HOVERING for 5 s; the timer resets
  whenever status leaves HOVERING; the loop logs `hover stable 5 s —
  autopilot live` once per hover episode (only when the autopilot is
  running). No retry logic added anywhere.
- Not-joystick hard stop: `SafeFlightController.last_reject` (None
  normally; cleared at each worker start; `"not-joystick"` on a
  Not-joystick error, else the lowercased error) — additive only, the
  existing reset to LANDED and the ground watchdog are untouched.
  Loop-side `check_not_joystick_stop` runs before the tick: running
  autopilot + `last_reject == "not-joystick"` -> `stop()`, log, and
  the operator page shows it. Message (`NOT_JOYSTICK_MESSAGE`):
  "Commands rejected ('error Not joystick') — reset the drone (hold
  power 5 s until yellow flash) and try again. Secondary check:
  force-close the Tello app completely (swipe it away from the app
  switcher)." The app is ruled out for the 04:06 run (never opened
  that day; DJITelloPy issue #201, also in pure SDK scripts), so the
  factory reset is the primary message.
- Relocalisation: on tracking loss with a loaded map,
  `try_relocalise(vio, frame, saved_map, diag)` throttled to ~2 s;
  success logs `relocalised against saved map (<n> PnP inliers)`. At
  shutdown, `save_map(None, vio)` if VIO initialised, guarded, logs
  points saved.
- Recorder: `recorder.write_frame(vis)` in the `show_drone_window`
  block after `vis` is drawn; shutdown stops it, logs frames
  written/dropped, removes/closes the file-log handler. Note: the
  recorder only writes while the drone window is open.
- Operator page: `/api/status` carries `auto_state`, `auto_message`,
  `height_cm`, `not_joystick`; the page shows Auto + Height lines and
  a red Not-joystick banner, and a Stop autopilot button POSTs to
  `/api/autopilot/stop` (same POST pattern as `/api/control`). Height
  is a guarded `tello.get_height()`, throttled to ~1 s, last good value
  kept on failure — LANDED with height > 0 reads as still-airborne.
  BRIDGE/DRONE/PERSON blocks intact.
- STATUS line: `format_status_line(..., auto_state=None,
  height_cm=None)` — byte-identical old format when both are None,
  otherwise appends `auto=<state>` and `height=<cm>cm`.
- `tests/test_wiring.py`: 16 tests — U toggles on/off, manual key
  stops autopilot before executing, Not-joystick rejection sets
  `last_reject` and the loop-side check stops the autopilot, 20 cm
  step in dispatch, relocalise/save/load wiring via fakes.

### Measured verification (2026-11-25)

`~/workspace/tello-search/.venv/bin/python -m pytest tests/test_live_fixes.py tests/test_doorway.py tests/test_autopilot.py tests/test_relocalisation.py tests/test_recorder.py tests/test_wiring.py -q` -> **101 passed** (32 + 9 + 25 + 10 + 9 + 16). Full 27-scenario virtual suite not re-run: live-path only. Nothing pushed yet.

### Gating dependency (stated plainly)

The command path is the gating dependency: the autopilot cannot fly
at all until the Not-joystick rejections are resolved. In the 04:06
run (no blanket) every move was rejected with 'error Not joystick'
while the battery stayed 72%; the drone drifted right uncontrolled for
~13 s and came down on a cabinet by itself. VIO never initialised in
that run: first low-displacement, then few-floor-points (floor_pts=0/8
— the camera faced the wall, the floor was not visible in the lower
frame), then few-tracked as the keyframe went stale (kf_age up to
143f); bridge invalid throughout. First thing to try is the factory
reset; the hover-stable gate also exists because commands sent too fast
while the drone is not ready are a reported cause. Manual flight is
blocked the same way — this is not an autopilot-only problem.

### What is live-testable in one room vs what needs a real doorway

One room (blanket as the wall, no second room free, no real doorway):
SCAN / SEARCH / TRACK states, the safety holds (tracking loss,
self-landing, Not-joystick, low battery), the start gate, the 5 s
hover gate, the 20 cm steps, relocalisation against the saved map,
recording. Needs a real doorway: AIM / APPROACH / PASS and the full
doorway flight — unit-tested only, never flown. Setup rules that
decide whether any of it works live: force-close the Tello app
completely (locking the phone is not enough — a background app can
steal the controls back mid-flight), take off facing the room with the
floor visible in the lower frame and good light, pin the blanket still
(never hand-held) and keep the drone clear of fabric, and remember the
saved map is only valid from the same takeoff spot. See
`docs/AUTONOMOUS_MODE.md`.
