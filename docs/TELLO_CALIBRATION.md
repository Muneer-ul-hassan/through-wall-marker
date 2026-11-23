# Tello State Calibration — procedure and results

**Status: NOT RUN ON A REAL DRONE.** The calibration constants in
`server/tello_state.py` (`TelloCalibration`) default to UNCALIBRATED
(velocity priors OFF). Do not enable velocity priors until this
procedure has been run on a real recording and the results below
have been filled in.

## Why this exists

DJI publishes **no accuracy spec** for any state field, and its own
docs disagree:

- SDK 3.0 says `vgx/vgy/vgz` are **dm/s** and `agx/agy/agz` are
  cm/s². Older material says **cm/s** and 0.001g. A 10× velocity
  error if assumed wrong.
- The velocity **frame** (body vs global) is undocumented.
- The front camera tilt (~13° down) is from a secondary source,
  unconfirmed.

The VIO uses the state as priors (yaw, pitch/roll, velocity, h+tof).
A wrong velocity unit/frame would inject a 10× error into the motion
model and the dead reckoning. So: **calibrate first, integrate never
before.**

## The pass criterion (read this first)

The calibration produces TWO independent verdicts. Both must be
understood before the real test:

1. **Velocity unit + frame** — one (unit, frame) hypothesis must
   integrate to the known endpoint (error < ~0.15 m) and beat the
   runner-up by >20% (`velocity_discriminated: true`). This decides
   whether velocity priors may be used at all.
2. **YAW GATE — the real-test gate.** The 1-DoF yaw PnP needs the
   attitude prior accurate to ~1 deg (round-3 fragility finding: a
   (yaw+2, pitch−1, roll+1) deg prior offset fails s1/s20/s4 at
   81.6–90.4 cm joint error). The real test may proceed **only if
   the measured yaw sigma is ≤ 1.5 deg AND the doorway bias is
   ≤ 1.5 deg in magnitude**. The script prints `YAW GATE: PASS` or
   `YAW GATE: FAIL` in plain words. **On FAIL the real test must NOT
   proceed** — fix the yaw source first (re-calibrate, move the
   flight line away from metal, or add a third yaw measurement).
   A good velocity calibration does NOT excuse a bad yaw prior.

## The manoeuvre (report §6.9)

Fly on a flat floor, in a room with a tape measure on the floor and
a clear 1 m straight run. Record the state stream (djitellopy,
UDP 8890, ~10 Hz) with laptop timestamps, and the video (for the
yaw-accuracy and latency checks). The whole manoeuvre takes ~1 min.

1. **Hover** at 1.3 m for 5 s (tape-measured height). This is the
   height segment (h and tof cross-check) and the VPS baseline.
2. **Straight 1 m**: fly forward 1.0 m in a straight line along the
   tape, at a steady ~0.3 m/s, yaw held at 0. Mark the start and
   end times. This is the velocity-UNIT segment.
3. **Yaw mid-translation**: from the end of the 1 m run, yaw 90°
   (to the left) while translating 0.5 m sideways (along the new
   heading), at a steady speed. Mark the start and end times. This
   is the velocity-FRAME segment — it is the only part that
   discriminates body vs global (a straight flight cannot).
4. **Yaw step**: a fast 90° yaw in place (for the latency
   cross-correlation with the video yaw). Hold still 3 s after.
5. **Hover** at 1.3 m for 5 s (second height segment).

## Recording

Capture the state stream to `state.jsonl` — one JSON object per
line: `{"t": <laptop time, seconds>, "state": "<djitellopy state
string>"}`. The laptop time is the receipt time (the state has no
timestamp). A minimal recorder:

```python
import socket, time, json
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 8890))
with open("state.jsonl", "w") as f:
    while True:
        data, _ = sock.recvfrom(2048)
        f.write(json.dumps({"t": time.time(),
                            "state": data.decode().strip()}) + "\n")
        f.flush()
```

Write `ground_truth.json` in the same directory:

```json
{"segments": [
  {"kind": "hover", "height_m": 1.3, "t_start": 0.0, "t_end": 5.0},
  {"kind": "straight", "distance_m": 1.0, "t_start": 5.0, "t_end": 8.3,
   "direction_deg": 0.0},
  {"kind": "yaw_turn", "angle_deg": 90, "t_start": 8.3, "t_end": 11.6,
   "distance_m": 0.5, "direction_deg": 90.0},
  {"kind": "hover", "height_m": 1.3, "t_start": 15.0, "t_end": 20.0}],
 "yaw_gt_deg": {"t": [...], "yaw_deg": [...]}}
```

- `t_start`/`t_end` are laptop times (same clock as `state.jsonl`).
- `direction_deg` is the flight direction in the world frame
  (0 = +Z, 90 = +X, clockwise from above — the DJI yaw convention).
- `yaw_gt_deg` is optional but recommended: the yaw per frame from
  the VIO (or the flight plan). Without it the yaw-accuracy check
  is skipped.
- Optionally add `video_yaw.json` (`{"t": [...], "yaw_deg": [...]}`,
  the VIO's yaw per video frame) for the latency check.

## Running it

```bash
python3 tools/calibrate_tello.py <recording-dir> --out calibration.json
```

It prints a report and, if the velocity hypothesis is clearly
discriminated and the yaw RMS is < 5°, writes the calibration JSON.
It also prints the YAW GATE verdict (PASS/FAIL against σ ≤ 1.5°
and |doorway bias| ≤ 1.5°) — the gate is what decides whether the
real test may proceed, and it is independent of whether the JSON
is written: a good velocity calibration does not excuse a bad yaw
prior. **Review the report before using the JSON.** Copy the
constants into `server/tello_state.py` (`TelloCalibration`) only
after reviewing.

## What the script checks

| Check | Pass criterion |
|---|---|
| Velocity unit + frame | One (unit, frame) hypothesis integrates to the known endpoint (error < ~0.15 m) and the runner-up is >20% worse (`velocity_discriminated: true`). A 10× unit error is unambiguous; a frame error shows up as a wrong direction on the yaw-turn segment. |
| Yaw accuracy (σ) | RMS error vs the GT yaw **≤ 1.5°** — this is the yaw gate (see above). `jump_rejects` counts state steps >10°/0.1 s (the gate in `StatePrior`) — a large count means the yaw stream is noisy, not that the gate is wrong. |
| Doorway bias | Mean signed yaw error during the doorway turn minus during straight flight, **|bias| ≤ 1.5°** — the second half of the yaw gate. A magnetometer biased by metal near the doorway is invisible in the overall RMS, so it is measured separately. Note the estimator's own noise: the bias is a difference of two segment means, good to ~±0.3° when the yaw sigma is ~1°, noisier when the yaw itself is noisy. |
| Height | `h` (barometric) and `tof` (IR) mean error vs the tape height. `tof` reads 10 cm when out of range — that sentinel is filtered, and `tof_n` is the count of valid samples. Expect `tof` to be the more accurate of the two indoors. |
| VPS-blind | Any straight segment where the mean |v| is <25% of the expected speed is flagged — the VPS is blind there (featureless floor) and the velocity is unusable in that stretch. |
| Latency | Cross-correlation of the yaw RATE between the state stream and the video yaw; the lag at peak correlation is the state latency. Expect ~0.05–0.2 s (UDP + laptop timestamping). |

## Synthetic validation

```bash
python3 tools/calibrate_tello.py --self-test
```

Generates synthetic logs with a KNOWN unit/frame (dm/s + global,
and cm/s + body) and verifies the script recovers them. This passed
2026-11-23 (dm/s/global err 0.090 m, cm/s/body err 0.014 m,
yaw RMS ~0°). Re-run it after any change to the script.

**Yaw-sigma proof (added 2026-11-23):** the self-test also
generates synthetic logs with a KNOWN yaw noise sigma and a KNOWN
doorway bias, and verifies the script recovers them and judges the
yaw gate correctly. Measured 2026-11-23 (venv python, exit 0):

| Injected | Recovered | Gate | Verdict |
|---|---|---|---|
| yaw σ = 1.0°, bias 0° | RMS 0.89°, bias 0.54° | PASS | OK — a good magnetometer passes |
| yaw σ = 2.5°, bias 0° | RMS 2.31°, bias 1.28° | FAIL | OK — a noisy magnetometer fails on sigma |
| yaw σ = 1.0°, bias +3.0° | RMS 2.47°, bias 3.44° | FAIL | OK — a doorway-biased magnetometer fails on bias |

(The 1 deg packet quantisation adds ~0.29° in quadrature, so a
1.0° injected sigma recovers as ~0.89–1.04°. The bias estimate's
own noise is the difference of two segment means, ~0.26 × σ_yaw —
on seed 0 the 2.5°-noise case shows a 1.28° spurious bias with
zero injected, which is pure sampling noise of the two means,
not a script error. With σ ~1° the bias estimate is good to
~±0.3°, so the 1.5° bias threshold is meaningful on a real run.)

## Results (fill in after a real run)

| Field | Value | Notes |
|---|---|---|
| velocity_unit | UNCALIBRATED | |
| velocity_frame | UNCALIBRATED | |
| yaw_sigma_deg | — | |
| state_latency_s | — | |
| h vs tof | — | |
| VPS-blind stretches | — | |
| Date / drone | — | |
