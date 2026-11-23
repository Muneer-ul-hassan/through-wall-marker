# Depth-model demo on two real web photos — behaviour, NOT proof

Run 2026-11-23 ~18:40 PKT. YOLOv8n-pose (same detector as the pipeline) +
Depth Anything V2 Small `predict()` (relative depth only) on two
downloaded web photos. Side-by-side PNGs sit next to this file.

## The honest caveat first (read this before the numbers)

A web photo has **no known true distance** — nobody measured how far the
camera was from the person. So **no error number and no accuracy number
can be claimed from this demo**. It shows what the model *does* on a real
crouching / half-hidden person, not how *right* it is. The number-proof
still needs a synthetic crouching scenario with a known distance
(the harness, where ground truth exists).

The full `estimate()` (metric depth in metres) was NOT used: it needs a
drone pose and camera matrix, which a web photo does not have. Only
`predict()` ran — relative model units, colourised with TURBO
(red/yellow = nearer, blue/dark = farther; the model's output grows
as things get closer, disparity-like).

## Photo 1 — crouching_person.jpg (640x480)

Source: David Noble blog, person bent over reaching into a rock cavity.
Photo: `crouching_person.jpg`. Visual: `crouching_person_depth.png`
(original | colourised depth map | skeleton + bbox overlay).

- YOLO found the person, **15/17 keypoints visible**. The skeleton fits
  the crouched pose well.
- **The bbox collapse is visible:** bbox 165.6 x 162.0 px, aspect
  width/height **1.02 — nearly square**. A standing person at the same
  distance would have a tall narrow box (aspect ~0.3-0.4). Keypoint
  vertical span is only 113.6 px. Bbox-height depth would read this
  person as much farther away than they are — this is exactly the case
  the depth-model voter is built for.
- Depth map: the person is a clear yellow-green blob, separated from
  the blue rock face behind them (nearer than the rock) and from the
  red foreground dirt (nearest). The model does pick the crouching
  person out as its own shape.
- Relative numbers (model units, NOT metres): person median 2.126,
  whole-frame background median 2.308, ratio 0.92. Do NOT read this as
  "the person is farther than the background" — the background median
  mixes near foreground dirt with the far rock, so the ratio is not a
  clean person-vs-what-is-behind-them number. The picture is the
  evidence here, not the ratio.
- Depth inference: **4603.9 ms** (see the timing note below).

## Photo 2 — peeking_over_wall.jpg (1695x2560)

Source: 46graus photo site, woman peering over a wall, only head and
one hand visible. Photo: `peeking_over_wall.jpg`.
Visual: `peeking_over_wall_depth.png`.

- YOLO found the person but only **8/17 keypoints visible** — and the
  skeleton is partly **hallucinated on the wall**: green lines cross
  the wall to keypoints the detector placed on the occluder (the hidden
  lower body). The bbox is also inflated to nearly the full image
  width (1582.5 px wide, aspect 0.89). On a half-hidden person, the
  detector still returns a skeleton — but part of it is guesswork on
  the thing hiding the person. Worth knowing for the real system.
- Depth map: her head is a clean blue dome, clearly separated from the
  dark far background; the wall itself reads nearest (yellow/red).
  Relative numbers: person median 2.177, background median 0.45,
  ratio **4.84** — the model cleanly separates her from the background.
- Depth inference: **3999.9 ms** (see the timing note below).

## Timing note

4.6 s and 4.0 s per frame here is **much slower than the ~410 ms**
measured in the earlier probe (tests/mono_depth_probe.py). These runs
happened while the UI-server agent was running the full test suite on
the same shared 1.5-CPU VM, so the times are under heavy CPU
contention — they are NOT the model's normal speed. Do not quote
4 seconds as the model's latency.

## Files

- `crouching_person.jpg`, `peeking_over_wall.jpg` — the downloaded photos
- `crouching_person_depth.png`, `peeking_over_wall_depth.png` — side-by-side visuals
- `run_depth_demo.py` — the script that made them
- `depth_demo_results.json` — the measured numbers above, machine-readable

## What this does and does not prove

- DOES show: the model separates a crouching person from the rock
  behind them, and a half-hidden person's head from a far background;
  the bbox collapse on a crouching person is real and visible; YOLO
  hallucinates keypoints on an occluder.
- Does NOT show: any distance in metres, any error in centimetres, or
  whether the voter would flip a decision — that still needs the
  synthetic crouching scenario with known ground truth.
