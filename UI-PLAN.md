# Wall-Marker Easy UI Plan — real data only

Researched and delivered to Muneer 2026-11-23 ~09:57 UTC (14:57 PKT), in
the main chat, on his order: "research kr le easy ui chahiya and noo demo
ya fake data". This file preserves that plan. It is a PLAN ONLY — nothing
here is built yet. Build is deferred until the bridge (VIO) work finishes,
because that work is editing the same files.

## Rules (his words)

- Easy UI.
- Real data only. No demo mode, no fake data, no placeholder markers.
  If there is no real detection, the screen shows that plainly instead
  of filling it in.

## What was found

- The iPhone viewer (`phone/viewer.html`) is already clean: no demo,
  fake, or placeholder wording anywhere in it.
- The one real-only violation is the server: `server/main.py` still ships
  a demo mode through the same viewer — virtual mode with a synthetic
  person. That must be split out of the product server into a test-only
  entry point. The product server must refuse to run without a real
  drone source.
- There is no laptop operator UI at all right now.

## iPhone viewer — one glance, one thumb, real only

- Camera fills the screen. One status pill at the top:
  - green "Person found"
  - amber "Stale"
  - grey "Watching"
  - red "No connection"
- The skeleton draws over the wall (skeleton, not a dot — his choice).
- Distance sits in one fixed pill at the bottom, always in the same
  place: `≈ 4.2 m ± 0.5 m`.
- One button only — **Start / Stop**, thumb-reachable. No settings,
  no menus.

### Every state, exact wording

- No person found: empty screen with a grey "Watching" pill — the empty
  camera *is* the message.
- Tracking lost: hide everything.
- Stale: amber, never lasts more than 0.5 s.
- Up to 3 people, each in its own colour, nearest first.
- Drone count shows only when there is more than one drone.

### Uncertainty is honest

- The `≈` and `±` stay.
- Rounding follows the uncertainty: one decimal under 1 m, whole metres
  above.
- The skeleton itself fades as uncertainty grows.
- No distance without a skeleton, and no skeleton without a distance.

## Laptop operator page — new (`/operator`)

Three big blocks:

- BRIDGE (OK / LOST)
- DRONE (LIVE / NO VIDEO)
- PERSON (FOUND / NONE)

Plus a 3-step checklist that ticks itself from real state, and one
Start / Stop button. No video on that page either.

## Open question (left open by the plan)

The plan assumes the phone is held still, portrait, one-handed. If he
wants to walk around pointing at different walls, the web viewer cannot
do that — that needs a native ARKit app. Ask him before building for
movement.

## Source

- Main-chat research delivery 2026-11-23T09:57:43Z (assistant message
  `assistant-msg-089a5168-de12-9ede-265e-7cd9ef691342`).
- Standing rules it implements: skeleton not dot; phone's own camera
  only; no drone feed on the phone (2026-11-22, main chat).
