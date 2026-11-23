#!/usr/bin/env python3
"""Probe: Depth Anything V2 Small on wall-marker scenario frames.

Measures, BEFORE any pipeline wiring:
* CPU inference time at several input sizes.
* The floor-plane scale anchor: per frame, fit true_z = a * model + b
  from floor pixels (true z from the ray / y=0 plane intersection using
  the known drone pose), robustly (iterative MAD rejection).
* The person depth from that anchor (median model value at the visible
  keypoint pixels) vs the ground-truth foot depth.

Run:  HF_HOME=$HOME/.cache/huggingface \
      ~/workspace/tello-search/.venv/bin/python tests/mono_depth_probe.py \
      [--scenarios s1_basic s4_dim ...] [--size 336] [--frames 30,40,50]
Ground / virtual only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

import torch

from server.detector import PoseDetector
from server.geometry import DRONE_K, project_point
from server.mono_depth import MODEL_ID, MonoDepthModel
from tests import scenarios

WEIGHTS = str(WALL_MARKER_DIR / "yolov8n-pose.pt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=[
        "s1_basic", "s4_dim", "s13_dim_noise", "s14_glare",
        "s15_occluded", "s18_moving", "s11_large_room_far"])
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--frames", default="30,40,50")
    ap.add_argument("--bench-sizes", default="256,336,518")
    args = ap.parse_args(argv)
    torch.set_num_threads(2)
    frame_ids = [int(x) for x in args.frames.split(",")]

    mono = MonoDepthModel(input_size=args.size, threads=2)
    mono.load()
    print(f"model: {MODEL_ID} (transformers, fp32, CPU, "
          f"threads={torch.get_num_threads()})")

    detector = PoseDetector(weights=WEIGHTS)
    detector.warmup()

    # --- inference-time benchmark on one real scenario frame ---
    frames0 = scenarios.get_frames("s1_basic", detector=detector,
                                   drone_id="drone1")
    bench_frame = frames0[30][0]
    print("\n== CPU inference benchmark (single frame, s1 f30) ==")
    for size in [int(x) for x in args.bench_sizes.split(",")]:
        m = MonoDepthModel(input_size=size, threads=2)
        m.load()
        m.predict(bench_frame)  # warm
        mss = [m.predict(bench_frame)[1] for _ in range(5)]
        print(f"  input {size}x{size * 3 // 4}: mean {np.mean(mss):.0f} ms "
              f"(min {min(mss):.0f}, max {max(mss):.0f}) over 5 runs")

    print(f"\n== floor-anchor probe (input {args.size}) ==")
    for name in args.scenarios:
        frames = scenarios.get_frames(name, detector=detector,
                                      drone_id="drone1")
        errs, rows = [], []
        for frame, pose, _tv, meta in frames:
            if meta["frame"] not in frame_ids or not meta["in_view"]:
                continue
            dets = detector.detect(frame)
            if not dets:
                rows.append((meta["frame"], "NO-DET", None, None))
                continue
            det = max(dets, key=lambda d: d["conf"])
            res = mono.estimate(frame, pose, DRONE_K, det["keypoints"],
                                det["kpt_conf"], det["bbox_xyxy"])
            true_depth = None
            if meta["person_world"] is not None:
                p = project_point(DRONE_K, pose[0], pose[1],
                                  np.array(meta["person_world"], float))
                true_depth = None if p is None else p[2]
            if res is None or true_depth is None:
                rows.append((meta["frame"], "NO-VOTE", None, None))
                continue
            err = res["depth_m"] - true_depth
            errs.append(err)
            rows.append((meta["frame"],
                         f"a={res['scale_a']:.3f} b={res['scale_b']:+.2f} "
                         f"rms={res['floor_rms_m']:.2f}m n={res['floor_n']} "
                         f"corr={res['floor_corr']:.2f} "
                         f"model_person={res['depth_m']:.2f}m "
                         f"sigma={res['sigma_m']:.2f}m "
                         f"true={true_depth:.2f}m "
                         f"err={err * 100:+.0f}cm",
                         res["ms"], err))
        for fr, txt, ms, _e in rows:
            print(f"  {name} f{fr:02d} {txt}"
                  + ("" if ms is None else f" [{ms:.0f} ms]"))
        if errs:
            errs = np.array(errs)
            print(f"  {name}: model-depth err mean {np.mean(errs) * 100:+.0f} cm, "
                  f"mean|err| {np.mean(np.abs(errs)) * 100:.0f} cm, "
                  f"worst {np.max(np.abs(errs)) * 100:.0f} cm (n={len(errs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
