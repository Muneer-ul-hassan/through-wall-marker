"""Depth-model demo on two real web photos (NOT a proof).

Runs YOLOv8n-pose (same detector as the pipeline) + Depth Anything V2
Small predict() (relative depth only — estimate() needs a drone pose
and camera matrix, which a web photo does not have) and saves
side-by-side PNGs: original | colourised depth map | skeleton overlay.

No ground truth exists for these photos, so NO distance or accuracy
number is claimed here. This shows behaviour only.
"""

import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/tello-search/wall-marker")

from server.mono_depth import MonoDepthModel  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BASE = "/home/hatch/workspace/tello-search/wall-marker"
WEIGHTS = f"{BASE}/yolov8n-pose.pt"
OUT = f"{BASE}/tests/web_images"

# COCO 17-keypoint skeleton edges.
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(img, kpts, conf):
    out = img.copy()
    for a, b in EDGES:
        if conf[a] >= 0.3 and conf[b] >= 0.3:
            cv2.line(out, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)),
                     (0, 255, 0), 2, cv2.LINE_AA)
    for i in range(17):
        if conf[i] >= 0.3:
            cv2.circle(out, tuple(kpts[i].astype(int)), 4, (0, 0, 255), -1)
    return out


def colourise_depth(depth_map):
    lo, hi = np.percentile(depth_map, [1, 99])
    norm = np.clip((depth_map - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)


def main():
    yolo = YOLO(WEIGHTS)
    model = MonoDepthModel()
    t0 = time.perf_counter()
    model.load()
    print(f"model load: {time.perf_counter() - t0:.1f} s")

    results = {}
    for name in ("crouching_person", "peeking_over_wall"):
        path = f"{OUT}/{name}.jpg"
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        r = yolo.predict(frame, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            print(f"{name}: NO PERSON DETECTED")
            results[name] = {"detected": False}
            continue
        idx = int(np.argmax(r.keypoints.conf.cpu().numpy().mean(axis=1)))
        kpts = r.keypoints.xy[idx].cpu().numpy().astype(float)
        conf = r.keypoints.conf[idx].cpu().numpy().astype(float)
        x1, y1, x2, y2 = r.boxes.xyxy[idx].cpu().numpy().astype(float)
        n_vis = int((conf >= 0.3).sum())
        depth_map, ms = model.predict(frame)

        # Relative depth: median model value at visible keypoints vs
        # median over the whole frame outside the bbox (background).
        dh, dw = depth_map.shape
        vals = []
        for i in range(17):
            if conf[i] >= 0.3:
                u, v = kpts[i]
                vals.append(float(depth_map[int(v * dh / h), int(u * dw / w)]))
        mask = np.ones((dh, dw), dtype=bool)
        bx1, by1 = int(x1 * dw / w), int(y1 * dh / h)
        bx2, by2 = int(x2 * dw / w), int(y2 * dh / h)
        mask[max(by1, 0):by2, max(bx1, 0):bx2] = False
        bg_med = float(np.median(depth_map[mask]))
        person_med = float(np.median(vals)) if vals else None

        # Panels at a common height.
        ph = 640
        def resize(img):
            s = ph / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * s), ph))
        p1 = resize(frame)
        depth_big = cv2.resize(colourise_depth(depth_map), (w, h))
        p2 = resize(depth_big)
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 0, 0), 2)
        overlay = draw_skeleton(overlay, kpts, conf)
        p3 = resize(overlay)
        side = np.hstack([p1, p2, p3])
        out_png = f"{OUT}/{name}_depth.png"
        cv2.imwrite(out_png, side)

        info = {
            "detected": True,
            "image_hw": [h, w],
            "bbox_xyxy_px": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "bbox_w_px": round(float(x2 - x1), 1),
            "bbox_h_px": round(float(y2 - y1), 1),
            "bbox_h_over_img_h": round(float((y2 - y1) / h), 3),
            "bbox_aspect_w_over_h": round(float((x2 - x1) / max(y2 - y1, 1)), 2),
            "visible_keypoints_ge_0.3": n_vis,
            "keypoint_y_span_px": round(
                float(kpts[conf >= 0.3, 1].max() - kpts[conf >= 0.3, 1].min()), 1)
            if n_vis else None,
            "depth_inference_ms": round(float(ms), 1),
            "person_median_model_value": None if person_med is None else round(person_med, 3),
            "background_median_model_value": round(bg_med, 3),
            "person_vs_background_ratio": None if person_med is None else round(person_med / bg_med, 2),
            "png": out_png,
        }
        results[name] = info
        print(json.dumps({name: info}, indent=2))

    with open(f"{OUT}/depth_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
