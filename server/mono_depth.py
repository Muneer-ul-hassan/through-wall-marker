"""Monocular depth model — the THIRD depth estimator (Depth Anything V2 Small).

Why this model (measured 2026-11-23, see tests/mono_depth_probe.py and
tests/TEST-REPORT.md "Depth-model tests"):
* Depth Anything V2 Small: 24.8M params, 95 MB weights (Apache-2.0 —
  the only DA-V2 checkpoint under a permissive licence), 60 ms on a
  V100 per the paper. On this laptop (CPU, 1.5 shared cores) it runs
  ~400 ms at 256x192 input with 2 threads.
* ZoeDepth (the metric alternative): ~1.37 GB fp32 weights
  (Intel/zoedepth-nyu-kitti, DPT-Large backbone) — ~14x larger than
  DA-V2-Small, larger than the ~1.2 GB free RAM on this machine, so it
  was excluded on measured download size / RAM before any inference
  run. Its metric output would have avoided the scale step, but a
  model that cannot be loaded cannot vote.
* DA-V2-Small outputs RELATIVE depth only, so its metric scale comes
  from a MEASURED anchor (below) — never a hard-coded scale constant.

Metric scale — the floor-plane anchor (measured per frame):
The drone camera height above the floor is known from its pose
(C_y, floor y = 0 — the same known-height fact the VIO scale method
uses). For floor pixels in the lower part of the image (excluding the
person bbox), the true camera-frame depth follows from the ray /
y = 0 plane intersection using the known drone pose. A robust linear
fit ``true_z = a * model + b`` over those pixels gives the scale for
THIS frame (``a`` is negative on this model: its output grows as
things get closer — disparity-like — the fit measures the sign, it is
not assumed). The fit is accepted only if it has >= 20 floor samples,
a residual RMS <= 0.35 m, and |Pearson r| >= 0.8 between model values
and true depths; otherwise there is NO model vote that frame. The
person depth is the fitted value at the median model value over the
visible keypoint pixels (keypoints are on the person; a bbox median
would mix in the wall behind them).

Sigma: sqrt(floor_fit_rms^2 + (|a| * 1.4826 * MAD of the keypoint
model values)^2), inflated for cache age (see server/main.py, which
holds the result in a short cache): sigma_age = 0.2 m/s * age_s,
added in quadrature. A model vote is therefore honest about how
stale it is.

Latency (the user's condition): the model NEVER runs on the marker
path. ``server/main.py`` runs it in a background thread, at most
every 10th detection frame (1 Hz at the 10 fps harness rate), on a
256x192 copy; the pipeline only ever reads the latest completed
result (age-gated to <= 2.0 s). If the worker is busy, the frame is
skipped, not queued. The model's torch thread count is left at the
process default so the YOLO detector's threading is identical with
the voter on or off. Measured latency / FPS before and after are in
tests/TEST-REPORT.md.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .geometry import unproject_ray

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_INPUT_SIZE = 256  # width; height = 3/4 * width (4:3 drone frame)

# Acceptance gates for a floor fit (measured, see module docstring).
MIN_FLOOR_SAMPLES = 20
MAX_FLOOR_RMS_M = 0.35
MIN_FLOOR_CORR = 0.8
MIN_VISIBLE_KEYPOINTS = 3


class MonoDepthModel:
    """Depth Anything V2 Small wrapper (lazy load, CPU only).

    ``load()`` downloads the ~95 MB weights from Hugging Face on first
    use (HF_HOME cache under home). ``estimate()`` returns a dict with
    ``depth_m`` / ``sigma_m`` (metric, floor-anchored) or ``None`` when
    the frame cannot support a vote. Thread-safe: ``estimate`` holds
    a lock so a background worker and a caller never run inference
    concurrently.
    """

    def __init__(self, model_id=MODEL_ID, input_size=DEFAULT_INPUT_SIZE,
                 threads=None):
        self.model_id = model_id
        self.input_size = int(input_size)
        # ``threads`` is None by default: torch's process-global thread
        # count is left untouched, so the YOLO detector's threading is
        # identical with the voter on or off (the before/after FPS
        # comparison in tests/run_tests.py is then clean). Pass an
        # explicit int only in a benchmark context — it calls
        # torch.set_num_threads, which is process-global and affects
        # every torch user in the process, YOLO included.
        self.threads = threads
        self._model = None
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_votes = 0
        self.ms_values = []

    def load(self):
        """Load weights (once). Raises on failure — the caller decides
        whether a missing model disables the voter or fails the run."""
        if self._model is not None:
            return self
        import torch
        if self.threads is not None:
            torch.set_num_threads(self.threads)
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)
        # The processor is loaded for its config only; inference feeds
        # normalised tensors directly (see predict) so the input size
        # stays under our control.
        AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id, dtype=torch.float32)
        model.eval()
        self._model = model
        self._torch = torch
        return self

    @property
    def loaded(self):
        return self._model is not None

    def predict(self, frame_bgr):
        """Run the model on a downscaled copy.

        Returns ``(depth_map (H', W') float64, ms)``. The map is in
        model units (relative depth — see module docstring); callers
        must anchor it before treating it as metres.
        """
        import cv2
        if self._model is None:
            self.load()
        w = self.input_size
        h = self.input_size * 3 // 4
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = self._torch.from_numpy(
            ((rgb - mean) / std).transpose(2, 0, 1)[None])
        t0 = time.perf_counter()
        with self._torch.no_grad():
            out = self._model(pixel_values=x).predicted_depth
        ms = (time.perf_counter() - t0) * 1000.0
        return out[0].numpy().astype(np.float64), ms

    # -- floor anchor + person depth -------------------------------------

    def floor_fit(self, depth_map, drone_pose, K, bbox_xyxy, img_hw):
        """Robust fit true_z_cam = a * model + b from floor pixels.

        Floor samples: grid over the lower 38% of the image, excluding
        the person bbox expanded by 30 px. True z for a floor pixel:
        intersect the world ray with the y = 0 plane, take the
        camera-frame z. Returns ``(a, b, rms_m, n, corr)`` or ``None``.
        """
        R, C = np.asarray(drone_pose[0], float), np.asarray(drone_pose[1], float)
        H, W = img_hw
        dh, dw = depth_map.shape
        sx, sy = dw / W, dh / H
        pts_m, pts_z = [], []
        for v in range(int(H * 0.62), H, 12):
            for u in range(0, W, 12):
                if bbox_xyxy is not None:
                    x1, y1, x2, y2 = bbox_xyxy
                    if x1 - 30 <= u <= x2 + 30 and y1 - 30 <= v <= y2 + 30:
                        continue
                ray_cam = unproject_ray(K, float(u), float(v))
                ray_w = R.T @ ray_cam
                if ray_w[1] >= -1e-6:  # ray does not hit the floor
                    continue
                s = -C[1] / ray_w[1]
                P = C + s * ray_w
                z_cam = float((R @ (P - C))[2])
                if z_cam <= 0.2 or z_cam > 15.0:
                    continue
                mv = float(depth_map[int(v * sy), int(u * sx)])
                if not np.isfinite(mv):
                    continue
                pts_m.append(mv)
                pts_z.append(z_cam)
        if len(pts_m) < MIN_FLOOR_SAMPLES:
            return None
        m = np.array(pts_m)
        z = np.array(pts_z)
        keep = np.ones(len(m), dtype=bool)
        a = b = 0.0
        for _ in range(3):
            if keep.sum() < MIN_FLOOR_SAMPLES:
                return None
            a, b = np.polyfit(m[keep], z[keep], 1)
            resid = z - (a * m + b)
            med = np.median(resid[keep])
            mad = np.median(np.abs(resid[keep] - med))
            thr = max(3.0 * 1.4826 * mad, 0.05)
            keep = np.abs(resid - med) <= thr
        if keep.sum() < MIN_FLOOR_SAMPLES:
            return None
        resid = z[keep] - (a * m[keep] + b)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        corr = float(np.corrcoef(m[keep], z[keep])[0, 1])
        if rms > MAX_FLOOR_RMS_M or abs(corr) < MIN_FLOOR_CORR:
            return None
        return float(a), float(b), rms, int(keep.sum()), corr

    def person_model_value(self, depth_map, keypoints, kpt_conf, img_hw):
        """(median, MAD) of the model values at visible keypoint pixels."""
        H, W = img_hw
        dh, dw = depth_map.shape
        vals = []
        for i in range(17):
            if float(kpt_conf[i]) < 0.3:
                continue
            u, v = float(keypoints[i, 0]), float(keypoints[i, 1])
            if 0 <= u < W and 0 <= v < H:
                vals.append(float(depth_map[int(v * dh / H), int(u * dw / W)]))
        if len(vals) < MIN_VISIBLE_KEYPOINTS:
            return None
        vals = np.array(vals)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        return med, mad

    def estimate(self, frame_bgr, drone_pose, K, keypoints, kpt_conf,
                 bbox_xyxy):
        """Floor-anchored metric person depth for one frame.

        Returns ``{"depth_m", "sigma_m", "scale_a", "scale_b",
        "floor_rms_m", "floor_n", "floor_corr", "ms"}`` or ``None``
        when the frame cannot support a vote (no floor fit, too few
        visible keypoints, or an implausible depth). The returned
        sigma does NOT include cache age — the caller (server/main.py)
        inflates it by age before voting.
        """
        with self._lock:
            self.n_calls += 1
            t0 = time.perf_counter()
            try:
                depth_map, ms = self.predict(frame_bgr)
            except Exception:
                return None
            H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
            fit = self.floor_fit(depth_map, drone_pose, K, bbox_xyxy,
                                 (H, W))
            if fit is None:
                return None
            a, b, rms, n, corr = fit
            pv = self.person_model_value(depth_map, keypoints, kpt_conf,
                                         (H, W))
            if pv is None:
                return None
            med, mad = pv
            depth = a * med + b
            if not np.isfinite(depth) or depth <= 0.5 or depth > 15.0:
                return None
            # Sigma floor (measured): the floor-fit residual + keypoint
            # spread alone UNDERSTATES the model's real error — the
            # probe (tests/mono_depth_probe.py, 21 frames over 7
            # scenarios) measured mean |err| 5-47 cm in clean scenes
            # and 57-71 cm occluded, while the fit-only sigma came out
            # ~0.15-0.25 m. The floor 0.15 * depth (~0.38 m at 2.5 m)
            # brackets the probe's clean-scene spread, so the vote's
            # stated uncertainty is honest. (The sigma no longer
            # weights a fusion — server/depth.py is pure arbitration
            # since the fusion was measured to hurt in s15_occluded —
            # but the vote is still recorded with an honest sigma.)
            sigma = max(
                float(np.sqrt(rms ** 2 + (abs(a) * 1.4826 * mad) ** 2)),
                0.15 * depth)
            self.n_votes += 1
            self.ms_values.append(ms)
            return {"depth_m": float(depth), "sigma_m": sigma,
                    "scale_a": a, "scale_b": b, "floor_rms_m": rms,
                    "floor_n": n, "floor_corr": corr, "ms": ms,
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0}

    def stats(self):
        """Measured call/vote counts + inference latency (ms)."""
        mss = self.ms_values
        return {"n_calls": self.n_calls, "n_votes": self.n_votes,
                "ms_mean": float(np.mean(mss)) if mss else None,
                "ms_min": float(np.min(mss)) if mss else None,
                "ms_max": float(np.max(mss)) if mss else None}
