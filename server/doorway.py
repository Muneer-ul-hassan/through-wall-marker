"""Doorway detection for the autopilot scan phase.

A doorway is a large vertical rectangle: two strong near-vertical
jambs of similar length, spanning a large fraction of frame height,
separated by a plausible doorway width (~0.8-1 m wide vs ~2 m high).
The autopilot calls :meth:`DoorwayDetector.detect` at most ~2-3 fps
/ on demand while scanning, never every frame — but ``detect`` is
kept cheap anyway: downscale to ~480 px wide, grayscale, Canny,
HoughLinesP, then cluster + pair the near-vertical segments. No ML
model, no network, no per-frame heavy work.

Standalone: imports only ``cv2``, ``numpy`` and ``dataclasses`` —
it never imports the live-loop module or any other project module, so
the autopilot (or a test) can use it without pulling in the live loop.

Returns ``None`` honestly when no doorway-shaped pair is visible
(plain wall, poster, window, single jamb, clutter). Confidence is
0..1 from jamb length / strength / symmetry / width plausibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- tuning constants (all in downscaled-frame coordinates) --------
TARGET_WIDTH = 480          # downscale width; keeps Canny+Hough cheap
MIN_JAMB_FRAC = 0.50        # a jamb must span >= 50% of frame height
MIN_GAP_FRAC = 0.08         # doorway width >= 8% of frame width
MAX_GAP_FRAC = 0.55         # doorway width <= 55% of frame width
VERT_TOL = 0.12             # |dx| <= 12% of segment length => vertical
CLUSTER_TOL = 15            # px (downscaled) to merge collinear segments
MIN_OVERLAP_FRAC = 0.60     # paired jambs must overlap vertically >= 60%


@dataclass
class Doorway:
    """A detected doorway, in ORIGINAL frame pixel coordinates."""

    center_x: float   # doorway centre, x in pixels (original frame)
    width: float      # doorway width in pixels (right_x - left_x)
    left_x: float     # left jamb x in pixels (original frame)
    right_x: float    # right jamb x in pixels (original frame)
    confidence: float  # 0..1, from line strength / length / symmetry


@dataclass
class _Jamb:
    x: float          # mean x of merged segments (downscaled coords)
    y0: float         # top of merged extent
    y1: float         # bottom of merged extent
    length: float     # total segment length contributing to the jamb


class DoorwayDetector:
    """CPU-light doorway detector: Canny + HoughLinesP + jamb pairing."""

    def __init__(
        self,
        target_width: int = TARGET_WIDTH,
        min_jamb_frac: float = MIN_JAMB_FRAC,
    ) -> None:
        self.target_width = target_width
        self.min_jamb_frac = min_jamb_frac

    # -- public API -------------------------------------------------
    def detect(self, frame: np.ndarray) -> Doorway | None:
        """Detect a doorway in a BGR frame; ``None`` if none visible.

        ``frame`` is a BGR ``numpy`` array, typically 960x720 from
        the Tello forward camera. Any size works; tiny frames
        (< 64 px on a side) return ``None`` instead of crashing.
        """
        if frame is None or frame.ndim != 3 or frame.shape[0] < 64 \
                or frame.shape[1] < 64:
            return None

        h, w = frame.shape[:2]
        scale = self.target_width / float(w)
        dw = self.target_width
        dh = max(8, int(round(h * scale)))
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=int(0.30 * dh),
            maxLineGap=20,
        )
        if lines is None:
            return None

        jambs = self._collect_jambs(lines, dw, dh)
        if len(jambs) < 2:
            return None

        best = self._best_pair(jambs, dw, dh)
        if best is None:
            return None
        left, right, confidence = best

        # Back to original-frame coordinates.
        inv = 1.0 / scale
        return Doorway(
            center_x=(left.x + right.x) * 0.5 * inv,
            width=(right.x - left.x) * inv,
            left_x=left.x * inv,
            right_x=right.x * inv,
            confidence=confidence,
        )

    # -- internals --------------------------------------------------
    def _collect_jambs(self, lines, dw: int, dh: int) -> list[_Jamb]:
        """Keep near-vertical, tall segments; merge collinear ones."""
        min_len = self.min_jamb_frac * dh
        segs = []  # (x_mid, y0, y1, length)
        # OpenCV 5 returns (N, 4); older versions return (N, 1, 4).
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue
            if dx > VERT_TOL * length:
                continue  # not near-vertical
            segs.append((0.5 * (x1 + x2), min(y1, y2), max(y1, y2), length))
        if not segs:
            return []

        # Cluster by x: segments whose x-mid is within CLUSTER_TOL of
        # the cluster mean and whose y-ranges overlap / touch belong
        # to the same jamb (one physical edge, split by Hough).
        segs.sort(key=lambda s: s[0])
        clusters: list[list[tuple]] = []
        for seg in segs:
            placed = False
            for cl in clusters:
                mean_x = sum(s[0] for s in cl) / len(cl)
                if abs(seg[0] - mean_x) <= CLUSTER_TOL:
                    cl.append(seg)
                    placed = True
                    break
            if not placed:
                clusters.append([seg])

        jambs = []
        for cl in clusters:
            total = sum(s[3] for s in cl)
            if total < min_len:
                continue
            # Weighted x by segment length; extent = union of y-ranges.
            x = sum(s[0] * s[3] for s in cl) / total
            y0 = min(s[1] for s in cl)
            y1 = max(s[2] for s in cl)
            jambs.append(_Jamb(x=x, y0=float(y0), y1=float(y1),
                               length=float(total)))
        jambs.sort(key=lambda j: j.x)
        return jambs

    def _best_pair(self, jambs, dw: int, dh: int):
        """Best (left, right, confidence) jamb pair, or None."""
        best = None  # (score, left, right, confidence)
        for i in range(len(jambs)):
            for j in range(i + 1, len(jambs)):
                left, right = jambs[i], jambs[j]
                gap = right.x - left.x
                if gap < MIN_GAP_FRAC * dw or gap > MAX_GAP_FRAC * dw:
                    continue
                # Vertical overlap must be large: both jambs span the
                # same tall region (a real doorway), not two unrelated
                # tall edges at different heights.
                overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
                shorter = min(left.y1 - left.y0, right.y1 - right.y0)
                if shorter <= 0 or overlap < MIN_OVERLAP_FRAC * shorter:
                    continue
                # Both jambs must individually be tall.
                if min(left.y1 - left.y0, right.y1 - right.y0) \
                        < self.min_jamb_frac * dh:
                    continue

                # Score: length (longer = stronger), symmetry of the
                # two jambs, and gap plausibility (a real doorway is
                # ~0.4-0.5x as wide as it is tall in the image).
                len_frac = min(1.0, min(left.length, right.length) / dh)
                sym = 1.0 - abs(left.length - right.length) \
                    / max(left.length, right.length)
                height = max(left.y1 - left.y0, right.y1 - right.y0)
                ratio = gap / max(height, 1.0)
                # Ideal width/height ratio ~0.45; penalise far from it.
                ratio_score = max(0.0, 1.0 - abs(ratio - 0.45) / 0.45)
                score = 0.5 * len_frac + 0.25 * sym + 0.25 * ratio_score
                confidence = float(min(1.0, max(0.0, score)))
                if best is None or score > best[0]:
                    best = (score, left, right, confidence)
        if best is None:
            return None
        return best[1], best[2], best[3]
