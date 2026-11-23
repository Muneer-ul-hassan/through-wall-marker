"""YOLOv8n-pose wrapper (ultralytics). Person class (COCO 0) only.

First use downloads ``yolov8n-pose.pt`` (~6 MB) from the ultralytics CDN;
call :meth:`PoseDetector.warmup` once up front so that happens before the
pipeline starts, with a clear log line.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class PoseDetector:
    """Detect people + COCO 17-keypoint poses in BGR frames."""

    def __init__(self, weights="yolov8n-pose.pt", conf=0.5, device="cpu"):
        self.weights = weights
        self.conf = float(conf)
        self.device = device
        self._model = None
        self._warned = False

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # imported lazily: heavy + downloads
            log.info("Loading YOLO pose weights: %s (device=%s)",
                     self.weights, self.device)
            self._model = YOLO(self.weights)
            log.info("YOLO pose weights ready: %s", self.weights)
        return self._model

    def warmup(self):
        """Run one dummy inference so weights download happens once, up front."""
        model = self._load()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, conf=self.conf, classes=[0], device=self.device,
              verbose=False, imgsz=640)
        log.info("PoseDetector warmup done (weights=%s)", self.weights)

    def detect(self, frame_bgr):
        """Detect people in a BGR frame.

        Returns a list of dicts, one per person (COCO class 0 only):
        ``{"bbox_xyxy": (x1, y1, x2, y2), "conf": float,
        "keypoints": (17, 2) float array in pixels,
        "kpt_conf": (17,) float array}``.
        Empty list when nobody is detected. Never raises on odd frames:
        an inference failure is logged once and yields ``[]``.
        """
        if frame_bgr is None:
            return []
        try:
            model = self._load()
            results = model(frame_bgr, conf=self.conf, classes=[0],
                            device=self.device, verbose=False, imgsz=640)
        except Exception as exc:  # noqa: BLE001 - must not raise on odd frames
            if not self._warned:
                log.warning("Pose detection failed (%s); returning no "
                            "detections for this frame", exc)
                self._warned = True
            return []
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            kpts = result.keypoints
            xy = kpts.xy.cpu().numpy() if kpts is not None else None
            kc = kpts.conf.cpu().numpy() if kpts is not None else None
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                if xy is None or kc is None:
                    keypoints = np.zeros((17, 2), dtype=float)
                    kpt_conf = np.zeros((17,), dtype=float)
                else:
                    keypoints = np.asarray(xy[i], dtype=float)
                    kpt_conf = np.asarray(kc[i], dtype=float)
                detections.append({
                    "bbox_xyxy": (float(xyxy[i][0]), float(xyxy[i][1]),
                                  float(xyxy[i][2]), float(xyxy[i][3])),
                    "conf": float(confs[i]),
                    "keypoints": keypoints,
                    "kpt_conf": kpt_conf,
                })
        return detections
