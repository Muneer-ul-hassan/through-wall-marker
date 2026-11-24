"""Saved-map persistence + relocalisation for the VIO (``server/vio.py``).

Problem (2026-11-25): the laptop server's monocular VIO builds a
sparse 3D map as it flies, but the map lives only in RAM. When
tracking is lost (camera faces a plain blanket/wall -> ``lk-empty``,
zero tracked points) the bridge goes invalid and stays lost: the
in-RAM relocalisation in ``MonocularVO._try_relocalise`` can only
re-observe the last good frame, and a from-scratch re-initialisation
is forbidden — it would silently teleport the drone into a new
arbitrary world frame. So a loss in a mapped room was the end of the
run.

This module fixes that by persisting the map:

* ``save_map(path, vio)`` — after a good flight (or at run end),
  serialise the VIO's sparse map (3D points + ORB descriptors +
  the keyframe poses needed to seed PnP) to ``maps/latest.npz``
  under the repo root (``DEFAULT_MAP_PATH``). Cheap: a few hundred
  points, tens of KB, ``np.savez``.
* ``load_map(path)`` — load it back as a ``MapData``.
* ``try_relocalise(vio, frame, map_data)`` — on tracking loss only
  (never every frame: ORB + RANSAC PnP is ~100 ms), match the
  current frame's ORB features against the saved map's descriptors,
  solve PnP through the VIO's own ``_solve_pnp`` (yaw1dof chain +
  full fallback, same camera model, same acceptance gate), and on
  success restore the pose via ``MonocularVO.adopt_saved_map_pose``
  so tracking resumes in the SAVED map's world frame. Returns
  True/False and never raises.

Frame convention: saved points and poses are in the VIO's INTERNAL
(OpenCV Y-down) world frame — the frame ``_solve_pnp`` works in.
A saved map is therefore only valid in a session anchored at the
same start pose (takeoff beside the phone, same spot); relocalising
against a map from a different takeoff spot would place the drone
in the wrong world frame. The wiring (server start / loss handler)
must load the map for the current session's takeoff spot and call
``try_relocalise`` only while tracking is lost.

This module never touches VIO tracking behaviour: it only reads the
map (``export_map_data``), solves PnP through the VIO's own solver,
and adopts a pose through the VIO's own recovery path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .vio import (CAM_HEIGHT_RANGE_M, MIN_PNP_THIN_INLIERS,
                  ORB_NFEATURES, ORB_RATIO,
                  PNP_REPROJECTION_ERROR_PX, _accept_pnp)

log = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Where the map file lives: maps/ under the repo root. The wiring
# worker saves after a good flight (``save_map()`` with no path
# argument) and loads at server start (``load_map()`` with no path
# argument); the filename is fixed so there is exactly one "latest"
# map per takeoff spot. One map per session is the honest scope:
# multi-spot maps would need a spot key and are out of scope.
DEFAULT_MAP_DIR = Path(__file__).resolve().parent.parent / "maps"
DEFAULT_MAP_PATH = DEFAULT_MAP_DIR / "latest.npz"


@dataclass
class MapData:
    """A persisted VIO sparse map, ready to relocalise against.

    ``points`` (N, 3) float64 — 3D map points in the INTERNAL
    (OpenCV Y-down) world frame, the frame ``_solve_pnp`` expects.
    ``descriptors`` (N, 32) uint8 + ``des_valid`` (N,) bool — the
    ORB descriptor of each point (from the keyframe / good frame
    where it was last observed); only ``des_valid`` points can be
    matched, the rest are carried for completeness. ``ids`` (N,)
    int64 — the VIO's persistent point IDs. ``K`` — intrinsics dict
    (fx, fy, cx, cy, width, height). ``ref_R``/``ref_C`` (internal
    frame) — the newest keyframe's pose, the PnP initialisation;
    ``keyframe_R``/``keyframe_C``/``keyframe_frame`` — every keyframe
    pose (newest last), tried as fallback PnP inits. ``start_R``/
    ``start_C`` — the session's start pose (geometry/external
    convention), the anchor the map's world frame is expressed in.
    """
    points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float))
    descriptors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 32), dtype=np.uint8))
    des_valid: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool))
    ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    K: dict = field(default_factory=dict)
    camera_height_m: float = 1.3
    ref_R: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float))
    ref_C: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float))
    ref_frame: int = 0
    keyframe_R: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3, 3), dtype=float))
    keyframe_C: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float))
    keyframe_frame: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=int))
    start_R: np.ndarray | None = None
    start_C: np.ndarray | None = None
    n_frames: int = 0
    n_keyframes: int = 0
    version: int = FORMAT_VERSION

    def __len__(self):
        return int(len(self.points))

    @property
    def n_descriptors(self):
        """Points usable for ORB relocalisation (have a descriptor)."""
        return int(np.asarray(self.des_valid, dtype=bool).sum())

    def to_dict(self):
        return {
            "version": np.int64(self.version),
            "points": np.asarray(self.points, dtype=float),
            "descriptors": np.asarray(self.descriptors, dtype=np.uint8),
            "des_valid": np.asarray(self.des_valid, dtype=bool),
            "ids": np.asarray(self.ids, dtype=np.int64),
            "K_fx": float(self.K.get("fx", 0.0)),
            "K_fy": float(self.K.get("fy", 0.0)),
            "K_cx": float(self.K.get("cx", 0.0)),
            "K_cy": float(self.K.get("cy", 0.0)),
            "K_width": int(self.K.get("width", 0)),
            "K_height": int(self.K.get("height", 0)),
            "camera_height_m": float(self.camera_height_m),
            "ref_R": np.asarray(self.ref_R, dtype=float),
            "ref_C": np.asarray(self.ref_C, dtype=float),
            "ref_frame": np.int64(self.ref_frame),
            "keyframe_R": np.asarray(self.keyframe_R, dtype=float),
            "keyframe_C": np.asarray(self.keyframe_C, dtype=float),
            "keyframe_frame": np.asarray(self.keyframe_frame, dtype=int),
            "start_R": (np.zeros((3, 3), dtype=float)
                        if self.start_R is None
                        else np.asarray(self.start_R, dtype=float)),
            "start_C": (np.zeros(3, dtype=float)
                        if self.start_C is None
                        else np.asarray(self.start_C, dtype=float)),
            "has_start_pose": np.bool_(self.start_R is not None),
            "n_frames": np.int64(self.n_frames),
            "n_keyframes": np.int64(self.n_keyframes),
        }

    @classmethod
    def from_export(cls, d):
        """Build from ``MonocularVO.export_map_data()`` (nested ``K``
        dict, ``start_R``/``start_C`` possibly None)."""
        return cls(
            points=np.asarray(d["points"], dtype=float),
            descriptors=np.asarray(d["descriptors"], dtype=np.uint8),
            des_valid=np.asarray(d["des_valid"], dtype=bool),
            ids=np.asarray(d["ids"], dtype=np.int64),
            K=dict(d["K"]),
            camera_height_m=float(d["camera_height_m"]),
            ref_R=np.asarray(d["ref_R"], dtype=float),
            ref_C=np.asarray(d["ref_C"], dtype=float),
            ref_frame=int(d["ref_frame"]),
            keyframe_R=np.asarray(d["keyframe_R"], dtype=float),
            keyframe_C=np.asarray(d["keyframe_C"], dtype=float),
            keyframe_frame=np.asarray(d["keyframe_frame"], dtype=int),
            start_R=(None if d.get("start_R") is None
                     else np.asarray(d["start_R"], dtype=float)),
            start_C=(None if d.get("start_C") is None
                     else np.asarray(d["start_C"], dtype=float)),
            n_frames=int(d["n_frames"]),
            n_keyframes=int(d["n_keyframes"]),
            version=FORMAT_VERSION,
        )

    @classmethod
    def from_dict(cls, d):
        def _arr(key, shape=None, dtype=float):
            a = np.asarray(d[key], dtype=dtype)
            if shape is not None:
                a = a.reshape(shape)
            return a

        md = cls(
            points=_arr("points", (-1, 3)),
            descriptors=_arr("descriptors", (-1, 32), dtype=np.uint8),
            des_valid=_arr("des_valid", (-1,), dtype=bool),
            ids=_arr("ids", (-1,), dtype=np.int64),
            K={"fx": float(d["K_fx"]), "fy": float(d["K_fy"]),
               "cx": float(d["K_cx"]), "cy": float(d["K_cy"]),
               "width": int(d["K_width"]),
               "height": int(d["K_height"])},
            camera_height_m=float(d["camera_height_m"]),
            ref_R=_arr("ref_R", (3, 3)),
            ref_C=_arr("ref_C", (3,)),
            ref_frame=int(d["ref_frame"]),
            keyframe_R=_arr("keyframe_R", (-1, 3, 3)),
            keyframe_C=_arr("keyframe_C", (-1, 3)),
            keyframe_frame=_arr("keyframe_frame", (-1,), dtype=int),
            n_frames=int(d["n_frames"]),
            n_keyframes=int(d["n_keyframes"]),
            version=int(d["version"]),
        )
        if bool(d["has_start_pose"]):
            md.start_R = _arr("start_R", (3, 3))
            md.start_C = _arr("start_C", (3,))
        return md


def save_map(path, vio):
    """Serialise the VIO's sparse map to ``path`` (``np.savez``).

    ``vio`` is a ``MonocularVO`` (anything with ``export_map_data``)
    or a ``MapData``. ``path`` defaults to ``DEFAULT_MAP_PATH``
    (``maps/latest.npz`` under the repo root) when None. Returns the
    number of map points saved. An empty map saves nothing and
    returns 0 (with a warning) — a 0-point file could never
    relocalise, so writing one would only mislead the loader.
    Read-only w.r.t. tracking: exporting does not mutate the VIO.
    """
    if isinstance(vio, MapData):
        data = vio
    else:
        data = MapData.from_export(vio.export_map_data())
    n = len(data)
    if n == 0:
        log.warning("save_map: VIO map is empty, nothing saved "
                    "(fly first, then save after a good flight)")
        return 0
    path = DEFAULT_MAP_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file handle explicitly: np.savez() appends ".npz" to
    # a bare path string, which would silently rename the file.
    with open(path, "wb") as fh:
        np.savez(fh, **data.to_dict())
    log.info("save_map: %d map points (%d with descriptors, %d "
             "keyframes) -> %s", n, data.n_descriptors,
             data.n_keyframes, path)
    return n


def load_map(path=None):
    """Load a saved map file back into a ``MapData``.

    ``path`` defaults to ``DEFAULT_MAP_PATH``. Raises
    ``FileNotFoundError`` when the file does not exist and
    ``ValueError`` when it is not a map file of a known version or
    its arrays are inconsistent — the caller (server start) must
    treat that as "no saved map", never as a map.
    """
    path = DEFAULT_MAP_PATH if path is None else Path(path)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "version" not in d or int(d["version"]) != FORMAT_VERSION:
        raise ValueError(
            f"load_map: {path} is not a map file version "
            f"{FORMAT_VERSION} (got "
            f"{d.get('version', 'missing')})")
    md = MapData.from_dict(d)
    n = len(md)
    if n != len(md.descriptors) or n != len(md.des_valid) \
            or n != len(md.ids):
        raise ValueError(
            f"load_map: {path} has inconsistent map arrays "
            f"(points={n}, descriptors={len(md.descriptors)}, "
            f"ids={len(md.ids)})")
    if md.ref_R.shape != (3, 3) or md.ref_C.shape != (3,):
        raise ValueError(f"load_map: {path} has a bad reference pose")
    log.info("load_map: %d map points (%d with descriptors, %d "
             "keyframes, %d frames) <- %s", n, md.n_descriptors,
             md.n_keyframes, md.n_frames, path)
    return md


def _to_gray(frame):
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
    return None


def _solve_pnp_cv2_fallback(obj, img, Kmat, R_init, C_init):
    """cv2.solvePnPRansac fallback when the VIO has no ``_solve_pnp``.

    Only used for duck-typed VIO stand-ins (the real MonocularVO
    always takes the ``_solve_pnp`` path). Returns ``(R, C,
    n_inliers)`` in the internal frame, or None. Cheirality is
    checked explicitly: the mirror solution the hand-rolled solver
    exists to avoid (see vio._lm_pnp) is rejected here by counting
    points in front of the camera.
    """
    obj = np.asarray(obj, dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    if len(obj) < 6:
        return None
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, Kmat, None, iterationsCount=100,
            reprojectionError=PNP_REPROJECTION_ERROR_PX,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok or rvec is None or tvec is None:
        return None
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ np.asarray(tvec, dtype=float).reshape(3)
    Pc = (R @ (obj - C).T).T
    if int((Pc[:, 2] > 0).sum()) < max(6, len(obj) // 2):
        return None  # mirror / behind-camera solution
    n_inliers = int(inl.reshape(-1).size) if inl is not None else 0
    if n_inliers == 0:
        err = np.linalg.norm(
            np.stack([Kmat[0, 0] * Pc[:, 0] / np.maximum(Pc[:, 2], 0.05)
                      + Kmat[0, 2],
                      Kmat[1, 1] * Pc[:, 1] / np.maximum(Pc[:, 2], 0.05)
                      + Kmat[1, 2]], axis=1) - img, axis=1)
        n_inliers = int((err <= PNP_REPROJECTION_ERROR_PX).sum())
    return R, C, n_inliers


def try_relocalise(vio, frame, map_data, diag=None):
    """Relocalise ``vio`` against a SAVED map from one frame.

    ``frame`` is a BGR (or gray) image viewing the mapped scene;
    ``map_data`` is a ``MapData`` (from ``load_map``). Matches the
    frame's ORB features against the saved map's descriptors
    (Lowe ratio test, same 0.75 as the VIO's own ORB recovery),
    solves PnP through the VIO's own ``_solve_pnp`` (yaw1dof chain
    + full fallback, same camera model, same acceptance gate), and
    on success restores the pose via
    ``MonocularVO.adopt_saved_map_pose`` so tracking resumes in the
    saved map's world frame. Call only while tracking is lost — this
    runs only on loss, never every frame (ORB + RANSAC PnP is
    ~100 ms). Returns True on success, False otherwise; never
    raises, and on failure the VIO is left exactly as it was (no
    pose adopted, no state mutated — the match + solve happen
    before anything is written back). ``diag`` (optional dict) is
    filled with ``orb_matches`` / ``inliers`` /
    ``relocalised="saved-map"`` on success for the status line.
    """
    if diag is None:
        diag = {}
    try:
        if map_data is None or len(map_data) == 0:
            return False
        if map_data.n_descriptors < MIN_PNP_THIN_INLIERS:
            diag["saved_map_descriptors"] = map_data.n_descriptors
            return False
        gray = _to_gray(frame)
        if gray is None or gray.size == 0:
            return False
        orb = getattr(vio, "_orb", None)
        if orb is None:
            orb = cv2.ORB_create(nfeatures=ORB_NFEATURES,
                                 scaleFactor=1.2, nlevels=8)
        kp, des_cur = orb.detectAndCompute(gray, None)
        if des_cur is None or len(kp) == 0:
            diag["orb_matches"] = 0
            return False
        kxy = np.array([k.pt for k in kp], dtype=np.float32)
        use = np.asarray(map_data.des_valid, dtype=bool)
        pts3d = np.asarray(map_data.points, dtype=float)
        des_ref = np.asarray(map_data.descriptors, dtype=np.uint8)[use]
        idx_ref = np.nonzero(use)[0]
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            pairs = bf.knnMatch(des_ref, des_cur, k=2)
        except cv2.error:
            return False
        obj, img = [], []
        for m_n in pairs:
            if len(m_n) != 2:
                continue
            m, n_ = m_n
            if m.distance < ORB_RATIO * n_.distance:
                obj.append(pts3d[idx_ref[m.queryIdx]])
                img.append(kxy[m.trainIdx])
        diag["orb_matches"] = len(obj)
        if len(obj) < MIN_PNP_THIN_INLIERS:
            return False
        obj = np.asarray(obj, dtype=float)
        img = np.asarray(img, dtype=np.float32)
        # PnP initialisations: the reference (newest keyframe) pose
        # first, then the two newest other keyframe poses. A stale
        # init can strand the LM in the wrong basin, so a second
        # viewpoint is worth one more cheap solve — still only on
        # loss, never per frame.
        inits = [(np.asarray(map_data.ref_R, dtype=float),
                  np.asarray(map_data.ref_C, dtype=float))]
        kf_R = np.asarray(map_data.keyframe_R, dtype=float)
        kf_C = np.asarray(map_data.keyframe_C, dtype=float)
        for j in range(len(kf_R) - 1, max(-1, len(kf_R) - 3), -1):
            Rk, Ck = kf_R[j], kf_C[j]
            if np.allclose(Rk, inits[0][0]) \
                    and np.allclose(Ck, inits[0][1]):
                continue
            inits.append((Rk, Ck))
        Kmat = np.array(
            [[map_data.K.get("fx", 800.0), 0.0,
              map_data.K.get("cx", 480.0)],
             [0.0, map_data.K.get("fy", 800.0),
              map_data.K.get("cy", 360.0)],
             [0.0, 0.0, 1.0]], dtype=float)
        solve_pnp = getattr(vio, "_solve_pnp", None)
        lo, hi = CAM_HEIGHT_RANGE_M
        for R_init, C_init in inits:
            if solve_pnp is not None:
                sol = solve_pnp(
                    obj, img, R_init, C_init,
                    thresh_px=PNP_REPROJECTION_ERROR_PX, diag=diag)
            else:
                sol = _solve_pnp_cv2_fallback(
                    obj, img, Kmat, R_init, C_init)
            if sol is None:
                continue
            R_new, C_new, n_inliers = sol
            diag["inliers"] = int(n_inliers)
            if not _accept_pnp(int(n_inliers), len(obj)):
                continue
            # Height guard only (same as _try_relocalise): the drone
            # kept moving while lost, so a jump guard against the
            # last estimate would reject genuine relocalisations.
            # The PnP inlier gate is the trust signal.
            if not (lo <= float(C_new[1]) <= hi):
                continue
            vio.adopt_saved_map_pose(img, obj, R_new, C_new, gray,
                                     diag)
            diag["relocalised"] = "saved-map"
            diag["saved_map_inliers"] = int(n_inliers)
            log.info("relocalised against saved map: %d PnP inliers "
                     "(%d ORB matches)", int(n_inliers), len(obj))
            return True
        return False
    except Exception as exc:  # never raise: a failed relocalisation
        # is an ordinary lost frame, not a crash (the live pipeline
        # thread must survive it — cf. the BUG 6 crash).
        log.debug("saved-map relocalisation failed: %s", exc)
        return False
