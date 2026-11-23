"""Synthetic two-room renderer for the wall-marker virtual test harness.

Pinhole-camera renderer (numpy + OpenCV only). Two rooms in world
coordinates, rendered from any camera pose with the DRONE_K / PHONE_K
models from ``server.geometry``.

Default room (identical to the original build — the original 7 scenarios
must render byte-identically):

* Room A: z in [0, 3], Room B: z in [3, 8]; both x in [-3, 3], y in [0, 3].
* Shared wall at z = 3.0 with a doorway opening (x in [-0.6, 0.6],
  y in [0, 2.1]).
* Floor: procedural checker + per-cell noise. Walls: plaster tones with
  a skirting board and a few rectangular pictures/posters (features for
  tracking). Ceiling with two light panels.
* Person: ONE standing person cropped from ultralytics' bundled bus.jpg,
  cut out with grabCut, pasted as a camera-facing billboard.

CRITICAL GEOMETRY RULE (SPEC.md): the billboard is placed at the TRUE
projected position — the person's 3D foot point is projected with
``geometry.project_point`` and the billboard's bottom-centre goes at
that pixel; ``height_px`` is the true projected height (foot row minus
head row, i.e. ``fy * H / z_cam`` with ``z_cam`` the foot's depth in
the camera frame); width follows the cutout aspect ratio. NO artificial
offset is ever added to the pasted position. If the billboard does not
fit fully inside the frame it is NOT pasted at all (the frame is then
marked not-in-view), so the harness never measures a half-visible
person against full-body ground truth.

Lighting is a single brightness factor applied to the whole frame
(1.0 normal, 0.5 dim) — never boosted on the person only. ``noise_sigma``
adds real per-pixel Gaussian sensor noise (deterministic per frame) on
top of the brightness factor — a dim 0.4x frame with noise is a much
harder test than a dim frame alone.

Room geometry is parameterised per scenario through ``room`` (see
``DEFAULT_ROOM``): wall distance, room size, doorway position/width,
posters (including a featureless doorway wall), furniture (clutter),
a bright window (glare), and foreground occluders (a wardrobe/doorframe
edge between the drone and the person, rendered AFTER the billboard so
it genuinely covers part of the person).

Multiple people: ``render`` accepts a list of person dicts
(``persons=[{"pos": (3,), "yaw": float, "width_scale": float}]``).
A sideways person is approximated by a narrowed billboard
(``width_scale`` < 1) with the ground-truth skeleton rotated to the
same yaw — a real sideways body would show a profile, so this is a
documented approximation, not a perfect sideways render.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from server.geometry import DRONE_K, project_point

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Room geometry (world coordinates, metres) — DEFAULT_ROOM is the original
# build's geometry; every scenario that does not override a key renders
# exactly as before.
# ---------------------------------------------------------------------------

WALL_Z = 3.0            # shared wall plane
ROOM_X = (-3.0, 3.0)
ROOM_Y = (0.0, 3.0)
ROOM_A_Z = (0.0, WALL_Z)
ROOM_B_Z = (WALL_Z, 8.0)
DOOR_X = (-0.6, 0.6)    # doorway opening in the shared wall
DOOR_Y = (0.0, 2.1)
PERSON_HEIGHT_M = 1.7

DEFAULT_ROOM = {
    "wall_z": WALL_Z,
    "room_x": ROOM_X,
    "room_y": ROOM_Y,
    "room_b_z": ROOM_B_Z,
    # room A z is always (0, wall_z)
    "door_x": DOOR_X,
    "door_y": DOOR_Y,
    # posters: "default" (the original list), "none", "featureless_door"
    # (drop every poster on the shared wall — blank wall at the doorway),
    # or an explicit list of (wall, u, v, w, h, inner BGR) tuples.
    "posters": "default",
    # furniture: list of boxes {"x": (x0, x1), "y": (y0, y1),
    # "z": (z0, z1), "color": BGR} rendered as -z face + top face.
    "furniture": [],
    # occluders: list of (4 world points, BGR colour) quads rendered
    # AFTER the person billboard — genuinely covers part of the person.
    "occluders": [],
    # window: (wall, u, v, w, h) bright quad on a wall, e.g. the room-B
    # back wall behind the person ("backB", u, v, w, h).
    "window": None,
    # glare: bloom strength (0 = off) applied around the window centre.
    "glare": 0.0,
}

# BGR colours (plaster / wood tones)
COL_WALL = (214, 207, 192)
COL_WALL_B = (206, 198, 182)   # room B walls, a touch darker
COL_CEILING = (232, 228, 220)
COL_FLOOR_LIGHT = (198, 183, 158)
COL_FLOOR_DARK = (163, 148, 128)
COL_SKIRT = (96, 84, 72)
COL_FRAME = (58, 48, 40)
COL_DOOR_MAT = (70, 62, 55)
COL_WINDOW = (250, 250, 244)   # bright window glass
COL_FURNITURE = (110, 85, 60)

# Posters: (wall, centre, size, inner colour). Wall ids: "A" = shared wall
# room-A side, "B" = shared wall room-B side, "backB" = room B back wall,
# "left"/"right" = side walls (x = -3 / +3).
_POSTERS = [
    # (wall, u_centre, v_centre, w, h, inner BGR)
    ("A", -1.8, 1.75, 0.9, 0.7, (120, 90, 60)),
    ("A", 1.8, 1.75, 0.9, 0.7, (70, 100, 130)),
    ("left", 1.5, 1.7, 1.0, 0.75, (90, 120, 80)),
    ("right", 1.5, 1.7, 1.0, 0.75, (130, 90, 90)),
    ("B", -1.8, 1.7, 1.0, 0.8, (80, 110, 140)),
    ("B", 1.8, 1.7, 1.0, 0.8, (140, 110, 70)),
    ("backB", -1.6, 1.8, 1.1, 0.85, (100, 80, 120)),
    ("backB", 1.6, 1.8, 1.1, 0.85, (70, 130, 110)),
    ("leftB", 5.5, 1.7, 1.0, 0.75, (120, 120, 90)),
    ("rightB", 5.5, 1.7, 1.0, 0.75, (90, 100, 140)),
]


def look_at_R(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation for a camera at ``eye`` looking at ``target``.

    Camera looks along +Z in its own frame, +Y up. Same convention as
    ``server.main._look_at_R``.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    z_cam = target - eye
    n = np.linalg.norm(z_cam)
    if n < 1e-9:
        raise ValueError("look_at_R: eye == target")
    z_cam /= n
    up = np.asarray(up, dtype=float)
    x_cam = np.cross(up, z_cam)
    nx = np.linalg.norm(x_cam)
    if nx < 1e-6:
        # Looking (nearly) straight up/down: up is parallel to the view
        # direction, so fall back to a different up vector instead of
        # producing a NaN rotation.
        up = np.array([1.0, 0.0, 0.0])
        x_cam = np.cross(up, z_cam)
        nx = np.linalg.norm(x_cam)
    x_cam /= nx
    y_cam = np.cross(z_cam, x_cam)
    return np.stack([x_cam, y_cam, z_cam])


# ---------------------------------------------------------------------------
# Person cutout from ultralytics' bundled bus.jpg
# ---------------------------------------------------------------------------

_CUTOUT_CACHE = None


def find_bus_jpg():
    """Locate ultralytics' bundled bus.jpg."""
    import ultralytics
    pkg = Path(ultralytics.__file__).resolve().parent
    for cand in (pkg / "assets" / "bus.jpg", pkg / "bus.jpg"):
        if cand.exists():
            return cand
    return None


def _grabcut_person(img, bbox):
    """Cut one person out of ``img`` with grabCut on ``bbox``.

    Returns (cutout_bgr, mask_uint8) cropped to the bbox, mask eroded
    slightly. Raises ValueError if the mask ends up empty.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (2, 2, max(3, w - 4), max(3, h - 4))
    cv2.grabCut(crop, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                  255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.erode(fg, kernel, iterations=2)
    if fg.max() == 0:
        raise ValueError("grabCut produced an empty foreground mask")
    return crop, fg


def load_person_cutout(detector):
    """Crop the largest person from bus.jpg; gate on standalone detection.

    Runs ``detector`` on bus.jpg, takes the largest person detection as
    the crop, cuts it out with grabCut, and verifies the isolated cutout
    (pasted on a plain background at ~500 px tall) is detected by the
    pose detector on its own. If no cutout passes, raises RuntimeError
    with a clear setup error — the harness must NOT continue with an
    undetectable person (SPEC.md).

    Result is cached for the process; returns (cutout_bgr, mask_uint8).
    """
    global _CUTOUT_CACHE
    if _CUTOUT_CACHE is not None:
        return _CUTOUT_CACHE
    bus = find_bus_jpg()
    if bus is None:
        raise RuntimeError(
            "SETUP ERROR: ultralytics bundled bus.jpg not found "
            "(looked under the installed ultralytics package). "
            "Cannot build the person billboard; harness stopped.")
    img = cv2.imread(str(bus))
    if img is None:
        raise RuntimeError(f"SETUP ERROR: could not read {bus}")
    dets = detector.detect(img)
    if not dets:
        raise RuntimeError(
            "SETUP ERROR: pose detector found no person in bus.jpg; "
            "cannot build the person billboard; harness stopped.")
    order = sorted(
        dets,
        key=lambda d: (d["bbox_xyxy"][2] - d["bbox_xyxy"][0])
        * (d["bbox_xyxy"][3] - d["bbox_xyxy"][1]),
        reverse=True)
    errors = []
    for rank, det in enumerate(order):
        try:
            cutout, mask = _grabcut_person(img, det["bbox_xyxy"])
        except ValueError as exc:
            errors.append(f"rank{rank}: grabCut failed ({exc})")
            continue
        # Standalone check: paste the cutout on a plain background and
        # require the pose detector to find it.
        h, w = cutout.shape[:2]
        target_h = 500
        tw = max(10, int(w * target_h / h))
        resized = cv2.resize(cutout, (tw, target_h),
                             interpolation=cv2.INTER_AREA)
        rmask = cv2.resize(mask, (tw, target_h),
                           interpolation=cv2.INTER_NEAREST)
        canvas = np.full((600, 700, 3), (150, 150, 150), dtype=np.uint8)
        y0, x0 = 50, (700 - tw) // 2
        roi = canvas[y0:y0 + target_h, x0:x0 + tw]
        roi[rmask > 0] = resized[rmask > 0]
        canvas[y0:y0 + target_h, x0:x0 + tw] = roi
        check = detector.detect(canvas)
        if check:
            log.info("Person cutout: rank%d detection from bus.jpg "
                     "(bbox %s, conf %.2f); standalone check passed "
                     "(conf %.2f, mean kpt conf %.2f)",
                     rank, det["bbox_xyxy"], det["conf"],
                     check[0]["conf"], float(check[0]["kpt_conf"].mean()))
            _CUTOUT_CACHE = (cutout, mask)
            return _CUTOUT_CACHE
        errors.append(
            f"rank{rank}: cutout NOT detected standalone "
            f"(bus.jpg conf was {det['conf']:.2f})")
    raise RuntimeError(
        "SETUP ERROR: no person cut from bus.jpg is detected by the "
        "pose detector on its own (" + "; ".join(errors) + "). "
        "Cannot build the person billboard; harness stopped — do not "
        "continue with an undetectable person.")


# ---------------------------------------------------------------------------
# Room renderer
# ---------------------------------------------------------------------------

def _quad(p1, p2, p3, p4):
    return np.array([p1, p2, p3, p4], dtype=float)


class RoomRenderer:
    """Render the two-room scene from a camera pose.

    ``K`` is a camera model dict (DRONE_K / PHONE_K). ``cutout``/``mask``
    are the person billboard from :func:`load_person_cutout`.
    ``brightness`` is a single factor applied to the whole frame.
    ``room`` overrides keys of ``DEFAULT_ROOM`` (wall distance, room
    size, doorway, posters, furniture, occluders, window, glare).
    ``noise_sigma`` adds real per-pixel Gaussian noise (0 = off);
    ``noise_seed`` makes the noise deterministic per frame.
    """

    def __init__(self, K, cutout, mask, person_height=PERSON_HEIGHT_M,
                 brightness=1.0, room=None, noise_sigma=0.0,
                 noise_seed=0):
        self.K = dict(K)
        self.W = int(K["width"])
        self.H = int(K["height"])
        self.cutout = cutout
        self.mask = mask
        self.person_height = float(person_height)
        self.brightness = float(brightness)
        self.room = dict(DEFAULT_ROOM)
        if room:
            self.room.update(room)
        self.noise_sigma = float(noise_sigma)
        self.noise_seed = int(noise_seed)
        self._frame_idx = 0
        self._polys = self._build_room_polys()

    # -- static room geometry ---------------------------------------------

    def _poster_list(self):
        """Effective poster list for this room config."""
        spec = self.room["posters"]
        if spec == "default":
            posters = list(_POSTERS)
        elif spec == "none":
            posters = []
        elif spec == "featureless_door":
            # Blank wall at the doorway: no posters on the shared wall
            # (either side). The rest of the room keeps its features.
            posters = [p for p in _POSTERS if p[0] not in ("A", "B")]
        else:
            posters = list(spec)
        # Extra posters (clutter) on top of the base list.
        posters.extend(self.room.get("posters_extra", []))
        return posters

    def _build_room_polys(self):
        """List of (points (N,3), BGR colour) for the empty rooms."""
        cfg = self.room
        polys = []
        x0, x1 = cfg["room_x"]
        y0, y1 = cfg["room_y"]
        wall_z = cfg["wall_z"]
        zA0, zA1 = 0.0, wall_z
        zB0, zB1 = cfg["room_b_z"]

        # Floor: 0.5 m checker cells + deterministic per-cell noise.
        rng = np.random.default_rng(7)
        nx, nz = 12, 16
        for ix in range(nx):
            for iz in range(nz):
                cx0 = x0 + (x1 - x0) * ix / nx
                cx1 = x0 + (x1 - x0) * (ix + 1) / nx
                cz0 = 0.0 + (zB1 - 0.0) * iz / nz
                cz1 = 0.0 + (zB1 - 0.0) * (iz + 1) / nz
                base = (COL_FLOOR_LIGHT if (ix + iz) % 2 == 0
                        else COL_FLOOR_DARK)
                noise = int(rng.integers(-9, 10))
                col = tuple(int(min(255, max(0, c + noise))) for c in base)
                polys.append((_quad((cx0, 0, cz0), (cx1, 0, cz0),
                                   (cx1, 0, cz1), (cx0, 0, cz1)), col))

        # Ceiling.
        polys.append((_quad((x0, y1, zA0), (x1, y1, zA0),
                           (x1, y1, zB1), (x0, y1, zB1)), COL_CEILING))
        # Ceiling light panels (bright quads, slightly below the ceiling).
        panel_zs = [1.5, 5.5]
        if zB1 > 9.0:
            panel_zs.append(8.5)
        if zB1 > 11.0:
            panel_zs.append(11.0)
        for zc in panel_zs:
            polys.append((_quad((-0.5, y1 - 0.01, zc - 0.6),
                               (0.5, y1 - 0.01, zc - 0.6),
                               (0.5, y1 - 0.01, zc + 0.6),
                               (-0.5, y1 - 0.01, zc + 0.6)),
                          (245, 244, 235)))

        # Walls. Shared wall has the doorway hole: left segment, right
        # segment, lintel above the door.
        dx0, dx1 = cfg["door_x"]
        dy1 = cfg["door_y"][1]
        # Back wall, room A (z = 0).
        polys.append((_quad((x0, y0, zA0), (x1, y0, zA0),
                           (x1, y1, zA0), (x0, y1, zA0)), COL_WALL))
        # Back wall, room B (z = room_b_z[1]).
        polys.append((_quad((x0, y0, zB1), (x1, y0, zB1),
                           (x1, y1, zB1), (x0, y1, zB1)), COL_WALL_B))
        # Shared wall segments (z = wall_z).
        polys.append((_quad((x0, y0, wall_z), (dx0, y0, wall_z),
                           (dx0, y1, wall_z), (x0, y1, wall_z)), COL_WALL))
        polys.append((_quad((dx1, y0, wall_z), (x1, y0, wall_z),
                           (x1, y1, wall_z), (dx1, y1, wall_z)), COL_WALL))
        polys.append((_quad((dx0, dy1, wall_z), (dx1, dy1, wall_z),
                           (dx1, y1, wall_z), (dx0, y1, wall_z)), COL_WALL))
        # Side walls (x = x0 / x1), full length.
        for xx in (x0, x1):
            polys.append((_quad((xx, y0, zA0), (xx, y0, zB1),
                               (xx, y1, zB1), (xx, y1, zA0)), COL_WALL))

        # Skirting boards (0.12 m tall) along the wall bases.
        sk = 0.12
        off = 0.015  # stand proud of the wall so painter's order is stable
        polys.append((_quad((x0, y0, zA0 + off), (x1, y0, zA0 + off),
                           (x1, sk, zA0 + off), (x0, sk, zA0 + off)),
                      COL_SKIRT))
        polys.append((_quad((x0, y0, zB1 - off), (x1, y0, zB1 - off),
                           (x1, sk, zB1 - off), (x0, sk, zB1 - off)),
                      COL_SKIRT))
        for xx, s in ((x0, 1), (x1, -1)):
            polys.append((_quad((xx + s * off, y0, zA0),
                               (xx + s * off, y0, zB1),
                               (xx + s * off, sk, zB1),
                               (xx + s * off, sk, zA0)), COL_SKIRT))
        # Shared wall skirting, both sides, skipping the doorway.
        for zz, s in ((wall_z - off, 1), (wall_z + off, -1)):
            for xa, xb in ((x0, dx0), (dx1, x1)):
                polys.append((_quad((xa, y0, zz), (xb, y0, zz),
                                   (xb, sk, zz), (xa, sk, zz)), COL_SKIRT))

        # Doorway threshold strip on the floor.
        polys.append((_quad((dx0, 0.005, wall_z - 0.1),
                           (dx1, 0.005, wall_z - 0.1),
                           (dx1, 0.005, wall_z + 0.1),
                           (dx0, 0.005, wall_z + 0.1)), COL_DOOR_MAT))

        # Bright window quad (glare source), if configured.
        if cfg["window"] is not None:
            wall, u, v, w, h = cfg["window"]
            pts = self._poster_quad(wall, u, v, w, h)
            if pts is not None:
                polys.append((pts, COL_WINDOW))

        # Furniture boxes (clutter): -z face (faces the doorway/drone)
        # + top face. They sit in the room as part of the painter's
        # order, so a box between drone and person would be painted
        # over by the billboard — furniture is placed clear of the
        # drone-person line of sight; real occlusion uses "occluders".
        for box in cfg["furniture"]:
            fx0, fx1 = box["x"]
            fy0, fy1 = box["y"]
            fz0, fz1 = box["z"]
            col = box.get("color", COL_FURNITURE)
            polys.append((_quad((fx0, fy0, fz0), (fx1, fy0, fz0),
                               (fx1, fy1, fz0), (fx0, fy1, fz0)), col))
            polys.append((_quad((fx0, fy1, fz0), (fx1, fy1, fz0),
                               (fx1, fy1, fz1), (fx0, fy1, fz1)), col))

        # Pictures / posters: frame quad + inner quad 0.015 m proud.
        for wall, u, v, w, h, inner in self._poster_list():
            pts = self._poster_quad(wall, u, v, w, h)
            if pts is None:
                continue
            polys.append((pts, COL_FRAME))
            pts2 = self._poster_quad(wall, u, v, w - 0.12, h - 0.12, 0.015)
            if pts2 is not None:
                polys.append((pts2, inner))
        return polys

    def _poster_quad(self, wall, u, v, w, h, proud=0.0):
        """Rectangle on a wall: u = along-wall coord, v = height.

        Returns 4 world points, or None for an unknown wall id.
        """
        cfg = self.room
        x0, x1 = cfg["room_x"]
        y0, _ = cfg["room_y"]
        wall_z = cfg["wall_z"]
        hw, hh = w / 2.0, h / 2.0
        if wall == "A":      # shared wall, room-A side (faces -Z)
            z = wall_z - proud
            return _quad((u - hw, v - hh, z), (u + hw, v - hh, z),
                         (u + hw, v + hh, z), (u - hw, v + hh, z))
        if wall == "B":      # shared wall, room-B side (faces +Z)
            z = wall_z + proud
            return _quad((u - hw, v - hh, z), (u + hw, v - hh, z),
                         (u + hw, v + hh, z), (u - hw, v + hh, z))
        if wall == "backB":  # room B back wall (faces -Z)
            z = cfg["room_b_z"][1] - proud
            return _quad((u - hw, v - hh, z), (u + hw, v - hh, z),
                         (u + hw, v + hh, z), (u - hw, v + hh, z))
        if wall == "left":   # x = x0, room A side (faces +X); u = z
            x = x0 + proud
            return _quad((x, v - hh, u - hw), (x, v - hh, u + hw),
                         (x, v + hh, u + hw), (x, v + hh, u - hw))
        if wall == "right":  # x = x1, room A side (faces -X); u = z
            x = x1 - proud
            return _quad((x, v - hh, u - hw), (x, v - hh, u + hw),
                         (x, v + hh, u + hw), (x, v + hh, u - hw))
        if wall == "leftB":  # x = x0, room B (faces +X); u = z
            x = x0 + proud
            return _quad((x, v - hh, u - hw), (x, v - hh, u + hw),
                         (x, v + hh, u + hw), (x, v + hh, u - hw))
        if wall == "rightB":  # x = x1, room B (faces -X); u = z
            x = x1 - proud
            return _quad((x, v - hh, u - hw), (x, v - hh, u + hw),
                         (x, v + hh, u + hw), (x, v + hh, u - hw))
        return None

    def _window_centre_world(self):
        """World point at the centre of the configured window, or None."""
        cfg = self.room
        if cfg["window"] is None:
            return None
        wall, u, v, _w, _h = cfg["window"]
        wall_z = cfg["wall_z"]
        x0, x1 = cfg["room_x"]
        if wall == "backB":
            return np.array([u, v, cfg["room_b_z"][1]], dtype=float)
        if wall == "B":
            return np.array([u, v, wall_z], dtype=float)
        if wall == "leftB":
            return np.array([x0, v, u], dtype=float)
        if wall == "rightB":
            return np.array([x1, v, u], dtype=float)
        return None

    # -- projection helpers -----------------------------------------------

    @staticmethod
    def _clip_near(p_cam, z_min=0.1):
        """Clip a polygon (in camera coords) against the z >= z_min plane."""
        if len(p_cam) == 0:
            return p_cam
        out = []
        n = len(p_cam)
        for i in range(n):
            cur = p_cam[i]
            prev = p_cam[i - 1]
            cur_in = cur[2] >= z_min
            prev_in = prev[2] >= z_min
            if cur_in:
                if not prev_in:
                    t = (z_min - prev[2]) / (cur[2] - prev[2])
                    out.append(prev + t * (cur - prev))
                out.append(cur)
            elif prev_in:
                t = (z_min - prev[2]) / (cur[2] - prev[2])
                out.append(prev + t * (cur - prev))
        return np.array(out, dtype=float) if out else np.zeros((0, 3))

    def _project_poly(self, R, C, pts_world):
        """Project a world polygon to pixel coords (or None if invisible)."""
        p_cam = (R @ (np.asarray(pts_world, dtype=float) - C).T).T
        p_cam = self._clip_near(p_cam)
        if len(p_cam) < 3:
            return None, None
        z = p_cam[:, 2]
        u = self.K["fx"] * p_cam[:, 0] / z + self.K["cx"]
        v = self.K["cy"] - self.K["fy"] * p_cam[:, 1] / z
        return np.stack([u, v], axis=1), float(np.mean(z))

    # -- render -------------------------------------------------------------

    def render(self, pose, person_pos=None, person_yaw=0.0, persons=None):
        """Render one frame.

        ``pose`` is (R, C) for the camera. ``persons`` is a list of
        ``{"pos": (3,) world foot, "yaw": float, "width_scale": float}``
        (backward compatible: a single ``person_pos``/``person_yaw``
        becomes a one-person list). Returns (frame_bgr, info) with
        info carrying, per person, ``foot_px``, ``head_px``,
        ``height_px``, ``billboard_box`` (x0, y0, x1, y1) or None, and
        ``in_view`` (True only when the whole person fits in the frame
        and is at least 80 px tall) — plus the same fields at top level
        for the first person (backward compat), and ``persons`` (list).
        """
        R, C = pose
        R = np.asarray(R, dtype=float)
        C = np.asarray(C, dtype=float)
        wall_z = self.room["wall_z"]
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Painter's algorithm: far polygons first.
        projected = []
        for pts, col in self._polys:
            px, mean_z = self._project_poly(R, C, pts)
            if px is not None:
                projected.append((mean_z, px, col))
        projected.sort(key=lambda item: -item[0])
        for _, px, col in projected:
            cv2.fillPoly(frame, [np.round(px).astype(np.int32)], col,
                         lineType=cv2.LINE_AA)

        if persons is None:
            if person_pos is None:
                persons = []
            else:
                persons = [{"pos": np.asarray(person_pos, dtype=float),
                            "yaw": float(person_yaw),
                            "width_scale": 1.0}]

        per_person = []
        for person in persons:
            foot = np.asarray(person["pos"], dtype=float)
            width_scale = float(person.get("width_scale", 1.0))
            entry = {"foot_px": None, "head_px": None, "height_px": 0.0,
                     "billboard_box": None, "in_view": False}
            if C[2] > wall_z + 0.05:
                # Only paste once the drone is inside room B: from room A
                # the wall blocks the view of the person.
                head = foot + np.array([0.0, self.person_height, 0.0])
                pf = project_point(self.K, R, C, foot)
                ph = project_point(self.K, R, C, head)
                if pf is not None and ph is not None:
                    uf, vf, zf = pf
                    uh, vh, _ = ph
                    height_px = vf - vh  # true projected height
                    if height_px > 0:
                        ch, cw = self.cutout.shape[:2]
                        width_px = height_px * cw / ch * width_scale
                        x0 = int(round(uf - width_px / 2.0))
                        y0 = int(round(vf - height_px))
                        x1 = x0 + int(round(width_px))
                        y1 = y0 + int(round(height_px))
                        entry.update(foot_px=(float(uf), float(vf)),
                                     head_px=(float(uh), float(vh)),
                                     height_px=float(height_px),
                                     billboard_box=(x0, y0, x1, y1))
                        if (0 <= x0 and 0 <= y0 and x1 <= self.W
                                and y1 <= self.H and height_px >= 80):
                            entry["in_view"] = True
                            self._paste_person(frame, x0, y0, x1, y1,
                                               uf, vf, width_px, height_px)
            per_person.append(entry)

        # Foreground occluders (wardrobe / doorframe edge): drawn AFTER
        # the billboards, so they genuinely cover part of the person.
        for pts, col in self.room.get("occluders", []):
            px, _ = self._project_poly(R, C, pts)
            if px is not None:
                cv2.fillPoly(frame, [np.round(px).astype(np.int32)], col,
                             lineType=cv2.LINE_AA)

        # Glare bloom around the bright window centre.
        if self.room.get("glare", 0.0) > 0.0:
            centre = self._window_centre_world()
            if centre is not None:
                p = project_point(self.K, R, C, centre)
                if p is not None and 0 <= p[0] < self.W and 0 <= p[1] < self.H:
                    radius = 200
                    glow = np.zeros((self.H, self.W), dtype=np.float32)
                    cv2.circle(glow, (int(p[0]), int(p[1])), radius, 1.0, -1)
                    glow = cv2.GaussianBlur(glow, (0, 0),
                                            sigmaX=radius / 2.5)
                    frame = np.clip(
                        frame.astype(np.float32)
                        + glow[..., None] * float(self.room["glare"]) * 255.0,
                        0, 255).astype(np.uint8)

        if self.brightness != 1.0:
            frame = np.clip(frame.astype(np.float32) * self.brightness,
                            0, 255).astype(np.uint8)
        if self.noise_sigma > 0:
            rng = np.random.default_rng(self.noise_seed + self._frame_idx)
            noise = rng.normal(0.0, self.noise_sigma,
                               frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise,
                            0, 255).astype(np.uint8)
        self._frame_idx += 1

        info = {"persons": per_person}
        if per_person:
            info.update(per_person[0])
        else:
            info.update({"foot_px": None, "head_px": None, "height_px": 0.0,
                         "billboard_box": None, "in_view": False})
        return frame, info

    def _paste_person(self, frame, x0, y0, x1, y1, uf, vf, width_px,
                      height_px):
        """Paste the billboard at the exact projected box + soft shadow."""
        # Soft contact shadow on the floor under the feet.
        shadow = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.ellipse(shadow, (int(round(uf)), int(round(vf)) + 3),
                    (max(4, int(width_px * 0.34)), max(3, int(height_px * 0.045))),
                    0, 0, 360, 255, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(2.0, width_px / 25.0))
        alpha = (shadow.astype(np.float32) / 255.0) * 0.30
        frame[:] = (frame.astype(np.float32) * (1.0 - alpha[..., None])
                    ).astype(np.uint8)
        # Billboard (nearest-neighbour mask keeps the cutout edge crisp).
        w_px, h_px = x1 - x0, y1 - y0
        resized = cv2.resize(self.cutout, (w_px, h_px),
                             interpolation=cv2.INTER_AREA)
        rmask = cv2.resize(self.mask, (w_px, h_px),
                           interpolation=cv2.INTER_NEAREST)
        roi = frame[y0:y1, x0:x1]
        roi[rmask > 0] = resized[rmask > 0]
        frame[y0:y1, x0:x1] = roi


def render_frame(K, pose, person_pos, cutout, mask, person_yaw=0.0,
                 person_height=PERSON_HEIGHT_M, brightness=1.0,
                 room=None, noise_sigma=0.0, noise_seed=0, persons=None):
    """One-shot convenience wrapper around :class:`RoomRenderer`."""
    renderer = RoomRenderer(K, cutout, mask, person_height, brightness,
                            room=room, noise_sigma=noise_sigma,
                            noise_seed=noise_seed)
    return renderer.render(pose, person_pos, person_yaw, persons=persons)
