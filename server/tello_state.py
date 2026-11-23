"""DJI Tello state packets as VIO priors (report §2/§6).

The Tello's own state packet (UDP 8890, ~10 Hz, parsed by djitellopy)
carries attitude (yaw/pitch/roll, integer degrees), velocities
(vgx/vgy/vgz), height (h, takeoff-relative) and ToF distance (tof,
above-ground). The research report (2026-11-23,
``research_notes/vio-rotation-tracking-fix-20261123-1014/report.md``)
audits every field and ranks them for this VIO:

  1. yaw — the only field measuring the failing degree of freedom
     (the turn); magnetometer-anchored, 1 deg quantisation.
  2. vgx/vgy/vgz — the only field measuring the surviving translation
     (2.7 cm/frame); demoted by the unit/frame dispute (see below).
  3. h + tof together — metric z + scale anchor.
  4. pitch/roll — collapse PnP rotation to 1 DoF (yaw).
  5. baro (delta only), 6. agx/agy/agz (turn cue only).

CRITICAL — CALIBRATION (report §6.9): the units of vgx/vgy/vgz are
DISPUTED between SDK versions (cm/s in SDK 1.3-era material, dm/s in
SDK 3.0 — a 10x error if assumed wrong) and the frame (body vs
global) is undocumented. DJI publishes NO accuracy spec for ANY
field. Therefore:

  * All calibration constants live in ONE place (``TelloCalibration``).
  * The default is UNCALIBRATED: velocity priors are OFF, and the
    only priors used are yaw/pitch/roll (attitude) and h/tof (height),
    whose units are unambiguous (degrees, cm).
  * Velocity is used ONLY when a real calibration has been run
    (``tools/calibrate_tello.py`` + ``docs/TELLO_CALIBRATION.md``)
    and the constants recorded here. NEVER bake in an assumed
    unit/frame — a 10x velocity error would silently corrupt every
    pose the prior touches.

Pose-source flags: every pose the VIO returns carries a source label
so the bridge never silently presents an attitude-only pose as visual
tracking (report §6.4):

  * ``POSE_VISION`` — full visual PnP against the 3D map.
  * ``POSE_ATTITUDE_AIDED`` — visual bearings (rotation-only mode) or
    PnP initialised/constrained by the yaw prior; the rotation comes
    from real measurements (bearings and/or the onboard yaw sensor).
  * ``POSE_DEAD_RECKONED`` — vision lost; pose integrated from yaw +
    velocity (if calibrated) + h/tof, bounded by time/distance.
  * ``POSE_INVALID`` — no pose (bridge invalidates, marker hidden).

The virtual harness feeds SYNTHETIC state packets derived from the
ground-truth trajectory (``SyntheticStateProvider``) so the prior
code path is testable without a real drone. Synthetic packets are
marked ``synthetic=True`` and are generated with a DOCUMENTED unit
and frame; the harness calibration is set to match them explicitly
— the real default stays UNCALIBRATED.

Coordinate conventions: exactly ``server.geometry`` — a pose is
``(R, C)`` with R world-to-camera and C the camera centre in world
coordinates, Y up. The Tello attitude is the BODY attitude in the
DJI convention (yaw = heading, pitch = nose-up positive, roll =
right-wing-down positive, all degrees). The front camera is tilted
~13 deg DOWN relative to the body (dronemaster, secondary source —
UNCONFIRMED, see ``CAMERA_TILT_DEG``), so:

  camera_pitch (harness, + = looking down) = tilt - body_pitch_dji
  camera_yaw = body_yaw, camera_roll = body_roll (tilt is pitch-only)

The conversion functions below implement exactly that, and are
round-trip tested (see ``tests/test_tello_state.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Pose-source flags (one per returned pose; see module docstring).
# ---------------------------------------------------------------------------
POSE_VISION = "vision"
POSE_ATTITUDE_AIDED = "attitude-aided"
POSE_DEAD_RECKONED = "dead-reckoned"
POSE_INVALID = "invalid"

# ---------------------------------------------------------------------------
# Priors: gating and sigmas (report §2/§6.4). These are working
# assumptions pending the §6.9 bench calibration, NOT datasheet
# values — DJI publishes no accuracy spec for any field.
# ---------------------------------------------------------------------------
# A yaw step bigger than this in one 0.1 s frame is not physical
# (the Tello yaws 30-60 deg/s); the -31 deg / +101 deg harness snaps
# must be gated out, and a magnetometer glitch likewise.
YAW_JUMP_MAX_DEG = 10.0
# Yaw prior sigma (deg): 1 deg quantisation + indoor magnetic
# disturbance, unquantified. 2 deg is the working assumption.
YAW_SIGMA_DEG = 2.0
# Height priors: h is fused/smooth (sigma ~2 cm), tof is direct AGL
# (sigma ~3-5 cm, worse on dark/absorbent floors).
H_SIGMA_M = 0.02
TOF_SIGMA_M = 0.04
# tof == 10 cm is the out-of-range sentinel, NOT a 10 cm floor.
TOF_SENTINEL_CM = 10
# Front-camera downward tilt vs the body (deg). From dronemaster
# (secondary source); UNCONFIRMED — confirm from calibration
# (§6.9.2: compare interpolated yaw/pitch against visual yaw on
# straight segments). A 13 deg error here corrupts every pitch prior.
CAMERA_TILT_DEG = 13.0
# Dead-reckoning budget (report §5.1 / §6.4): invalidate if vision
# does not return within this time or integrated distance.
DEAD_RECKON_MAX_S = 2.0
DEAD_RECKON_MAX_M = 1.0
# State-packet rate is ~10 Hz; the laptop timestamps on receipt and
# interpolates to the video frame time (no timestamp in the packet).
STATE_RATE_HZ = 10.0


# ---------------------------------------------------------------------------
# Calibration — ONE clearly-marked place (report §6.9).
# ---------------------------------------------------------------------------
@dataclass
class TelloCalibration:
    """Calibration constants for the Tello state priors.

    Default is UNCALIBRATED: velocity priors are OFF and must stay
    off until a real calibration run (``tools/calibrate_tello.py``)
    has measured the velocity unit and frame. The attitude (yaw /
    pitch / roll) and height (h / tof) priors do NOT depend on this
    calibration — their units (degrees, cm) are unambiguous — so
    they are usable uncalibrated, with the sigmas above.
    """
    # Velocity unit: "unknown" (default), "cm/s", or "dm/s".
    velocity_unit: str = "unknown"
    # Velocity frame: "unknown" (default), "body", or "global".
    velocity_frame: str = "unknown"
    # Yaw offset (deg) between the Tello yaw zero and the session
    # start yaw. Set at takeoff (yaw is reset to 0 on power-on, not
    # on landing): yaw_session = yaw_tello - yaw_offset.
    yaw_offset_deg: float = 0.0
    # Measured yaw sigma (deg) from the calibration run
    # (tools/calibrate_tello.py, yaw RMS vs GT yaw). Default 2.0
    # is the working assumption (1 deg quantisation + indoor
    # magnetic disturbance) until a real calibration measures it.
    # The yaw gate for the real test is sigma <= 1.5 deg AND
    # |doorway bias| <= 1.5 deg (see docs/TELLO_CALIBRATION.md).
    yaw_sigma_deg: float = 2.0
    # Front-camera tilt (deg, down positive). See CAMERA_TILT_DEG.
    camera_tilt_deg: float = CAMERA_TILT_DEG
    # State-vs-video latency (s): the video (H.264 + Wi-Fi) lags the
    # state packet by an unmeasured offset; estimate once by
    # cross-correlating yaw steps with visual rotation (§6.9.5).
    video_latency_s: float = 0.0
    # Set True ONLY by a real calibration run (see
    # tools/calibrate_tello.py). While False, velocity priors are
    # disabled regardless of velocity_unit/velocity_frame.
    calibrated: bool = False

    @property
    def velocity_ready(self) -> bool:
        """True only when a real calibration has fixed unit + frame."""
        return (self.calibrated
                and self.velocity_unit in ("cm/s", "dm/s")
                and self.velocity_frame in ("body", "global"))

    @property
    def velocity_scale_m_per_unit(self) -> float:
        """Metres per second per raw velocity unit (calibrated only)."""
        if self.velocity_unit == "cm/s":
            return 0.01
        if self.velocity_unit == "dm/s":
            return 0.1
        raise ValueError(
            "velocity unit uncalibrated — velocity priors are OFF")


# The single default calibration: UNCALIBRATED. Real runs must pass
# an explicitly calibrated instance; the harness sets one up for its
# synthetic packets (see SyntheticStateProvider).
DEFAULT_CALIBRATION = TelloCalibration()


# ---------------------------------------------------------------------------
# State packet.
# ---------------------------------------------------------------------------
@dataclass
class TelloStatePacket:
    """One Tello state packet (djitellopy ``get_current_state`` dict).

    ``t`` is the LAPTOP receipt time (seconds, same clock as the
    video frame times) — the packet itself carries no timestamp.
    ``synthetic`` marks packets derived from ground truth in the
    virtual harness (never a real drone measurement).
    """
    t: float
    yaw: int = 0        # deg, -180..180, magnetometer heading
    pitch: int = 0      # deg, nose-up positive (DJI convention)
    roll: int = 0       # deg, right-wing-down positive
    vgx: int = 0        # raw velocity units (see TelloCalibration)
    vgy: int = 0
    vgz: int = 0
    tof: int = 0        # cm above ground (10 = out-of-range sentinel)
    h: int = 0          # cm above takeoff point (fused)
    baro: float = 0.0   # cm above sea level (absolute; delta-only use)
    agx: float = 0.0    # specific force incl. gravity (see §6.2 #8)
    agy: float = 0.0
    agz: float = 0.0
    time: int = 0       # motor-on time, s
    bat: int = 0        # battery, %
    synthetic: bool = False


def parse_state_string(s: str, t: float) -> TelloStatePacket:
    """Parse a djitellopy state string into a packet.

    Format (SDK 2.0): ``pitch:%d;roll:%d;yaw:%d;vgx:%d;vgy:%d;vgz:%d;
    templ:%d;temph:%d;tof:%d;h:%d;bat:%d;baro:%.2f;time:%d;agx:%.2f;
    agy:%.2f;agz:%.2f;`` — mission-pad fields (mid/x/y/z/mpry) are
    FORBIDDEN and are ignored if present (they are never parsed into
    a pose). ``t`` is the laptop receipt time.
    """
    kv = {}
    for part in s.strip().rstrip(";").split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        kv[k.strip()] = v.strip()

    def _int(key, default=0):
        try:
            return int(float(kv.get(key, default)))
        except (TypeError, ValueError):
            return default

    def _float(key, default=0.0):
        try:
            return float(kv.get(key, default))
        except (TypeError, ValueError):
            return default

    return TelloStatePacket(
        t=float(t),
        yaw=_int("yaw"), pitch=_int("pitch"), roll=_int("roll"),
        vgx=_int("vgx"), vgy=_int("vgy"), vgz=_int("vgz"),
        tof=_int("tof"), h=_int("h"),
        baro=_float("baro"), agx=_float("agx"), agy=_float("agy"),
        agz=_float("agz"), time=_int("time"), bat=_int("bat"))


# ---------------------------------------------------------------------------
# Attitude <-> camera rotation (geometry convention, Y up).
# ---------------------------------------------------------------------------
def yaw_pitch_roll_from_R(R):
    """Body attitude (yaw, pitch, roll) in deg from a camera rotation R.

    Yaw = heading of the camera forward axis (== body yaw, the tilt
    is pitch-only). Pitch is returned in the DJI convention
    (nose-up positive): camera_pitch (harness, + = looking down) =
    tilt - pitch_dji, so pitch_dji = tilt - camera_pitch. Roll is
    the rotation about the forward axis (see module docstring).
    """
    R = np.asarray(R, dtype=float)
    z = R[2]
    yaw = math.degrees(math.atan2(z[0], z[2]))
    cam_pitch = math.degrees(math.asin(float(np.clip(-z[1], -1.0, 1.0))))
    pitch_dji = CAMERA_TILT_DEG - cam_pitch
    roll = math.degrees(math.atan2(R[0, 1], R[1, 1]))
    return yaw, pitch_dji, roll


def R_from_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg,
                          tilt_deg=CAMERA_TILT_DEG):
    """Camera rotation R from body attitude (yaw/pitch/roll, deg).

    Inverse of ``yaw_pitch_roll_from_R`` (with the same tilt): the
    camera looks down ``tilt - pitch_dji`` from level, at heading
    ``yaw``, rolled by ``roll`` about its forward axis. Roll is
    applied by rotating the camera right/up vectors about the
    forward axis (exact for any roll; the look-at construction
    gives the roll=0 case).
    """
    yaw = math.radians(yaw_deg)
    cam_pitch = math.radians(tilt_deg - pitch_deg)
    roll = math.radians(roll_deg)
    # Forward axis from yaw + camera pitch.
    z = np.array([math.sin(yaw) * math.cos(cam_pitch),
                  -math.sin(cam_pitch),
                  math.cos(yaw) * math.cos(cam_pitch)])
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:  # looking (nearly) straight up/down
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / nx
    y = np.cross(z, x)
    if abs(roll) > 1e-12:
        # Rotate the right/up vectors about the forward axis.
        c, s = math.cos(roll), math.sin(roll)
        x, y = c * x + s * y, -s * x + c * y
    return np.stack([x, y, z])


def unwrap_deg(angles_deg):
    """Unwrap a sequence of degrees to a continuous (no +/-180 jump) series."""
    return np.degrees(np.unwrap(np.radians(
        np.asarray(angles_deg, dtype=float))))


# ---------------------------------------------------------------------------
# StatePrior — time-ordered packet buffer with interpolation + gates.
# ---------------------------------------------------------------------------
class StatePrior:
    """Tello state packets as VIO priors, timestamped on receipt.

    Feed packets with :meth:`add` (in time order). :meth:`attitude`
    interpolates yaw/pitch/roll to a video frame time (yaw unwrapped,
    linear interpolation); :meth:`velocity` returns the velocity in
    m/s ONLY when calibrated (else None — velocity priors are OFF);
    :meth:`height` returns (h_m, tof_m) with the tof sentinel
    filtered. Every accessor applies the report §6.4 gates: yaw jumps
    > ``YAW_JUMP_MAX_DEG`` between consecutive packets are rejected
    (the packet is dropped), ``tof == 10`` is treated as missing,
    and a zero velocity while vision reports motion is flagged (the
    VPS-blind case — see ``velocity_is_blind``).
    """

    def __init__(self, calibration=None):
        self.calibration = calibration or TelloCalibration()
        self._packets = []  # TelloStatePacket, time order
        self._yaw_unwrapped = []  # deg, aligned with _packets

    def __len__(self):
        return len(self._packets)

    def add(self, packet: TelloStatePacket) -> bool:
        """Add a packet; returns False if it was rejected by a gate.

        The yaw-jump gate drops a packet whose yaw moved more than
        ``YAW_JUMP_MAX_DEG`` from the previous accepted packet (a
        magnetometer glitch or a non-physical snap — the harness
        -31 deg / +101 deg snaps are exactly this).
        """
        if self._packets:
            prev = self._packets[-1]
            dyaw = abs((packet.yaw - prev.yaw + 180.0) % 360.0 - 180.0)
            dt = packet.t - prev.t
            # Gate on the per-frame jump (packets are ~10 Hz, frames
            # are 10 fps, so one packet per frame at most).
            if dt > 0 and dyaw / max(dt, 1e-9) > YAW_JUMP_MAX_DEG / 0.1:
                return False
        self._packets.append(packet)
        if len(self._yaw_unwrapped) == 0:
            self._yaw_unwrapped.append(float(packet.yaw))
        else:
            prev_u = self._yaw_unwrapped[-1]
            dyaw = (packet.yaw - (prev_u % 360.0) + 180.0) % 360.0 - 180.0
            # Unwrap against the previous UNWRAPPED value: the raw
            # yaw wraps at +/-180, so compare in wrapped space then
            # accumulate the signed step.
            raw_prev = self._packets[-2].yaw
            step = (packet.yaw - raw_prev + 180.0) % 360.0 - 180.0
            self._yaw_unwrapped.append(prev_u + step)
        return True

    def _interp_index(self, t):
        """Bracket indices (i0, i1, frac) for time t, or None."""
        if len(self._packets) < 1:
            return None
        ts = [p.t for p in self._packets]
        if t <= ts[0]:
            return 0, 0, 0.0
        if t >= ts[-1]:
            return len(ts) - 1, len(ts) - 1, 0.0
        import bisect
        i1 = bisect.bisect_right(ts, t)
        i0 = i1 - 1
        frac = (t - ts[i0]) / max(ts[i1] - ts[i0], 1e-9)
        return i0, i1, frac

    def attitude(self, t):
        """(yaw, pitch, roll) in deg at time t (interpolated, yaw unwrapped).

        Yaw is returned unwrapped (continuous, may exceed +/-180)
        and offset by the calibration yaw offset. Returns None if no
        packets are buffered.
        """
        br = self._interp_index(t)
        if br is None:
            return None
        i0, i1, frac = br
        yaw = (self._yaw_unwrapped[i0] * (1.0 - frac)
               + self._yaw_unwrapped[i1] * frac)
        pitch = (self._packets[i0].pitch * (1.0 - frac)
                 + self._packets[i1].pitch * frac)
        roll = (self._packets[i0].roll * (1.0 - frac)
                + self._packets[i1].roll * frac)
        return (yaw - self.calibration.yaw_offset_deg, pitch, roll)

    def R_prior(self, t):
        """Camera rotation R from the attitude prior at time t, or None."""
        att = self.attitude(t)
        if att is None:
            return None
        yaw, pitch, roll = att
        return R_from_yaw_pitch_roll(yaw, pitch, roll,
                                     tilt_deg=self.calibration.camera_tilt_deg)

    def velocity(self, t):
        """Velocity (vx, vy, vz) in m/s (world frame, Y up) at time t.

        Returns None when UNCALIBRATED (velocity priors OFF) — the
        unit/frame dispute (§6.9) means an assumed velocity is a
        10x error waiting to happen. When calibrated, converts the
        raw units with the calibrated scale and rotates body-frame
        velocities into the world frame using the yaw prior.

        Frame convention (ASSUMED — verify by calibration §6.9.1):
        the DJI frame is x=forward, y=right, z=up (right-handed).
        In the codebase world frame (Y up, Z forward) that maps as
        DJI-x -> world-Z, DJI-y -> world-X, DJI-z -> world-Y. A
        body-frame velocity is rotated by yaw; a global-frame
        velocity is the takeoff-frame velocity in that mapping.
        """
        if not self.calibration.velocity_ready:
            return None
        br = self._interp_index(t)
        if br is None:
            return None
        i0, i1, frac = br
        scale = self.calibration.velocity_scale_m_per_unit
        v = np.array([
            (self._packets[i0].vgx * (1.0 - frac)
             + self._packets[i1].vgx * frac),
            (self._packets[i0].vgy * (1.0 - frac)
             + self._packets[i1].vgy * frac),
            (self._packets[i0].vgz * (1.0 - frac)
             + self._packets[i1].vgz * frac)]) * scale
        if self.calibration.velocity_frame == "body":
            att = self.attitude(t)
            if att is None:
                return None
            yaw = math.radians(att[0])
            sy, cy = math.sin(yaw), math.cos(yaw)
            # DJI body: v[0]=forward, v[1]=right, v[2]=up.
            # forward -> world (sin, 0, cos); right -> (cos, 0, -sin).
            wx = v[0] * sy + v[1] * cy
            wy = v[2]
            wz = v[0] * cy - v[1] * sy
            return np.array([wx, wy, wz])
        # Global (takeoff) frame, DJI axes -> world axes:
        # DJI-x -> world-Z, DJI-y -> world-X, DJI-z -> world-Y.
        return np.array([v[1], v[2], v[0]])

    def height(self, t):
        """(h_m, tof_m) at time t; either may be None if unavailable.

        ``h`` is takeoff-relative (fused, smooth); ``tof`` is direct
        above-ground distance. The tof sentinel (10 cm) is filtered
        to None. Both are in metres.
        """
        br = self._interp_index(t)
        if br is None:
            return None, None
        i0, i1, frac = br
        h = (self._packets[i0].h * (1.0 - frac)
             + self._packets[i1].h * frac) / 100.0
        tof_vals = []
        for i in (i0, i1):
            if self._packets[i].tof != TOF_SENTINEL_CM:
                tof_vals.append(self._packets[i].tof)
        tof = (float(np.mean(tof_vals)) / 100.0) if tof_vals else None
        return h, tof

    def velocity_is_blind(self, t, visual_speed_m_per_frame,
                          frame_dt=0.1):
        """True if the velocity reads 0 while vision says it is moving.

        A zero velocity over a featureless / dark / reflective floor
        is a blind zero (VPS cannot see the floor), not a measurement
        of stillness — trust vision and flag it (§6.2 #2).
        """
        v = self.velocity(t)
        if v is None:
            return False
        return (float(np.linalg.norm(v)) < 1e-6
                and visual_speed_m_per_frame > 0.05)


# ---------------------------------------------------------------------------
# SyntheticStateProvider — GT trajectory -> state packets (harness only).
# ---------------------------------------------------------------------------
class SyntheticStateProvider:
    """Derive Tello state packets from a ground-truth pose sequence.

    Harness-only: makes the prior code path testable without a real
    drone. Packets are marked ``synthetic=True`` and are generated
    with a DOCUMENTED unit and frame (``velocity_unit`` /
    ``velocity_frame``) — the harness calibration is set to match
    them explicitly (``calibration_for``), so the prior is exercised
    end to end. The REAL default stays UNCALIBRATED.

    Quantisation matches the real packet: yaw/pitch/roll are integer
    degrees, velocities are integer raw units, tof/h are integer cm.
    A packet is emitted per frame (10 Hz), timestamped at the frame
    time — the same rate the real state packet arrives at.
    """

    def __init__(self, poses, times, velocity_unit="dm/s",
                 velocity_frame="global",
                 tilt_deg=CAMERA_TILT_DEG):
        """``poses``: list of (R, C) GT poses; ``times``: list of t."""
        self.poses = list(poses)
        self.times = list(times)
        self.velocity_unit = velocity_unit
        self.velocity_frame = velocity_frame
        self.tilt_deg = tilt_deg
        self._packets = None

    def calibration_for(self) -> TelloCalibration:
        """The calibration matching these synthetic packets (harness use)."""
        return TelloCalibration(
            velocity_unit=self.velocity_unit,
            velocity_frame=self.velocity_frame,
            camera_tilt_deg=self.tilt_deg,
            calibrated=True)

    def packets(self):
        """List of TelloStatePacket, one per frame."""
        if self._packets is not None:
            return self._packets
        scale = 0.1 if self.velocity_unit == "dm/s" else 0.01
        packets = []
        for k, ((R, C), t) in enumerate(zip(self.poses, self.times)):
            yaw, pitch, roll = yaw_pitch_roll_from_R(R)
            # Velocity from the GT trajectory (finite difference).
            if 0 < k < len(self.poses) - 1:
                dt = self.times[k + 1] - self.times[k - 1]
                v_world = (self.poses[k + 1][1] - self.poses[k - 1][1]) / dt
            elif k == 0 and len(self.poses) > 1:
                dt = self.times[1] - self.times[0]
                v_world = (self.poses[1][1] - self.poses[0][1]) / dt
            else:
                v_world = np.zeros(3)
            if self.velocity_frame == "body":
                # DJI body frame (x=forward, y=right, z=up) from the
                # world velocity, rotated by yaw. Inverse of the
                # body->world rotation in StatePrior.velocity.
                yaw_rad = math.radians(yaw)
                sy, cy = math.sin(yaw_rad), math.cos(yaw_rad)
                v_body = np.array([
                    sy * v_world[0] + cy * v_world[2],
                    cy * v_world[0] - sy * v_world[2],
                    v_world[1]])
                v_raw = v_body / scale
            else:
                # Global (takeoff) frame, DJI axes: world -> DJI is
                # the inverse of the DJI -> world mapping in
                # StatePrior.velocity: DJI-x = world-Z, DJI-y =
                # world-X, DJI-z = world-Y.
                v_raw = np.array(
                    [v_world[2], v_world[0], v_world[1]]) / scale
            h_cm = int(round(float(C[1]) * 100.0))
            packets.append(TelloStatePacket(
                t=float(t),
                yaw=int(round(yaw)),
                pitch=int(round(pitch)),
                roll=int(round(roll)),
                vgx=int(round(v_raw[0])),
                vgy=int(round(v_raw[1])),
                vgz=int(round(v_raw[2])),
                tof=h_cm, h=h_cm,
                baro=15000.0 + (float(C[1]) - 1.3) * 100.0,
                time=int(round(t)),
                bat=100,
                synthetic=True))
        self._packets = packets
        return packets
