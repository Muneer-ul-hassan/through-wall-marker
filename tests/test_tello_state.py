"""Unit tests for server/tello_state.py (run: python -m pytest tests/test_tello_state.py -v).

Covers the attitude <-> rotation round-trip, the velocity frame/unit
conversions (body and global, cm/s and dm/s), the yaw-jump gate, the
tof sentinel, and the synthetic provider. These are pure-math tests —
no drone, no video.
"""
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from server.tello_state import (
    CAMERA_TILT_DEG, POSE_INVALID, POSE_VISION, R_from_yaw_pitch_roll,
    StatePrior, SyntheticStateProvider, TelloCalibration,
    TelloStatePacket, parse_state_string, unwrap_deg,
    yaw_pitch_roll_from_R,
)


def _R_yaw_pitch(yaw_deg, pitch_deg):
    """Camera R for a yaw + camera-pitch (roll 0), via the harness formula."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    z = np.array([math.sin(yaw) * math.cos(pitch),
                  -math.sin(pitch),
                  math.cos(yaw) * math.cos(pitch)])
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def test_attitude_roundtrip():
    # Body pitch 0 (level) => camera pitch = tilt (13 deg down).
    for yaw in (-150.0, -90.0, -30.0, 0.0, 45.0, 120.0, 179.0):
        for pitch_dji in (-10.0, 0.0, 5.0, 10.0):
            for roll in (-8.0, 0.0, 6.0):
                R = R_from_yaw_pitch_roll(yaw, pitch_dji, roll)
                y2, p2, r2 = yaw_pitch_roll_from_R(R)
                # Yaw wraps at +/-180; compare in wrapped space.
                dy = (y2 - yaw + 180.0) % 360.0 - 180.0
                assert abs(dy) < 1e-9, (yaw, pitch_dji, roll, y2)
                assert abs(p2 - pitch_dji) < 1e-9, (yaw, pitch_dji, roll, p2)
                assert abs(r2 - roll) < 1e-9, (yaw, pitch_dji, roll, r2)


def test_attitude_matches_harness_yaw_pitch():
    # The yaw/pitch extracted from a harness look-at R must equal the
    # yaw/pitch that built it (roll 0, pitch_dji = tilt - cam_pitch).
    for yaw, cam_pitch in [(20.0, 5.0), (-60.0, 10.0), (120.0, -3.0)]:
        R = _R_yaw_pitch(yaw, cam_pitch)
        y2, p2, r2 = yaw_pitch_roll_from_R(R)
        assert abs((y2 - yaw + 180) % 360 - 180) < 1e-9
        assert abs(p2 - (CAMERA_TILT_DEG - cam_pitch)) < 1e-9
        assert abs(r2) < 1e-9


def test_unwrap():
    assert list(unwrap_deg([170, 175, -175, -170])) == pytest.approx(
        [170, 175, 185, 190])
    assert list(unwrap_deg([0, 10, 20])) == pytest.approx([0, 10, 20])


def test_parse_state_string():
    s = ("pitch:2;roll:-1;yaw:-45;vgx:15;vgy:-3;vgz:0;templ:50;temph:55;"
         "tof:130;h:130;bat:80;baro:150.25;time:12;agx:0.01;agy:-0.02;"
         "agz:-1.00;")
    p = parse_state_string(s, t=1.5)
    assert p.t == 1.5 and p.yaw == -45 and p.pitch == 2 and p.roll == -1
    assert p.vgx == 15 and p.vgy == -3 and p.vgz == 0
    assert p.tof == 130 and p.h == 130 and p.bat == 80
    assert abs(p.baro - 150.25) < 1e-9 and abs(p.agz + 1.00) < 1e-9
    # Mission-pad fields are ignored (forbidden), never parsed to a pose.
    p2 = parse_state_string("mid:1;x:10;y:20;z:30;mpry:0,0,0;yaw:5;", t=0.0)
    assert p2.yaw == 5


def _synthetic_poses():
    # A short GT trajectory: fly forward 1 m, then yaw 90 deg while
    # still translating (an arc), at 10 fps.
    poses, times = [], []
    for k in range(30):
        t = k / 10.0
        C = np.array([0.1 * k * 0.1, 1.3, 0.4 + 0.1 * k * 0.1])
        yaw = 0.0 if k < 15 else (k - 15) * 6.0
        R = _R_yaw_pitch(yaw, 5.0)
        poses.append((R, C))
        times.append(t)
    return poses, times


def test_synthetic_velocity_roundtrip():
    poses, times = _synthetic_poses()
    for unit in ("cm/s", "dm/s"):
        for frame in ("body", "global"):
            prov = SyntheticStateProvider(
                poses, times, velocity_unit=unit, velocity_frame=frame)
            cal = prov.calibration_for()
            assert cal.velocity_ready
            prior = StatePrior(calibration=cal)
            for pkt in prov.packets():
                assert prior.add(pkt)
            # Check the velocity at a mid-flight time against GT.
            k = 10
            t = times[k]
            v_prior = prior.velocity(t)
            assert v_prior is not None
            v_gt = (poses[k + 1][1] - poses[k - 1][1]) / (times[k + 1] - times[k - 1])
            assert np.allclose(v_prior, v_gt, atol=0.02), (unit, frame, v_prior, v_gt)


def test_velocity_off_when_uncalibrated():
    poses, times = _synthetic_poses()
    prov = SyntheticStateProvider(poses, times)
    prior = StatePrior()  # default calibration: UNCALIBRATED
    for pkt in prov.packets():
        prior.add(pkt)
    assert prior.velocity(times[10]) is None
    # Attitude and height still work uncalibrated (units unambiguous).
    assert prior.attitude(times[10]) is not None
    h, tof = prior.height(times[10])
    assert h is not None and tof is not None
    assert abs(h - 1.3) < 0.011 and abs(tof - 1.3) < 0.011


def test_yaw_jump_gate():
    prior = StatePrior()
    assert prior.add(TelloStatePacket(t=0.0, yaw=0))
    assert prior.add(TelloStatePacket(t=0.1, yaw=3))
    # A +101 deg jump in one frame is not physical: rejected.
    assert not prior.add(TelloStatePacket(t=0.2, yaw=104))
    assert len(prior) == 2
    # A normal continuation is accepted.
    assert prior.add(TelloStatePacket(t=0.2, yaw=6))


def test_tof_sentinel_filtered():
    prior = StatePrior()
    prior.add(TelloStatePacket(t=0.0, tof=130, h=130))
    prior.add(TelloStatePacket(t=0.1, tof=10, h=130))  # sentinel
    h, tof = prior.height(0.05)
    assert h is not None and abs(h - 1.3) < 1e-9
    # Only the valid tof sample is used (the sentinel is dropped).
    assert tof is not None and abs(tof - 1.3) < 1e-9


def test_attitude_interpolation_and_unwrap():
    prior = StatePrior()
    prior.add(TelloStatePacket(t=0.0, yaw=175, pitch=0, roll=0))
    prior.add(TelloStatePacket(t=0.1, yaw=-175, pitch=2, roll=0))
    yaw, pitch, roll = prior.attitude(0.05)
    # Unwrapped: 175 -> 185, so the midpoint is 180.
    assert abs(yaw - 180.0) < 1e-9
    assert abs(pitch - 1.0) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
