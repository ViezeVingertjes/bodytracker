"""Tests for the coordinate maths.

Every bug this file guards against actually happened during development, and each
one was invisible in the preview window -- which is exactly why they need tests
rather than eyeballing.
"""

import numpy as np
import pytest

from transform import (
    TRACKER_ROLES,
    OneEuroFilter,
    RotationSmoother,
    TrackerPredictor,
    basis_to_unity_euler,
    to_unity,
    unity_euler_to_matrix,
)


class TestAxisConversion:
    """Camera space (X right, Y down, Z forward) -> Unity (X right, Y up, Z fwd).

    A sign error here mirrors or inverts the avatar and looks completely correct
    in the preview, because the preview draws image pixels, not Unity space.
    """

    def test_head_above_hips_stays_above(self):
        # camera space: +Y is DOWN, so the head has the more negative Y
        head = to_unity(np.array([0.0, -0.7, 2.5]))
        hips = to_unity(np.array([0.0, 0.0, 2.5]))
        assert head[1] > hips[1], "head must be above hips in Unity space"

    def test_stepping_back_moves_backward(self):
        near = to_unity(np.array([0.0, 0.0, 2.0]))
        far = to_unity(np.array([0.0, 0.0, 3.0]))
        assert far[2] < near[2], "moving away from the camera must move -Z"

    def test_left_right_not_mirrored(self):
        # The subject's right hand appears at camera -X.
        right_hand = to_unity(np.array([-0.4, 0.0, 2.5]))
        # For a character facing +Z in Unity, the right hand is at +X.
        assert right_hand[0] > 0, "left/right must not be mirrored"

    def test_is_a_pure_negation(self):
        point = np.array([0.3, -0.4, 2.5])
        assert np.allclose(to_unity(point), -point)


class TestEulerConversion:
    """VRChat applies euler angles Z, X, Y. A wrong order looks almost right at
    small angles and diverges badly at large ones."""

    def test_identity_basis_is_zero_rotation(self):
        euler = basis_to_unity_euler([1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert np.allclose(euler, [0, 0, 0], atol=1e-9)

    def test_round_trip_over_random_rotations(self):
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(2000):
            euler = rng.uniform(-180, 180, 3)
            euler[0] = rng.uniform(-89, 89)  # away from gimbal lock
            matrix = unity_euler_to_matrix(euler)
            back = basis_to_unity_euler(matrix[:, 0], matrix[:, 1], matrix[:, 2])
            # Euler triples are not unique; compare the MATRICES they produce.
            worst = max(worst, np.abs(matrix - unity_euler_to_matrix(back)).max())
        assert worst < 1e-9, f"round-trip error {worst}"

    @pytest.mark.parametrize("x", [90.0, -90.0, 89.9999])
    def test_gimbal_lock_is_finite_and_correct(self, x):
        matrix = unity_euler_to_matrix([x, 37.0, 21.0])
        euler = basis_to_unity_euler(matrix[:, 0], matrix[:, 1], matrix[:, 2])
        assert np.all(np.isfinite(euler)), "gimbal lock must not produce NaN"
        assert np.abs(matrix - unity_euler_to_matrix(euler)).max() < 1e-6


class TestRotationSmoother:
    """Regression: rejecting frames that differ from a STALE stored value locked
    out genuine rotation permanently -- a real 90 degree foot turn was rejected
    forever and the foot never turned again."""

    def test_single_frame_glitch_is_suppressed(self):
        smoother = RotationSmoother()
        for _ in range(4):
            smoother({3: np.array([0.0, 10.0, 0.0])})
        out = smoother({3: np.array([0.0, 170.0, 0.0])})
        assert abs(out[3][1] - 10.0) < 5.0, "one-frame flip must be rejected"

    def test_sustained_rotation_is_eventually_accepted(self):
        smoother = RotationSmoother()
        for _ in range(3):
            smoother({3: np.array([0.0, 0.0, 0.0])})
        outputs = [smoother({3: np.array([0.0, 90.0, 0.0])})[3][1]
                   for _ in range(10)]
        assert max(outputs) > 80.0, "a real sustained turn must be accepted"

    def test_small_motion_passes_through(self):
        smoother = RotationSmoother()
        outputs = [smoother({3: np.array([0.0, float(i * 5), 0.0])})[3][1]
                   for i in range(6)]
        assert outputs[-1] > outputs[0], "normal motion must not be blocked"


class TestTrackerPredictor:
    def test_predicts_along_measured_velocity(self):
        predictor = TrackerPredictor()
        for i in range(12):
            predictor.update({1: np.array([i * 0.033, 0.0, 0.0])}, i * 0.033)
        now = 11 * 0.033
        base = predictor.at(now, 0.0)[1]
        led = predictor.at(now, 0.050)[1]
        # ~1 m/s for 50 ms
        assert 0.03 < led[0] - base[0] < 0.07

    def test_horizon_is_clamped(self):
        predictor = TrackerPredictor()
        for i in range(12):
            predictor.update({1: np.array([i * 0.033, 0.0, 0.0])}, i * 0.033)
        far = predictor.at(11 * 0.033, 10.0)[1]
        assert far[0] < 1.0, "an absurd lead must not fling a tracker away"

    def test_speed_is_clamped(self):
        predictor = TrackerPredictor(max_speed=1.0)
        predictor.update({1: np.array([0.0, 0.0, 0.0])}, 0.0)
        predictor.update({1: np.array([100.0, 0.0, 0.0])}, 0.033)  # absurd jump
        out = predictor.at(0.033, 0.1)[1]
        assert np.isfinite(out).all()
        assert out[0] < 101.0


class TestOneEuroFilter:
    def test_first_sample_passes_through(self):
        f = OneEuroFilter()
        assert np.allclose(f(np.array([1.0, 2.0, 3.0]), 0.0), [1.0, 2.0, 3.0])

    def test_converges_to_a_constant_input(self):
        f = OneEuroFilter()
        for i in range(200):
            out = f(np.array([5.0, 0.0, 0.0]), i * 0.033)
        assert abs(out[0] - 5.0) < 0.01

    def test_non_increasing_time_does_not_explode(self):
        f = OneEuroFilter()
        f(np.array([1.0, 0.0, 0.0]), 1.0)
        out = f(np.array([2.0, 0.0, 0.0]), 1.0)  # same timestamp
        assert np.all(np.isfinite(out))


def test_tracker_roles_match_slimevr_convention():
    """SlimeVR's VRCOSCHandler is the de-facto standard. VRChat's own prose lists
    the roles in a DIFFERENT order, and taking that literally gives chest=2 and
    feet=3,4, which is wrong."""
    assert TRACKER_ROLES == {
        "hip": 1, "left_foot": 2, "right_foot": 3,
        "left_knee": 4, "right_knee": 5, "chest": 6,
        "left_elbow": 7, "right_elbow": 8,
    }
