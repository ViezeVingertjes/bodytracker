"""Tests for the OSC wire format and the stabilisation stages.

The OSC tests exist because a refactor once made the sender transmit *nothing*
while looking entirely correct: messages were queued into a buffer that a
separate flush() sent, and any caller who forgot to flush emitted zero bytes with
no error. These assert on what actually reaches a socket.
"""

import socket
import threading
import time

import numpy as np
import pytest
from pythonosc.osc_packet import OscPacket

import skeleton as S
from osc_out import PoseSender, TrackerSender
from stabilize import BoneLengthModel, OcclusionFiller, OutlierGate, SkeletonStabilizer
from transform import RotationPredictor, TrackerPredictor


def receive_one(port, timeout=2.0):
    """Capture a single UDP datagram and decode every message inside it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(timeout)
    received = {}

    def listen():
        try:
            data, _ = sock.recvfrom(65535)
            for timed in OscPacket(data).messages:
                received[timed.message.address] = timed.message.params
        except TimeoutError:
            pass

    thread = threading.Thread(target=listen)
    thread.start()
    return sock, thread, received


class Recorder:
    """Stands in for TrackerSender, keeping every frame it was handed."""

    def __init__(self):
        self.frames = []

    def send_frame(self, poses, head=None):
        self.frames.append((dict(poses), head))


class TestPoseSender:
    @staticmethod
    def _wired(hz=0, lead=0.0, clock=None, include_head=True):
        recorder = Recorder()
        sender = PoseSender(recorder, TrackerPredictor(), RotationPredictor(),
                            lead=lead, hz=hz, clock=clock,
                            include_head=include_head)
        return sender, recorder

    def test_inline_mode_sends_nothing_on_its_own(self):
        sender, recorder = self._wired(hz=0)
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        sender.start()          # must be a no-op at hz=0
        time.sleep(0.05)
        sender.stop()
        assert recorder.frames == []

    def test_poses_at_pairs_positions_with_rotations(self):
        sender, _ = self._wired()
        sender.update({1: np.array([1.0, 2.0, 3.0])},
                      {1: np.array([0.0, 90.0, 0.0])}, None, 0.0)
        poses, head = sender.poses_at(0.0)
        assert head is None
        position, rotation = poses[1]
        assert np.allclose(position, [1.0, 2.0, 3.0])
        assert np.allclose(rotation, [0.0, 90.0, 0.0], atol=1e-6)

    def test_position_without_rotation_gets_identity(self):
        sender, _ = self._wired()
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        poses, _ = sender.poses_at(0.0)
        assert np.allclose(poses[1][1], [0.0, 0.0, 0.0])

    def test_head_is_separated_from_body_trackers(self):
        sender, _ = self._wired()
        sender.update({1: np.zeros(3)}, {}, np.array([0.0, 1.7, 0.0]), 0.0)
        poses, head = sender.poses_at(0.0)
        assert set(poses) == {1}
        assert np.allclose(head[0], [0.0, 1.7, 0.0])

    def test_head_can_be_suppressed(self):
        sender, _ = self._wired(include_head=False)
        sender.update({1: np.zeros(3)}, {}, np.array([0.0, 1.7, 0.0]), 0.0)
        poses, head = sender.poses_at(0.0)
        assert head is None
        assert set(poses) == {1}, "suppressing the head must not drop trackers"

    def test_send_once_transmits_one_frame(self):
        sender, recorder = self._wired()
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        sender.send_once(0.0)
        assert len(recorder.frames) == 1

    def test_threaded_mode_sends_without_new_measurements(self):
        # The whole point: one update, many sends. A camera-locked sender would
        # emit exactly once here.
        sender, recorder = self._wired(hz=200)
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        with sender:
            time.sleep(0.25)
        # 200Hz for 0.25s is ~50 sends; timing on a loaded box is loose, so
        # assert the order of magnitude rather than an exact count.
        assert 15 <= len(recorder.frames) <= 90, len(recorder.frames)

    def test_stop_is_idempotent_and_joins(self):
        sender, _ = self._wired(hz=100)
        sender.start()
        sender.stop()
        sender.stop()
        assert not sender.is_running()

    def test_concurrent_updates_never_tear_a_pose(self):
        # A torn read pairs frame N's position with frame N-1's rotation. Both
        # are written from the same counter, so they must agree.
        #
        # The sender's clock is pinned to the writer's timebase and published
        # only AFTER each update lands, so it never runs ahead of the newest
        # measurement. Prediction is then a no-op (the horizon clips at 0) and
        # any disagreement can only be a torn read, which is what this tests.
        now = [0.0]
        sender, recorder = self._wired(hz=500, clock=lambda: now[0])
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                i += 1
                v = float(i % 90)
                t = i * 1e-4
                sender.update({1: np.array([v, v, v])},
                              {1: np.array([0.0, v, 0.0])}, None, t)
                now[0] = t

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            with sender:
                time.sleep(0.3)
        finally:
            stop.set()
            thread.join()

        assert recorder.frames, "sender produced nothing"
        for poses, _ in recorder.frames:
            position, rotation = poses[1]
            assert np.isclose(position[0], position[1])
            assert np.isclose(rotation[1], position[0], atol=1e-6), (
                "rotation came from a different update than the position")

    def test_stale_keys_unions_both_predictors(self):
        sender, _ = self._wired()
        sender.update({1: np.zeros(3)}, {2: np.zeros(3)}, None, 0.0)
        assert sender.stale_keys(0.5) == set()
        assert sender.stale_keys(2.0) == {1, 2}


class TestOscWireFormat:
    def test_send_frame_actually_transmits(self):
        sock, thread, got = receive_one(9911)
        TrackerSender("127.0.0.1", 9911).send_frame(
            {1: (np.array([0.1, 1.0, -2.0]), (0.0, 15.0, 0.0))},
            (np.array([0.0, 1.7, -2.0]), (0.0, 0.0, 0.0)))
        thread.join()
        sock.close()
        assert "/tracking/trackers/1/position" in got
        assert "/tracking/trackers/1/rotation" in got
        assert "/tracking/trackers/head/position" in got

    def test_values_survive_the_wire(self):
        sock, thread, got = receive_one(9912)
        TrackerSender("127.0.0.1", 9912).send_frame(
            {1: np.array([0.25, 1.5, -2.75])})
        thread.join()
        sock.close()
        params = got["/tracking/trackers/1/position"]
        assert len(params) == 3
        assert np.allclose(params, [0.25, 1.5, -2.75], atol=1e-5)

    def test_bare_position_gets_identity_rotation(self):
        sock, thread, got = receive_one(9913)
        TrackerSender("127.0.0.1", 9913).send_frame({1: np.array([0.0, 1.0, 0.0])})
        thread.join()
        sock.close()
        assert got["/tracking/trackers/1/rotation"] == [0.0, 0.0, 0.0]

    def test_whole_frame_is_one_datagram(self):
        """Eight trackers plus head sent separately would be 18 datagrams and
        540/second, enough for a busy link to reorder and tear a pose."""
        sock, thread, got = receive_one(9914)
        TrackerSender("127.0.0.1", 9914).send_frame(
            {i: np.array([0.0, float(i), 0.0]) for i in range(1, 9)},
            np.array([0.0, 1.7, 0.0]))
        thread.join()
        sock.close()
        assert len(got) == 18, "all 9 trackers x 2 addresses in a single datagram"

    def test_empty_frame_sends_nothing(self):
        sock, thread, got = receive_one(9915, timeout=0.4)
        TrackerSender("127.0.0.1", 9915).send_frame({}, None)
        thread.join()
        sock.close()
        assert got == {}


class TestOutlierGate:
    def test_plausible_motion_passes(self):
        gate = OutlierGate()
        gate({S.L_HIP: np.array([0.0, 0.0, 2.5])}, 0.0)
        out = gate({S.L_HIP: np.array([0.02, 0.0, 2.5])}, 0.033)
        assert S.L_HIP in out

    def test_teleport_is_rejected(self):
        gate = OutlierGate()
        gate({S.L_HIP: np.array([0.0, 0.0, 2.5])}, 0.0)
        out = gate({S.L_HIP: np.array([5.0, 0.0, 2.5])}, 0.033)
        # held at the previous value rather than accepting a 150 m/s jump
        assert not np.allclose(out.get(S.L_HIP, [5, 0, 2.5]), [5.0, 0.0, 2.5])


class TestBoneLengthModel:
    def test_learns_and_corrects_a_stretched_limb(self):
        model = BoneLengthModel(min_samples=5)
        for _ in range(10):
            model.observe({S.L_HIP: np.array([0.0, 0.0, 2.5]),
                           S.L_KNEE: np.array([0.0, 0.45, 2.5])})
        stretched = {S.L_HIP: np.array([0.0, 0.0, 2.5]),
                     S.L_KNEE: np.array([0.0, 0.90, 2.5])}
        fixed = model.apply(stretched)
        length = np.linalg.norm(fixed[S.L_KNEE] - fixed[S.L_HIP])
        assert abs(length - 0.45) < 0.05

    def test_grossly_wrong_joint_is_flagged_not_corrected(self):
        """A mildly wrong length means noisy depth; a grossly wrong one means the
        landmark itself is somewhere else entirely, and correcting its length
        just moves a wrong point to a wrong point at the right distance."""
        model = BoneLengthModel(min_samples=5)
        for _ in range(10):
            model.observe({S.L_HIP: np.array([0.0, 0.0, 2.5]),
                           S.L_KNEE: np.array([0.0, 0.45, 2.5])})
        bad = {S.L_HIP: np.array([0.0, 0.0, 2.5]),
               S.L_KNEE: np.array([0.0, 1.5, 2.5])}
        assert S.L_KNEE in model.implausible(bad)


class TestOcclusionFiller:
    def test_missing_joint_is_reconstructed(self):
        filler = OcclusionFiller()
        full = {S.L_HIP: np.array([-0.1, 0.0, 2.5]),
                S.R_HIP: np.array([0.1, 0.0, 2.5]),
                S.L_SHOULDER: np.array([-0.15, -0.5, 2.5]),
                S.R_SHOULDER: np.array([0.15, -0.5, 2.5]),
                S.L_ANKLE: np.array([-0.1, 0.9, 2.5])}
        filler(full, 0.0)
        without_ankle = {k: v for k, v in full.items() if k != S.L_ANKLE}
        out = filler(without_ankle, 0.033)
        assert S.L_ANKLE in out, "an occluded joint must be reconstructed"

    def test_reconstruction_follows_the_body(self):
        """Holding a missing joint in WORLD space strands it while its owner
        walks away; it must be held in the body frame instead."""
        filler = OcclusionFiller()

        def body(offset):
            return {S.L_HIP: np.array([-0.1 + offset, 0.0, 2.5]),
                    S.R_HIP: np.array([0.1 + offset, 0.0, 2.5]),
                    S.L_SHOULDER: np.array([-0.15 + offset, -0.5, 2.5]),
                    S.R_SHOULDER: np.array([0.15 + offset, -0.5, 2.5]),
                    S.L_ANKLE: np.array([-0.1 + offset, 0.9, 2.5])}

        filler(body(0.0), 0.0)
        moved = {k: v for k, v in body(0.5).items() if k != S.L_ANKLE}
        out = filler(moved, 0.033)
        assert out[S.L_ANKLE][0] > 0.1, "held joint must travel with the body"


class TestSkeletonStabilizer:
    def test_does_not_mutate_the_input(self):
        stabilizer = SkeletonStabilizer()
        joints = {S.L_HIP: np.array([0.0, 0.0, 2.5])}
        before = dict(joints)
        stabilizer(joints, 0.0)
        assert set(joints) == set(before), "caller's dict must not be mutated"

    def test_reports_which_joints_were_inferred(self):
        stabilizer = SkeletonStabilizer()
        assert isinstance(stabilizer.inferred, set)

    @pytest.mark.parametrize("flag", ["enable_bones", "enable_gate", "enable_fill"])
    def test_each_stage_can_be_disabled(self, flag):
        stabilizer = SkeletonStabilizer(**{flag: False})
        out = stabilizer({S.L_HIP: np.array([0.0, 0.0, 2.5])}, 0.0)
        assert S.L_HIP in out
