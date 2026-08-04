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
    """Stands in for TrackerSender, keeping every frame it was handed.

    Mirrors the real contract: returns True only when something would actually
    go on the wire, so callers counting datagrams behave as they do in
    production.
    """

    def __init__(self):
        self.frames = []

    def send_frame(self, poses, head=None):
        self.frames.append((dict(poses), head))
        return bool(poses) or head is not None


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

    def test_a_failing_send_does_not_kill_the_thread(self):
        # The failure this guards against is not a crash -- it is SILENCE. A
        # dead sender thread leaves the capture loop running happily while
        # every tracker in VRChat freezes in mid-air, because OSC has no
        # tracker-removed message. python-osc's socket is non-blocking, so
        # EAGAIN and ENETUNREACH are live possibilities at 90Hz over WiFi.
        calls = []

        class Flaky:
            def send_frame(self, poses, head=None):
                calls.append(1)
                if len(calls) == 3:
                    raise OSError(101, "Network is unreachable")
                return bool(poses) or head is not None

        positions, rotations = TrackerPredictor(), RotationPredictor()
        sender = PoseSender(Flaky(), positions, rotations, hz=200)
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        with sender:
            time.sleep(0.2)
            alive = sender.is_running()

        assert alive, "one send error killed the sender thread"
        assert len(calls) > 5, "thread stopped sending after the error"
        assert sender.send_errors == 1
        assert isinstance(sender.last_error, OSError)

    def test_frames_sent_counts_datagrams_not_calls(self):
        # An empty frame transmits nothing, so counting it inflates the
        # achieved-Hz figure that exists to reveal the bug above.
        sender, recorder = self._wired()
        sender.send_once(0.0)                      # nothing measured yet
        assert sender.frames_sent == 0, "counted a frame that sent no datagram"
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        sender.send_once(0.0)
        assert sender.frames_sent == 1
        assert len(recorder.frames) == 2           # called twice, sent once

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

    def test_fast_and_slow_joints_have_different_limits(self):
        """The per-joint split is the class's entire reason to exist.

        Both existing tests used L_HIP only, so swapping the two limits, or
        emptying FAST_JOINTS altogether, changed nothing they could see.
        """
        gate = OutlierGate()          # slow 2.0 m/s, fast 5.0 m/s
        start, dt = np.array([0.0, 0.0, 2.5]), 0.033
        moved = np.array([0.1, 0.0, 2.5])          # 3.0 m/s -- between the two
        gate({S.L_WRIST: start, S.L_KNEE: start}, 0.0)
        out = gate({S.L_WRIST: moved, S.L_KNEE: moved}, dt)

        assert np.allclose(out[S.L_WRIST], moved), (
            "a wrist at 3 m/s is normal and must pass")
        assert np.allclose(out[S.L_KNEE], start), (
            "a knee at 3 m/s exceeds the slow limit and must be held")

    def test_a_held_joint_is_reported_as_held(self):
        gate = OutlierGate()
        gate({S.L_HIP: np.array([0.0, 0.0, 2.5])}, 0.0)
        gate({S.L_HIP: np.array([5.0, 0.0, 2.5])}, 0.033)
        assert gate.held == {S.L_HIP}

    def test_hold_expires_rather_than_lasting_for_ever(self):
        gate = OutlierGate(max_hold=0.25)
        gate({S.L_HIP: np.array([0.0, 0.0, 2.5])}, 0.0)
        # Same implausible position, but long after the last good measurement.
        out = gate({S.L_HIP: np.array([5.0, 0.0, 2.5])}, 1.0)
        assert S.L_HIP not in out, (
            "past max_hold the joint must be dropped, not held for ever")


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
        # WHICH end moved matters and was unasserted: the proximal joint is
        # nearer the torso and better observed, so the distal one must absorb
        # the correction. Checking only the length let a fix that drags the hip
        # instead of the knee pass unnoticed.
        assert np.allclose(fixed[S.L_HIP], stretched[S.L_HIP]), (
            "the proximal joint must not move")
        assert not np.allclose(fixed[S.L_KNEE], stretched[S.L_KNEE])

    def test_hallucinated_frames_do_not_retrain_the_model(self):
        """A limb held at the wrong length must not become the learned length.

        The gate cannot defend this: it catches SPEED, and a landmark parked at
        double length has none. Observed unfiltered, the median crosses after
        ~1.5s and implausible() stops firing -- then apply() enforces the wrong
        length on the real joint long after the occlusion clears.
        """
        stabilizer = SkeletonStabilizer(enable_gate=False, enable_fill=False)
        good = {S.L_HIP: np.array([0.0, 0.0, 2.5]),
                S.L_KNEE: np.array([0.0, 0.45, 2.5])}
        for i in range(60):
            stabilizer(good, i / 30)
        learned = stabilizer.bones.length((S.L_HIP, S.L_KNEE))

        bad = {S.L_HIP: np.array([0.0, 0.0, 2.5]),
               S.L_KNEE: np.array([0.0, 0.90, 2.5])}
        for i in range(60, 150):          # 3 s of a hallucinated limb
            stabilizer(bad, i / 30)

        after = stabilizer.bones.length((S.L_HIP, S.L_KNEE))
        assert abs(after - learned) < 0.02, (
            f"learned length drifted {learned:.3f} -> {after:.3f} on bad frames")
        assert S.L_KNEE in stabilizer.bones.implausible(bad), (
            "the model stopped recognising the hallucination as implausible")

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

    def test_a_returning_joint_is_ramped_not_snapped(self):
        """The blend was algebraically a no-op and nothing noticed.

        `_local` was overwritten from the incoming measurement at the top of
        the frame, then read back to compute the held position -- and
        basis.T followed by basis is the identity, so `held` came back exactly
        equal to the measurement. held*(1-w) + measured*w == measured for every
        w, so a joint returning 15 cm away jumped in one frame: the precise
        glitch this class exists to remove.
        """
        filler = OcclusionFiller(blend_time=0.3)

        def body(ankle_x=None):
            j = {S.L_HIP: np.array([-0.1, 0.0, 2.5]),
                 S.R_HIP: np.array([0.1, 0.0, 2.5]),
                 S.L_SHOULDER: np.array([-0.2, -0.5, 2.5]),
                 S.R_SHOULDER: np.array([0.2, -0.5, 2.5])}
            if ankle_x is not None:
                j[S.L_ANKLE] = np.array([ankle_x, 0.9, 2.5])
            return j

        for i in range(5):                       # seen at -0.10
            filler(body(-0.10), i / 30)
        for i in range(5, 10):                   # occluded
            filler(body(), i / 30)

        # Returns 15 cm away. Sample partway through the ramp.
        first = filler(body(0.05), 10 / 30)[S.L_ANKLE][0]
        mid = filler(body(0.05), 10 / 30 + 0.15)[S.L_ANKLE][0]
        end = filler(body(0.05), 10 / 30 + 0.30)[S.L_ANKLE][0]

        assert first < -0.05, f"snapped straight to the measurement ({first:+.3f})"
        assert -0.10 < mid < 0.05, f"midpoint not between held and measured ({mid:+.3f})"
        assert abs(end - 0.05) < 1e-6, "must reach the measurement by blend_time"

    def test_reocclusion_midway_restarts_the_ramp(self):
        """Leaving the clock running meant elapsed >= blend_time on return, so
        the joint snapped anyway -- reintroducing the bug the ramp prevents."""
        filler = OcclusionFiller(blend_time=0.3)

        def body(ankle_x=None):
            j = {S.L_HIP: np.array([-0.1, 0.0, 2.5]),
                 S.R_HIP: np.array([0.1, 0.0, 2.5]),
                 S.L_SHOULDER: np.array([-0.2, -0.5, 2.5]),
                 S.R_SHOULDER: np.array([0.2, -0.5, 2.5])}
            if ankle_x is not None:
                j[S.L_ANKLE] = np.array([ankle_x, 0.9, 2.5])
            return j

        filler(body(-0.10), 0.0)
        filler(body(), 0.1)                  # occluded
        filler(body(0.05), 0.2)              # returns, ramp starts, holds -0.10
        filler(body(), 0.3)                  # gone again mid-ramp, holds 0.05
        # Returns much later, somewhere new. The ramp must restart from where
        # it was being held; a stale clock makes elapsed >= blend_time so it
        # snaps to the measurement instead.
        out = filler(body(0.30), 1.0)
        assert out[S.L_ANKLE][0] < 0.20, (
            f"snapped to the new measurement ({out[S.L_ANKLE][0]:+.3f}) instead "
            "of restarting the ramp from the held position")


def torso(offset=0.0):
    return {S.L_HIP: np.array([-0.1 + offset, 0.0, 2.5]),
            S.R_HIP: np.array([0.1 + offset, 0.0, 2.5]),
            S.L_SHOULDER: np.array([-0.2 + offset, -0.5, 2.5]),
            S.R_SHOULDER: np.array([0.2 + offset, -0.5, 2.5])}


class TestSkeletonStabilizer:
    def test_does_not_mutate_the_input(self):
        stabilizer = SkeletonStabilizer()
        joints = {**torso(), S.L_KNEE: np.array([-0.1, 0.45, 2.5])}
        before = {k: v.copy() for k, v in joints.items()}
        for i in range(30):
            stabilizer(joints, i / 30)
        assert set(joints) == set(before), "caller's dict must not be mutated"
        # Compare VALUES too. Checking keys alone let in-place mutation of the
        # caller's numpy arrays pass unnoticed, which is the mutation that
        # actually corrupts the solver's own skeleton.
        for key, original in before.items():
            assert np.array_equal(joints[key], original), f"{key} mutated in place"

    def test_reports_a_gate_held_joint_as_inferred(self):
        # `set(output) - set(input)` cannot see this: the joint IS in the input,
        # its value is just not the measured one. The preview colours inferred
        # joints differently precisely so this failure is visible.
        stabilizer = SkeletonStabilizer(enable_bones=False, enable_fill=False)
        joints = {**torso(), S.L_KNEE: np.array([-0.1, 0.45, 2.5])}
        for i in range(5):
            stabilizer(joints, i / 30)
        teleported = {**torso(), S.L_KNEE: np.array([1.2, 0.45, 2.5])}
        stabilizer(teleported, 5 / 30)
        assert S.L_KNEE in stabilizer.inferred, (
            "a gate-held joint is not a measurement and must be marked")

    def test_reports_an_implausible_then_refilled_joint_as_inferred(self):
        # The case the drop calls worse than absence: present, confidently
        # wrong, dropped, then reinvented by the filler -- and it stayed in the
        # input keyset the whole time, so it looked measured.
        stabilizer = SkeletonStabilizer(enable_gate=False)
        good = {**torso(), S.L_KNEE: np.array([-0.1, 0.45, 2.5]),
                S.L_ANKLE: np.array([-0.1, 0.9, 2.5])}
        for i in range(40):
            stabilizer(good, i / 30)
        hallucinated = {**good, S.L_ANKLE: np.array([-0.1, 2.4, 2.5])}
        stabilizer(hallucinated, 40 / 30)
        assert S.L_ANKLE in stabilizer.inferred

    def test_a_normal_frame_reports_nothing_inferred(self):
        stabilizer = SkeletonStabilizer()
        joints = {**torso(), S.L_KNEE: np.array([-0.1, 0.45, 2.5])}
        for i in range(30):
            stabilizer(joints, i / 30)
        assert stabilizer.inferred == set(), "clean tracking must not be flagged"

    @pytest.mark.parametrize("flag", ["enable_bones", "enable_gate", "enable_fill"])
    def test_each_stage_can_be_disabled(self, flag):
        # A single joint at a single timestamp passed through every stage
        # unchanged whether the stages existed or not, so this could not detect
        # the flags being ignored. Drive a case each stage demonstrably alters.
        joints = {**torso(), S.L_KNEE: np.array([-0.1, 0.45, 2.5])}
        teleported = {**torso(), S.L_KNEE: np.array([1.2, 0.45, 2.5])}

        attr = {"enable_bones": "bones", "enable_gate": "gate",
                "enable_fill": "filler"}[flag]
        off = SkeletonStabilizer(**{flag: False})
        on = SkeletonStabilizer()

        # The direct check: the flag is honoured at construction. Ignoring it
        # builds the stage anyway, which this catches and the old single-joint
        # pass-through could not.
        assert getattr(off, attr) is None
        assert getattr(on, attr) is not None

        for i in range(40):
            off(joints, i / 30)
        out = off(teleported, 40 / 30)
        assert S.L_KNEE in out, "disabling a stage must not drop joints"
