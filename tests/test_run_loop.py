"""End-to-end tests for the run loop's output path.

The unit tests cover PoseSender in isolation. These cover the thing that
actually broke for a year: TrackerPredictor advertised sending faster than the
camera and cmd_run simply never called it that way. A wiring regression here is
silent -- everything still sends, just at the wrong rate -- so it needs a test
that counts datagrams coming out of the real loop rather than one that inspects
objects.

No camera and no pose model: both are faked, because the assertion is about the
send path, not about tracking quality.
"""

import contextlib
import socket
import threading
import time

import numpy as np
import pytest

import skeleton as S
from skeleton import Skeleton


class FakeCam:
    fps = 30
    sensor_warnings = None
    visual_preset_applied = True

    def __init__(self, deadline):
        self.deadline = deadline

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        if time.monotonic() > self.deadline:
            raise KeyboardInterrupt      # what ctrl-c does to the real loop
        time.sleep(1 / 30)
        return (np.zeros((480, 848, 3), np.uint8),
                np.zeros((480, 848), np.uint16), None)

    def health_warning(self, _delivery_fps):
        return None

    def light_margin(self):
        return None


class FakeSolver:
    """A body that moves, so prediction has a velocity to work with."""

    last_rejection = None
    LAYOUT = ((S.L_HIP, -0.1, 0.0), (S.R_HIP, 0.1, 0.0),
              (S.L_SHOULDER, -0.2, -0.5), (S.R_SHOULDER, 0.2, -0.5),
              (S.L_ANKLE, -0.1, 0.8), (S.R_ANKLE, 0.1, 0.8),
              (S.L_FOOT, -0.1, 0.9), (S.R_FOOT, 0.1, 0.9),
              (S.L_EAR, -0.05, -0.7), (S.R_EAR, 0.05, -0.7))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def solve(self, _cam, _color, _depth, ms):
        t = ms / 1000.0
        joints = {idx: np.array([x + 0.3 * t, y, 2.0]) for idx, x, y in self.LAYOUT}
        return Skeleton(joints, dict.fromkeys(joints, 1.0), {}, {}, 2.0)


def count_datagrams(monkeypatch, port, send_hz, seconds=1.2):
    """Run the real cmd_run for `seconds`, return how many datagrams it sent."""
    import bodytracker

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.2)
    received = []
    stop = threading.Event()

    def listen():
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                received.append(sock.recvfrom(65535)[0])

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()

    deadline = time.monotonic() + seconds
    monkeypatch.setattr(bodytracker, "open_camera", lambda _a: FakeCam(deadline))
    monkeypatch.setattr(bodytracker, "PoseSolver", lambda *_a, **_k: FakeSolver())

    parser = bodytracker.build_parser()
    args = parser.parse_args(["run", "127.0.0.1", "--port", str(port),
                              "--send-hz", str(send_hz)])
    # The fake camera raises KeyboardInterrupt to end the loop, exactly as
    # ctrl-c does -- which also exercises the sender thread's shutdown path.
    with contextlib.suppress(KeyboardInterrupt):
        args.func(args, parser)

    time.sleep(0.3)          # let anything already in flight arrive
    stop.set()
    listener.join()
    sock.close()
    return len(received)


class TestRunLoopSendRate:
    def test_default_sends_once_per_camera_frame(self, monkeypatch):
        sent = count_datagrams(monkeypatch, 9961, send_hz=0)
        # ~1.2s of a 30fps camera. Wide bounds: this asserts "camera rate", not
        # a precise number, because the fake camera's sleep is not exact.
        assert 10 <= sent <= 50, sent

    def test_send_hz_decouples_output_from_camera_rate(self, monkeypatch):
        camera_rate = count_datagrams(monkeypatch, 9962, send_hz=0)
        decoupled = count_datagrams(monkeypatch, 9963, send_hz=90)
        # The comparison is between two runs of the same loop on the same box,
        # so it survives a slow or loaded machine in a way an absolute Hz
        # threshold would not.
        assert decoupled > camera_rate * 1.5, (camera_rate, decoupled)

    def test_sender_thread_does_not_outlive_the_loop(self, monkeypatch):
        count_datagrams(monkeypatch, 9964, send_hz=90)
        leaked = [t for t in threading.enumerate()
                  if t.is_alive() and t is not threading.current_thread()
                  and "PoseSender" in repr(t)]
        assert leaked == []
        # Nothing should still be transmitting after cmd_run returned.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 9964))
        sock.settimeout(0.4)
        with pytest.raises(TimeoutError):
            sock.recvfrom(65535)
        sock.close()
