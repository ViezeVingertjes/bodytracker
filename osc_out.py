"""VRChat OSC tracker output.

Sends tracker poses to VRChat's OSC tracker API. VRChat listens on UDP 9000.

Coordinate contract (from VRChat's OSC tracker spec):
  - Unity space: left-handed, +Y up, 1.0 == 1 metre
  - rotations are euler degrees, applied Z, X, Y
  - indices 1..8 are body trackers; "head" is the space-alignment anchor, not a
    body tracker -- VRChat shifts the whole tracking space each frame so the head
    point lands on the avatar's head bone (yaw is lerped over ~10s)

Index -> role assignment is NOT confirmed by VRChat's docs; the spec only lists
which roles exist (hip, chest, 2x feet, 2x knees, 2x elbows). See RESEARCH.md
section 2 -- settle it empirically before relying on a particular numbering.

Everything for one frame goes out as a single OSC BUNDLE rather than as loose
messages. With 8 trackers plus the head that is 18 messages per frame, 540
datagrams per second at 30 Hz -- enough for a busy WiFi link to reorder or drop
some, which would tear a pose across frames (a hip from frame N with a foot from
N-1). A bundle is one datagram: it arrives whole or not at all.
"""

import threading
import time

from pythonosc import osc_bundle_builder, osc_message_builder
from pythonosc.udp_client import UDPClient

VRCHAT_OSC_PORT = 9000


def _message(address, values):
    builder = osc_message_builder.OscMessageBuilder(address=address)
    for value in values:
        builder.add_arg(float(value))
    return builder.build()


class TrackerSender:
    """One call sends one frame. There is deliberately no queue to forget to flush.

    An earlier version had send_tracker() append to a buffer that a separate
    flush() transmitted. A caller that forgot flush() sent absolutely nothing,
    silently and forever -- which is exactly what happened during development.
    Taking the whole frame in one call makes that mistake unrepresentable.
    """

    def __init__(self, host: str, port: int = VRCHAT_OSC_PORT):
        self.host = host
        self.port = port
        self._client = UDPClient(host, port)

    def send_frame(self, trackers, head=None):
        """Send a whole frame as a single OSC bundle.

        trackers: {index -> position} or {index -> (position, rotation)}, where
                  index is 1..8.
        head:     position, or (position, rotation), or None.
        """
        # IMMEDIATELY: VRChat should apply the pose on arrival, not schedule it.
        # A timestamped bundle would rely on clock agreement between this machine
        # and the Quest, which is not something we control.
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

        entries = list(trackers.items())
        if head is not None:
            entries.append(("head", head))

        empty = True
        for index, pose in entries:
            position, rotation = _split_pose(pose)
            bundle.add_content(
                _message(f"/tracking/trackers/{index}/position", position)
            )
            bundle.add_content(
                _message(f"/tracking/trackers/{index}/rotation", rotation)
            )
            empty = False

        if not empty:
            self._client.send(bundle.build())


class PoseSender:
    """Owns the predictors and decides when a frame goes out.

    Two modes, one implementation. With `hz=0` the caller pumps it once per
    camera frame, which is what this pipeline has always done. With `hz>0` a
    thread sends on its own clock, independent of the camera.

    The second mode is the point. VRChat applies tracker data per RENDERED
    frame, so 30Hz updates on a 72-90Hz headset are held for two or three
    frames -- staleness stacked on top of the pipeline latency that --lead-ms
    already exists to cancel. The predictors can produce a pose for any moment,
    so a faster sender needs no new measurements: it re-extrapolates the ones it
    has. Sub-stepping the capture loop instead would not work, because at ~30fps
    with ~33ms of work per iteration that loop has no idle time to sub-step
    into -- which is what makes a thread the only option rather than a
    preference.

    Prediction is evaluated on the SENDING side, not at update time. A pose
    extrapolated when its frame arrived would be exactly as stale as the frame
    by the time it actually went out.
    """

    def __init__(self, sender, positions, rotations, lead=0.0, hz=0,
                 clock=None, include_head=True):
        self._sender = sender
        self._positions = positions
        self._rotations = rotations
        self.lead = lead
        self.hz = hz
        self.include_head = include_head
        self._clock = clock or time.monotonic
        # ONE lock over BOTH predictors. A lock per predictor would let a send
        # land between the position and rotation updates of a single frame,
        # pairing this frame's hip with last frame's hip rotation -- precisely
        # the inconsistency RotationPredictor was added to remove.
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        # Datagrams actually transmitted. A requested rate is not an achieved
        # one: the sender thread competes for the GIL with inference and
        # overlay drawing, so asking for 90Hz can quietly deliver 60. Reporting
        # the request alone would let a user conclude a rate "did not help"
        # when they never received it.
        self.frames_sent = 0

    def update(self, trackers, rotations, head, t):
        """Hand the predictors one frame of measurements."""
        with self._lock:
            self._positions.update(trackers, t)
            self._rotations.update(rotations, t)
            if head is not None:
                self._positions.update({"head": head}, t)

    def poses_at(self, t):
        """({index: (position, rotation)}, head_pose or None) at time `t`.

        Emits EVERY tracker seen so far, not just those measured this frame.
        OSC is stateless and VRChat has no "tracker removed" message: stop
        sending an address and VRChat keeps whatever it last received, so going
        quiet does not hide a missing foot -- it freezes that foot in mid-air,
        indefinitely. With ankles measured at 71-87% presence, silence would
        mean feet freezing several times a second, which reads as broken
        tracking rather than as a dropped frame.

        Trackers not measured recently are reported by stale_keys() so callers
        can say so honestly, but they keep being sent: a slightly out-of-date
        foot beats a frozen one, and both beat VRChat inventing its own.
        """
        with self._lock:
            positions = self._positions.at(t, self.lead)
            rotations = self._rotations.at(t, self.lead)
        poses, head_pose = {}, None
        for key, position in positions.items():
            rotation = rotations.get(key, (0.0, 0.0, 0.0))
            if key == "head":
                if self.include_head:
                    head_pose = (position, rotation)
            else:
                poses[key] = (position, rotation)
        return poses, head_pose

    def send_once(self, t):
        poses, head_pose = self.poses_at(t)
        # Sent outside the lock: this is network I/O, and holding the lock
        # across it would stall the capture thread behind a slow socket.
        self._sender.send_frame(poses, head_pose)
        self.frames_sent += 1

    def stale_keys(self, t):
        with self._lock:
            return (set(self._positions.stale_keys(t))
                    | set(self._rotations.stale_keys(t)))

    def start(self):
        if self.hz <= 0 or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.hz
        next_tick = self._clock()
        while not self._stop.is_set():
            self.send_once(self._clock())
            next_tick += period
            delay = next_tick - self._clock()
            if delay < 0:
                # Fell behind. Resync instead of accumulating debt, which would
                # otherwise be repaid as a burst of back-to-back sends.
                next_tick = self._clock()
                delay = 0.0
            self._stop.wait(delay)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        return False


def _split_pose(pose):
    """Accept either a bare position or a (position, rotation) pair."""
    if (
        isinstance(pose, tuple)
        and len(pose) == 2
        and hasattr(pose[0], "__len__")
        and len(pose[0]) == 3
    ):
        return pose[0], pose[1]
    return pose, (0.0, 0.0, 0.0)
