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
