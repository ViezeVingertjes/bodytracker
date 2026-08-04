"""Landmark indices and the Skeleton container.

Split out of solver.py so that everything downstream -- stabilisation, the VRChat
transform, the overlay, the tests -- depends only on "a set of 3D joints in
camera space", not on how those joints were produced. Importing solver pulls in
MediaPipe and roughly a second of start-up; nothing here needs that.

Indices are MediaPipe's own numbering, kept as-is: they are already the de-facto
identifiers throughout the codebase.
"""


# MediaPipe pose landmark indices we care about. Left/right are the SUBJECT's.
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32  # foot index (toe) -- gives foot direction

# Landmarks we need a real depth reading for. Everything else is ignored.
#
# Wrists are tracked but NOT sent as trackers -- VRChat has no hand or wrist
# tracker slot (the controllers drive the hands). They are here because elbow
# orientation comes from the shoulder->elbow->wrist chain, and because seeing the
# forearm in the preview is the quickest read on whether tracking is healthy.
NEEDED = (
    NOSE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
    L_FOOT, R_FOOT,
)



class Skeleton:
    """3D joints in camera space (metres). Missing joints are absent from `joints`."""

    def __init__(self, joints, visibility, pixels=None, rejected=None, body_depth=None):
        self.joints = joints          # {landmark_index: np.array([X, Y, Z])}
        self.visibility = visibility  # {landmark_index: float 0..1}
        self.pixels = pixels or {}    # {landmark_index: (px, py)} -- for overlays
        # Landmarks MediaPipe found but we dropped, with the reason. Worth
        # surfacing: a joint silently missing looks identical to a joint the
        # model never saw, and they need different fixes.
        self.rejected = rejected or {}
        self.body_depth = body_depth  # torso depth reference, metres

    def has(self, *indices):
        return all(i in self.joints for i in indices)

    def midpoint(self, a, b):
        if not self.has(a, b):
            return None
        return (self.joints[a] + self.joints[b]) * 0.5

    def get(self, index):
        return self.joints.get(index)
