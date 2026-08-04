"""Camera space -> VRChat space, plus the smoothing that makes it usable.

THE AXIS CONVERSION
-------------------
Camera space is right-handed: +X right, +Y down, +Z forward from the lens.
(Verified on the real camera, see capture.deproject.)

VRChat/Unity is left-handed: +X right, +Y up, +Z forward, metres.

Two things have to happen at once:

  1. handedness flip, because Unity is left-handed
  2. a 180 deg turn about Y, because the user FACES the camera -- their forward
     is -Z in camera space, but VRChat assumes a head rotation of (0,0,0) means
     forward-looking

Both together are simply the negation of all three axes:

    unity = (-Xc, -Yc, -Zc)

Sanity checks that this is right, not just plausible:
  - head is up      -> Yc negative -> unity Y positive        (head above hips)
  - step back       -> Zc larger   -> unity Z more negative   (moves backward)
  - user's right hand is at camera -X -> unity +X, and for a character facing
    +Z in Unity the right hand IS at +X, so left/right are NOT mirrored

Getting a sign wrong here is the mirrored/inverted-avatar bug, and it looks
perfectly fine in the preview window -- which is why the reasoning is written out
rather than left as a one-line negation.
"""

import math

import numpy as np

import skeleton as S


def to_unity(point):
    """Camera-space metres -> VRChat/Unity-space metres."""
    return np.array([-point[0], -point[1], -point[2]], dtype=float)


def basis_to_unity_euler(right, up, forward):
    """Orthonormal Unity-space axes -> VRChat euler angles in degrees.

    VRChat's spec says euler angles are "applied Z, X, Y". That is Unity's
    convention, where the composed rotation matrix is M = Ry * Rx * Rz with the
    axes as columns. Expanding that product:

        M = [[cy*cz + sy*sx*sz,  -cy*sz + sy*sx*cz,  sy*cx],
             [cx*sz,              cx*cz,             -sx  ],
             [-sy*cz + cy*sx*sz,  sy*sz + cy*sx*cz,  cy*cx]]

    which inverts to:
        x = asin(-M[1][2])
        z = atan2(M[1][0], M[1][1])
        y = atan2(M[0][2], M[2][2])

    Derived rather than copied, because a wrong Euler order produces rotations
    that look almost right at small angles and diverge badly at large ones --
    the kind of bug that survives casual inspection.
    """
    m = np.stack([right, up, forward], axis=1)

    sx = -m[1][2]
    sx = float(np.clip(sx, -1.0, 1.0))
    x = math.asin(sx)

    # Gimbal lock: cos(x) ~ 0 leaves y and z degenerate. Fold the rotation into
    # y and pin z, which is stable rather than producing NaNs or wild spins.
    if abs(math.cos(x)) < 1e-6:
        y = math.atan2(-m[2][0], m[0][0])
        z = 0.0
    else:
        z = math.atan2(m[1][0], m[1][1])
        y = math.atan2(m[0][2], m[2][2])

    return np.degrees([x, y, z])


def unity_euler_to_matrix(euler_deg):
    """Inverse of basis_to_unity_euler, used to verify the derivation."""
    x, y, z = np.radians(euler_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return ry @ rx @ rz


def _orthonormal(primary, secondary):
    """Build a right/up/forward basis from two roughly-perpendicular vectors.

    `primary` is trusted as the dominant axis (up), `secondary` only fixes the
    roll about it -- so a noisy secondary tilts the result slightly rather than
    corrupting the main direction.
    """
    up = np.asarray(primary, dtype=float)
    n = np.linalg.norm(up)
    if n < 1e-6:
        return None
    up = up / n

    right = np.asarray(secondary, dtype=float)
    right = right - up * float(np.dot(right, up))
    n = np.linalg.norm(right)
    if n < 1e-6:
        return None
    right = right / n

    forward = np.cross(right, up)
    return right, up, forward


class OneEuroFilter:
    """Low-lag adaptive smoothing.

    A plain low-pass forces a choice between jitter when still and lag when
    moving. One Euro adapts its cutoff to speed: heavy smoothing when nearly
    stationary, light smoothing when moving fast. That matters here because
    VRChat applies the head anchor per frame with no interpolation of its own,
    so unsmoothed input reads as visible jitter.
    """

    def __init__(self, min_cutoff=1.0, beta=0.4, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self._x_prev is None:
            self._x_prev, self._dx_prev, self._t_prev = x, np.zeros_like(x), t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


# Which VRChat tracker index each role uses.
#
# This follows SlimeVR's mapping, which is the de-facto standard. From
# SlimeVR-Server, osc/VRCOSCHandler.kt, getVRCOSCTrackersId():
#
#     HIP -> 1;  LEFT_FOOT -> 2;  RIGHT_FOOT -> 3
#     LEFT_UPPER_LEG -> 4;  RIGHT_UPPER_LEG -> 5
#     UPPER_CHEST -> 6;  LEFT_UPPER_ARM -> 7;  RIGHT_UPPER_ARM -> 8
#
# carrying the comment: "Don't change as third party applications may rely on
# this for mapping trackers to body parts."
#
# VRChat's own docs never state an index->role mapping -- they only list which
# roles exist, in the order "hip, chest, 2x feet, 2x knees, 2x elbows". Reading
# that order literally gives chest=2 and feet=3,4, which is what this used to do
# and is WRONG relative to every other tool in the ecosystem.
#
# VRChat itself most likely does not care about the index at all: its docs say
# the system "should function similarly to our existing implementation for
# SteamVR trackers", and Auto-center OSC Trackers works by finding "the two
# lowest trackers on the y axis" and assuming those are the feet -- i.e. roles
# are inferred from POSITION at calibration, not from the address number. But
# matching the ecosystem convention costs nothing and is what other OSC
# receivers, overlays and debugging tools expect to see.
TRACKER_ROLES = {
    "hip": 1,
    "left_foot": 2,
    "right_foot": 3,
    "left_knee": 4,
    "right_knee": 5,
    "chest": 6,
    "left_elbow": 7,
    "right_elbow": 8,
}

# VRChat's docs warn that fewer trackers often track better, because its IK
# compensates well for a MISSING point and badly for a WRONG one. That is a
# warning about reliability, not a rule about count -- so the roles were measured
# rather than guessed. Per-role, over a 446-frame recording of real movement:
#
#     role          idx  measured  jitter
#     hip             1      100%   4.2mm   core
#     left_foot       2       99%   7.9mm   standard FBT
#     right_foot      3      100%   4.8mm   standard FBT
#     left_knee       4      100%   6.7mm   good, but IK usually solves knees fine
#     right_knee      5      100%   5.6mm   "
#     chest           6      100%   3.1mm   core -- our STEADIEST tracker
#     left_elbow      7      100%   5.5mm   shares the arm with your controller
#     right_elbow     8       94%   6.7mm   "
#
# Default is hip + chest + both feet. Chest earns its place on the numbers: it is
# the single most stable point we produce, and it gives VRChat torso lean and
# twist that hip-only cannot.
#
# Knees are available and track well (--roles ...,left_knee,right_knee); they are
# not default because VRChat's IK already infers knee position from hip and foot,
# so they add little unless you kneel or sit a lot.
#
# Elbows are deliberately NOT default. Your hands come from the Quest controllers,
# so an elbow tracker constrains an arm whose endpoint is already authoritatively
# tracked -- a slightly-wrong elbow fights the controller rather than helping.
DEFAULT_ROLES = ("hip", "chest", "left_foot", "right_foot")


def _foot(skeleton, ankle, toe):
    """Foot position, preferring the ankle but accepting the toe.

    Measured presence: ankles 71%/87%, toes 92%/94%. The toe is seen more often
    because it sits against the floor with good stereo texture, while the ankle
    is frequently the lowest-contrast part of the leg. Falling back to the toe
    keeps a foot tracker alive through ankle dropouts, and the small forward
    offset that introduces is far less visible than a foot that stops updating.
    """
    point = skeleton.get(ankle)
    if point is not None:
        return point
    return skeleton.get(toe)


def build_trackers(skeleton, roles=DEFAULT_ROLES, with_rotations=True):
    """Skeleton (camera space) -> ({index: position}, {index: rotation}, head).

    Roles whose joints are missing are absent from the result. Deciding what to
    do about that is main.py's job via TrackerLatch -- omitting an address does
    NOT retract a tracker in VRChat, it freezes it, so the decision needs the
    latch's history and cannot be made here.
    """
    sources = {
        "hip": lambda: skeleton.midpoint(S.L_HIP, S.R_HIP),
        "chest": lambda: skeleton.midpoint(S.L_SHOULDER, S.R_SHOULDER),
        "left_foot": lambda: _foot(skeleton, S.L_ANKLE, S.L_FOOT),
        "right_foot": lambda: _foot(skeleton, S.R_ANKLE, S.R_FOOT),
        "left_knee": lambda: skeleton.get(S.L_KNEE),
        "right_knee": lambda: skeleton.get(S.R_KNEE),
        "left_elbow": lambda: skeleton.get(S.L_ELBOW),
        "right_elbow": lambda: skeleton.get(S.R_ELBOW),
    }

    trackers = {}
    for role in roles:
        point = sources[role]()
        if point is not None:
            trackers[TRACKER_ROLES[role]] = to_unity(point)

    rotations = build_rotations(skeleton, roles) if with_rotations else {}

    # Head anchor: ear midpoint is steadier than the nose, which swings with
    # head yaw. Falls back to the nose when an ear is occluded.
    head = skeleton.midpoint(S.L_EAR, S.R_EAR)
    if head is None:
        head = skeleton.get(S.NOSE)
    return trackers, rotations, (to_unity(head) if head is not None else None)


def build_rotations(skeleton, roles):
    """Per-role euler rotations in VRChat space, for roles we can determine.

    Only rotations that are genuinely observable are produced. A knee's twist
    about its own axis, for example, is not recoverable from three points, so
    knees and elbows get an orientation aligned to the limb and nothing more --
    which is honest, and better than inventing a roll that will look wrong.
    """
    u = to_unity
    rotations = {}

    def emit(role, basis):
        if basis is not None and role in roles:
            rotations[TRACKER_ROLES[role]] = basis_to_unity_euler(*basis)

    hips = skeleton.midpoint(S.L_HIP, S.R_HIP)
    chest = skeleton.midpoint(S.L_SHOULDER, S.R_SHOULDER)

    # Hip and chest: spine gives "up", the hip/shoulder line gives the facing.
    # The subject's right is Unity +X (see the module docstring), and MediaPipe's
    # R_* landmarks are the SUBJECT's right, so R-minus-L is the right axis.
    if hips is not None and chest is not None:
        if skeleton.has(S.L_HIP, S.R_HIP):
            emit("hip", _orthonormal(
                u(chest) - u(hips),
                u(skeleton.get(S.R_HIP)) - u(skeleton.get(S.L_HIP))))
        if skeleton.has(S.L_SHOULDER, S.R_SHOULDER):
            # Explicit None check: `a or b` on numpy arrays raises, because the
            # truth value of a multi-element array is ambiguous.
            head = skeleton.midpoint(S.L_EAR, S.R_EAR)
            if head is None:
                head = skeleton.get(S.NOSE)
            spine = (u(head) - u(chest)) if head is not None else (u(chest) - u(hips))
            emit("chest", _orthonormal(
                spine,
                u(skeleton.get(S.R_SHOULDER)) - u(skeleton.get(S.L_SHOULDER))))

    # Feet: the ankle->toe vector is the foot's forward. World up is used as the
    # secondary reference because foot roll is not observable from two points,
    # and assuming a level foot is right far more often than it is wrong.
    for role, ankle, toe in (("left_foot", S.L_ANKLE, S.L_FOOT),
                             ("right_foot", S.R_ANKLE, S.R_FOOT)):
        if skeleton.has(ankle, toe):
            forward = u(skeleton.get(toe)) - u(skeleton.get(ankle))
            n = float(np.linalg.norm(forward))
            if n > 1e-6:
                forward = forward / n
                right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
                if float(np.linalg.norm(right)) > 1e-6:
                    basis = _orthonormal(np.cross(forward, right), right)
                    emit(role, basis)

    # Limbs: aligned along the bone, twist left at zero.
    for role, distal, proximal, a, b in (
        ("left_knee", S.L_KNEE, S.L_HIP, S.L_HIP, S.R_HIP),
        ("right_knee", S.R_KNEE, S.R_HIP, S.L_HIP, S.R_HIP),
        ("left_elbow", S.L_ELBOW, S.L_SHOULDER, S.L_SHOULDER, S.R_SHOULDER),
        ("right_elbow", S.R_ELBOW, S.R_SHOULDER, S.L_SHOULDER, S.R_SHOULDER),
    ):
        if skeleton.has(distal, proximal, a, b):
            emit(role, _orthonormal(
                u(skeleton.get(proximal)) - u(skeleton.get(distal)),
                u(skeleton.get(b)) - u(skeleton.get(a))))

    return rotations


class RotationSmoother:
    """Smooths rotations without ever smoothing euler angles directly.

    Averaging euler angles is wrong at the wrap: a foot oscillating either side
    of 180 deg would average to 0 deg -- pointing exactly backwards. So the
    smoothing happens on the rotation MATRIX, which has no discontinuity, and
    euler is only ever produced at the very end.

    Measured on a 446-frame recording: hips and chest are already steady
    (0.3-0.7 deg/frame), while feet show occasional large flips because the
    ankle->toe vector is only ~22 cm long, so small positional noise swings its
    direction a lot. Those flips are what this removes.
    """

    def __init__(self, alpha=0.25, max_step_deg=45.0, max_rejects=3):
        self.alpha = alpha
        self.max_step_deg = max_step_deg
        # A rejection must not be able to last forever. Rejecting purely on
        # "differs from the stored value" locks out any GENUINE large rotation:
        # turn a foot 90 degrees and every subsequent frame differs from the
        # stale stored matrix by 90 degrees, so it is rejected in perpetuity and
        # the foot never turns again. After this many consecutive rejections the
        # new orientation is accepted as real. Verified by test -- a sustained
        # 90 degree change now converges instead of sticking at 0.
        self.max_rejects = max_rejects
        self._matrices = {}
        self._rejects = {}

    @staticmethod
    def _orthonormalize(m):
        # Blending two rotation matrices leaves a near-rotation; SVD projects it
        # back onto the closest true rotation, which keeps scale and shear out.
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:  # reflection, not rotation
            u[:, -1] *= -1
            r = u @ vt
        return r

    def __call__(self, rotations):
        out = {}
        for key, euler in rotations.items():
            m = unity_euler_to_matrix(euler)
            previous = self._matrices.get(key)
            if previous is not None:
                # Angle between the two rotations, from the trace of prevT*m.
                cos_angle = (np.trace(previous.T @ m) - 1.0) / 2.0
                angle = math.degrees(math.acos(float(np.clip(cos_angle, -1, 1))))
                if angle > self.max_step_deg:
                    rejects = self._rejects.get(key, 0) + 1
                    if rejects <= self.max_rejects:
                        # A one-frame flip -- almost always the short ankle->toe
                        # vector snapping. Keep the old value and wait.
                        self._rejects[key] = rejects
                        out[key] = matrix_to_unity_euler(previous)
                        continue
                    # Persisted across several frames, so it is a real rotation,
                    # not a glitch. Accept it and resume smoothing.
                    self._rejects[key] = 0
                else:
                    self._rejects[key] = 0
                    m = self._orthonormalize(
                        previous * (1 - self.alpha) + m * self.alpha
                    )
            self._matrices[key] = m
            out[key] = matrix_to_unity_euler(m)
        return out


def matrix_to_unity_euler(m):
    return basis_to_unity_euler(m[:, 0], m[:, 1], m[:, 2])


class TrackerPredictor:
    """Latch plus velocity, so a pose can be produced for any moment in time.

    The pipeline is consistently late: measured 9.1ms waiting for a frame, 23.9ms
    of inference, and a frame that is ~50ms old by the time it could be displayed.
    Trackers therefore show where you WERE, and the error that causes was measured
    at 9.0mm at display time.

    Extrapolating forward by roughly that latency cancels most of it. Measured on
    a 446-frame recording, against where the body actually is at display time:

        lead      jitter    display-time error
          0ms      4.5mm         9.0mm
         50ms      5.4mm         3.8mm
         65ms      5.7mm         3.7mm
         85ms      6.3mm         4.9mm   <- overshooting

    58% less error for 20% more jitter, with the optimum landing on the measured
    latency rather than somewhere arbitrary -- which is what distinguishes
    cancelling a real lag from fitting noise.

    Velocity is smoothed with an EMA rather than used raw: it is a difference of
    two noisy positions, so it carries roughly sqrt(2) times the position noise
    and multiplying that by a lead time would amplify it straight into the output.

    Prediction also lets output run faster than the camera. VRChat applies tracker
    data per rendered frame, so 30Hz updates on a 72-90Hz headset are held for two
    or three frames at a time; predicting at each send smooths that.
    """

    def __init__(self, velocity_alpha=0.35, max_speed=4.0):
        self.velocity_alpha = velocity_alpha
        # Hard ceiling on extrapolated speed. A bad velocity estimate would
        # otherwise throw a tracker metres away for one frame.
        self.max_speed = max_speed
        self._value = {}
        self._velocity = {}
        self._time = {}

    def update(self, trackers, t):
        for key, position in trackers.items():
            previous, previous_t = self._value.get(key), self._time.get(key)
            if previous is not None and previous_t is not None and t > previous_t:
                raw = (position - previous) / (t - previous_t)
                speed = float(np.linalg.norm(raw))
                if speed > self.max_speed:
                    raw = raw * (self.max_speed / speed)
                old = self._velocity.get(key)
                self._velocity[key] = (raw if old is None else
                                       old * (1 - self.velocity_alpha)
                                       + raw * self.velocity_alpha)
            self._value[key] = position
            self._time[key] = t

    def at(self, t, lead=0.0):
        """Every tracker seen so far, extrapolated to `t + lead`."""
        out = {}
        for key, position in self._value.items():
            velocity = self._velocity.get(key)
            if velocity is None:
                out[key] = position
                continue
            horizon = (t - self._time[key]) + lead
            # Never predict further ahead than a few frames; beyond that the
            # linear model stops being a reasonable description of a limb.
            horizon = float(np.clip(horizon, 0.0, 0.12))
            out[key] = position + velocity * horizon
        return out

    def stale_keys(self, t, max_stale=1.0):
        return [k for k, ts in self._time.items() if t - ts > max_stale]


class TrackerLatch:
    """Keeps every enabled tracker emitting, even when its joint is missing.

    OSC is stateless and VRChat has no "tracker removed" message: if we simply
    stop sending an address, VRChat keeps whatever position it last received.
    Going silent therefore does NOT hide a missing foot -- it freezes that foot
    in mid-air wherever it last was, indefinitely.

    With ankles measured at 71-87% presence, silence would mean feet freezing
    several times a second, which reads as broken tracking rather than as a
    dropped frame. So once a tracker has been seen we always emit something for
    it, and the only question is what.

    Held values are marked stale after `max_stale` so callers can report the
    situation honestly, but they keep being sent -- a slightly out-of-date foot
    is better than a frozen one, and both beat VRChat inventing its own.
    """

    def __init__(self, max_stale=1.0):
        self.max_stale = max_stale
        self._last = {}
        self._last_t = {}

    def update(self, trackers, t):
        for key, value in trackers.items():
            self._last[key] = value
            self._last_t[key] = t

    def all_current(self):
        """Every tracker seen so far, latest known value. Never shrinks."""
        return dict(self._last)

    def stale_keys(self, t):
        return [k for k, ts in self._last_t.items() if t - ts > self.max_stale]
