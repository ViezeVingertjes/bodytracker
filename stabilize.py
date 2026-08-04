"""Joint-level stabilisation: outlier gating, smoothing, and skeletal constraints.

Depth post-processing (capture.py) fixes per-pixel noise. This fixes the failures
that survive it, which are structural rather than statistical:

  1. A landmark momentarily lands on the background or on the other leg, and the
     joint teleports half a metre for one frame. Averaging cannot fix a
     teleport -- it just smears it over several frames. It has to be REJECTED.

  2. Depth noise along the view axis stretches and compresses limbs, so a forearm
     that is 26 cm one frame is 34 cm the next. The human body does not do this,
     and enforcing that fact removes a whole class of wobble that no amount of
     temporal filtering removes, because the error is not zero-mean per joint.

Order matters: gate, then constrain, then smooth. Gating first stops an outlier
from poisoning the learned bone lengths; smoothing last means the constraint
solver never sees the lag that smoothing introduces.
"""

import collections

import numpy as np

import solver as S
from transform import OneEuroFilter

# Bones whose length is genuinely fixed on a real body. Ordered proximal ->
# distal so corrections propagate outward from the torso, which is the part the
# depth camera sees most reliably.
BONE_CHAINS = [
    [(S.L_HIP, S.L_KNEE), (S.L_KNEE, S.L_ANKLE)],
    [(S.R_HIP, S.R_KNEE), (S.R_KNEE, S.R_ANKLE)],
    [(S.L_SHOULDER, S.L_ELBOW), (S.L_ELBOW, S.L_WRIST)],
    [(S.R_SHOULDER, S.R_ELBOW), (S.R_ELBOW, S.R_WRIST)],
]

# Rigid torso spans, used as-is (no chain propagation).
TORSO_BONES = [
    (S.L_SHOULDER, S.R_SHOULDER),
    (S.L_HIP, S.R_HIP),
    (S.L_SHOULDER, S.L_HIP),
    (S.R_SHOULDER, S.R_HIP),
]

ALL_BONES = [b for chain in BONE_CHAINS for b in chain] + TORSO_BONES


class BoneLengthModel:
    """Learns each bone's true length from observation, then enforces it.

    Uses a median over a rolling window rather than a mean: a mean is dragged by
    exactly the outliers this is meant to defend against, and limb length has no
    reason to drift, so there is nothing for a mean to track that a median misses.
    """

    def __init__(self, window=90, min_samples=20, tolerance=0.12):
        self._history = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self.min_samples = min_samples
        # Fractional deviation tolerated before a bone is corrected. Some slack
        # is wanted: real landmarks shift slightly against the skeleton as a
        # person moves, and clamping hard makes motion look robotic.
        self.tolerance = tolerance

    def observe(self, joints):
        for a, b in ALL_BONES:
            if a in joints and b in joints:
                self._history[(a, b)].append(float(np.linalg.norm(joints[a] - joints[b])))

    def length(self, bone):
        samples = self._history[bone]
        if len(samples) < self.min_samples:
            return None
        return float(np.median(samples))

    def implausible(self, joints, limit=0.35):
        """Joints whose bone length is so wrong the LANDMARK itself is untrustworthy.

        There are two different failures here and they need opposite treatment:

          - mild length error: the landmark is roughly right, depth noise has
            stretched it along the view axis. apply() fixes that by correcting
            length while keeping the direction.
          - gross length error: MediaPipe has hallucinated an occluded joint
            somewhere else entirely -- behind a chair, or on the other leg. The
            DIRECTION is wrong too, so correcting length just moves a wrong point
            to a wrong point at the right distance.

        The second case is what makes tracking "freak out" when a limb is briefly
        covered: the joint is present, so occlusion filling never engages, and a
        confidently-wrong position sails through. Returning it here lets the
        caller drop it and reconstruct from the body frame instead.
        """
        bad = set()
        for chain in BONE_CHAINS:
            for proximal, distal in chain:
                if proximal not in joints or distal not in joints:
                    continue
                target = self.length((proximal, distal))
                if target is None or target <= 1e-6:
                    continue
                actual = float(np.linalg.norm(joints[distal] - joints[proximal]))
                if abs(actual - target) / target > limit:
                    # Blame the distal joint: the proximal one is nearer the
                    # torso and better observed.
                    bad.add(distal)
        return bad

    def apply(self, joints):
        """Return joints with over/under-stretched bones pulled back to length."""
        out = dict(joints)

        for chain in BONE_CHAINS:
            for proximal, distal in chain:
                if proximal not in out or distal not in out:
                    continue
                target = self.length((proximal, distal))
                if target is None or target <= 1e-6:
                    continue

                vector = out[distal] - out[proximal]
                actual = float(np.linalg.norm(vector))
                if actual < 1e-6:
                    continue
                if abs(actual - target) / target <= self.tolerance:
                    continue

                # Keep the DIRECTION the pose model gave us -- that comes from the
                # image and is reliable -- and correct only the length, which is
                # what depth noise corrupts. Move the distal joint; the proximal
                # one is closer to the torso and better observed.
                out[distal] = out[proximal] + vector * (target / actual)

        # Torso spans are rigid too, and were previously measured but never
        # corrected -- which made the aggregate metrics look like the constraint
        # was doing nothing. Correct them symmetrically: unlike a limb, neither
        # end of the shoulder line is "more proximal", so splitting the
        # correction avoids biasing the body to one side.
        for a, b in TORSO_BONES:
            if a not in out or b not in out:
                continue
            target = self.length((a, b))
            if target is None or target <= 1e-6:
                continue
            vector = out[b] - out[a]
            actual = float(np.linalg.norm(vector))
            if actual < 1e-6 or abs(actual - target) / target <= self.tolerance:
                continue
            centre = (out[a] + out[b]) * 0.5
            half = vector * (target / actual) * 0.5
            out[a] = centre - half
            out[b] = centre + half
        return out


class OutlierGate:
    """Rejects joints that move faster than a human limb can.

    A per-joint speed limit, not a global one: a wrist genuinely moves several
    times faster than a hip, so one threshold would either let wrist teleports
    through or clamp real hip motion.
    """

    FAST_JOINTS = {S.L_WRIST, S.R_WRIST, S.L_ELBOW, S.R_ELBOW,
                   S.L_ANKLE, S.R_ANKLE, S.L_FOOT, S.R_FOOT}

    def __init__(self, slow_limit=2.0, fast_limit=5.0, max_hold=0.25):
        # Tuned by sweep on a 446-frame recording. Tightening further makes the
        # worst glitch WORSE, not better: an over-eager gate holds a joint, then
        # releases it further from where it was, turning one medium step into a
        # larger one. 5.0/2.0 was the minimum of that curve (218mm worst, vs
        # 277mm at 8.0/3.0 and 287mm at 3.0/1.5).
        self.slow_limit = slow_limit   # m/s for torso, hips, knees, head
        self.fast_limit = fast_limit   # m/s for hands and feet
        # How long a rejected joint keeps reporting its last good value before
        # being dropped. Brief holds bridge single-frame glitches; a long hold
        # would freeze a limb in mid-air after the person actually moved it.
        self.max_hold = max_hold
        self._last = {}
        self._last_t = {}

    def __call__(self, joints, t):
        out = {}
        for idx, point in joints.items():
            limit = self.fast_limit if idx in self.FAST_JOINTS else self.slow_limit
            previous = self._last.get(idx)
            if previous is not None:
                dt = t - self._last_t[idx]
                if dt > 0:
                    speed = float(np.linalg.norm(point - previous)) / dt
                    if speed > limit:
                        # Implausible jump: hold the last good value, briefly.
                        if dt <= self.max_hold:
                            out[idx] = previous
                        continue
            out[idx] = point
            self._last[idx] = point
            self._last_t[idx] = t
        return out


def torso_frame(joints):
    """Orthonormal body frame from the torso: (origin, 3x3 basis) or None.

    The torso is the right anchor because hips and shoulders are the best-observed
    joints on the body (measured at 100% presence, versus 71-87% for ankles), so
    it is available almost exactly when anything else is.
    """
    if not all(k in joints for k in (S.L_HIP, S.R_HIP, S.L_SHOULDER, S.R_SHOULDER)):
        return None

    hips = (joints[S.L_HIP] + joints[S.R_HIP]) * 0.5
    chest = (joints[S.L_SHOULDER] + joints[S.R_SHOULDER]) * 0.5

    x = joints[S.R_HIP] - joints[S.L_HIP]
    nx = np.linalg.norm(x)
    y = chest - hips
    ny = np.linalg.norm(y)
    if nx < 1e-6 or ny < 1e-6:
        return None
    x = x / nx
    y = y / ny

    # Gram-Schmidt: the hip line and spine are not exactly perpendicular on a
    # real body, and a non-orthonormal basis would shear reconstructed joints.
    x = x - y * float(np.dot(x, y))
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x = x / nx
    z = np.cross(x, y)

    return hips, np.stack([x, y, z], axis=1)  # columns are the axes


class OcclusionFiller:
    """Keeps missing joints moving with the body instead of freezing in space.

    The naive fallback -- hold a missing joint at its last WORLD position -- looks
    broken the moment the person moves: the foot stays behind while its owner
    walks away. Holding the joint's position in the BODY frame instead means an
    unseen foot travels, turns and bobs with the hips, which is both physically
    right and visually stable.

    Reappearance is blended rather than snapped. A joint that returns 15 cm from
    where it was held would otherwise jump in a single frame, which is exactly
    the glitch this is meant to remove -- so the correction is spread over a
    short ramp.
    """

    def __init__(self, blend_time=0.3, max_hold=3.0):
        self.blend_time = blend_time
        # Beyond this, a joint is considered genuinely gone rather than briefly
        # occluded, and we stop inventing it.
        self.max_hold = max_hold
        self._local = {}        # joint -> position in torso frame
        self._world = {}        # joint -> last known world position (fallback)
        self._last_seen = {}
        self._returned_at = {}
        self._was_missing = set()

    def __call__(self, joints, t):
        frame = torso_frame(joints)
        out = dict(joints)

        for idx in joints:
            self._last_seen[idx] = t
            self._world[idx] = joints[idx]

        if frame is not None:
            origin, basis = frame
            for idx, point in joints.items():
                self._local[idx] = basis.T @ (point - origin)

        # Reconstruct anything we know about but cannot currently see.
        for idx, last_t in list(self._last_seen.items()):
            if idx in joints:
                continue
            age = t - last_t
            if age > self.max_hold:
                continue
            if frame is not None and idx in self._local:
                origin, basis = frame
                out[idx] = origin + basis @ self._local[idx]
            elif idx in self._world:
                out[idx] = self._world[idx]  # no torso: world hold is all we have
            self._was_missing.add(idx)

        # Blend a returning joint back in over blend_time.
        for idx in list(self._was_missing):
            if idx not in joints:
                continue
            self._returned_at.setdefault(idx, t)
            elapsed = t - self._returned_at[idx]
            if elapsed >= self.blend_time:
                self._was_missing.discard(idx)
                self._returned_at.pop(idx, None)
                continue
            held = None
            if frame is not None and idx in self._local:
                origin, basis = frame
                held = origin + basis @ self._local[idx]
            elif idx in self._world:
                held = self._world[idx]
            if held is not None:
                w = elapsed / self.blend_time
                out[idx] = held * (1.0 - w) + joints[idx] * w

        return out

    def reconstructed(self, joints):
        """Which joints in the last output were invented rather than measured."""
        return set(self._was_missing) - set(joints)


class SkeletonStabilizer:
    """gate -> bone constraints -> smooth. See module docstring for why that order."""

    def __init__(self, min_cutoff=0.5, beta=0.35, enable_bones=True,
                 enable_gate=True, enable_fill=True):
        self.gate = OutlierGate() if enable_gate else None
        self.bones = BoneLengthModel() if enable_bones else None
        self.filler = OcclusionFiller() if enable_fill else None
        self._filters = {}
        self._min_cutoff = min_cutoff
        self._beta = beta
        self.inferred = set()

    def __call__(self, joints, t):
        # Copy up front: this method removes implausible joints, and the caller's
        # dict must not be mutated. (The gate happens to return a fresh dict, but
        # relying on that would break the moment gating is disabled.)
        measured = set(joints)
        joints = dict(joints)

        if self.gate is not None:
            joints = self.gate(joints, t)

        if self.bones is not None:
            # Observe BEFORE correcting, or the model learns from its own output
            # and slowly locks to whatever it first believed. Observing before
            # the filler runs also keeps reconstructed joints out of the model,
            # which would otherwise train on its own inventions.
            self.bones.observe(joints)

            # Drop joints the skeleton says cannot be where the model put them,
            # so the filler treats them as occluded rather than trusting a
            # hallucinated position. Without this, a briefly-covered limb is
            # "present but wrong", which is worse than absent -- absent gets
            # reconstructed, wrong gets sent.
            for idx in self.bones.implausible(joints):
                joints.pop(idx, None)

        if self.filler is not None:
            joints = self.filler(joints, t)

        if self.bones is not None:
            joints = self.bones.apply(joints)

        smoothed = {}
        for idx, point in joints.items():
            if idx not in self._filters:
                self._filters[idx] = OneEuroFilter(self._min_cutoff, self._beta)
            smoothed[idx] = self._filters[idx](point, t)

        # Which joints in this output were inferred rather than measured. The
        # preview colours these differently -- a held joint that looks identical
        # to a tracked one hides exactly the failure you are trying to watch for.
        self.inferred = set(smoothed) - measured
        return smoothed
