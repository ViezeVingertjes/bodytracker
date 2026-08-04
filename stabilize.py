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

Order matters, and the full order is:

    gate -> drop implausible -> observe -> fill -> constrain -> smooth

Judging before observing is what protects the learned bone lengths. Gating alone
does not: the gate catches SPEED, and a landmark parked at double length has
none, so observing first let ~1.5s of a hallucinated limb drag the median far
enough that the joint stopped being recognised as implausible at all -- after
which the constraint enforced the wrong length on the real joint. Filling sits
between the drop and the constraint so a dropped joint is reconstructed from the
body frame rather than left absent. Smoothing last means the constraint solver
never sees the lag that smoothing introduces.
"""

import collections

import numpy as np

import skeleton as S
from transform import OneEuroFilter

# Bones whose length is genuinely fixed on a real body. Ordered proximal ->
# distal so corrections propagate outward from the torso, which is the part the
# depth camera sees most reliably.
#
# ankle->toe is included and matters more than its length suggests. It is the
# shortest vector on the body used as a DIRECTION -- transform.build_rotations
# takes each foot's forward from it -- so it has the worst angular sensitivity
# to depth noise of anything here: ~2cm of toe depth error is 7.6 deg of foot
# yaw, and 4cm is 14.9 deg, both routine where the depth patch straddles the
# floor. Constraining its length also lets implausible() reject a toe that has
# landed on the floor behind the foot, which nothing previously could.
BONE_CHAINS = [
    [(S.L_HIP, S.L_KNEE), (S.L_KNEE, S.L_ANKLE), (S.L_ANKLE, S.L_FOOT)],
    [(S.R_HIP, S.R_KNEE), (S.R_KNEE, S.R_ANKLE), (S.R_ANKLE, S.R_FOOT)],
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
        # Joints whose reported value this frame is a HELD one, not a
        # measurement. Surfaced so callers can mark them: a substituted joint
        # that looks identical to a tracked one hides the failure entirely.
        self.held = set()

    def __call__(self, joints, t):
        out = {}
        self.held = set()
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
                            self.held.add(idx)
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
        # Where a returning joint is blending FROM, captured once at the moment
        # it returns and held fixed for the ramp. Recomputing it per frame from
        # the live pose is what made the ramp a no-op.
        self._return_from = {}     # body-frame offset
        self._return_world = {}    # world fallback when there is no torso

    def _held_at(self, idx, frame, local):
        """Where a held joint sits now, given a body-frame offset."""
        if frame is not None and idx in local:
            origin, basis = frame
            return origin + basis @ local[idx]
        return self._world.get(idx)  # no torso: world hold is all we have

    def __call__(self, joints, t):
        frame = torso_frame(joints)
        out = dict(joints)

        # Snapshot where each RETURNING joint was being held, before the
        # incoming measurements overwrite the stored pose below.
        #
        # Reading it afterwards made the whole ramp a no-op: `_local` had just
        # been rewritten from the measurement, and basis.T then basis is the
        # identity, so `held` came back exactly equal to `joints[idx]` and
        # held*(1-w) + measured*w == measured for every w. The joint snapped in
        # one frame -- precisely the glitch this class exists to remove.
        for idx in self._was_missing & set(joints):
            if idx not in self._return_from:
                held = self._held_at(idx, frame, self._local)
                if held is not None:
                    # Stored in the BODY frame, so the blend target keeps
                    # travelling with the person for the length of the ramp
                    # instead of being left behind in world space.
                    self._return_from[idx] = (
                        frame[1].T @ (held - frame[0]) if frame is not None
                        else None)
                    self._return_world[idx] = held

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
            if t - last_t > self.max_hold:
                # Genuinely gone rather than briefly occluded. Stop claiming it
                # so `_was_missing` cannot grow without bound.
                self._was_missing.discard(idx)
                self._returned_at.pop(idx, None)
                self._return_from.pop(idx, None)
                self._return_world.pop(idx, None)
                continue
            held = self._held_at(idx, frame, self._local)
            if held is not None:
                out[idx] = held
            self._was_missing.add(idx)

        # Blend a returning joint back in over blend_time.
        for idx in list(self._was_missing):
            if idx not in joints:
                # Occluded again mid-ramp. Drop the clock so the next return
                # restarts the blend; leaving it stale meant `elapsed` was
                # already past blend_time and the joint snapped anyway.
                self._returned_at.pop(idx, None)
                self._return_from.pop(idx, None)
                self._return_world.pop(idx, None)
                continue
            self._returned_at.setdefault(idx, t)
            elapsed = t - self._returned_at[idx]
            if elapsed >= self.blend_time:
                self._was_missing.discard(idx)
                self._returned_at.pop(idx, None)
                self._return_from.pop(idx, None)
                self._return_world.pop(idx, None)
                continue
            local = self._return_from.get(idx)
            if local is not None and frame is not None:
                origin, basis = frame
                held = origin + basis @ local
            else:
                held = self._return_world.get(idx)
            if held is not None:
                w = elapsed / self.blend_time
                out[idx] = held * (1.0 - w) + joints[idx] * w

        return out


class SkeletonStabilizer:
    """gate -> drop -> observe -> fill -> constrain -> smooth.

    See the module docstring for why that order, in particular why observing
    the bone model AFTER the implausible-drop is what keeps a hallucinated limb
    from becoming the learned limb.
    """

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

        # Joints in the output that are NOT what the model measured this frame,
        # even though they were present in the input. Tracked explicitly because
        # set-difference against the input cannot see them.
        substituted = set()

        if self.gate is not None:
            joints = self.gate(joints, t)
            substituted |= self.gate.held

        if self.bones is not None:
            # Drop joints the skeleton says cannot be where the model put them,
            # so the filler treats them as occluded rather than trusting a
            # hallucinated position. Without this, a briefly-covered limb is
            # "present but wrong", which is worse than absent -- absent gets
            # reconstructed, wrong gets sent.
            dropped = self.bones.implausible(joints)
            for idx in dropped:
                joints.pop(idx, None)
            substituted |= dropped

            # Observe AFTER the drop, not before. Gating first does NOT protect
            # the learned lengths as the module docstring claimed: the gate
            # catches speed, and a limb MediaPipe places at double length and
            # holds there has no speed error at all. Observed first, ~1.5s of
            # such a hallucination drags the 90-sample median far enough that
            # implausible() stops firing -- and then apply() enforces the wrong
            # length on the real joint until 45 more good frames wash through.
            #
            # Gate-held joints are excluded for a related reason: a held
            # endpoint paired with a fresh one measures a bone that never
            # existed. Correcting before observing is still avoided, so the
            # model never trains on its own output.
            self.bones.observe({idx: point for idx, point in joints.items()
                                if idx not in substituted})

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
        #
        # `set(smoothed) - measured` alone finds only joints MediaPipe never
        # sent. It misses both the cases that matter most: a joint the gate
        # held, and a joint dropped as implausible then reinvented by the
        # filler -- the very case the drop above calls worse than absence. Both
        # stay in `measured`, so both looked healthy in the preview.
        self.inferred = (set(smoothed) - measured) | (substituted & set(smoothed))
        return smoothed
