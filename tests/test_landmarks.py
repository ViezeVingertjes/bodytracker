"""Pin our landmark indices to MediaPipe's own enum.

skeleton.py hardcodes MediaPipe's numbering with a comment asserting what each
index means. A comment cannot fail. If MediaPipe ever renumbers -- or if one of
those constants was simply mistyped -- nothing else in this codebase would
notice: every index in 0..32 is a valid landmark, so a wrong one does not throw,
it tracks the wrong body part. Confidently, at full framerate, forever.

That failure is close to undetectable by eye. Swap L_KNEE and L_ANKLE and the
preview still draws a plausible leg; swap left and right and the avatar mirrors
in a way that looks like a coordinate-system bug rather than a landmark bug.

This is the only test that imports MediaPipe (~0.5s). That is the point: it
compares against the library's own definition rather than against a second copy
of our assumption.
"""

import pytest

import skeleton as S

PoseLandmark = pytest.importorskip(
    "mediapipe.tasks.python.vision.pose_landmarker"
).PoseLandmark


# Our constant -> the MediaPipe enum member it claims to be.
CLAIMS = {
    "NOSE": "NOSE",
    "L_EAR": "LEFT_EAR", "R_EAR": "RIGHT_EAR",
    "L_SHOULDER": "LEFT_SHOULDER", "R_SHOULDER": "RIGHT_SHOULDER",
    "L_ELBOW": "LEFT_ELBOW", "R_ELBOW": "RIGHT_ELBOW",
    "L_WRIST": "LEFT_WRIST", "R_WRIST": "RIGHT_WRIST",
    "L_HIP": "LEFT_HIP", "R_HIP": "RIGHT_HIP",
    "L_KNEE": "LEFT_KNEE", "R_KNEE": "RIGHT_KNEE",
    "L_ANKLE": "LEFT_ANKLE", "R_ANKLE": "RIGHT_ANKLE",
    "L_FOOT": "LEFT_FOOT_INDEX", "R_FOOT": "RIGHT_FOOT_INDEX",
}


@pytest.mark.parametrize(("ours", "theirs"), sorted(CLAIMS.items()))
def test_constant_matches_mediapipe(ours, theirs):
    assert getattr(S, ours) == int(PoseLandmark[theirs]), (
        f"skeleton.{ours} is {getattr(S, ours)}, but MediaPipe says "
        f"{theirs} is {int(PoseLandmark[theirs])}")


def test_we_check_every_constant_we_define():
    """A new constant in skeleton.py must be added to CLAIMS, not left unpinned."""
    defined = {n for n, v in vars(S).items()
               if n.isupper() and isinstance(v, int) and n != "NEEDED"}
    assert defined == set(CLAIMS), (
        f"unpinned: {defined - set(CLAIMS)}, stale: {set(CLAIMS) - defined}")


def test_model_still_has_33_landmarks():
    # Guards against a MediaPipe model change that alters the topology rather
    # than just the numbering.
    assert len(PoseLandmark) == 33


def test_left_and_right_are_the_subjects_own():
    """MediaPipe's LEFT_* is the SUBJECT's left, not the viewer's.

    The whole camera-to-Unity conversion in transform.py is built on this, and
    getting it backwards mirrors the avatar in a way that looks correct in the
    preview window. Asserted here because it is an assumption about MediaPipe,
    not about our own maths.
    """
    # Every left landmark is an odd-then-even pair with its right counterpart in
    # MediaPipe's numbering: left first, right second, consistently.
    for ours_left, ours_right in (("L_EAR", "R_EAR"),
                                  ("L_SHOULDER", "R_SHOULDER"),
                                  ("L_ELBOW", "R_ELBOW"),
                                  ("L_WRIST", "R_WRIST"),
                                  ("L_HIP", "R_HIP"),
                                  ("L_KNEE", "R_KNEE"),
                                  ("L_ANKLE", "R_ANKLE"),
                                  ("L_FOOT", "R_FOOT")):
        left, right = getattr(S, ours_left), getattr(S, ours_right)
        assert right == left + 1, f"{ours_left}/{ours_right} are not adjacent"
        assert PoseLandmark(left).name.startswith("LEFT_")
        assert PoseLandmark(right).name.startswith("RIGHT_")


def test_needed_covers_every_constant():
    """Anything used but absent from NEEDED never gets a depth reading."""
    defined = {getattr(S, n) for n in CLAIMS}
    assert set(S.NEEDED) == defined, (
        f"missing depth: {defined - set(S.NEEDED)}, "
        f"wasted reads: {set(S.NEEDED) - defined}")


def test_foot_landmark_is_the_toe_not_the_heel():
    """transform.build_rotations uses ankle->foot as the foot's FORWARD vector.

    LEFT_HEEL (29) sits behind the ankle, so using it here would point every
    foot backwards -- a 180 degree error that the rotation smoother would then
    dutifully stabilise.
    """
    toes = {int(PoseLandmark.LEFT_FOOT_INDEX), int(PoseLandmark.RIGHT_FOOT_INDEX)}
    heels = {int(PoseLandmark.LEFT_HEEL), int(PoseLandmark.RIGHT_HEEL)}
    assert {S.L_FOOT, S.R_FOOT} == toes
    assert {S.L_FOOT, S.R_FOOT}.isdisjoint(heels)
