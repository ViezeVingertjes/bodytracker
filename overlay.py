"""Shared preview rendering, used by both `main.py --preview` and `tools/preview.py`.

Kept separate so the live tracker and the diagnostic tool draw the SAME picture.
If they had their own copies, the window you use while tuning would drift away
from the window you use while actually tracking, and the diagnostic would stop
telling you about the thing you run.

cv2 is imported lazily by `require_cv2()` -- importing it costs ~200 ms and pulls
in Qt, which a headless tracking run has no use for.
"""

import os
import sys

import numpy as np

import solver as S

GREEN = (0, 220, 0)
AMBER = (0, 170, 255)
GREY = (130, 130, 130)
WHITE = (255, 255, 255)
RED = (60, 60, 255)
CYAN = (255, 220, 0)

BONES = [
    (S.L_SHOULDER, S.R_SHOULDER),
    (S.L_SHOULDER, S.L_ELBOW),
    (S.R_SHOULDER, S.R_ELBOW),
    (S.L_ELBOW, S.L_WRIST),
    (S.R_ELBOW, S.R_WRIST),
    (S.L_SHOULDER, S.L_HIP),
    (S.R_SHOULDER, S.R_HIP),
    (S.L_HIP, S.R_HIP),
    (S.L_HIP, S.L_KNEE),
    (S.R_HIP, S.R_KNEE),
    (S.L_KNEE, S.L_ANKLE),
    (S.R_KNEE, S.R_ANKLE),
    (S.L_ANKLE, S.L_FOOT),
    (S.R_ANKLE, S.R_FOOT),
]

NAMES = {
    S.NOSE: "nose",
    S.L_SHOULDER: "L shoulder", S.R_SHOULDER: "R shoulder",
    S.L_ELBOW: "L elbow", S.R_ELBOW: "R elbow",
    S.L_WRIST: "L wrist", S.R_WRIST: "R wrist",
    S.L_HIP: "L hip", S.R_HIP: "R hip",
    S.L_KNEE: "L knee", S.R_KNEE: "R knee",
    S.L_ANKLE: "L ankle", S.R_ANKLE: "R ankle",
    S.L_FOOT: "L toe", S.R_FOOT: "R toe",
}

_cv2 = None


def require_cv2():
    """Import cv2 with the Qt workarounds applied. Cached after first call."""
    global _cv2
    if _cv2 is None:
        # LINUX ONLY. OpenCV's Qt build ships no Wayland plugin, so it runs
        # under XWayland regardless and Qt warns about "ignoring
        # XDG_SESSION_TYPE=wayland" every launch. Pinning the platform alone
        # does not silence it -- Qt warns as long as it sees the variable -- so
        # drop it for this process.
        #
        # Must NOT run on Windows or macOS: there is no xcb plugin there, and
        # forcing it makes Qt fail to start at all, killing the preview window.
        if sys.platform.startswith("linux"):
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
            os.environ.pop("XDG_SESSION_TYPE", None)
        import cv2
        _cv2 = cv2
    return _cv2


def draw_skeleton(canvas, skeleton, camera=None, inferred=()):
    """Overlay the pose exactly as it is being SENT.

    When `camera` is given, joints are re-projected from their final stabilised
    3D positions rather than drawn at the original landmark pixels. That matters:
    after stabilisation and occlusion filling a joint no longer sits on the pixel
    it came from, and a reconstructed joint has no landmark pixel at all -- so
    drawing raw pixels would show a preview that disagrees with the OSC data.

    green  measured this frame, real depth
    cyan   INFERRED -- held or reconstructed from the body frame, not measured
    amber  found by MediaPipe but depth rejected -- labelled with the reason
    grey   not found this frame
    """
    cv2 = require_cv2()
    if skeleton is None:
        return

    def pixel(idx):
        if camera is not None and idx in skeleton.joints:
            try:
                px, py = camera.project(skeleton.joints[idx])
                return (int(px), int(py))
            except Exception:  # noqa: BLE001 - fall back to the landmark pixel
                pass
        if idx in skeleton.pixels:
            px, py = skeleton.pixels[idx]
            return (int(px), int(py))
        return None

    for a, b in BONES:
        if a in skeleton.joints and b in skeleton.joints:
            pa, pb = pixel(a), pixel(b)
            if pa and pb:
                colour = CYAN if (a in inferred or b in inferred) else GREEN
                cv2.line(canvas, pa, pb, colour, 2)

    drawn = set()
    for idx in skeleton.joints:
        p = pixel(idx)
        if p is None:
            continue
        drawn.add(idx)
        if idx in inferred:
            cv2.circle(canvas, p, 6, CYAN, 2)
            cv2.putText(canvas, "held", (p[0] + 7, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, CYAN, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, p, 5, GREEN, -1)
            cv2.putText(canvas, f"{skeleton.joints[idx][2]:.2f}", (p[0] + 7, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREEN, 1, cv2.LINE_AA)

    for idx, (px, py) in skeleton.pixels.items():
        if idx in drawn:
            continue
        p = (int(px), int(py))
        if idx in skeleton.rejected:
            cv2.circle(canvas, p, 5, AMBER, 1)
            cv2.putText(canvas, skeleton.rejected[idx], (p[0] + 7, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, AMBER, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, p, 4, GREY, 1)


def draw_lines(canvas, lines):
    """Write (text, colour) pairs down the top-left corner."""
    cv2 = require_cv2()
    y = 20
    for text, colour in lines:
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)
        y += 20


def depth_colormap(depth_raw):
    cv2 = require_cv2()
    vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_TURBO
    )
    vis[depth_raw == 0] = (0, 0, 0)
    return vis


def sanity_metrics(skeleton):
    """Physical plausibility -- the check that caught a false detection early on.

    A spurious pose fitted to furniture reads as an obviously impossible body
    (an early one was 0.10 m across the shoulders), which is glaring in metres
    and invisible in normalised image coordinates.
    """
    out = []
    if skeleton is None:
        return out
    ls, rs = skeleton.get(S.L_SHOULDER), skeleton.get(S.R_SHOULDER)
    if ls is not None and rs is not None:
        d = float(np.linalg.norm(ls - rs))
        out.append((f"shoulder width {d:.2f} m", 0.12 <= d <= 0.80))
    sh = skeleton.midpoint(S.L_SHOULDER, S.R_SHOULDER)
    hp = skeleton.midpoint(S.L_HIP, S.R_HIP)
    if sh is not None and hp is not None:
        d = float(np.linalg.norm(sh - hp))
        out.append((f"torso length {d:.2f} m", 0.15 <= d <= 1.00))
        out.append(("shoulders above hips", sh[1] < hp[1]))  # +Y is down
    return out


def framing_status(skeleton, height):
    """Is the whole body in frame? The RESEARCH.md section 3 kill criterion."""
    if skeleton is None:
        return None
    head = skeleton.pixels.get(S.NOSE)
    feet = [skeleton.pixels.get(i) for i in
            (S.L_ANKLE, S.R_ANKLE, S.L_FOOT, S.R_FOOT)]
    feet = [f for f in feet if f is not None]
    head_ok = head is not None and head[1] > 6
    feet_ok = bool(feet) and max(f[1] for f in feet) < height - 6
    return head_ok, feet_ok


def status_lines(skeleton, fps, extra=(), frame_height=None):
    """The standard readout shared by every preview surface.

    frame_height must be the ACTUAL canvas height. It was hardcoded to 480 at
    first, which silently mis-reported "FEET in frame" at any other resolution --
    the one readout people rely on to position themselves.
    """
    lines = [(f"{fps:4.1f} fps", WHITE)]
    if skeleton is None:
        lines.append(("NO POSE DETECTED", RED))
        return lines

    lines.append((f"joints {len(skeleton.joints)}/{len(S.NEEDED)}   "
                  f"body depth {skeleton.body_depth:.2f} m", WHITE))
    framing = (framing_status(skeleton, frame_height)
               if frame_height else None)
    if framing:
        head_ok, feet_ok = framing
        lines.append((f"HEAD {'in' if head_ok else 'OUT OF'} frame",
                      GREEN if head_ok else RED))
        lines.append((f"FEET {'in' if feet_ok else 'OUT OF'} frame",
                      GREEN if feet_ok else RED))
    for text, ok in sanity_metrics(skeleton):
        lines.append((f"{'ok ' if ok else 'BAD'} {text}", GREEN if ok else RED))
    lines.extend(extra)
    return lines
