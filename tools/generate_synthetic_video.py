"""Generate a fully synthetic multi-fault demo video (proof-of-mechanism only).

No real camera footage is used anywhere. The tool procedurally draws a
"busy" fake room scene (checkerboard floor, window grid, door, bookshelf,
monitor, plant, barcode block, text, random shapes, fine-detail noise),
renders a clean 5 s baseline segment, then applies controlled fault
transforms (tilt, tampering/obstruction, low light, blur) in eight labeled
segments and writes a ground-truth JSON describing exactly what was
injected where.

This clip exists to demonstrate and sanity-check the detector/decision
stack's *mechanism*: parameters were chosen so each fault's signature
clears the production detectors' thresholds. It is NOT real-world
validation material — real deployment validation requires physically
staged footage per the project rules.

Usage:
    python tools/generate_synthetic_video.py
    python tools/generate_synthetic_video.py --resolution 640x360
    python tools/generate_synthetic_video.py --verify
    python tools/generate_synthetic_video.py --out <path> --seed 42

Outputs (defaults):
    data/test_footage/synthetic_multifault_demo.mp4
    data/test_footage/synthetic_multifault_demo_ground_truth.json

Fault physics / cross-triggering notes (relevant to the decision layer):
  * Low light is a relative detector: it flags when the dark-pixel ratio
    rises >= 50% over baseline. The engine CONFIRMS it at confidence >= 0.5,
    which requires >50% of pixels below the V=60 threshold -- i.e. a
    genuinely dark frame. Such a frame also drops Laplacian sharpness below
    the blur floor, so darkened segments co-fire the blur detector. The
    engine resolves this with its relation classes: tilt -> blur is
    "always" (suppressed), low_light -> blur only activates at near-black
    (>= 0.8), and tampering -> blur needs area conservation.
  * Tampering is a localized-structure-loss detector. The engine's
    confirmation floor (confidence >= 0.5) corresponds to one contiguous
    lost-structure cluster of ~57% of meaningful baseline blocks, so the
    "large" obstruction (~65% coverage) confirms cleanly while the "small"
    (~18%) fires the per-frame candidate only.
  * Blur strong enough to confirm also removes the Canny edges the
    tampering detector measures, so tampering is masked in the tampering+blur
    segments -- an honest cross-trigger the decision layer must tolerate.
  * The near-black low-light segment causally explains the co-injected blur
    (low_light -> blur activates at confidence >= 0.8), so the engine
    reports low_light alone there.

Expected decision-engine output per segment (single-frame fusion, after
temporal confirmation): baseline {} -> seg1 {tilt} -> seg2 {low_light, tilt}
-> seg3 {blur} -> seg4 {tampering} -> seg5 {blur} -> seg6 {blur} ->
seg7 {low_light, blur} -> seg8 {low_light}. Notes on segments whose
injected faults do not reach engine confirmation (verified detector-level
outcomes via --verify):
  * seg1: small tampering fires a per-frame candidate only; the engine's
    tampering floor needs ~57% contiguous obstruction.
  * seg3: tampering (conf ~0.4) and low_light (conf ~0.1) both fire as
    candidates but neither clears its 0.5 confirmation floor; the marginal
    blur cross-fire (conf ~0.51, from the darkened flat occlusion) is the
    only event the engine would emit -- the honest cross-trigger tradeoff
    this clip exists to surface.
  * seg5/seg6: the strong blur removes the Canny edges tampering needs, so
    tampering is masked (rejected as global loss); the engine reports blur.
  * seg8: the near-black low-light causally explains the blur (near-black
    relation activates at confidence >= 0.8), so the engine reports
    low_light alone.

All tunable parameters are named constants at the top of this module --
adjust them if a segment does not cross its detection threshold, then
re-run with --verify to confirm.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

# --- sys.path bootstrap ---------------------------------------------------
# `python tools/generate_synthetic_video.py` inserts tools/ (not the project
# root) at sys.path[0], so `from config import ...` would fail without this.
# It also lets `python -m tools.generate_synthetic_video` work from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import BASELINES_DIR, PROJECT_ROOT, validate_config  # noqa: E402

logger = logging.getLogger(__name__)

# --- Output / clip constants ----------------------------------------------
OUTPUT_FILENAME = "synthetic_multifault_demo.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test_footage" / OUTPUT_FILENAME
GROUND_TRUTH_PATH = DEFAULT_OUTPUT.with_name("synthetic_multifault_demo_ground_truth.json")

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_SEED = 20260102
DEFAULT_CAMERA_ID = "synth_cam"

BASELINE_DURATION_SECONDS = 5.0
SEGMENT_DURATION_SECONDS = 5.0

# --- Scene constants -------------------------------------------------------
# Fine-detail noise amplitude. This layer is the sharpness source the blur
# detector measures (it is high-spatial-frequency, so any Gaussian blur
# kills it and the Laplacian variance drops well below the 50% floor), yet
# its gradients stay below the tampering detector's Canny threshold, so the
# two detectors' edge/sharpness references come from different structure.
DETAIL_NOISE_SIGMA = 13.0

# --- Fault injection parameters (all tunable) ------------------------------
# Low-light darken factors (multiplier applied to the HSV value channel).
#   MODERATE        co-occurs with tilt (segment 2) and blur (segment 7).
#                   Strong enough for the decision engine to CONFIRM the
#                   low-light event (confidence ~0.6 >= the 0.5 emission
#                   floor) yet below the 0.8 gate floor, so tilt still runs
#                   in segment 2. The blur co-firing it causes in segment 2
#                   is suppressed by tilt (tilt -> blur is an "always"
#                   relation); in segment 7 the low_light -> blur relation
#                   needs near-black (>=0.8), so both survive there.
#   TAMPERING       co-occurs with tampering (segment 3). Must stay mild
#                   enough that the scene's Canny edges survive H.264/MPEG-4
#                   re-encoding for the tampering detector (at factor <= ~0.6
#                   the darkening itself removes edges globally and tampering
#                   is rejected; 0.75 measured too close to the 0.75
#                   total-loss ceiling once encoding noise is included). The
#                   low-light candidate fires but stays well below the
#                   engine's 0.5 confirmation floor -- the honest
#                   cross-triggering tradeoff the decision layer sees.
#   NEAR_BLACK      essentially black; trips the low-light gate (>=0.8),
#                   which is the near-black scenario the decision layer
#                   models (segment 8).
LOWLIGHT_MODERATE_FACTOR = 0.37
LOWLIGHT_TAMPERING_FACTOR = 0.78
LOWLIGHT_NEAR_BLACK_FACTOR = 0.08

# Blur: Gaussian blur sigma (pixels). Set so the fine-detail noise layer is
# destroyed (Laplacian variance falls to ~0.2% of baseline) -> an
# unambiguous blur event. Note: this strength of blur also removes the Canny
# edges the tampering detector measures, so tampering is masked by blur in
# the tampering+blur segments -- a documented cross-triggering reality the
# decision layer must tolerate.
BLUR_SIGMA = 2.0

# Tilt: affine rotation (degrees) plus a small translation (px), mirroring
# the physically plausible "camera knocked" signature. Rendered with
# INTER_NEAREST deliberately: linear-interpolation resampling of the
# fine-noise layer measurably reduces Laplacian sharpness (measured
# sharpness ratio ~0.32), which would cross-trigger the blur detector
# during a tilt-only segment. The median matched-keypoint displacement must
# exceed 10% of the frame diagonal (1280x720 -> ~147 px); with 20 deg + a
# (200, 70) px translation the measured shift is ~0.12 of the diagonal
# (confidence ~0.81), comfortably above the candidate threshold while DISK
# matching stays reliable (500+ matches).
TILT_ROTATION_DEGREES = 20.0
TILT_TRANSLATION_PX = (200.0, 70.0)

# Tampering: a structure-free occlusion rectangle over a fraction of the
# frame area, with a soft blurred edge so it registers as structure loss
# rather than a new crisp edge ring.
#   SMALL  ~18% of frame area: fires the tampering detector's per-frame
#          candidate but stays below the decision engine's tampering
#          confirmation floor (~57% contiguous structure loss needed).
#   LARGE  ~65% of frame area: clears that floor with margin (and stays
#          under the detector's 0.75 global-loss ceiling).
# The fill is a mid-gray solid: dark enough to read as an obstruction but
# with HSV value (~88) above the low-light dark-pixel threshold (60), so it
# does not perturb the dark-pixel ratio (the previous near-black fill pushed
# the low-light detector past its confirmation floor on a 60% obstruction).
# Flatness still reduces Laplacian sharpness, so a large obstruction
# co-fires the blur detector -- a genuine physical cross-trigger the engine
# resolves via the tampering -> blur area-conservation suppression.
TAMPER_SMALL_FRACTION = 0.18
TAMPER_LARGE_FRACTION = 0.65
TAMPER_FILL_COLOR = (78, 82, 88)
TAMPER_EDGE_BLUR_SIGMA = 8.0

# --- Verification sampling --------------------------------------------------
VERIFY_FRAMES_PER_SEGMENT = 3

# Codec fallback order (first that writes AND reads back cleanly wins).
_CODECS: tuple[tuple[str, str], ...] = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("MJPG", ".mp4"),
    ("XVID", ".mp4"),
)
_MAX_FRAME_COUNT_TOLERANCE = 2


@dataclass(frozen=True)
class Segment:
    """One timeline segment: what to inject and what it means for ground truth.

    ``inject`` names renderer transforms (keys of ``_TRANSFORMS``); ``faults``
    holds the canonical fault labels recorded in the ground-truth JSON.
    """

    name: str
    inject: tuple[str, ...]
    faults: tuple[str, ...]
    start_s: float = 0.0
    end_s: float = 0.0


# Timeline after the clean baseline, in the order requested:
#   1 tilt + tampering (small)          5 tampering (small) + blur
#   2 tilt + low_light                  6 tampering (large) + blur
#   3 tampering (small) + low_light     7 low_light (moderate) + blur
#   4 tampering (large), normal light   8 low_light (near-black) + blur
_SEGMENTS: tuple[Segment, ...] = (
    Segment(name="baseline", inject=(), faults=()),
    Segment(name="tilt_tampering_small", inject=("tilt", "tampering_small"), faults=("tilt", "tampering")),
    Segment(name="tilt_low_light", inject=("tilt", "low_light_moderate"), faults=("tilt", "low_light")),
    Segment(name="tampering_small_low_light", inject=("tampering_small", "low_light_tampering"), faults=("tampering", "low_light")),
    Segment(name="tampering_large", inject=("tampering_large",), faults=("tampering",)),
    Segment(name="tampering_small_blur", inject=("tampering_small", "blur"), faults=("tampering", "blur")),
    Segment(name="tampering_large_blur", inject=("tampering_large", "blur"), faults=("tampering", "blur")),
    Segment(name="low_light_moderate_blur", inject=("low_light_moderate", "blur"), faults=("low_light", "blur")),
    Segment(name="low_light_near_black_blur", inject=("low_light_near_black", "blur"), faults=("low_light", "blur")),
)

# Renderer transforms keyed by injection name.
_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tampering_small": lambda f: _apply_tampering(f, TAMPER_SMALL_FRACTION),
    "tampering_large": lambda f: _apply_tampering(f, TAMPER_LARGE_FRACTION),
    "low_light_moderate": lambda f: _apply_low_light(f, LOWLIGHT_MODERATE_FACTOR),
    "low_light_tampering": lambda f: _apply_low_light(f, LOWLIGHT_TAMPERING_FACTOR),
    "low_light_near_black": lambda f: _apply_low_light(f, LOWLIGHT_NEAR_BLACK_FACTOR),
    "blur": lambda f: _apply_blur(f, BLUR_SIGMA),
    "tilt": lambda f: _apply_tilt(f, TILT_ROTATION_DEGREES, TILT_TRANSLATION_PX),
}

# Fixed physical application order: camera motion first, then lens
# obstruction, then optical degradation (defocus, then light).
_TRANSFORM_ORDER = {
    "tilt": 0,
    "tampering_small": 1,
    "tampering_large": 1,
    "blur": 2,
    "low_light_moderate": 3,
    "low_light_tampering": 3,
    "low_light_near_black": 3,
}


def _render_base_scene(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Draw the static, deterministic "busy room" background (BGR uint8).

    The scene must provide three different kinds of measurable structure:
      * strong high-contrast edges (books, window grid, door, barcode) for
        the tampering detector's Canny edge maps;
      * fine high-frequency detail (DETAIL_NOISE_SIGMA) for the blur
        detector's Laplacian variance;
      * abundant distinctive corners/patches (text, grid lines, book
        spines) for DISK keypoint matching during tilt detection.
    """
    frame = np.zeros((height, width, 3), np.uint8)

    # --- Wall and floor ---------------------------------------------------
    wall_bottom = int(height * 0.62)
    cv2.rectangle(frame, (0, 0), (width - 1, wall_bottom - 1), (168, 170, 176), -1)
    # Subtle horizontal wall banding (extra horizontal edges).
    for y in range(0, wall_bottom, 96):
        cv2.rectangle(frame, (0, y), (width - 1, min(y + 3, wall_bottom - 1)), (142, 145, 152), -1)

    # Floor with a large checkerboard (dense grid lines = edges + corners).
    cv2.rectangle(frame, (0, wall_bottom), (width - 1, height - 1), (96, 86, 74), -1)
    tile = 56
    for row in range(wall_bottom, height, tile):
        for col in range(0, width, tile):
            light = ((row // tile) + (col // tile)) % 2 == 0
            color = (132, 120, 102) if light else (70, 62, 52)
            cv2.rectangle(
                frame,
                (col, row),
                (min(col + tile, width - 1), min(row + tile, height - 1)),
                color,
                -1,
            )

    # --- Window with cross grid ---------------------------------------------
    wx0, wy0 = int(width * 0.10), int(height * 0.08)
    wx1, wy1 = int(width * 0.34), int(height * 0.48)
    cv2.rectangle(frame, (wx0, wy0), (wx1, wy1), (70, 74, 82), -1)               # frame
    cv2.rectangle(frame, (wx0 + 8, wy0 + 8), (wx1 - 8, wy1 - 8), (150, 172, 200), -1)  # glass
    midx, midy = (wx0 + wx1) // 2, (wy0 + wy1) // 2
    cv2.rectangle(frame, (midx - 4, wy0 + 8), (midx + 4, wy1 - 8), (70, 74, 82), -1)  # vertical mullion
    cv2.rectangle(frame, (wx0 + 8, midy - 4), (wx1 - 8, midy + 4), (70, 74, 82), -1)  # horizontal mullion
    cv2.rectangle(frame, (wx0 + 22, wy0 + 22), (wx0 + 76, wy0 + 118), (205, 220, 238), -1)  # "reflection"
    cv2.rectangle(frame, (wx0 + 110, wy0 + 44), (wx0 + 162, wy0 + 140), (188, 204, 224), -1)

    # --- Door with panels and handle ------------------------------------------
    dx0, dy0 = int(width * 0.40), wall_bottom - int(height * 0.52)
    dx1, dy1 = int(width * 0.52), wall_bottom - 1
    cv2.rectangle(frame, (dx0, dy0), (dx1, dy1), (122, 100, 76), -1)
    cv2.rectangle(frame, (dx0 + 12, dy0 + 16), (dx1 - 12, dy0 + int(height * 0.17)), (100, 82, 60), -1)
    cv2.rectangle(frame, (dx0 + 12, dy1 - int(height * 0.22)), (dx1 - 12, dy1 - 12), (100, 82, 60), -1)
    cv2.circle(frame, (dx1 - 16, (dy0 + dy1) // 2), 5, (232, 212, 178), -1)

    # --- Framed picture --------------------------------------------------------
    px0, py0 = int(width * 0.58), int(height * 0.10)
    px1, py1 = int(width * 0.70), int(height * 0.30)
    cv2.rectangle(frame, (px0, py0), (px1, py1), (58, 50, 42), -1)
    cv2.rectangle(frame, (px0 + 10, py0 + 10), (px1 - 10, py1 - 10), (206, 201, 186), -1)
    cv2.rectangle(frame, (px0 + 38, py0 + 46), (px1 - 38, py1 - 30), (92, 122, 152), -1)
    cv2.circle(frame, (px0 + 74, py0 + 118), 15, (242, 222, 132), -1)
    cv2.rectangle(frame, (px0 + 60, py0 + 108), (px1 - 30, py1 - 46), (120, 150, 90), -1)


    # --- Bookshelf with colored book spines -------------------------------------
    bx0, by0 = int(width * 0.76), int(height * 0.08)
    bx1, by1 = int(width * 0.94), int(height * 0.56)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (82, 70, 56), -1)
    shelf_ys = list(range(by0 + 10, by1, (by1 - by0) // 5))
    for sy in shelf_ys:
        cv2.rectangle(frame, (bx0 + 6, sy), (bx1 - 6, min(sy + 5, by1 - 2)), (56, 48, 38), -1)
    book_colors = [
        (205, 85, 75), (70, 142, 205), (85, 185, 95), (232, 205, 95),
        (165, 95, 168), (95, 95, 95), (232, 142, 62), (70, 120, 170),
    ]
    for sy0, sy1 in zip(shelf_ys[:-1], shelf_ys[1:]):
        x = bx0 + 12
        while x < bx1 - 28:
            book_w = int(rng.integers(14, 30))
            book_h = sy1 - sy0 - 10
            cv2.rectangle(
                frame,
                (x, sy0 + 8),
                (min(x + book_w, bx1 - 8), sy0 + 8 + book_h),
                book_colors[x % len(book_colors)],
                -1,
            )
            x += book_w + 2

    # --- Barcode-style block (dense high-frequency edges) ------------------------
    bc0, bc1 = int(width * 0.44), int(width * 0.79)
    by = int(height * 0.10)
    for i, x in enumerate(range(bc0, bc1, 9)):
        bw = 3 if i % 2 == 0 else 7
        cv2.rectangle(frame, (x, by), (min(x + bw, bc1), by + int(height * 0.20)), (40, 42, 48), -1)

    # --- Table + monitor -----------------------------------------------------------
    tx0, ty0 = int(width * 0.08), int(height * 0.66)
    tx1, ty1 = int(width * 0.44), int(height * 0.70)
    cv2.rectangle(frame, (tx0, ty0), (tx1, ty1), (140, 110, 80), -1)
    cv2.rectangle(frame, (tx0 + 8, ty1), (tx0 + 18, ty1 + int(height * 0.16)), (92, 72, 52), -1)
    cv2.rectangle(frame, (tx1 - 22, ty1), (tx1 - 12, ty1 + int(height * 0.16)), (92, 72, 52), -1)
    cv2.rectangle(frame, (tx0 + int(width * 0.05), ty0 - int(height * 0.12)),
                  (tx0 + int(width * 0.20), ty0 - 4), (30, 32, 36), -1)
    cv2.rectangle(frame, (tx0 + int(width * 0.10), ty0 - int(height * 0.26)),
                  (tx0 + int(width * 0.26), ty0 - int(height * 0.11)), (18, 20, 24), -1)
    cv2.rectangle(frame, (tx0 + int(width * 0.105), ty0 - int(height * 0.255)),
                  (tx0 + int(width * 0.255), ty0 - int(height * 0.115)), (150, 192, 222), -1)
    for i in range(3):
        yy = ty0 - int(height * 0.24) + i * int(height * 0.05)
        cv2.rectangle(frame, (tx0 + int(width * 0.12), yy), (tx0 + int(width * 0.22), yy + 3), (40, 62, 92), -1)

    # --- Plant (pot, trunk, foliage circles) ------------------------------------------
    pxx, pyy = int(width * 0.56), int(height * 0.86)
    cv2.rectangle(frame, (pxx - 9, pyy), (pxx + 9, pyy + 18), (112, 82, 62), -1)
    cv2.rectangle(frame, (pxx - 3, pyy - 26), (pxx + 3, pyy), (82, 62, 42), -1)
    for cx, cy, r in ((pxx - 30, pyy - 70, 22), (pxx + 10, pyy - 80, 26), (pxx + 34, pyy - 60, 18), (pxx - 6, pyy - 96, 20)):
        cv2.circle(frame, (cx, cy), r, (40, 122, 72), -1)
        cv2.circle(frame, (cx - 4, cy - 4), max(r - 7, 4), (62, 152, 92), -1)

    # --- Random colored shapes (deterministic under the fixed seed) ----------------------
    for _ in range(26):
        kind = int(rng.integers(0, 3))
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        color = tuple(int(v) for v in rng.integers(70, 240, size=3))
        if kind == 0:
            cv2.circle(frame, (cx, cy), int(rng.integers(6, 28)), color, -1)
        elif kind == 1:
            w, h = int(rng.integers(12, 64)), int(rng.integers(8, 44))
            cv2.rectangle(frame, (cx, cy), (min(cx + w, width - 1), min(cy + h, height - 1)), color, -1)
        else:
            pts = np.array(
                [[cx, cy], [cx + int(rng.integers(12, 52)), cy],
                 [cx + int(rng.integers(4, 22)), cy + int(rng.integers(12, 54))]],
                np.int32,
            )
            cv2.fillPoly(frame, [pts], color)

    # --- Text labels (distinctive DISK keypoints) -----------------------------------------
    cv2.putText(frame, "SYNTHETIC ROOM", (int(width * 0.36), int(height * 0.055)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 44, 52), 3, cv2.LINE_AA)
    cv2.putText(frame, "CAM 07", (int(width * 0.82), int(height * 0.955)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 44, 52), 2, cv2.LINE_AA)

    # --- Fine-detail noise + gentle vertical lighting gradient --------------------------------
    detail = rng.standard_normal((height, width), dtype=np.float32) * DETAIL_NOISE_SIGMA
    gradient = np.linspace(-18.0, 22.0, height, dtype=np.float32)[:, None]
    base = frame.astype(np.float32) + detail[:, :, None] + gradient[:, :, None]
    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def _apply_low_light(frame: np.ndarray, factor: float) -> np.ndarray:
    """Darken the whole frame by scaling the HSV value channel."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    scaled = np.clip(hsv[:, :, 2].astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
    hsv[:, :, 2] = scaled
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _apply_blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian defocus."""
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def _apply_tilt(
    frame: np.ndarray,
    angle_degrees: float = TILT_ROTATION_DEGREES,
    translation_px: tuple[float, float] = TILT_TRANSLATION_PX,
) -> np.ndarray:
    """Affine rotation (rotation + small translation), like a knocked camera.

    INTER_NEAREST is used deliberately: linear resampling of the fine-noise
    layer reduces Laplacian sharpness (measured ratio ~0.32) and would
    cross-trigger the blur detector during a tilt-only segment. BORDER_REPLICATE
    keeps the border brightness signature unchanged so the low-light detector
    stays quiet purely from the tilt. Most of the frame stays visible so DISK
    keypoint matching between baseline and current frame remains reliable.
    """
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    matrix[0, 2] += translation_px[0]
    matrix[1, 2] += translation_px[1]
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_tampering(
    frame: np.ndarray,
    fraction: float,
    fill: tuple[int, int, int] = TAMPER_FILL_COLOR,
    edge_blur_sigma: float = TAMPER_EDGE_BLUR_SIGMA,
) -> np.ndarray:
    """Draw a flat, structure-free occlusion covering ``fraction`` of the frame.

    The rectangle is centered and keeps the frame's aspect ratio. Its edge
    is softly blurred so the obstruction registers as *lost structure*
    rather than a crisp new edge ring. The interior is texture-free, so the
    grid blocks underneath lose their edge density — the tampering
    detector's signature.
    """
    height, width = frame.shape[:2]
    aspect = width / height
    # Width/height fractions whose product equals the requested area fraction.
    w_fraction = math.sqrt(fraction * aspect)
    h_fraction = math.sqrt(fraction / aspect)
    box_w = int(round(width * w_fraction))
    box_h = int(round(height * h_fraction))
    x0, y0 = (width - box_w) // 2, (height - box_h) // 2
    # May extend slightly past the borders (large case); cv2 clips the draw.

    mask = np.zeros((height, width), np.float32)
    cv2.rectangle(mask, (x0, y0), (x0 + box_w, y0 + box_h), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), edge_blur_sigma)
    occlusion = np.full((height, width, 3), fill, dtype=np.uint8)
    alpha = np.clip(mask, 0.0, 1.0)[:, :, None]
    return (frame * (1.0 - alpha) + occlusion * alpha).astype(np.uint8)


def _render_frame(base: np.ndarray, segment: Segment) -> np.ndarray:
    """Render one frame by applying the segment's fault transforms in order."""
    frame = base
    for name in sorted(segment.inject, key=lambda n: _TRANSFORM_ORDER[n]):
        frame = _TRANSFORMS[name](frame)
    return frame


def _frame_stream(
    height: int,
    width: int,
    fps: int,
    segments: Iterable[Segment],
    seed: int,
) -> Callable[[], Iterable[np.ndarray]]:
    """Return a factory producing a fresh, deterministic frame generator.

    A factory (not a plain generator) lets the codec fallback loop re-render
    identical frames for each codec attempt without holding ~1350 frames in
    memory. Deterministic under the fixed seed.
    """
    def _frames() -> Iterable[np.ndarray]:
        rng = np.random.default_rng(seed)
        base = _render_base_scene(rng, height, width)
        for segment in segments:
            frames = int(round((segment.end_s - segment.start_s) * fps))
            for _ in range(frames):
                yield _render_frame(base, segment)

    return _frames


def _build_timeline(
    segments: tuple[Segment, ...],
    baseline_duration: float,
    segment_duration: float,
) -> tuple[Segment, ...]:
    """Assign start/end times: clean baseline first, then fixed-length segments."""
    timed: list[Segment] = []
    t = 0.0
    for i, segment in enumerate(segments):
        duration = baseline_duration if i == 0 else segment_duration
        timed.append(Segment(segment.name, segment.inject, segment.faults, start_s=t, end_s=t + duration))
        t += duration
    return tuple(timed)


def _smoke_check(video_path: Path, expected_frames: int, expected_shape: tuple[int, int]) -> int:
    """Reopen the produced video and verify it decodes correctly.

    Returns the decoded frame count. Raises RuntimeError on any failure:
    unreadable frame rate, empty/None frames, wrong dimensions, no content,
    or a frame count off by more than the tolerance.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"No valid frame rate read back from {video_path}.")
        decoded = 0
        first_mean: float | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Frame {decoded} decoded as empty.")
            if tuple(frame.shape[:2]) != expected_shape:
                raise RuntimeError(
                    f"Frame {decoded} has shape {tuple(frame.shape[:2])}, expected {expected_shape}."
                )
            if first_mean is None:
                first_mean = float(frame.mean())
            decoded += 1
        if first_mean is None or first_mean < 1.0:
            raise RuntimeError(f"{video_path} contains no measurable content.")
        if abs(decoded - expected_frames) > _MAX_FRAME_COUNT_TOLERANCE:
            raise RuntimeError(
                f"Decoded {decoded} frames, expected ~{expected_frames} "
                f"(tolerance {_MAX_FRAME_COUNT_TOLERANCE})."
            )
        return decoded
    finally:
        cap.release()


def _write_video(
    video_path: Path,
    frame_stream_factory: Callable[[], Iterable[np.ndarray]],
    fps: int,
    resolution: tuple[int, int],
    expected_frames: int,
) -> tuple[str, int]:
    """Write the clip with codec fallback; return (fourcc, decoded_frames)."""
    width, height = resolution
    errors: list[str] = []
    for fourcc_name, _suffix in _CODECS:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        if video_path.exists():
            video_path.unlink()
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            errors.append(f"{fourcc_name}: writer failed to open")
            continue
        try:
            written = 0
            for frame in frame_stream_factory():
                writer.write(frame)
                written += 1
        except Exception as exc:  # noqa: BLE001 - report and fall through
            writer.release()
            errors.append(f"{fourcc_name}: write failed: {exc}")
            continue
        writer.release()
        try:
            decoded = _smoke_check(video_path, expected_frames, (height, width))
        except RuntimeError as exc:
            errors.append(f"{fourcc_name}: smoke check failed: {exc}")
            continue
        logger.info("Encoded clip with fourcc %r (%d/%d frames decoded, %.1f s).",
                    fourcc_name, decoded, written, written / fps)
        return fourcc_name, decoded
    raise RuntimeError(f"All video codecs failed for {video_path}: " + "; ".join(errors))


def _write_ground_truth(
    video_path: Path,
    ground_truth_path: Path,
    camera_id: str,
    resolution: tuple[int, int],
    fps: int,
    segments: tuple[Segment, ...],
) -> None:
    """Write the per-segment ground truth plus a schema-compatible fault list.

    ``faults`` uses the same shape the existing validation scripts consume
    (``{"type", "start_s", "end_s"}``); ``segments`` carries the per-segment
    breakdown requested for decision-engine checking.
    """
    try:
        relative = video_path.resolve().relative_to(PROJECT_ROOT.resolve())
        video_ref = str(relative).replace("\\", "/")
    except ValueError:
        video_ref = str(video_path)

    fault_windows: list[dict[str, object]] = []
    for segment in segments:
        for fault in segment.faults:
            fault_windows.append({"type": fault, "start_s": segment.start_s, "end_s": segment.end_s})
    fault_windows.sort(key=lambda f: (f["start_s"], f["type"]))

    payload = {
        "video": video_ref,
        "camera_id": camera_id,
        "generator": "tools/generate_synthetic_video.py",
        "description": (
            "Fully synthetic, procedurally generated clip — no real camera footage. "
            "Proof-of-mechanism only, not real-world validation material."
        ),
        "resolution": list(resolution),
        "fps": fps,
        "segments": [
            {
                "name": segment.name,
                "start_s": round(segment.start_s, 3),
                "end_s": round(segment.end_s, 3),
                "faults": list(segment.faults),
            }
            for segment in segments
        ],
        "faults": fault_windows,
    }
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_segment_summary(segments: tuple[Segment, ...], fps: int) -> None:
    """Print the requested summary table of segments and time ranges."""
    width = max(len(segment.name) for segment in segments)
    header = f"{'Segment':<{width}}  {'Start (s)':>10}  {'End (s)':>10}  Injected faults"
    print(header)
    print("-" * len(header))
    for segment in segments:
        faults = ", ".join(segment.faults) if segment.faults else "(none)"
        print(f"{segment.name:<{width}}  {segment.start_s:>10.2f}  {segment.end_s:>10.2f}  {faults}")
    total = segments[-1].end_s
    print("-" * len(header))
    print(f"Total duration: {total:.1f} s at {fps} fps "
          f"({int(round(total * fps))} frames).")


def _verify_detections(
    video_path: Path,
    camera_id: str,
    segments: tuple[Segment, ...],
    frames_per_segment: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the production detectors on sampled frames and grade each segment.

    Baselines are captured from the clip's clean opening segment with the
    production ``pipeline.capture_baseline`` code path, then the four
    detectors evaluate sampled frames from every fault segment. Returns a
    per-segment/per-detector summary dict (``candidate_rate``,
    ``mean_confidence``). Heavy imports (torch/kornia) happen only here, so
    plain generation never pays the DISK load cost.
    """
    from detectors import blur as blur_detector
    from detectors import brightness, tampering, tilt
    from pipeline.capture_baseline import capture_baseline

    record = capture_baseline(camera_id, str(video_path))
    baseline_edges = cv2.imread(str(BASELINES_DIR / f"{camera_id}_edges.png"), cv2.IMREAD_GRAYSCALE)
    baseline_frame = cv2.imread(str(BASELINES_DIR / f"{camera_id}.jpg"))
    if baseline_edges is None or baseline_frame is None:
        raise RuntimeError("Baseline files missing after capture_baseline.")

    baseline_keypoints, baseline_descriptors = tilt.extract_features(baseline_frame)
    baseline_shape = baseline_frame.shape[:2]
    dark_ratio = float(record["lowlight_dark_pixel_ratio"])
    sharpness = float(record["blur_baseline_sharpness"])

    detectors: dict[str, Callable[[np.ndarray], object]] = {
        "low_light": lambda f: brightness.evaluate(f, dark_ratio),
        "tampering": lambda f: tampering.evaluate(f, baseline_edges),
        "blur": lambda f: blur_detector.evaluate(f, sharpness),
        "tilt": lambda f: tilt.evaluate(f, baseline_keypoints, baseline_descriptors, baseline_shape),
    }

    # --- Sample frames per fault segment from the produced file -------------
    samples: dict[str, list[np.ndarray]] = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"No valid frame rate read back from {video_path}.")
        for segment in segments:
            if not segment.faults:
                continue
            frames: list[np.ndarray] = []
            for k in range(frames_per_segment):
                t = segment.start_s + (k + 0.5) * (segment.end_s - segment.start_s) / frames_per_segment
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not sample frame at t={t:.2f}s in {segment.name}.")
                frames.append(frame)
            samples[segment.name] = frames
    finally:
        cap.release()

    # --- Run detectors and aggregate ----------------------------------------
    results: dict[str, dict[str, dict[str, float]]] = {}
    for segment in segments:
        if segment.name not in samples:
            continue
        per_detector: dict[str, dict[str, float]] = {}
        for det_name, detector in detectors.items():
            candidate_count = 0
            confidences: list[float] = []
            for frame in samples[segment.name]:
                try:
                    result = detector(frame)
                    if bool(result.is_candidate):
                        candidate_count += 1
                    confidences.append(float(result.confidence))
                except Exception as exc:  # noqa: BLE001 - surface in the report
                    logger.warning("Detector %s failed on %s sample: %s", det_name, segment.name, exc)
                    confidences.append(float("nan"))
            mean_conf = float(np.nanmean(confidences)) if confidences else 0.0
            per_detector[det_name] = {
                "candidate_rate": candidate_count / len(samples[segment.name]),
                "mean_confidence": mean_conf,
            }
        results[segment.name] = per_detector

    _print_verification_report(segments, results)
    return results


def _print_verification_report(
    segments: tuple[Segment, ...],
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Pretty-print the verification table with expected vs detected faults."""
    print()
    print("Verification report (production detector paths; sampled frames per segment):")
    header = (f"{'Segment':<28} {'Expected':<28} "
              f"{'low_light':>14} {'tampering':>14} {'blur':>14} {'tilt':>14}")
    print(header)
    print("-" * len(header))
    for segment in segments:
        if segment.name not in results:
            continue
        expected = ", ".join(segment.faults) if segment.faults else "(none)"
        cells = []
        for det_name in ("low_light", "tampering", "blur", "tilt"):
            det = results[segment.name].get(det_name, {"candidate_rate": 0.0, "mean_confidence": 0.0})
            cells.append(f"{det['candidate_rate']:.2f}/{det['mean_confidence']:.2f}")
        print(f"{segment.name:<28} {expected:<28} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14}")
    print("-" * len(header))
    print("Cells are candidate_rate / mean_confidence. A fault's detector should "
          "show candidate_rate > 0 where it was injected.")


def verify_existing_clip(
    video_path: str | Path,
    camera_id: str = DEFAULT_CAMERA_ID,
    fps: int = DEFAULT_FPS,
    frames_per_segment: int = VERIFY_FRAMES_PER_SEGMENT,
) -> dict:
    """Verify an already-generated clip with the production detectors.

    Uses the tool's own timeline (same segment durations), so the clip must
    have been produced by this generator. Returns the per-segment
    verification dict and prints the expected-vs-detected report.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video to verify not found: {video_path}")
    timed = _build_timeline(_SEGMENTS, BASELINE_DURATION_SECONDS, SEGMENT_DURATION_SECONDS)
    return _verify_detections(video_path, camera_id, timed, frames_per_segment)


def generate_synthetic_video(
    video_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    camera_id: str = DEFAULT_CAMERA_ID,
    fps: int = DEFAULT_FPS,
    resolution: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    seed: int = DEFAULT_SEED,
    verify: bool = False,
) -> dict:
    """Generate the synthetic multi-fault demo video and its ground truth.

    Returns a metadata dict (paths, codec, frame counts, verification).
    Raises RuntimeError/ValueError on any failure so CI-style callers fail
    loudly instead of silently running with a broken clip.
    """
    video_path = Path(video_path) if video_path is not None else DEFAULT_OUTPUT
    ground_truth_path = Path(ground_truth_path) if ground_truth_path is not None else GROUND_TRUTH_PATH
    width, height = resolution
    if width <= 0 or height <= 0 or fps <= 0 or BASELINE_DURATION_SECONDS <= 0 or SEGMENT_DURATION_SECONDS <= 0:
        raise ValueError("resolution, fps, and durations must be positive.")

    timed = _build_timeline(_SEGMENTS, BASELINE_DURATION_SECONDS, SEGMENT_DURATION_SECONDS)
    expected_frames = int(round(timed[-1].end_s * fps))
    video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc, decoded = _write_video(
        video_path,
        _frame_stream(height, width, fps, timed, seed),
        fps,
        (width, height),
        expected_frames,
    )

    _write_ground_truth(video_path, ground_truth_path, camera_id, resolution, fps, timed)
    _print_segment_summary(timed, fps)

    verification: dict | None = None
    if verify:
        verification = _verify_detections(video_path, camera_id, timed, VERIFY_FRAMES_PER_SEGMENT)

    return {
        "video_path": str(video_path),
        "ground_truth_path": str(ground_truth_path),
        "camera_id": camera_id,
        "fourcc": fourcc,
        "frames_written": expected_frames,
        "frames_decoded": decoded,
        "duration_seconds": timed[-1].end_s,
        "fps": fps,
        "resolution": list(resolution),
        "seed": seed,
        "verification": verification,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the generator CLI."""
    parser = argparse.ArgumentParser(
        prog="generate_synthetic_video",
        description="Generate a fully synthetic multi-fault demo video and ground-truth JSON.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Output video path (default: data/test_footage/synthetic_multifault_demo.mp4).")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH,
                        help="Output ground-truth JSON path (default: alongside the video).")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID,
                        help="Camera id used for verification baselines (default: synth_cam).")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resolution", default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
                        help="Frame resolution WIDTHxHEIGHT (default: 1280x720; "
                             "use 640x360 to match the CI fixture videos).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Deterministic scene seed (default: %(default)s).")
    parser.add_argument("--verify", action="store_true",
                        help="After generating, run the production detectors on sampled frames "
                             "and print an expected-vs-detected report (slow: loads DISK).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip generation; verify the existing clip at --out with the "
                             "production detectors (must have been produced by this tool).")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 success, 1 failure, 2 usage error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.verify and args.verify_only:
        print("error: --verify and --verify-only are mutually exclusive.", file=sys.stderr)
        return 2
    try:
        width_s, height_s = args.resolution.lower().split("x")
        resolution = (int(width_s), int(height_s))
    except ValueError:
        print(f"error: invalid --resolution {args.resolution!r}; expected WIDTHxHEIGHT", file=sys.stderr)
        return 2

    try:
        validate_config()
        if args.verify_only:
            verification = verify_existing_clip(
                video_path=args.out,
                camera_id=args.camera_id,
                fps=args.fps,
                frames_per_segment=VERIFY_FRAMES_PER_SEGMENT,
            )
            result = {"verification": verification}
        else:
            result = generate_synthetic_video(
                video_path=args.out,
                ground_truth_path=args.ground_truth,
                camera_id=args.camera_id,
                fps=args.fps,
                resolution=resolution,
                seed=args.seed,
                verify=args.verify,
            )
    except (ValueError, RuntimeError) as exc:
        logger.error("Synthetic video generation failed: %s", exc)
        return 1

    if args.verify_only:
        logger.info("Verification complete for %s (camera %s).", args.out, args.camera_id)
    else:
        logger.info(
            "Generated %s (%d frames, %.1f s, fourcc %s); ground truth at %s",
            result["video_path"], result["frames_written"], result["duration_seconds"],
            result["fourcc"], result["ground_truth_path"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

