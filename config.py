"""
Central configuration for the camera health monitoring system.

All tunable values live here, not scattered inside detector logic.
This is the single place to retune thresholds after reviewing real
deployment data.
"""

from pathlib import Path

# --- Project paths -----------------------------------------------------
# Built with pathlib, not raw string concatenation, so paths work
# correctly regardless of OS.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BASELINES_DIR = DATA_DIR / "baselines"
EVENT_FRAMES_DIR = DATA_DIR / "event_frames"
EVENTS_DB_PATH = DATA_DIR / "events.db"

# --- Low-light detector --------------------------------------------------
# Brightness is measured using the HSV "V" (Value) channel rather than
# plain grayscale, because V is more stable when colored light sources
# (e.g. sodium streetlights, colored IR/LED illumination) are present.

LOWLIGHT_DARK_PIXEL_THRESHOLD = 60
# A pixel's V value (0-255) below this is counted as "dark".
# Chosen as a starting point representing a clearly dim pixel, not
# tuned to any specific test clip. Subject to revision once real
# deployment data is available.

LOWLIGHT_BASELINE_DROP_RATIO = 0.5
# Fraction (0-1). If the percentage of dark pixels in the current frame
# exceeds the camera's baseline dark-pixel percentage by this much
# (as a relative increase), the frame is flagged as a low-light
# candidate. E.g. 0.5 means "50% more dark pixels than baseline".

STREAM_RECONNECT_DELAY_SECONDS = 2.0
# Wait time before retrying a dropped connection, so we don't hammer
# the stream source with rapid reconnect attempts.

STREAM_MAX_CONSECUTIVE_FAILURES = 15
# Roughly half a second of failed reads at 30fps before we treat it
# as a dropped connection rather than a transient decode error.

BASELINE_CAPTURE_SECONDS = 3.0
# Duration of the averaging window used when capturing a per-camera
# baseline, chosen to smooth out single-frame noise while staying
# short enough to capture manually without difficulty.

TEST_RUN_DURATION_SECONDS = 60.0
# How long a blind validation run scores frames before stopping.
# Set slightly longer than the test clip's length so the full clip
# (including its loop-back point) is covered at least once.

TEST_RUNS_DIR = DATA_DIR / "test_runs"
# Output location for blind-test CSV logs — generated data, not code.

DEBUG_FRAMES_DIR = DATA_DIR / "debug_frames"
# Saved frames from moments a detector flagged a candidate, for manual
# visual review during testing. Not used in production runs.

# --- Tampering/obstruction detector ------------------------------------
# Detects lens obstruction by comparing edge density in grid blocks of
# the current frame against a camera's baseline, then requiring the
# largest connected cluster of "structure lost" blocks to exceed a
# size threshold. Chosen over raw pixel-difference because obstruction
# is fundamentally a loss of visible structure, not a brightness
# change. Known limitation: cannot fully separate real obstruction
# from blur/low-light using this signal alone — see
# tests/test_tampering_detector.py and the tampering summary doc.

TAMPERING_CANNY_LOW_THRESHOLD = 50
TAMPERING_CANNY_HIGH_THRESHOLD = 150
# Standard Canny edge-detector thresholds (low/high hysteresis bounds).
# Conventional starting values for general-purpose edge detection,
# not tuned to this specific footage.

TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION = 0.15
# A tampering candidate requires the SINGLE LARGEST connected cluster
# of "structure lost" blocks to cover at least this fraction of a
# camera's meaningful baseline structure — not just any blocks adding
# up to a high total. Real obstruction is physically localized to one
# region; blur and low-light degrade structure scattered across the
# whole frame, so they should not form one large contiguous cluster.

TAMPERING_GRID_BLOCK_SIZE = 32
# Frame is divided into blocks of this size (pixels) for edge-density
# comparison, rather than comparing individual pixels. Absorbs natural
# frame-to-frame jitter on fine texture (e.g. blinds, fences) that
# would otherwise register as false structural change. Generic to any
# resolution — blocks are computed from actual frame dimensions.

TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY = 0.02
# A block must have at least this fraction of edge pixels in the
# baseline to count as meaningful structure. Blocks with little/no
# baseline structure (e.g. a blank wall) are excluded from scoring,
# since they have nothing to "disappear."

TAMPERING_BLOCK_DENSITY_DROP_RATIO = 0.5
# A block is flagged as "disappeared" if its current edge density
# drops to this fraction (or less) of its baseline density. E.g. 0.5
# means the block lost at least half its edge structure.

TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO = 0.5
# Fraction of baseline-window frames in which a pixel must register as
# an edge for it to count as stable baseline structure, rather than
# frame-to-frame noise. Reasoned starting point, not tuned to any
# specific footage — applies identically to any camera/baseline window.

DEBUG_FRAME_SAMPLE_INTERVAL_SECONDS = 1.0
# Sampling interval for diagnostic edge-comparison dumps — one frame
# per second gives a representative timeline without flooding the
# debug folder. Diagnostic use only, not used in production scoring.

# --- Blur/dirty-lens detector --------------------------------------------
# Detects loss of image sharpness (out-of-focus, smudged/dirty lens) via
# Variance of Laplacian: the variance of a Laplacian-filtered frame is
# high when an image has abundant sharp edges, and low when detail has
# been smoothed away by blur. The standard, well-validated technique
# for this exact problem in both general CV and device-health-monitoring
# contexts. Baseline-relative, like the other detectors, since a
# naturally low-texture scene (e.g. a plain wall) has lower Laplacian
# variance even when perfectly in focus.

BLUR_SHARPNESS_DROP_RATIO = 0.5
# Fraction (0-1). A frame is flagged as a blur candidate when its
# Laplacian variance drops to this fraction (or less) of the camera's
# baseline sharpness. E.g. 0.5 means "sharpness fell to half or less
# of normal." Reasoned starting point, not tuned to any specific footage.