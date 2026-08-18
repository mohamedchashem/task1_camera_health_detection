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