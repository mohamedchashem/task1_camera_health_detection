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

STREAM_RECONNECT_BASE_SECONDS = 2.0
# Initial wait before the first reconnect attempt after a dropped
# connection. Each further consecutive reconnect waits longer (base *
# factor^(n-1), capped by MAX), so a recovering source is not hammered
# with rapid reconnect attempts.

STREAM_RECONNECT_MAX_SECONDS = 30.0
# Ceiling on the exponential-backoff wait: an extended outage must not
# push reconnect delays to unbounded lengths.

STREAM_RECONNECT_BACKOFF_FACTOR = 2.0
# Per-attempt multiplier applied to the reconnect delay.

STREAM_MAX_CONSECUTIVE_FAILURES = 15
# Roughly half a second of failed reads at 30fps before we treat it
# as a dropped connection rather than a transient decode error.

BASELINE_CAPTURE_SECONDS = 3.0
# Duration of the averaging window used when capturing a per-camera
# baseline, chosen to smooth out single-frame noise while staying
# short enough to capture manually without difficulty.

BASELINE_MAX_DARK_RATIO = 0.15
# Capture-time quality gate (warning, not a hard failure): a baseline
# segment whose mean dark-pixel ratio is at or above this ceiling was
# captured in improper lighting, making its edge structure and sharpness
# references unreliable. Logged as a quality warning on the saved record.
# Empirical starting point (cam_02's bad baseline measured 0.136).

BASELINE_MIN_SHARPNESS = 500.0
# Capture-time quality gate (warning, not a hard failure): a baseline
# segment whose mean Laplacian sharpness is at or below this floor was
# captured too blurry to serve as a trustworthy reference. Logged as a
# quality warning on the saved record. Empirical starting point
# (cam_02's bad baseline measured 398.6 vs cam_01's 3806.6).

TEST_RUNS_DIR = DATA_DIR / "test_runs"
# Output location for blind-test CSV logs — generated data, not code.

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

TAMPERING_MAX_GLOBAL_LOSS_FRACTION = 0.75
# Guard against global (non-localized) structure loss masquerading as
# obstruction. Real obstruction is localized: the largest lost cluster
# covers part of the lens while the rest of the scene keeps its
# structure, so the TOTAL disappeared fraction of meaningful blocks
# stays well below 1. Blur, low-light, and rotation resampling remove
# edges nearly everywhere, so the disappeared mask becomes one giant
# cluster covering most of the frame. This ceiling rejects those
# global-degradation cases (a real obstruction covering up to ~75% of
# meaningful structure still passes). Empirical starting point, not
# tuned to any specific footage.

TAMPERING_MIN_COMPACTNESS_RATIO = 0.6
# A tampering candidate must also have its single largest lost cluster
# account for at least this fraction of ALL lost structure. This
# separates a physical obstruction from camera rotation, whose edge
# changes look like tampering on the raw largest-cluster metric alone.
# A real obstruction is one contiguous object over one region of the
# lens, so essentially all lost structure sits in one cluster
# (largest/total ~ 1.0). Rotation displaces edges everywhere, scattering
# the lost blocks into many disconnected patches where the largest
# cluster is only a fraction of the total (measured 0.50-0.52 on the
# synthetic fixture vs 1.00 during its tampering window). Requiring the
# largest cluster to hold a clear majority of the loss rejects that
# scattered rotation signature without weakening genuine obstruction
# detection. Empirical starting point with wide margin on both sides.

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

TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION = 0.5
# A tampering baseline must contain meaningful structure in at least this
# fraction of its grid blocks. A baseline captured in dark/blurry
# conditions has almost no stable structure to "lose", so any ambient
# change can masquerade as a compact obstruction cluster. Below this
# fraction the detector reports a "degraded_baseline" result (never a
# candidate) until a usable baseline is captured. Empirical starting
# point; measured 0.99 (cam_01, clean) vs 0.30 (cam_02, poor lighting).

TAMPERING_BLOCK_DENSITY_DROP_RATIO = 0.5
# A block is flagged as "disappeared" if its current edge density
# drops to this fraction (or less) of its baseline density. E.g. 0.5
# means the block lost at least half its edge structure.

TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO = 0.5
# Fraction of baseline-window frames in which a pixel must register as
# an edge for it to count as stable baseline structure, rather than
# frame-to-frame noise. Reasoned starting point, not tuned to any
# specific footage — applies identically to any camera/baseline window.

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

# --- Tilt/angle detector -------------------------------------------------
# Detects camera rotation via learned local feature matching (DISK,
# accessed through Kornia) between the current frame and baseline
# reference. Rotation is measured as the median displacement of
# matched keypoints, normalized by frame diagonal -- deliberately NOT
# via homography or affine transform fitting. Model fitting (including
# using RANSAC purely as an outlier filter) proved numerically unstable
# on real footage: sparse/clustered matches during faults produce
# degenerate fits, which silently discarded real tilt matches along
# with bad ones. Outlier rejection instead uses median absolute
# deviation (MAD) on the displacement values directly, which requires
# no geometric model of the scene.

TILT_MAX_KEYPOINTS = 2048
# Max keypoints to extract per frame, passed to DISK's detector.

TILT_MIN_RELIABLE_MATCHES = 10
# Minimum matched keypoint pairs required to trust a rotation estimate.
# Below this, the frame is skipped rather than risk a meaningless
# angle from too few points. Applied both to raw matches and to
# MAD-filtered inlier matches.

TILT_MIN_MATCH_RATIO = 0.05
# Relative match-volume guard: a rotation estimate is trusted only when the
# number of matches is at least this fraction of the SMALLER of the two
# frames' keypoint counts. On severely blurred/obscured/black frames DISK
# can still produce a handful of spurious correspondences that pass the
# absolute floor (TILT_MIN_RELIABLE_MATCHES) but represent a tiny fraction
# of the available features; their median displacement is meaningless and
# would otherwise read as a huge false tilt. Empirical starting point,
# validated on the fixture plus test_video2 footage.

TILT_MIN_INLIER_RATIO = 0.5
# After MAD outlier rejection, at least this fraction of the matches must
# survive. A genuine camera displacement moves features coherently, so most
# matches agree; blur- or noise-induced garbage is scattered and MAD rejects
# most of it. A guard on the raw match count alone is not enough.

TILT_MEDIAN_SHIFT_THRESHOLD_RATIO = 0.1
# A frame is flagged as a tilt candidate when the median displacement
# of matched keypoints reaches this fraction of the frame's diagonal
# length or more (e.g. 0.1 = median keypoint shift of 10% of the
# diagonal). Scale-independent, so it applies the same way regardless
# of camera resolution. Reasoned starting point, not tuned to this
# footage.

TILT_SHIFT_CONFIDENCE_CEILING_RATIO = 0.15
# Median-shift ratio at which confidence saturates to 1.0. The previous
# 0.3 (a keypoint move of 30% of the frame diagonal) compressed real
# faults into the low half of the scale: a clearly visible camera
# rotation (median keypoint shift ~12% of the diagonal on the synthetic
# fixture) read as only ~0.4 confidence. 0.15 = a median keypoint move
# of 15% of the diagonal, well past the candidate threshold (10%), is
# treated as an unambiguous camera move; confidence therefore scales
# visibly across the range of realistic tilt magnitudes instead of
# saturating only for extreme displacements.

TILT_MAD_REJECTION_THRESHOLD = 3.0
# Number of MADs (median absolute deviations) a matched point's
# displacement may differ from the median before it's rejected as an
# outlier. 3.0 follows the "X84 rule," a standard robust-statistics
# default (~2 standard deviations under Gaussian noise), also reported
# as effective for outlier rejection in feature-tracking specifically.

TILT_MAD_EPSILON = 1e-6
# Floor value for MAD when computing outlier z-scores. Guards against
# division by a near-zero MAD, which happens when matched-point
# displacements are already nearly identical (e.g. a static scene) --
# that case has nothing to reject, not everything to reject.

TILT_MATCH_RATIO_THRESHOLD = 1.0
# Threshold for match_smnn's nearest-neighbor ratio test. Loosened from
# an initial 0.9 after diagnosing that large real perspective changes
# make correct keypoint matches look less similar (higher ratio) than
# under mild viewpoint changes -- a real, generalizable property of
# matching under significant camera movement, not specific to one
# video. RANSAC (downstream) filters remaining false matches, so this
# threshold's job is only to avoid discarding genuine matches too
# aggressively before RANSAC gets a chance to see them.

TILT_MODEL_NAME = "depth"
# Pretrained DISK checkpoint name passed to kornia's DISK.from_pretrained()
# (weights downloaded once by torch.hub into its cache, then reused) when no
# local weights path is configured. "depth" is kornia's default DISK variant.

TILT_DISK_WEIGHTS_PATH: Path | None = None
# Optional absolute path to a local DISK checkpoint file ("depth-save.pth"
# format, i.e. a dict with an "extractor" state-dict key). When set, the
# tilt detector loads weights from this file instead of kornia's pretrained
# download -- the offline/hermetic path. None means use the pretrained
# default (TILT_MODEL_NAME) via torch.hub's cache.

TILT_DEVICE: str | None = None
# Optional explicit torch device for tilt feature extraction ("cpu",
# "cuda:0", ...). None means auto-select: CUDA if available, else CPU.

# --- Phase 3 operational pipeline ----------------------------------------
# Device selection precedence: CLI --device > TILT_DEVICE > DEFAULT_DEVICE.
# Fail-fast policy: no silent CPU fallback. If the resolved device is a
# CUDA device and CUDA is unavailable, startup must fail unless
# ALLOW_CPU_FALLBACK (or the --allow-cpu-fallback CLI flag) explicitly
# opts into CPU. Availability is resolved at startup after CLI parsing
# (main.py), not inside validate_config(), because it depends on CLI
# overrides.

DEFAULT_DEVICE = "cuda"
# Default torch device for detector execution when neither the CLI nor
# TILT_DEVICE overrides it. "cuda" enforces the GPU mandate by default.

ALLOW_CPU_FALLBACK = False
# Opt-in only: allow falling back to CPU when the resolved CUDA device is
# unavailable. False makes an unavailable CUDA device fail fast instead
# of silently degrading performance.

FRAME_QUEUE_CAPACITY = 3
# Per-camera bounded queue between the reader sub-thread and the
# processor: capacity in frames. When full, the newest frame evicts the
# oldest (drop-oldest), capping memory and bounding latency on slow feeds.

MAX_PROCESSING_LAG_SECONDS = 2.0
# A frame dequeued from a live stream is dropped without processing when
# its capture timestamp is older than this many seconds (wall-clock vs
# frame time), i.e. the processor has fallen too far behind the stream.

TILT_SAMPLE_INTERVAL_SECONDS = 0.0
# Tilt detector cadence in seconds of video time. 0.0 = run on every
# frame (the GPU policy); >0 runs tilt at most once per interval, and
# frames where tilt is sub-sampled produce a "skipped" observation that
# is NOT counted by the temporal confirmation tracker.

METRICS_LOG_INTERVAL_SECONDS = 5.0
# Interval between CameraMetrics snapshots emitted to system.jsonl.

SHUTDOWN_TIMEOUT_SECONDS = 5.0
# Grace period for cooperative shutdown (stop signal -> flush -> drain).

# --- Automated ground-truth test thresholds ------------------------------
# Pass bars used ONLY by the automated ground-truth validation tests
# (tests/test_*_detector.py) to grade blind detector output against
# human-confirmed fault windows. These are NOT detector thresholds.
# Rates differ per detector because each detector's raw signal quality
# differs: tampering's structural-loss signal is noisier in practice, so
# its required true-positive rate is set lower than the other detectors'.
TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE = 0.7  # low-light, blur, tilt
TAMPERING_TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE = 0.6
TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE = 0.05

# --- Decision layer (Phase 1) ------------------------------------------------
# Severity precedence for primary-fault resolution: higher priority first.
# A detected physical cause outranks its own symptoms: rotation resampling
# (tilt) genuinely lowers Laplacian sharpness, so a real tilt event also
# trips the blur detector. tilt is ranked above blur so the cause wins the
# primary label instead of its symptom.
DECISION_PRECEDENCE = ("tampering", "low_light", "tilt", "blur")

# Which faults a primary fault's signal causally explains (cross-trigger map).
DECISION_SUPPRESSION_MAP = {
    "tampering": {"low_light", "blur", "tilt"},
    "low_light": {"blur", "tilt"},
    "tilt": {"blur"},
    "blur": set(),
}

# A suppressor must reach this confidence before it can override a
# lower-priority fault in precedence disputes. Detector confidences are
# not cross-normalized, so this prevents a weak signal from dominating.
# Raised from 0.3 to 0.5 so that a suppressor below the per-fault emission
# floor (DECISION_CONFIRM_MIN_CONFIDENCE) can never hijack a stronger
# signal. Empirical starting point.
DECISION_SUPPRESSOR_MIN_CONFIDENCE = 0.5

# Relative suppression margin: an active suppressor (confidence >=
# DECISION_SUPPRESSOR_MIN_CONFIDENCE) additionally needs
# ``suppressor_confidence * DECISION_SUPPRESSION_MARGIN >=
# suppressed_confidence`` to override the lower-priority candidate. With
# 1.5, a suppressor at the confidence gate can only override candidates up
# to 0.75 confidence; a 1.0-confidence signal requires a >= ~0.67
# suppressor. This stops weak tampering noise (measured 0.32-0.35 on the
# old scale at t=47.49s of test_video2) from overriding a 1.0-confidence
# tilt. Empirical starting point.
DECISION_SUPPRESSION_MARGIN = 1.5

# Per-fault emission floor: a candidate frame counts toward temporal
# confirmation only when its confidence is at least the fault's floor.
# Values are on each detector's normalized confidence scale. tampering's
# floor matches its recalibrated scale (0.50 = ~57.5% of baseline structure
# lost in one contiguous cluster); low_light/blur/tilt floors sit below the
# confidence values real faults reach so only noise is filtered.
DECISION_CONFIRM_MIN_CONFIDENCE = {
    "tampering": 0.5,
    "low_light": 0.5,
    "blur": 0.5,
    "tilt": 0.5,
}

# Execution order for running detectors, distinct from the fusion
# precedence (DECISION_PRECEDENCE): cheap signal detectors first, expensive
# structural detectors last, so a high-confidence gate can short-circuit
# expensive work on frames it already explains. Contains exactly the four
# fault types (validated).
DECISION_EXECUTION_ORDER = ("low_light", "blur", "tampering", "tilt")

# A gate detector that fires a candidate at or above its gate floor causes
# the detectors listed under it to be skipped for that frame (observation
# status "skipped", reason "suppressed_by_gate"). Skipped detectors are
# excluded from the confirmation tracker, so their windows age out instead
# of being polluted by unmeasurable frames. High-confidence low-light
# (near-black) makes both structural detectors unmeasurable; blur stays
# active as the primary optical signal.
DECISION_GATE_CONFIDENCE = 0.8
# Default gate floor: a gate detector must fire a candidate at confidence
# >= this to activate. Individual gates may demand a higher bar via
# DECISION_GATE_CONFIDENCE_BY_GATE.

DECISION_GATE_CONFIDENCE_BY_GATE = {
    "blur": 0.9,
}
# Per-gate floors that override DECISION_GATE_CONFIDENCE for that gate.
# Blur only gates the structural tilt detector at >= 0.90 (severe blur):
# below that the frame is still measurable for keypoint matching, so a
# real tilt must not be missed. During severe blur DISK can still emit a
# handful of spurious correspondences whose median displacement reads as
# a large false tilt (observed on test_video2, cam_02: tilt confidence 1.0
# at blur ~0.99), so tilt is skipped and its confirmation window ages out.

DECISION_GATE_SKIP_MAP = {
    "low_light": ("tampering", "tilt"),
    "blur": ("tilt",),
}

# Temporal confirmation window (time-based, matches BASELINE_CAPTURE_SECONDS).
DECISION_CONFIRMATION_WINDOW_SECONDS = 3.0
# Fraction of observed frames in the window that must be candidates.
DECISION_CONFIRMATION_MIN_POSITIVE_RATIO = 0.5
# Minimum observed frames in the window (floor for very low frame rates).
DECISION_CONFIRMATION_MIN_WINDOW_FRAMES = 3

# Fault-isolation backoff.
DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS = 30
DECISION_DETECTOR_ERROR_BACKOFF_SECONDS = 5.0

# Minimum gap between confirmed events of the same fault (spam guard).
DECISION_MIN_EVENT_GAP_SECONDS = 10.0

# --- Frame logs & persistence (Phase 2) --------------------------------------
# Frame-level evaluation log (JSON Lines, one JSON object per logged frame).
FRAME_LOG_PATH = DATA_DIR / "logs" / "frame_log.jsonl"
# 1 = log every frame; >1 samples every Nth frame. Frames where a detector
# errored are always logged regardless of sampling.
FRAME_LOG_INTERVAL_FRAMES = 1
# Frame log files are rotated and pruned by age.
FRAME_LOG_RETENTION_DAYS = 7
# Size cap for a single frame log file before it rolls to a timestamped
# backup (which is then pruned by FRAME_LOG_RETENTION_DAYS). Bounds disk
# growth within one long run.
FRAME_LOG_MAX_BYTES = 10 * 1024 * 1024

# Operational metrics log (JSON Lines): one CameraMetrics snapshot per interval.
SYSTEM_LOG_PATH = DATA_DIR / "logs" / "system.jsonl"
# system.jsonl rolls by size with RotatingFileHandler-style numbered backups
# (.1 newest ... .N oldest); the active file plus SYSTEM_LOG_BACKUP_COUNT
# backups bound the total disk the metrics log can occupy.
SYSTEM_LOG_MAX_BYTES = 10 * 1024 * 1024
SYSTEM_LOG_BACKUP_COUNT = 3

# Application log (python logging): mirrors stderr on disk and is size
# rotated so a long run cannot fill the disk with log records.
APP_LOG_FILE = DATA_DIR / "logs" / "app.log"
APP_LOG_MAX_BYTES = 10 * 1024 * 1024
APP_LOG_BACKUP_COUNT = 5

# Annotated frame snapshots on confirmed faults (bounded ring, oldest evicted).
EVENT_FRAMES_MAX_TOTAL = 200


def validate_config() -> None:
    """Fail fast at startup if the configuration is internally inconsistent.

    Call once from each entry point (CLI scripts and main.py). Checks:
    - all threshold ranges stay within their documented bounds;
    - TILT_MODEL_NAME is a checkpoint kornia's DISK knows;
    - TILT_DEVICE, when set, parses as a torch device;
    - TILT_DISK_WEIGHTS_PATH, when set, points at an existing file.

    Data directories (baselines, test runs, debug frames, ...) are
    intentionally NOT required to pre-exist: they are created on demand
    by the code that writes to them.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    # Low-light detector
    require(
        0 <= LOWLIGHT_DARK_PIXEL_THRESHOLD <= 255,
        f"LOWLIGHT_DARK_PIXEL_THRESHOLD must be in [0, 255]; got {LOWLIGHT_DARK_PIXEL_THRESHOLD}.",
    )
    require(
        0 < LOWLIGHT_BASELINE_DROP_RATIO,
        f"LOWLIGHT_BASELINE_DROP_RATIO must be positive; got {LOWLIGHT_BASELINE_DROP_RATIO}.",
    )

    # Streams / timing
    require(
        STREAM_RECONNECT_BASE_SECONDS > 0,
        f"STREAM_RECONNECT_BASE_SECONDS must be > 0; got {STREAM_RECONNECT_BASE_SECONDS}.",
    )
    require(
        STREAM_RECONNECT_MAX_SECONDS >= STREAM_RECONNECT_BASE_SECONDS,
        f"STREAM_RECONNECT_MAX_SECONDS must be >= STREAM_RECONNECT_BASE_SECONDS; "
        f"got max={STREAM_RECONNECT_MAX_SECONDS}, base={STREAM_RECONNECT_BASE_SECONDS}.",
    )
    require(
        STREAM_RECONNECT_BACKOFF_FACTOR > 1,
        f"STREAM_RECONNECT_BACKOFF_FACTOR must be > 1 (strict exponential growth); "
        f"got {STREAM_RECONNECT_BACKOFF_FACTOR}.",
    )
    require(
        STREAM_MAX_CONSECUTIVE_FAILURES > 0,
        f"STREAM_MAX_CONSECUTIVE_FAILURES must be > 0; got {STREAM_MAX_CONSECUTIVE_FAILURES}.",
    )
    require(
        BASELINE_CAPTURE_SECONDS > 0,
        f"BASELINE_CAPTURE_SECONDS must be > 0; got {BASELINE_CAPTURE_SECONDS}.",
    )
    require(
        0 < BASELINE_MAX_DARK_RATIO <= 1,
        f"BASELINE_MAX_DARK_RATIO must be in (0, 1]; got {BASELINE_MAX_DARK_RATIO}.",
    )
    require(
        BASELINE_MIN_SHARPNESS > 0,
        f"BASELINE_MIN_SHARPNESS must be > 0; got {BASELINE_MIN_SHARPNESS}.",
    )
    # Tampering/obstruction detector
    require(
        0 <= TAMPERING_CANNY_LOW_THRESHOLD < TAMPERING_CANNY_HIGH_THRESHOLD <= 255,
        f"TAMPERING_CANNY thresholds must satisfy 0 <= low < high <= 255; "
        f"got low={TAMPERING_CANNY_LOW_THRESHOLD}, high={TAMPERING_CANNY_HIGH_THRESHOLD}.",
    )
    require(
        TAMPERING_GRID_BLOCK_SIZE > 0,
        f"TAMPERING_GRID_BLOCK_SIZE must be > 0; got {TAMPERING_GRID_BLOCK_SIZE}.",
    )
    require(
        0 <= TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY <= 1,
        f"TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY must be in [0, 1]; "
        f"got {TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY}.",
    )
    require(
        0 < TAMPERING_BLOCK_DENSITY_DROP_RATIO <= 1,
        f"TAMPERING_BLOCK_DENSITY_DROP_RATIO must be in (0, 1]; got {TAMPERING_BLOCK_DENSITY_DROP_RATIO}.",
    )
    require(
        0 < TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION <= 1,
        f"TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION must be in (0, 1]; "
        f"got {TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION}.",
    )
    require(
        0 < TAMPERING_MAX_GLOBAL_LOSS_FRACTION <= 1,
        f"TAMPERING_MAX_GLOBAL_LOSS_FRACTION must be in (0, 1]; "
        f"got {TAMPERING_MAX_GLOBAL_LOSS_FRACTION}.",
    )
    require(
        0 < TAMPERING_MIN_COMPACTNESS_RATIO <= 1,
        f"TAMPERING_MIN_COMPACTNESS_RATIO must be in (0, 1]; "
        f"got {TAMPERING_MIN_COMPACTNESS_RATIO}.",
    )
    require(
        0 <= TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO <= 1,
        f"TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO must be in [0, 1]; "
        f"got {TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO}.",
    )
    require(
        0 < TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION <= 1,
        f"TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION must be in (0, 1]; "
        f"got {TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION}.",
    )

    # Blur/dirty-lens detector
    require(
        0 < BLUR_SHARPNESS_DROP_RATIO < 1,
        f"BLUR_SHARPNESS_DROP_RATIO must be in (0, 1); got {BLUR_SHARPNESS_DROP_RATIO}.",
    )

    # Tilt detector
    require(TILT_MAX_KEYPOINTS > 0, f"TILT_MAX_KEYPOINTS must be > 0; got {TILT_MAX_KEYPOINTS}.")
    require(
        TILT_MIN_RELIABLE_MATCHES > 0,
        f"TILT_MIN_RELIABLE_MATCHES must be > 0; got {TILT_MIN_RELIABLE_MATCHES}.",
    )
    require(
        0 < TILT_MIN_MATCH_RATIO <= 1,
        f"TILT_MIN_MATCH_RATIO must be in (0, 1]; got {TILT_MIN_MATCH_RATIO}.",
    )
    require(
        0 < TILT_MIN_INLIER_RATIO <= 1,
        f"TILT_MIN_INLIER_RATIO must be in (0, 1]; got {TILT_MIN_INLIER_RATIO}.",
    )
    require(
        0 < TILT_MEDIAN_SHIFT_THRESHOLD_RATIO <= 1,
        f"TILT_MEDIAN_SHIFT_THRESHOLD_RATIO must be in (0, 1]; "
        f"got {TILT_MEDIAN_SHIFT_THRESHOLD_RATIO}.",
    )
    require(
        0 < TILT_SHIFT_CONFIDENCE_CEILING_RATIO <= 1,
        f"TILT_SHIFT_CONFIDENCE_CEILING_RATIO must be in (0, 1]; "
        f"got {TILT_SHIFT_CONFIDENCE_CEILING_RATIO}.",
    )
    require(
        TILT_MAD_REJECTION_THRESHOLD > 0,
        f"TILT_MAD_REJECTION_THRESHOLD must be > 0; got {TILT_MAD_REJECTION_THRESHOLD}.",
    )
    require(TILT_MAD_EPSILON > 0, f"TILT_MAD_EPSILON must be > 0; got {TILT_MAD_EPSILON}.")
    require(
        TILT_MATCH_RATIO_THRESHOLD > 0,
        f"TILT_MATCH_RATIO_THRESHOLD must be > 0; got {TILT_MATCH_RATIO_THRESHOLD}.",
    )
    require(
        TILT_MODEL_NAME in {"depth", "epipolar"},
        f"TILT_MODEL_NAME must be 'depth' or 'epipolar'; got {TILT_MODEL_NAME!r}.",
    )
    if TILT_DEVICE is not None:
        import torch

        try:
            torch.device(TILT_DEVICE)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"TILT_DEVICE is not a valid torch device: {TILT_DEVICE!r}.") from exc
    if TILT_DISK_WEIGHTS_PATH is not None and not TILT_DISK_WEIGHTS_PATH.exists():
        raise ValueError(
            f"TILT_DISK_WEIGHTS_PATH does not exist: {TILT_DISK_WEIGHTS_PATH}. "
            "Unset it to use kornia's pretrained download instead."
        )

    # Phase 3 operational pipeline
    require(
        isinstance(DEFAULT_DEVICE, str) and len(DEFAULT_DEVICE) > 0,
        f"DEFAULT_DEVICE must be a non-empty string; got {DEFAULT_DEVICE!r}.",
    )
    require(
        isinstance(ALLOW_CPU_FALLBACK, bool),
        f"ALLOW_CPU_FALLBACK must be a bool; got {ALLOW_CPU_FALLBACK!r}.",
    )
    require(
        FRAME_QUEUE_CAPACITY >= 1,
        f"FRAME_QUEUE_CAPACITY must be >= 1; got {FRAME_QUEUE_CAPACITY}.",
    )
    require(
        MAX_PROCESSING_LAG_SECONDS > 0,
        f"MAX_PROCESSING_LAG_SECONDS must be > 0; got {MAX_PROCESSING_LAG_SECONDS}.",
    )
    require(
        TILT_SAMPLE_INTERVAL_SECONDS >= 0,
        f"TILT_SAMPLE_INTERVAL_SECONDS must be >= 0; got {TILT_SAMPLE_INTERVAL_SECONDS}.",
    )
    require(
        METRICS_LOG_INTERVAL_SECONDS > 0,
        f"METRICS_LOG_INTERVAL_SECONDS must be > 0; got {METRICS_LOG_INTERVAL_SECONDS}.",
    )
    require(
        SHUTDOWN_TIMEOUT_SECONDS > 0,
        f"SHUTDOWN_TIMEOUT_SECONDS must be > 0; got {SHUTDOWN_TIMEOUT_SECONDS}.",
    )
    import torch

    try:
        default_device = torch.device(DEFAULT_DEVICE)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"DEFAULT_DEVICE is not a valid torch device: {DEFAULT_DEVICE!r}."
        ) from exc
    require(
        default_device.type in ("cpu", "cuda"),
        f"DEFAULT_DEVICE must be 'cpu' or a CUDA device; got {DEFAULT_DEVICE!r}.",
    )

    # Decision layer
    require(
        set(DECISION_PRECEDENCE) == {"tampering", "low_light", "blur", "tilt"},
        f"DECISION_PRECEDENCE must contain exactly the four fault types; got {DECISION_PRECEDENCE}.",
    )
    for suppressor, suppressed in DECISION_SUPPRESSION_MAP.items():
        require(
            suppressor in DECISION_PRECEDENCE,
            f"DECISION_SUPPRESSION_MAP key {suppressor!r} is not in DECISION_PRECEDENCE.",
        )
        require(
            set(suppressed) <= set(DECISION_PRECEDENCE) - {suppressor},
            f"DECISION_SUPPRESSION_MAP[{suppressor!r}] must reference known fault types "
            f"other than itself; got {sorted(suppressed)}.",
        )
    require(
        0 <= DECISION_SUPPRESSOR_MIN_CONFIDENCE <= 1,
        f"DECISION_SUPPRESSOR_MIN_CONFIDENCE must be in [0, 1]; got {DECISION_SUPPRESSOR_MIN_CONFIDENCE}.",
    )
    require(
        DECISION_SUPPRESSION_MARGIN > 0,
        f"DECISION_SUPPRESSION_MARGIN must be > 0; got {DECISION_SUPPRESSION_MARGIN}.",
    )
    require(
        set(DECISION_CONFIRM_MIN_CONFIDENCE) == {"tampering", "low_light", "blur", "tilt"},
        f"DECISION_CONFIRM_MIN_CONFIDENCE must cover exactly the four fault types; "
        f"got {sorted(DECISION_CONFIRM_MIN_CONFIDENCE)}.",
    )
    for fault, floor in DECISION_CONFIRM_MIN_CONFIDENCE.items():
        require(
            0 <= floor <= 1,
            f"DECISION_CONFIRM_MIN_CONFIDENCE[{fault!r}] must be in [0, 1]; got {floor}.",
        )
    require(
        set(DECISION_EXECUTION_ORDER) == {"tampering", "low_light", "blur", "tilt"}
        and len(DECISION_EXECUTION_ORDER) == 4,
        f"DECISION_EXECUTION_ORDER must contain exactly the four fault types once each; "
        f"got {DECISION_EXECUTION_ORDER}.",
    )
    for gate, skipped in DECISION_GATE_SKIP_MAP.items():
        require(
            gate in DECISION_PRECEDENCE,
            f"DECISION_GATE_SKIP_MAP key {gate!r} is not in DECISION_PRECEDENCE.",
        )
        require(
            set(skipped) <= set(DECISION_PRECEDENCE) - {gate},
            f"DECISION_GATE_SKIP_MAP[{gate!r}] must reference known fault types "
            f"other than itself; got {sorted(skipped)}.",
        )
    for gate, floor in DECISION_GATE_CONFIDENCE_BY_GATE.items():
        require(
            gate in DECISION_GATE_SKIP_MAP,
            f"DECISION_GATE_CONFIDENCE_BY_GATE key {gate!r} has no entry in "
            f"DECISION_GATE_SKIP_MAP and cannot gate anything.",
        )
        require(
            0 < floor <= 1,
            f"DECISION_GATE_CONFIDENCE_BY_GATE[{gate!r}] must be in (0, 1]; got {floor}.",
        )
    require(
        0 < DECISION_GATE_CONFIDENCE <= 1,
        f"DECISION_GATE_CONFIDENCE must be in (0, 1]; got {DECISION_GATE_CONFIDENCE}.",
    )
    require(
        DECISION_CONFIRMATION_WINDOW_SECONDS > 0,
        f"DECISION_CONFIRMATION_WINDOW_SECONDS must be > 0; got {DECISION_CONFIRMATION_WINDOW_SECONDS}.",
    )
    require(
        0 < DECISION_CONFIRMATION_MIN_POSITIVE_RATIO <= 1,
        f"DECISION_CONFIRMATION_MIN_POSITIVE_RATIO must be in (0, 1]; "
        f"got {DECISION_CONFIRMATION_MIN_POSITIVE_RATIO}.",
    )
    require(
        DECISION_CONFIRMATION_MIN_WINDOW_FRAMES >= 1,
        f"DECISION_CONFIRMATION_MIN_WINDOW_FRAMES must be >= 1; got {DECISION_CONFIRMATION_MIN_WINDOW_FRAMES}.",
    )
    require(
        DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS >= 1,
        f"DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS must be >= 1; "
        f"got {DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS}.",
    )
    require(
        DECISION_DETECTOR_ERROR_BACKOFF_SECONDS > 0,
        f"DECISION_DETECTOR_ERROR_BACKOFF_SECONDS must be > 0; "
        f"got {DECISION_DETECTOR_ERROR_BACKOFF_SECONDS}.",
    )
    require(
        DECISION_MIN_EVENT_GAP_SECONDS > 0,
        f"DECISION_MIN_EVENT_GAP_SECONDS must be > 0; got {DECISION_MIN_EVENT_GAP_SECONDS}.",
    )

    # Frame logs & persistence
    require(
        FRAME_LOG_INTERVAL_FRAMES >= 1,
        f"FRAME_LOG_INTERVAL_FRAMES must be >= 1; got {FRAME_LOG_INTERVAL_FRAMES}.",
    )
    require(
        FRAME_LOG_RETENTION_DAYS >= 1,
        f"FRAME_LOG_RETENTION_DAYS must be >= 1; got {FRAME_LOG_RETENTION_DAYS}.",
    )
    require(
        FRAME_LOG_MAX_BYTES >= 1,
        f"FRAME_LOG_MAX_BYTES must be >= 1; got {FRAME_LOG_MAX_BYTES}.",
    )
    require(
        SYSTEM_LOG_MAX_BYTES >= 1,
        f"SYSTEM_LOG_MAX_BYTES must be >= 1; got {SYSTEM_LOG_MAX_BYTES}.",
    )
    require(
        SYSTEM_LOG_BACKUP_COUNT >= 1,
        f"SYSTEM_LOG_BACKUP_COUNT must be >= 1; got {SYSTEM_LOG_BACKUP_COUNT}.",
    )
    require(
        APP_LOG_MAX_BYTES >= 1,
        f"APP_LOG_MAX_BYTES must be >= 1; got {APP_LOG_MAX_BYTES}.",
    )
    require(
        APP_LOG_BACKUP_COUNT >= 1,
        f"APP_LOG_BACKUP_COUNT must be >= 1; got {APP_LOG_BACKUP_COUNT}.",
    )
    require(
        EVENT_FRAMES_MAX_TOTAL >= 1,
        f"EVENT_FRAMES_MAX_TOTAL must be >= 1; got {EVENT_FRAMES_MAX_TOTAL}.",
    )

    # Ground-truth test thresholds
    require(
        0 < TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE <= 1,
        f"TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE must be in (0, 1]; "
        f"got {TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE}.",
    )
    require(
        0 < TAMPERING_TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE <= 1,
        f"TAMPERING_TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE must be in (0, 1]; "
        f"got {TAMPERING_TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE}.",
    )
    require(
        0 <= TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE < 1,
        f"TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE must be in [0, 1); "
        f"got {TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE}.",
    )