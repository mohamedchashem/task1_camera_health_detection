"""Tampering/obstruction fault detector.

Detects lens obstruction by finding contiguous regions of a camera's
baseline structure that have gone missing in the current frame.
Obstruction is physically localized (one object blocking one region of
the lens); blur and low-light degrade structure scattered across the
whole frame instead. Two invariants separate real obstruction from
those other faults: the lost structure must form one large connected
cluster, and the TOTAL fraction of lost structure must stay bounded --
global degradation removes edges nearly everywhere and would otherwise
form one giant cluster covering the whole frame.

Approved tampering-candidacy redesign (subtasks 6-7/10, Approach A): the
absolute total-loss ceiling is replaced by a RELATIVE ambient-retention
model — per-block retention ratios (current/baseline edge density)
compared against the frame's own upper-tail ambient level, so a real
obstruction stays a statistical outlier even when co-occurring blur
collapses the ambient level. The pure scoring functions implement the
model (subtask 6) and ``evaluate()`` runs it (subtask 7).

Approach C, subtask 8 (engine side): the fully-ambiguous ``degraded_ambient``
case -- the frame's ambient retention collapsed below the degeneracy floor, so
no reliable relative comparison is possible -- is treated by the decision
engine as a per-frame non-measurement (observation status ``unavailable``),
never a clean negative, and resolved against the frame's own blur/low_light to
record whether the collapse is explained by contamination.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    TAMPERING_CANNY_LOW_THRESHOLD,
    TAMPERING_CANNY_HIGH_THRESHOLD,
    TAMPERING_GRID_BLOCK_SIZE,
    TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY,
    TAMPERING_BLOCK_DENSITY_DROP_RATIO,
    TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION,
    TAMPERING_MIN_COMPACTNESS_RATIO,
    TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION,
    TAMPERING_AMBIENT_RETENTION_QUANTILE,
    TAMPERING_OBSTRUCTION_DEPTH_RATIO,
    TAMPERING_AMBIENT_MIN_RETENTION,
)

# Detector-level diagnostic reason: the camera's baseline cannot support
# structure-loss measurement (dark/blurry/unstructured capture), so
# ``evaluate()`` emits this on every frame and never a candidate. Single
# source of truth shared with the decision engine, which treats it as a
# non-measurement (observation status ``unavailable``, excluded from the
# confirmation tracker's observed set).
DEGRADED_BASELINE_REASON = "degraded_baseline"

# Detector-level diagnostic reason for the CURRENT frame's ambient retention
# collapsing below the degeneracy floor (TAMPERING_AMBIENT_MIN_RETENTION):
# there is so little structure left anywhere in the frame that the relative
# retention comparison cannot run meaningfully -- the information-
# theoretically ambiguous case, deliberately never confirmed. Unlike
# DEGRADED_BASELINE_REASON (a static, per-camera property fixed by the
# baseline quality), this is a dynamic, per-frame outcome of what is happening
# in the frame right now. Single source of truth shared with the decision
# engine, which treats it as a non-measurement (observation status
# ``unavailable``) and resolves whether a co-occurring global fault explains
# it (Approach C, subtask 8).
DEGRADED_AMBIENT_REASON = "degraded_ambient"

# Mapping of ``ObstructionAssessment.reject_reason`` onto the detector's
# public ``reason`` diagnostic. Two reasons are non-measurements for the
# decision engine (observation status ``unavailable``, excluded from the
# tracker's observed set): DEGRADED_BASELINE_REASON (static, per-camera) and
# DEGRADED_AMBIENT_REASON (dynamic, per-frame). Every other reason below is a
# measured negative or diagnostic (status OK). Plain negatives -- no
# obstruction, or obstruction below reportable size -- keep ``reason`` None
# exactly as the legacy model did. ``no_meaningful_blocks`` is unreachable
# through ``evaluate()`` (the baseline-quality guard above catches every
# sparse baseline first) but maps defensively to the same non-measurement
# reason.
_TAMPERING_REASON_BY_REJECT: dict[str | None, str | None] = {
    None: None,                              # candidate
    "no_meaningful_blocks": DEGRADED_BASELINE_REASON,
    "degraded_ambient": DEGRADED_AMBIENT_REASON,
    "no_obstruction_blocks": None,           # clean negative (no obstruction)
    "min_size": None,                        # below reportable size (plain negative)
    "depth": "tampering_shallow_depth",
    "compactness": "tampering_scattered",
}


@dataclass(frozen=True)
class TamperingResult:
    largest_contiguous_loss_fraction: float  # raw score, 0-1
    confidence: float                         # normalized 0-1 score
    is_candidate: bool                         # single-frame flag; temporal confirmation happens upstream
    reason: str | None = None                  # optional diagnostic ("degraded_baseline", ...)
    meaningful_block_fraction: float = 0.0     # baseline grid coverage of usable structure, 0-1
    total_loss_fraction: float = 0.0           # obstruction-block coverage of meaningful blocks, 0-1
                                               # (relative-outlier blocks from the ambient-retention model;
                                               #  consumed by the decision engine's tampering suppression
                                               #  predicates as the obstruction area)


def compute_edge_map(frame: np.ndarray) -> np.ndarray:
    """Binary edge map of a frame via Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, TAMPERING_CANNY_LOW_THRESHOLD, TAMPERING_CANNY_HIGH_THRESHOLD)


def _compute_block_density(edge_map: np.ndarray, block_size: int) -> np.ndarray:
    """Fraction of edge pixels in each block of a grid over the edge map."""
    height, width = edge_map.shape
    rows, cols = height // block_size, width // block_size
    cropped = edge_map[: rows * block_size, : cols * block_size]
    blocks = cropped.reshape(rows, block_size, cols, block_size)
    return (blocks > 0).mean(axis=(1, 3))


def meaningful_block_fraction(baseline_edges: np.ndarray) -> float:
    """Fraction of grid blocks with enough baseline edge structure to count
    as meaningful for obstruction scoring.

    A baseline with almost no meaningful structure cannot support
    structure-loss detection: with a handful of meaningful blocks, any small
    ambient/background shift can look like a compact "lost" cluster. Callers
    should treat fractions below ``TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION``
    as a degraded baseline and refuse to emit tampering candidates; see
    ``baseline_degradation_info`` for the single shared way to make that
    decision.
    """
    baseline_density = _compute_block_density(baseline_edges, TAMPERING_GRID_BLOCK_SIZE)
    total = baseline_density.size
    if total == 0:
        return 0.0
    meaningful = np.count_nonzero(
        baseline_density >= TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY
    )
    return meaningful / total


def baseline_degradation_info(baseline_edges: np.ndarray) -> tuple[float, bool, str | None]:
    """Single authoritative definition of a "degraded" tampering baseline.

    Returns ``(meaningful_block_fraction, is_degraded, warning_text)``.
    ``is_degraded`` is True when the edge map cannot support structure-loss
    detection (``meaningful_block_fraction <
    TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION``); ``warning_text`` is the
    capture-time quality-warning message when degraded and None otherwise.

    Shared by all three places that need this decision — capture-time
    gating (pipeline.capture_baseline), startup-time enforcement
    (main.py), and this module's own runtime check in ``evaluate()`` — so
    the definition and the wording can never drift apart.
    """
    fraction = meaningful_block_fraction(baseline_edges)
    if fraction >= TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION:
        return fraction, False, None
    warning_text = (
        f"tampering meaningful-block fraction {fraction:.3f} is below "
        f"TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION "
        f"({TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION}); structure-loss detection is unreliable"
    )
    return fraction, True, warning_text


def confidence_from_largest_fraction(largest_fraction: float) -> float:
    """Normalize the largest lost-cluster fraction onto the detector's
    confidence scale.

    ``largest_fraction`` spans the candidate threshold
    (``TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION``, the smallest reportable
    obstruction) up to 1.0 (the whole baseline blocked). Mapping that range
    to 0..1 keeps the scale comparable with the other detectors and makes
    the confidence floor meaningful: 0.50 on this scale corresponds to
    ~57.5% of the baseline's structure blocked in one contiguous cluster.
    """
    span = 1.0 - TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION
    if span <= 0.0:
        return float(np.clip(largest_fraction, 0.0, 1.0))
    normalized = (
        largest_fraction - TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION
    ) / span
    return float(np.clip(normalized, 0.0, 1.0))


def compute_loss_fractions(
    baseline_edges: np.ndarray, current_edges: np.ndarray
) -> tuple[float, float]:
    """Return (largest_contiguous_loss_fraction, total_loss_fraction).

    ``largest_contiguous_loss_fraction`` is the single largest connected
    cluster of "structure lost" blocks as a fraction of the baseline's
    meaningful-structure blocks. ``total_loss_fraction`` is the fraction of
    ALL meaningful baseline blocks that lost their structure.

    A genuine obstruction is localized: the largest cluster covers part of
    the lens while the rest of the scene keeps its structure, so the total
    stays well below 1. Global degradation -- blur, low-light, rotation
    resampling -- removes edges nearly everywhere, pushing the total toward
    1 even though the largest connected cluster is also large.

    The two edge maps must share the same shape: block grids are derived
    from the actual dimensions, so mismatched resolutions would silently
    miscompare (or broadcast-fail) instead of failing loudly.
    """
    if baseline_edges.shape != current_edges.shape:
        raise ValueError(
            f"Baseline edge map shape {baseline_edges.shape} does not match "
            f"current edge map shape {current_edges.shape}. Re-capture the "
            "baseline at the camera's current resolution."
        )

    baseline_density = _compute_block_density(baseline_edges, TAMPERING_GRID_BLOCK_SIZE)
    current_density = _compute_block_density(current_edges, TAMPERING_GRID_BLOCK_SIZE)

    meaningful_blocks = baseline_density >= TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY
    meaningful_count = np.count_nonzero(meaningful_blocks)
    if meaningful_count == 0:
        return 0.0, 0.0  # baseline has no real structure to lose; can't signal obstruction this way

    disappeared = meaningful_blocks & (current_density <= baseline_density * TAMPERING_BLOCK_DENSITY_DROP_RATIO)
    total_loss_fraction = np.count_nonzero(disappeared) / meaningful_count

    disappeared_mask = disappeared.astype(np.uint8)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(disappeared_mask, connectivity=4)

    if num_labels <= 1:
        return 0.0, total_loss_fraction  # label 0 is background; no disappeared blocks at all

    # stats[0] is the background component; skip it when finding the largest real cluster.
    largest_component_size = stats[1:, cv2.CC_STAT_AREA].max()
    return largest_component_size / meaningful_count, total_loss_fraction


def compute_largest_contiguous_loss_fraction(baseline_edges: np.ndarray, current_edges: np.ndarray) -> float:
    """Largest single connected cluster of 'structure lost' blocks, as a
    fraction of the baseline's total meaningful-structure blocks.

    Convenience wrapper returning only the cluster metric; see
    ``compute_loss_fractions`` for the full (cluster, total) pair.
    """
    largest, _total = compute_loss_fractions(baseline_edges, current_edges)
    return largest


# --- Approach A: relative ambient-retention obstruction model ----------------
# Pure scoring functions for the approved tampering-candidacy redesign
# (subtask 6 of 10), consumed by ``evaluate()`` since subtask 7. Each is a
# standalone pure function (unit-tested with synthetic per-block density
# arrays), and ``assess_obstruction_candidacy`` runs the whole pipeline
# end-to-end on per-block density grids.
#
# Model summary: every meaningful baseline block gets a CONTINUOUS retention
# ratio (current edge density / baseline edge density). The frame's ambient
# retention level is estimated from the UPPER TAIL of those ratios (a high
# quantile), which stays at the true ambient level even when an obstruction
# covers most of the frame. A block is an obstruction block when its
# retention is far below the ambient level (<= depth_ratio * ambient) -- a
# comparison to the frame's own current state, never a fixed absolute
# threshold. Candidacy then requires the largest connected obstruction
# cluster to be large enough, compact enough, and genuinely deep (a
# statistical outlier relative to the ambient level).
#
# The ONE absolute constant is TAMPERING_AMBIENT_MIN_RETENTION: a degeneracy
# floor below which the whole frame has collapsed to near-zero structure and
# the relative comparison is no longer meaningful (that fully-ambiguous case
# is delegated to Approach C in a later subtask).


@dataclass(frozen=True)
class ObstructionAssessment:
    """Full output of the Approach A obstruction-candidacy decision on a pair
    of per-block edge-density grids (no image code involved).

    ``is_candidate`` is True only when every gate passes; ``reject_reason``
    names the first gate that failed (or None). ``largest_cluster_fraction``
    and ``compactness`` are fractions of the meaningful block count / of all
    obstruction blocks respectively, matching the legacy detector's metrics.
    """

    meaningful_block_fraction: float
    ambient_retention: float
    obstruction_block_fraction: float
    largest_cluster_fraction: float
    compactness: float
    cluster_median_retention: float
    relative_depth: float
    is_candidate: bool
    reject_reason: str | None


def compute_retention_ratios(
    baseline_density: np.ndarray,
    current_density: np.ndarray,
    min_baseline_density: float = TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous retention ratio (current/baseline edge density) per block.

    Returns ``(retention, meaningful_mask)``. ``retention`` has the shape of
    the input density grids and is NaN for non-meaningful blocks (baseline
    density below ``min_baseline_density``): those blocks have no baseline
    structure to retain and must never enter ambient estimation, obstruction
    classification, or clustering. The ratio is kept CONTINUOUS (never
    thresholded to a binary lost/kept flag), because the ambient-retention
    model needs the actual magnitude of the loss, not just "lost or not".
    """
    if baseline_density.shape != current_density.shape:
        raise ValueError(
            f"Baseline density grid shape {baseline_density.shape} does not match "
            f"current density grid shape {current_density.shape}."
        )
    meaningful_mask = baseline_density >= min_baseline_density
    retention = np.full(baseline_density.shape, np.nan, dtype=float)
    # baseline density is >= min_baseline_density > 0 on meaningful blocks
    # (by default; errstate guards a misconfigured 0 floor), so the ratio is
    # finite and in [0, 50] for meaningful blocks.
    with np.errstate(divide="ignore", invalid="ignore"):
        retention[meaningful_mask] = (
            current_density[meaningful_mask] / baseline_density[meaningful_mask]
        )
    return retention, meaningful_mask


def estimate_ambient_retention(
    retention: np.ndarray,
    meaningful_mask: np.ndarray,
    quantile: float = TAMPERING_AMBIENT_RETENTION_QUANTILE,
) -> float:
    """Frame's ambient retention level: the upper-tail quantile of the
    per-block retention ratios over meaningful blocks.

    The upper tail (not the mean/median) is deliberate: obstruction blocks
    sit in the lower tail, so the estimate stays at the true ambient level
    even when an obstruction covers most of the frame. Valid while
    obstruction coverage stays below ``(1 - quantile)`` of the meaningful
    blocks (~90% at the default). Returns 0.0 when there are no meaningful
    blocks -- callers must treat that as unusable.
    """
    values = retention[meaningful_mask]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, quantile))


def classify_obstruction_blocks(
    retention: np.ndarray,
    meaningful_mask: np.ndarray,
    ambient: float,
    depth_ratio: float = TAMPERING_OBSTRUCTION_DEPTH_RATIO,
) -> np.ndarray:
    """Boolean mask of obstruction blocks: meaningful blocks whose retention
    is far below the frame's own ambient level (``retention <= depth_ratio *
    ambient``). This is a RELATIVE comparison to the frame's current state,
    never a fixed absolute threshold.

    With ``ambient <= 0`` the comparison is undefined and nothing is flagged
    (degenerate frame; callers guard this via ``TAMPERING_AMBIENT_MIN_RETENTION``).
    """
    mask = np.zeros(retention.shape, dtype=bool)
    if ambient <= 0.0:
        return mask
    threshold = depth_ratio * ambient
    mask[meaningful_mask] = retention[meaningful_mask] <= threshold
    return mask


def find_largest_obstruction_cluster(
    obstruction_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Largest 4-connected cluster of obstruction blocks.

    Returns ``(largest_cluster_size, largest_cluster_mask)``. Non-meaningful
    blocks are never True in ``obstruction_mask``, so they break connectivity
    exactly as in the legacy ``compute_loss_fractions``. Returns
    ``(0, all-False mask)`` when no obstruction block exists.
    """
    empty_mask = np.zeros(obstruction_mask.shape, dtype=bool)
    if not obstruction_mask.any():
        return 0, empty_mask
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        obstruction_mask.astype(np.uint8), connectivity=4
    )
    if num_labels <= 1:
        return 0, empty_mask
    # stats[0] is the background component; skip it when finding the largest real cluster.
    largest_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_size = int(stats[largest_index + 1, cv2.CC_STAT_AREA])
    cluster_mask = labels == (largest_index + 1)
    return largest_size, cluster_mask


def obstruction_depth_info(
    cluster_mask: np.ndarray,
    retention: np.ndarray,
    ambient: float,
    depth_ratio: float = TAMPERING_OBSTRUCTION_DEPTH_RATIO,
) -> tuple[float, float, bool]:
    """Outlier-depth information for an obstruction cluster.

    Returns ``(cluster_median_retention, relative_depth, is_outlier)`` where
    ``relative_depth = cluster_median_retention / ambient``. The cluster is a
    genuine statistical outlier when its median retention is at most
    ``depth_ratio`` times the frame's ambient level -- i.e. the cluster as a
    whole sits far below the frame's own current state, not just a few noisy
    blocks on the classification boundary. ``is_outlier`` is always False
    when ``ambient <= 0`` (undefined comparison).

    Note: under the current single-ratio parameterization this gate is
    implied by per-block classification (every cluster block already
    satisfies the same inequality). It is kept as an explicit, independently
    testable gate per the approved plan, and it is the knob a future
    calibration can tighten independently of the (looser) per-block
    classification ratio -- classification for recall, depth for precision.
    """
    cluster_values = retention[cluster_mask]
    if cluster_values.size == 0:
        return 0.0, 0.0, False
    cluster_median = float(np.median(cluster_values))
    if ambient <= 0.0:
        return cluster_median, 0.0, False
    relative_depth = cluster_median / ambient
    return cluster_median, relative_depth, relative_depth <= depth_ratio


def assess_obstruction_candidacy(
    baseline_density: np.ndarray,
    current_density: np.ndarray,
    min_baseline_density: float = TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY,
    ambient_quantile: float = TAMPERING_AMBIENT_RETENTION_QUANTILE,
    depth_ratio: float = TAMPERING_OBSTRUCTION_DEPTH_RATIO,
    min_ambient: float = TAMPERING_AMBIENT_MIN_RETENTION,
    min_cluster_fraction: float = TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION,
    min_compactness: float = TAMPERING_MIN_COMPACTNESS_RATIO,
) -> ObstructionAssessment:
    """Full Approach A obstruction-candidacy decision on per-block
    edge-density grids. Pure function: no images, no evaluate() involvement
    (the wiring subtask will feed it real edge maps).

    Gate order:
      1. no meaningful baseline structure  -> reject ("no_meaningful_blocks")
      2. ambient below the degeneracy floor -> reject ("degraded_ambient");
         the whole frame has collapsed, the case is fully ambiguous and
         belongs to Approach C
      3. no obstruction blocks             -> reject ("no_obstruction_blocks");
         global degradation sits AT the ambient level, so it flags nothing
      4. largest cluster below min size    -> reject ("min_size")
      5. cluster not a genuine depth outlier -> reject ("depth")
      6. cluster not compact enough        -> reject ("compactness")
    """
    retention, meaningful_mask = compute_retention_ratios(
        baseline_density, current_density, min_baseline_density
    )
    meaningful_count = int(np.count_nonzero(meaningful_mask))
    meaningful_fraction = (
        meaningful_count / meaningful_mask.size if meaningful_mask.size else 0.0
    )

    ambient = 0.0
    obstruction_fraction = 0.0
    largest_fraction = 0.0
    compactness = 0.0
    cluster_median = 0.0
    relative_depth = 0.0

    def _assessment(reason: str | None, **overrides: float) -> ObstructionAssessment:
        return ObstructionAssessment(
            meaningful_block_fraction=meaningful_fraction,
            ambient_retention=ambient,
            obstruction_block_fraction=obstruction_fraction,
            largest_cluster_fraction=largest_fraction,
            compactness=compactness,
            cluster_median_retention=cluster_median,
            relative_depth=relative_depth,
            is_candidate=reason is None,
            reject_reason=reason,
            **overrides,
        )

    if meaningful_count == 0:
        return _assessment("no_meaningful_blocks")

    ambient = estimate_ambient_retention(retention, meaningful_mask, ambient_quantile)
    if ambient < min_ambient:
        return _assessment("degraded_ambient")

    obstruction_mask = classify_obstruction_blocks(
        retention, meaningful_mask, ambient, depth_ratio
    )
    obstruction_count = int(np.count_nonzero(obstruction_mask))
    obstruction_fraction = obstruction_count / meaningful_count
    if obstruction_count == 0:
        return _assessment("no_obstruction_blocks")

    largest_size, cluster_mask = find_largest_obstruction_cluster(obstruction_mask)
    largest_fraction = largest_size / meaningful_count
    if largest_fraction < min_cluster_fraction:
        return _assessment("min_size")

    cluster_median, relative_depth, is_outlier = obstruction_depth_info(
        cluster_mask, retention, ambient, depth_ratio
    )
    if not is_outlier:
        return _assessment("depth")

    compactness = largest_size / obstruction_count
    if compactness < min_compactness:
        return _assessment("compactness")

    return _assessment(None)


def evaluate(frame: np.ndarray, baseline_edges: np.ndarray) -> TamperingResult:
    """Score a frame's obstruction level relative to a camera's baseline edge map.

    Runs the approved Approach A ambient-retention model
    (``assess_obstruction_candidacy``): per-block retention ratios compared
    against the frame's own upper-tail ambient level, so a real obstruction
    stays a statistical outlier even when co-occurring blur collapses the
    ambient level -- the exact case the legacy absolute total-loss ceiling
    wrongly rejected. ``confidence`` is the RAW obstruction-magnitude score
    and is always preserved, even when ``is_candidate`` is False (the
    decision engine keeps it as ``raw_confidence`` and zeroes the reportable
    confidence downstream).
    """
    # Baseline-quality guard: a baseline captured in dark/blurry conditions
    # contains meaningful structure in only a small fraction of its blocks,
    # so its structure-loss metric is dominated by ambient/background noise
    # rather than real obstruction. Refuse to emit candidates (never a false
    # tampering event) until a usable baseline is captured. The decision
    # engine surfaces this as the detector's "reason".
    baseline_meaningful_fraction, baseline_is_degraded, _ = baseline_degradation_info(
        baseline_edges
    )
    if baseline_is_degraded:
        return TamperingResult(
            largest_contiguous_loss_fraction=0.0,
            confidence=0.0,
            is_candidate=False,
            reason=DEGRADED_BASELINE_REASON,
            meaningful_block_fraction=baseline_meaningful_fraction,
        )

    current_edges = compute_edge_map(frame)
    assessment = assess_obstruction_candidacy(
        _compute_block_density(baseline_edges, TAMPERING_GRID_BLOCK_SIZE),
        _compute_block_density(current_edges, TAMPERING_GRID_BLOCK_SIZE),
    )

    # Obstruction magnitude -> confidence: the largest connected obstruction
    # cluster as a fraction of meaningful blocks, mapped through the same
    # [TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION, 1.0] -> [0, 1] scale as the
    # legacy model, so the scale, the emission floors, and the cam_08
    # evidence (~0.8 for an ~83% obstruction) are preserved. Depth and
    # compactness are candidacy gates, not magnitude signals, so they must
    # not double-count into the magnitude score.
    confidence = confidence_from_largest_fraction(assessment.largest_cluster_fraction)

    return TamperingResult(
        largest_contiguous_loss_fraction=assessment.largest_cluster_fraction,
        confidence=confidence,
        is_candidate=assessment.is_candidate,
        reason=_TAMPERING_REASON_BY_REJECT.get(assessment.reject_reason),
        meaningful_block_fraction=assessment.meaningful_block_fraction,
        total_loss_fraction=assessment.obstruction_block_fraction,
    )