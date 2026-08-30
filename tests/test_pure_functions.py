"""Fast unit tests for pure helper functions used by the detectors.

These functions are side-effect free and need no camera footage or
baseline data, so edge cases can be tested exhaustively and quickly.
Heavy end-to-end behavior is covered separately by the ground-truth
tests (tests/test_*_detector.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from detectors.tampering import (
    _compute_block_density,
    assess_obstruction_candidacy,
    baseline_degradation_info,
    classify_obstruction_blocks,
    compute_largest_contiguous_loss_fraction,
    compute_loss_fractions,
    compute_retention_ratios,
    confidence_from_largest_fraction,
    estimate_ambient_retention,
    find_largest_obstruction_cluster,
    meaningful_block_fraction,
    obstruction_depth_info,
)
from detectors.tilt import (
    _mad_inlier_mask,
    inlier_ratio_sufficient,
    match_volume_sufficient,
)
from config import (
    LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR,
    TAMPERING_AMBIENT_MIN_RETENTION,
    TAMPERING_AMBIENT_RETENTION_QUANTILE,
    TAMPERING_BLUR_AREA_SLACK,
    TAMPERING_LOW_LIGHT_AREA_SLACK,
    TAMPERING_MIN_COMPACTNESS_RATIO,
    TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION,
    TAMPERING_OBSTRUCTION_DEPTH_RATIO,
)
from pipeline.decision_engine import (
    _area_conserved,
    _is_near_black,
    _pair_should_suppress,
)
from pipeline.paths import validate_camera_id


# --- detectors.tilt._mad_inlier_mask --------------------------------------


def test_mad_inlier_mask_identical_displacements_all_inliers() -> None:
    displacements = np.array([3.0, 3.0, 3.0, 3.0])
    mask = _mad_inlier_mask(displacements)
    assert mask.dtype == bool
    assert mask.all()


def test_mad_inlier_mask_rejects_far_outlier() -> None:
    displacements = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    mask = _mad_inlier_mask(displacements)
    assert not mask[-1]
    assert mask[:-1].all()


def test_mad_inlier_mask_keeps_tight_cluster() -> None:
    displacements = np.array([1.0, 1.1, 1.2, 0.9, 1.15])
    assert _mad_inlier_mask(displacements).all()


# --- detectors.tilt.match_volume_sufficient ---------------------------------


def test_match_volume_sufficient_requires_absolute_floor() -> None:
    assert not match_volume_sufficient(5, 500, 500)  # below TILT_MIN_RELIABLE_MATCHES


def test_match_volume_sufficient_requires_relative_ratio() -> None:
    # 20 matches from 2000/2000 keypoints is 1% -- far below the guard: a
    # handful of spurious correspondences must not be trusted.
    assert not match_volume_sufficient(20, 2000, 2000)


def test_match_volume_sufficient_passes_on_healthy_volume() -> None:
    assert match_volume_sufficient(500, 2000, 1800)


def test_match_volume_sufficient_rejects_degenerate_frames() -> None:
    # A frame with zero keypoints has nothing to match against.
    assert not match_volume_sufficient(0, 2000, 0)
    assert not match_volume_sufficient(10, 2000, 0)


# --- detectors.tilt.inlier_ratio_sufficient ---------------------------------


def test_inlier_ratio_sufficient_passes_healthy_fraction() -> None:
    assert inlier_ratio_sufficient(480, 500)


def test_inlier_ratio_sufficient_rejects_scattered_matches() -> None:
    # Blur/obscuration garbage is scattered: MAD rejects most matches.
    assert not inlier_ratio_sufficient(5, 50)


def test_inlier_ratio_sufficient_rejects_empty_matches() -> None:
    assert not inlier_ratio_sufficient(0, 0)


# --- detectors.tampering.meaningful_block_fraction --------------------------


def test_meaningful_block_fraction_full_structure_is_one() -> None:
    edges = np.full((64, 64), 255, dtype=np.uint8)
    assert meaningful_block_fraction(edges) == pytest.approx(1.0)


def test_meaningful_block_fraction_empty_structure_is_zero() -> None:
    edges = np.zeros((64, 64), dtype=np.uint8)
    assert meaningful_block_fraction(edges) == pytest.approx(0.0)


def test_meaningful_block_fraction_sparse_structure() -> None:
    # One of four 32x32 blocks full of edges -> 0.25, below the 0.5 guard:
    # a sparse-structure baseline must be treated as degraded.
    edges = np.zeros((64, 64), dtype=np.uint8)
    edges[0:32, 0:32] = 255
    assert meaningful_block_fraction(edges) == pytest.approx(0.25)


# --- detectors.tampering.baseline_degradation_info ---------------------------
# Single shared definition of a "degraded" baseline; must agree with
# meaningful_block_fraction and pin the capture-time warning wording.


def test_baseline_degradation_info_empty_edges_is_degraded() -> None:
    edges = np.zeros((64, 64), dtype=np.uint8)
    fraction, is_degraded, warning_text = baseline_degradation_info(edges)
    assert fraction == pytest.approx(0.0)
    assert is_degraded is True
    assert warning_text == (
        "tampering meaningful-block fraction 0.000 is below "
        "TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION (0.5); "
        "structure-loss detection is unreliable"
    )


def test_baseline_degradation_info_full_edges_is_clean() -> None:
    edges = np.full((64, 64), 255, dtype=np.uint8)
    fraction, is_degraded, warning_text = baseline_degradation_info(edges)
    assert fraction == pytest.approx(1.0)
    assert is_degraded is False
    assert warning_text is None


def test_baseline_degradation_info_sparse_structure_is_degraded() -> None:
    edges = np.zeros((64, 64), dtype=np.uint8)
    edges[0:32, 0:32] = 255  # 0.25 meaningful, below the 0.5 gate
    fraction, is_degraded, _warning = baseline_degradation_info(edges)
    assert fraction == pytest.approx(0.25)
    assert is_degraded is True


# --- detectors.tampering.confidence_from_largest_fraction -------------------


def test_tampering_confidence_maps_threshold_to_zero() -> None:
    # At the smallest reportable obstruction the normalized confidence is 0.
    assert confidence_from_largest_fraction(0.15) == pytest.approx(0.0)


def test_tampering_confidence_maps_half_blocked_to_half() -> None:
    # 0.575 of the baseline structure lost in one cluster -> confidence 0.5,
    # which is the emission floor for tampering.
    assert confidence_from_largest_fraction(0.575) == pytest.approx(0.5, abs=1e-9)


def test_tampering_confidence_full_block_is_one() -> None:
    assert confidence_from_largest_fraction(1.0) == pytest.approx(1.0)


def test_tampering_confidence_clamps_below_threshold() -> None:
    assert confidence_from_largest_fraction(0.0) == pytest.approx(0.0)
    assert confidence_from_largest_fraction(0.1) == pytest.approx(0.0)


# --- detectors.tampering._compute_block_density ----------------------------


def test_block_density_all_zeros() -> None:
    density = _compute_block_density(np.zeros((64, 64), dtype=np.uint8), 32)
    assert density.shape == (2, 2)
    assert not density.any()


def test_block_density_full_block_is_one() -> None:
    edge_map = np.zeros((64, 64), dtype=np.uint8)
    edge_map[0:32, 0:32] = 1
    density = _compute_block_density(edge_map, 32)
    assert density[0, 0] == pytest.approx(1.0)
    assert density[0, 1] == pytest.approx(0.0)
    assert density[1, 0] == pytest.approx(0.0)
    assert density[1, 1] == pytest.approx(0.0)


def test_block_density_half_filled_block() -> None:
    edge_map = np.zeros((64, 64), dtype=np.uint8)
    edge_map[0:16, 0:32] = 1  # top-left half of the top-left block
    density = _compute_block_density(edge_map, 32)
    assert density[0, 0] == pytest.approx(0.5)


def test_block_density_crops_non_divisible_dimensions() -> None:
    edge_map = np.zeros((70, 70), dtype=np.uint8)  # 70 % 32 != 0
    density = _compute_block_density(edge_map, 32)
    assert density.shape == (2, 2)  # crops to 64x64, discards leftover 6px


# --- detectors.tampering.compute_largest_contiguous_loss_fraction ---------


def test_loss_fraction_identical_maps_is_zero() -> None:
    edges = np.full((64, 64), 255, dtype=np.uint8)
    assert compute_largest_contiguous_loss_fraction(edges, edges) == 0.0


def test_loss_fraction_no_baseline_structure_is_zero() -> None:
    baseline = np.zeros((64, 64), dtype=np.uint8)
    current = np.full((64, 64), 255, dtype=np.uint8)
    assert compute_largest_contiguous_loss_fraction(baseline, current) == 0.0


def test_loss_fraction_partial_block_disappearance() -> None:
    baseline = np.zeros((64, 64), dtype=np.uint8)
    baseline[0:32, 0:64] = 255  # two meaningful blocks (top row)
    current = np.zeros((64, 64), dtype=np.uint8)
    current[0:32, 32:64] = 255  # only the top-right block remains
    assert compute_largest_contiguous_loss_fraction(baseline, current) == pytest.approx(0.5)


def test_loss_fraction_full_disappearance_is_one() -> None:
    baseline = np.full((64, 64), 255, dtype=np.uint8)
    current = np.zeros((64, 64), dtype=np.uint8)
    assert compute_largest_contiguous_loss_fraction(baseline, current) == pytest.approx(1.0)


def test_loss_fractions_partial_disappearance_total_matches() -> None:
    baseline = np.zeros((64, 64), dtype=np.uint8)
    baseline[0:32, 0:64] = 255  # two meaningful blocks (top row)
    current = np.zeros((64, 64), dtype=np.uint8)
    current[0:32, 32:64] = 255  # only the top-right block remains
    largest, total = compute_loss_fractions(baseline, current)
    assert largest == pytest.approx(0.5)
    assert total == pytest.approx(0.5)


def test_loss_fractions_full_disappearance_total_is_one() -> None:
    baseline = np.full((64, 64), 255, dtype=np.uint8)
    current = np.zeros((64, 64), dtype=np.uint8)
    largest, total = compute_loss_fractions(baseline, current)
    assert largest == pytest.approx(1.0)
    assert total == pytest.approx(1.0)


def test_loss_fractions_no_baseline_structure_is_zero() -> None:
    baseline = np.zeros((64, 64), dtype=np.uint8)
    current = np.full((64, 64), 255, dtype=np.uint8)
    assert compute_loss_fractions(baseline, current) == (0.0, 0.0)


def test_loss_fraction_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_largest_contiguous_loss_fraction(
            np.zeros((32, 32), dtype=np.uint8),
            np.zeros((64, 64), dtype=np.uint8),
        )


# --- pipeline.paths.validate_camera_id --------------------------------------


def test_validate_camera_id_accepts_safe_ids() -> None:
    for camera_id in ("cam1", "CAM-1", "camera_2", "A"):
        assert validate_camera_id(camera_id) == camera_id


def test_validate_camera_id_rejects_path_traversal() -> None:
    for camera_id in ("..", "../etc", "../../etc/passwd", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            validate_camera_id(camera_id)


def test_validate_camera_id_rejects_invalid_characters() -> None:
    for camera_id in ("", "a b", "a.b", "café", "a#b"):
        with pytest.raises(ValueError):
            validate_camera_id(camera_id)

# --- pipeline.decision_engine co-occurrence predicates ------------------------
# Multi-fault co-occurrence (Phase N): standalone predicates from Subtask 3.
# Not wired into fusion yet; tested in isolation with synthetic inputs.


def test_area_conserved_true_when_increase_fits_within_slack() -> None:
    # 0.5 dark-increase is explainable by a 0.4 obstruction + 0.2 slack.
    assert _area_conserved(obstruction_coverage=0.4, measured_increase=0.5, slack=0.2)


def test_area_conserved_false_when_increase_exceeds_slack() -> None:
    # 0.8 dark-increase cannot be caused by a 0.4 obstruction (+0.2 slack).
    assert not _area_conserved(obstruction_coverage=0.4, measured_increase=0.8, slack=0.2)


def test_area_conserved_boundary_is_conserved() -> None:
    # Exact equality (increase == coverage + slack) still counts as conserved.
    assert _area_conserved(obstruction_coverage=0.4, measured_increase=0.6, slack=0.2)


def test_area_conserved_zero_coverage_requires_zero_increase() -> None:
    assert _area_conserved(obstruction_coverage=0.0, measured_increase=0.0, slack=0.0)
    assert not _area_conserved(obstruction_coverage=0.0, measured_increase=0.01, slack=0.0)


def test_is_near_black_at_or_above_floor() -> None:
    assert _is_near_black(0.95, floor=0.8)
    assert _is_near_black(0.8, floor=0.8)  # boundary equality


def test_is_near_black_below_floor() -> None:
    assert not _is_near_black(0.5, floor=0.8)


# --- _pair_should_suppress relation-class routing ------------------------------


def test_pair_should_suppress_always_uses_margin_only() -> None:
    # tilt->blur is "always": margin decides and metrics are irrelevant.
    assert _pair_should_suppress(
        "tilt", "blur", {}, {}, suppressor_confidence=0.9, suppressed_confidence=1.0
    )
    assert not _pair_should_suppress(
        "tilt", "blur", {}, {}, suppressor_confidence=0.5, suppressed_confidence=1.0
    )


def test_pair_should_suppress_conditional_tampering_low_light() -> None:
    # Margin met AND area conserved (0.5 <= 0.4 + slack): suppress.
    assert _pair_should_suppress(
        "tampering", "low_light",
        {"total_loss_fraction": 0.4}, {"relative_increase": 0.5},
        suppressor_confidence=0.8, suppressed_confidence=0.6,
    )
    # Margin met but area NOT conserved (0.8 > 0.4 + slack): do not suppress.
    assert not _pair_should_suppress(
        "tampering", "low_light",
        {"total_loss_fraction": 0.4}, {"relative_increase": 0.8},
        suppressor_confidence=0.8, suppressed_confidence=0.6,
    )


def test_pair_should_suppress_conditional_tampering_low_light_missing_metrics() -> None:
    # Older observations carry no metrics: the predicate is unverifiable.
    assert not _pair_should_suppress(
        "tampering", "low_light", {}, {},
        suppressor_confidence=0.8, suppressed_confidence=0.6,
    )


def test_pair_should_suppress_conditional_tampering_blur() -> None:
    # Blur's observed increase is the sharpness fraction lost (1 - ratio):
    # 0.4 <= 0.5 + slack -> conserved -> suppress.
    assert _pair_should_suppress(
        "tampering", "blur",
        {"total_loss_fraction": 0.5}, {"sharpness_ratio": 0.6},
        suppressor_confidence=0.8, suppressed_confidence=0.6,
    )
    # Severe global blur (0.8 lost) exceeds what a 0.5 obstruction explains.
    assert not _pair_should_suppress(
        "tampering", "blur",
        {"total_loss_fraction": 0.5}, {"sharpness_ratio": 0.2},
        suppressor_confidence=0.8, suppressed_confidence=0.6,
    )


def test_pair_should_suppress_conditional_tampering_blur_missing_metrics() -> None:
    assert not _pair_should_suppress(
        "tampering", "blur", {}, {},
        suppressor_confidence=0.8, suppressed_confidence=0.6,
    )


def test_pair_should_suppress_conditional_low_light_blur_near_black() -> None:
    floor = LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR
    # Margin met AND low-light at the near-black floor: suppress.
    assert _pair_should_suppress(
        "low_light", "blur", {}, {},
        suppressor_confidence=floor, suppressed_confidence=0.5,
    )
    # Margin met but the scene is not near-black: do not suppress.
    assert not _pair_should_suppress(
        "low_light", "blur", {}, {},
        suppressor_confidence=floor - 0.05, suppressed_confidence=0.5,
    )


def test_pair_should_suppress_conditional_margin_failure_short_circuits() -> None:
    # Even with metrics that would conserve area, a failed margin check
    # prevents suppression.
    assert not _pair_should_suppress(
        "tampering", "low_light",
        {"total_loss_fraction": 0.9}, {"relative_increase": 0.1},
        suppressor_confidence=0.4, suppressed_confidence=1.0,
    )


def test_pair_should_suppress_absent_pairs_never_suppress() -> None:
    # tampering->tilt and low_light->tilt are intentionally absent from
    # DECISION_SUPPRESSION_RULES (no physical pathway): never suppress.
    assert not _pair_should_suppress(
        "tampering", "tilt", {"total_loss_fraction": 0.9}, {"median_shift_ratio": 0.01},
        suppressor_confidence=1.0, suppressed_confidence=0.5,
    )
    assert not _pair_should_suppress(
        "low_light", "tilt", {"dark_pixel_ratio": 0.99}, {"median_shift_ratio": 0.01},
        suppressor_confidence=1.0, suppressed_confidence=0.5,
    )


def test_pair_should_suppress_reversed_order_pair_never_suppresses() -> None:
    # A lower-precedence fault never suppresses a higher-precedence one.
    assert not _pair_should_suppress(
        "blur", "tampering", {}, {},
        suppressor_confidence=1.0, suppressed_confidence=0.1,
    )


def test_pair_should_suppress_uses_config_slack_constants() -> None:
    # Tampering->low_light wiring reads the configured slack value.
    coverage = 0.4
    assert _pair_should_suppress(
        "tampering", "low_light",
        {"total_loss_fraction": coverage},
        {"relative_increase": coverage + TAMPERING_LOW_LIGHT_AREA_SLACK},
        0.8, 0.6,
    )
    assert not _pair_should_suppress(
        "tampering", "low_light",
        {"total_loss_fraction": coverage},
        {"relative_increase": coverage + TAMPERING_LOW_LIGHT_AREA_SLACK + 1e-6},
        0.8, 0.6,
    )
    # Tampering->blur wiring reads its own slack constant.
    assert _pair_should_suppress(
        "tampering", "blur",
        {"total_loss_fraction": coverage},
        {"sharpness_ratio": 1.0 - coverage - TAMPERING_BLUR_AREA_SLACK},
        0.8, 0.6,
    )


# --- detectors.tampering Approach A: relative ambient-retention model ---------
# Pure scoring functions for the approved tampering-candidacy redesign
# (subtask 6 of 10). Tests use synthetic per-block density grids only --
# no images. evaluate() is intentionally unchanged this subtask; these
# functions are the standalone building blocks the wiring subtask consumes.


def _density_grid(rows: int, cols: int, value: float) -> np.ndarray:
    return np.full((rows, cols), value, dtype=float)


def _rectangle(rows: int, cols: int, row_slice: slice, col_slice: slice) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    mask[row_slice, col_slice] = True
    return mask


# --- compute_retention_ratios -------------------------------------------------


def test_retention_ratios_continuous_ratio_per_block() -> None:
    baseline = _density_grid(4, 4, 0.5)
    current = _density_grid(4, 4, 0.25)  # half the baseline density
    retention, meaningful = compute_retention_ratios(baseline, current)
    assert meaningful.all()
    assert retention[0, 0] == pytest.approx(0.5)
    # retained structure -> ratio 1.0, never thresholded to a binary flag
    current[2, 2] = 0.5
    retention, _ = compute_retention_ratios(baseline, current)
    assert retention[2, 2] == pytest.approx(1.0)
    # structure growth stays continuous (ratio > 1), never clipped
    current[3, 3] = 1.0
    retention, _ = compute_retention_ratios(baseline, current)
    assert retention[3, 3] == pytest.approx(2.0)


def test_retention_ratios_excludes_non_meaningful_blocks() -> None:
    baseline = _density_grid(4, 4, 0.5)
    baseline[0, 0] = 0.01  # below TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY
    retention, meaningful = compute_retention_ratios(baseline, _density_grid(4, 4, 0.0))
    assert not meaningful[0, 0]
    assert np.isnan(retention[0, 0])
    assert meaningful[1, 1]
    assert retention[1, 1] == pytest.approx(0.0)


def test_retention_ratios_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_retention_ratios(np.zeros((4, 4)), np.zeros((3, 3)))


# --- estimate_ambient_retention ----------------------------------------------


def test_ambient_retention_uses_upper_tail_not_mean() -> None:
    # 120 of 144 blocks obstructed (retention 0.05), 24 ambient (retention
    # 0.95). The mean is ~0.20 -- the wrong answer. The 90th-percentile upper
    # tail is 0.95, because the obstruction sits entirely in the lower tail.
    retention = _density_grid(12, 12, 0.95)
    retention[_rectangle(12, 12, slice(0, 10), slice(0, 12))] = 0.05
    meaningful = np.ones((12, 12), dtype=bool)
    ambient = estimate_ambient_retention(retention, meaningful)
    assert ambient == pytest.approx(0.95)
    assert ambient != pytest.approx(0.2)  # a mean/median would sit at the obstruction level


def test_ambient_retention_no_meaningful_blocks_is_zero() -> None:
    retention = np.full((4, 4), np.nan)
    meaningful = np.zeros((4, 4), dtype=bool)
    assert estimate_ambient_retention(retention, meaningful) == 0.0


def test_ambient_retention_collapses_with_uniform_global_degradation() -> None:
    # Pure blur/low-light: every block drops to the same retention level, so
    # the ambient estimate collapses with them -- there is no higher tail left.
    retention = _density_grid(12, 12, 0.1)
    meaningful = np.ones((12, 12), dtype=bool)
    assert estimate_ambient_retention(retention, meaningful) == pytest.approx(0.1)


# --- classify_obstruction_blocks ---------------------------------------------


def test_classify_obstruction_blocks_relative_threshold() -> None:
    retention = np.array([[0.05, 0.2], [0.4, 0.5]], dtype=float)
    meaningful = np.ones((2, 2), dtype=bool)
    # ambient 0.4, depth ratio 0.35 -> threshold 0.14: only 0.05 is an outlier.
    mask = classify_obstruction_blocks(retention, meaningful, ambient=0.4)
    assert mask[0, 0]
    assert not mask[0, 1]
    assert not mask[1, 0]
    assert not mask[1, 1]
    # Higher ambient -> proportionally higher threshold (relative, not absolute).
    mask_high = classify_obstruction_blocks(retention, meaningful, ambient=1.0)
    assert mask_high[0, 0] and mask_high[0, 1]  # 0.05, 0.2 <= 0.35
    assert not mask_high[1, 0] and not mask_high[1, 1]  # 0.4, 0.5 > 0.35


def test_classify_obstruction_blocks_zero_ambient_flags_nothing() -> None:
    # Degenerate: with ambient == 0 the relative comparison is undefined and
    # nothing may be flagged (0 <= 0 would otherwise flag every block).
    retention = np.zeros((4, 4), dtype=float)
    meaningful = np.ones((4, 4), dtype=bool)
    mask = classify_obstruction_blocks(retention, meaningful, ambient=0.0)
    assert not mask.any()


def test_classify_obstruction_blocks_respects_meaningful_mask() -> None:
    retention = _density_grid(4, 4, 0.01)
    meaningful = np.zeros((4, 4), dtype=bool)
    meaningful[0:2, 0:2] = True
    mask = classify_obstruction_blocks(retention, meaningful, ambient=0.1)
    assert mask[0:2, 0:2].all()
    assert not mask[2:, :].any()  # non-meaningful blocks are never obstruction


# --- find_largest_obstruction_cluster ----------------------------------------


def test_find_largest_cluster_returns_area_and_mask() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:3, 0:3] = True   # 9-block cluster
    mask[0:2, 5:8] = True   # 6-block cluster, disconnected
    size, cluster = find_largest_obstruction_cluster(mask)
    assert size == 9
    assert cluster.sum() == 9


def test_find_largest_cluster_empty_mask() -> None:
    size, cluster = find_largest_obstruction_cluster(np.zeros((8, 8), dtype=bool))
    assert size == 0
    assert not cluster.any()


def test_find_largest_cluster_single_block() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 2] = True
    size, cluster = find_largest_obstruction_cluster(mask)
    assert size == 1
    assert cluster[2, 2]


# --- obstruction_depth_info ---------------------------------------------------


def test_depth_info_deep_cluster_is_outlier() -> None:
    retention = _density_grid(8, 8, 1.0)
    cluster = _rectangle(8, 8, slice(0, 2), slice(0, 4))
    retention[cluster] = 0.05
    median, depth, is_outlier = obstruction_depth_info(cluster, retention, ambient=1.0)
    assert median == pytest.approx(0.05)
    assert depth == pytest.approx(0.05)
    assert is_outlier  # 0.05 <= TAMPERING_OBSTRUCTION_DEPTH_RATIO * 1.0


def test_depth_info_shallow_cluster_is_not_outlier() -> None:
    retention = _density_grid(8, 8, 1.0)
    cluster = _rectangle(8, 8, slice(0, 2), slice(0, 4))
    retention[cluster] = 0.5
    _median, depth, is_outlier = obstruction_depth_info(cluster, retention, ambient=1.0)
    assert depth == pytest.approx(0.5)
    assert not is_outlier  # 0.5 > depth ratio 0.35


def test_depth_info_zero_ambient_is_never_outlier() -> None:
    retention = _density_grid(8, 8, 0.0)
    cluster = _rectangle(8, 8, slice(0, 2), slice(0, 4))
    _median, depth, is_outlier = obstruction_depth_info(cluster, retention, ambient=0.0)
    assert is_outlier is False


# --- assess_obstruction_candidacy: full-scenario decisions --------------------


def test_scenario_pure_global_degradation_no_false_candidate() -> None:
    # Blur/low-light alone: EVERY block degrades to the same retention (0.25).
    # Ambient collapses with them, no block sits below depth_ratio * ambient,
    # so nothing is classified as obstruction -- must never produce a candidate.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.125)
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "no_obstruction_blocks"
    assert assessment.obstruction_block_fraction == 0.0


def test_scenario_global_degradation_with_scattered_noise_stays_rejected() -> None:
    # Global degradation plus a few scattered low-retention noise blocks: the
    # noise IS below the relative threshold, but stays isolated -- no large
    # connected cluster can form, so no false candidate.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.125)  # retention 0.25
    for r, c in ((1, 1), (5, 3), (10, 7)):
        current[r, c] = 0.01  # retention 0.02, well below 0.35 * 0.25
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "min_size"
    assert assessment.obstruction_block_fraction == pytest.approx(3 / 144)
    assert assessment.largest_cluster_fraction == pytest.approx(1 / 144)


def test_scenario_large_obstruction_no_other_degradation_is_candidate() -> None:
    # Genuine large obstruction (120 of 144 blocks, ~83%) with no co-occurring
    # degradation. The upper-tail ambient stays at 1.0, the obstruction is a
    # clear outlier, and the single contiguous cluster passes every gate.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.5)
    current[0:10, :] = 0.025  # retention 0.05 over the obstruction rectangle
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is True
    assert assessment.reject_reason is None
    assert assessment.ambient_retention == pytest.approx(1.0)
    assert assessment.obstruction_block_fraction == pytest.approx(120 / 144)
    assert assessment.largest_cluster_fraction >= TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION
    assert assessment.compactness == pytest.approx(1.0)
    assert assessment.compactness >= TAMPERING_MIN_COMPACTNESS_RATIO


def test_scenario_large_obstruction_with_severe_global_degradation_is_candidate() -> None:
    # THE BUG CASE: ~83% obstruction co-occurring with severe blur that pushes
    # ambient retention down to 0.4. The legacy absolute ceiling rejected this
    # (total loss 0.83 > 0.75). The relative model keeps the obstruction a
    # detectable outlier against the collapsed ambient: 0.05 vs 0.4.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.2)   # ambient retention 0.4 (blur)
    current[0:10, :] = 0.025                # obstruction retention 0.05
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is True
    assert assessment.ambient_retention == pytest.approx(0.4)
    assert assessment.relative_depth == pytest.approx(0.05 / 0.4)
    assert assessment.largest_cluster_fraction == pytest.approx(120 / 144)
    assert assessment.reject_reason is None


def test_scenario_degenerate_near_zero_ambient_does_not_confirm() -> None:
    # Degenerate: ambient collapses to near-zero everywhere INCLUDING where
    # the obstruction is (retention 0.01 ambient / 0.008 obstruction). The
    # relative model must NOT confidently confirm -- this is the fully
    # ambiguous case reserved for Approach C. Both the degeneracy floor
    # (0.01 < TAMPERING_AMBIENT_MIN_RETENTION) and the relative threshold
    # (0.008 > 0.35 * 0.01) reject it.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.005)  # retention 0.01 everywhere
    current[0:10, :] = 0.004                # obstruction retention 0.008
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "degraded_ambient"
    assert assessment.ambient_retention == pytest.approx(0.01)
    assert assessment.ambient_retention < TAMPERING_AMBIENT_MIN_RETENTION


def test_scenario_degenerate_zero_retention_everywhere_does_not_confirm() -> None:
    # Completely black current frame: every retention ratio is exactly 0, so
    # ambient == 0. Without the degeneracy guard, "0 <= k * 0" would flag
    # every block and produce a giant false candidate; the guard rejects it.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.0)
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "degraded_ambient"
    assert assessment.ambient_retention == 0.0


def test_scenario_no_meaningful_baseline_blocks_rejected() -> None:
    baseline = _density_grid(12, 12, 0.01)  # all below the meaningful floor
    current = _density_grid(12, 12, 0.0)
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "no_meaningful_blocks"
    assert assessment.meaningful_block_fraction == 0.0


def test_scenario_small_obstruction_fails_min_size() -> None:
    # A physically tiny obstruction (10 of 144 blocks) passes ambient/depth but
    # must not reach candidacy: below TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.5)
    current[0:2, 0:5] = 0.025
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "min_size"
    assert assessment.largest_cluster_fraction == pytest.approx(10 / 144)


def test_scenario_scattered_obstruction_regions_fail_compactness() -> None:
    # Two large but disconnected obstruction regions (30 blocks each): the
    # largest cluster is 30/144 >= min size and is a depth outlier, but holds
    # only 50% of ALL obstruction blocks -- below TAMPERING_MIN_COMPACTNESS_RATIO.
    # Proves compactness is NOT redundant with connected-cluster identification.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.5)
    current[0:3, 0:10] = 0.025
    current[9:12, 0:10] = 0.025
    assessment = assess_obstruction_candidacy(baseline, current)
    assert assessment.is_candidate is False
    assert assessment.reject_reason == "compactness"
    assert assessment.compactness == pytest.approx(0.5)
    assert assessment.largest_cluster_fraction >= TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION


def test_assessment_parameterized_and_config_independent() -> None:
    # The scoring functions are parameterized, not hard-wired to config: the
    # same densities produce different decisions under a looser depth ratio,
    # which is exactly the knob a future calibration would tune.
    baseline = _density_grid(12, 12, 0.5)
    current = _density_grid(12, 12, 0.25)  # ambient retention 0.5
    current[0:5, 0:6] = 0.1                # 30 blocks at retention 0.2
    default = assess_obstruction_candidacy(baseline, current)
    assert default.reject_reason == "no_obstruction_blocks"  # 0.2 > 0.35 * 0.5
    loose = assess_obstruction_candidacy(baseline, current, depth_ratio=0.5)
    assert loose.is_candidate is True  # 0.2 <= 0.5 * 0.5 -> obstruction cluster





