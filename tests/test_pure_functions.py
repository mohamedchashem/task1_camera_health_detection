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
    compute_largest_contiguous_loss_fraction,
    compute_loss_fractions,
    confidence_from_largest_fraction,
    meaningful_block_fraction,
)
from detectors.tilt import (
    _mad_inlier_mask,
    inlier_ratio_sufficient,
    match_volume_sufficient,
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
