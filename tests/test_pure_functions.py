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
)
from detectors.tilt import _mad_inlier_mask
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
