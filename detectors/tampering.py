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
    TAMPERING_MAX_GLOBAL_LOSS_FRACTION,
    TAMPERING_MIN_COMPACTNESS_RATIO,
)


@dataclass(frozen=True)
class TamperingResult:
    largest_contiguous_loss_fraction: float  # raw score, 0-1
    confidence: float                         # normalized 0-1 score
    is_candidate: bool                         # single-frame flag; temporal confirmation happens upstream


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


def evaluate(frame: np.ndarray, baseline_edges: np.ndarray) -> TamperingResult:
    """Score a frame's obstruction level relative to a camera's baseline edge map."""
    current_edges = compute_edge_map(frame)
    largest_fraction, total_fraction = compute_loss_fractions(baseline_edges, current_edges)

    # Localization guards turn the raw largest-cluster metric into a
    # physical-obstruction decision. Two independent invariants separate a
    # real obstruction from faults that also remove edges:
    # 1. Total-loss ceiling: global degradation (blur/low-light) removes
    #    structure everywhere, pushing total loss toward the whole frame.
    # 2. Compactness: the single largest lost cluster must hold at least
    #    TAMPERING_MIN_COMPACTNESS_RATIO of ALL lost structure. A real
    #    obstruction is one contiguous object, so essentially all loss sits
    #    in one cluster (largest/total ~ 1.0). Camera rotation relocates
    #    edges instead of removing them, scattering the loss into many
    #    disconnected patches whose largest cluster is only a fraction of
    #    the total (measured 0.50-0.52 vs 1.00 on the synthetic fixture).
    localized = (
        total_fraction <= TAMPERING_MAX_GLOBAL_LOSS_FRACTION
        and largest_fraction >= TAMPERING_MIN_COMPACTNESS_RATIO * total_fraction
    )
    # Note: with total_fraction == 0 there is no lost structure, the
    # compactness check degenerates to 0 >= 0 and the min-contiguous check
    # below (largest >= 0.15) correctly rejects the frame.
    is_candidate = localized and largest_fraction >= TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION
    confidence = float(np.clip(largest_fraction, 0.0, 1.0))

    return TamperingResult(
        largest_contiguous_loss_fraction=largest_fraction,
        confidence=confidence,
        is_candidate=is_candidate,
    )