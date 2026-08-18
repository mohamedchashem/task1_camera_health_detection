"""Tampering/obstruction fault detector.

Detects lens obstruction by finding contiguous regions of a camera's
baseline structure that have gone missing in the current frame.
Obstruction is physically localized (one object blocking one region of
the lens); blur and low-light degrade structure scattered across the
whole frame instead. Requiring a large connected cluster of loss,
rather than just a high total percentage, distinguishes real
obstruction from those other faults.
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


def compute_largest_contiguous_loss_fraction(baseline_edges: np.ndarray, current_edges: np.ndarray) -> float:
    """Largest single connected cluster of 'structure lost' blocks, as a
    fraction of the baseline's total meaningful-structure blocks.
    """
    baseline_density = _compute_block_density(baseline_edges, TAMPERING_GRID_BLOCK_SIZE)
    current_density = _compute_block_density(current_edges, TAMPERING_GRID_BLOCK_SIZE)

    meaningful_blocks = baseline_density >= TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY
    meaningful_count = np.count_nonzero(meaningful_blocks)
    if meaningful_count == 0:
        return 0.0  # baseline has no real structure to lose; can't signal obstruction this way

    disappeared = meaningful_blocks & (current_density <= baseline_density * TAMPERING_BLOCK_DENSITY_DROP_RATIO)

    disappeared_mask = disappeared.astype(np.uint8)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(disappeared_mask, connectivity=4)

    if num_labels <= 1:
        return 0.0  # label 0 is background; no disappeared blocks at all

    # stats[0] is the background component; skip it when finding the largest real cluster.
    largest_component_size = stats[1:, cv2.CC_STAT_AREA].max()
    return largest_component_size / meaningful_count


def evaluate(frame: np.ndarray, baseline_edges: np.ndarray) -> TamperingResult:
    """Score a frame's obstruction level relative to a camera's baseline edge map."""
    current_edges = compute_edge_map(frame)
    loss_fraction = compute_largest_contiguous_loss_fraction(baseline_edges, current_edges)

    is_candidate = loss_fraction >= TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION
    confidence = float(np.clip(loss_fraction, 0.0, 1.0))

    return TamperingResult(
        largest_contiguous_loss_fraction=loss_fraction,
        confidence=confidence,
        is_candidate=is_candidate,
    )