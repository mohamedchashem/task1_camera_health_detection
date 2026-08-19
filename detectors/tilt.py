"""Tilt / angle-change fault detector.

Detects camera physical rotation via learned local feature matching
(DISK, accessed through Kornia) between the current frame and a
camera's baseline reference frame. Measures the median displacement
of matched keypoints directly, rather than fitting a full homography
or affine transform — geometric model fitting proved numerically
unstable on real footage where sparse/clustered matches during faults
produced degenerate fits (condition numbers in the millions), making
any transform-based metric unreliable, including using a fitted
model purely as a RANSAC outlier filter (RANSAC's internal fit is
the same degenerate model, so it silently discarded real tilt
matches along with the bad ones).

Outlier rejection instead uses median absolute deviation (MAD): the
median of matched-point displacements is inherently robust to
outliers, and MAD-based rejection (the "X84 rule") filters remaining
noise using only the displacement distribution itself, with no
geometric model of the scene involved at any point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import kornia.feature as KF
import numpy as np
import torch

from config import (
    TILT_MAX_KEYPOINTS,
    TILT_MIN_RELIABLE_MATCHES,
    TILT_MEDIAN_SHIFT_THRESHOLD_RATIO,
    TILT_MAD_REJECTION_THRESHOLD,
    TILT_MAD_EPSILON,
    TILT_SHIFT_CONFIDENCE_CEILING_RATIO,
    TILT_MATCH_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Tilt detector using device: %s", _device)

_disk_model = KF.DISK.from_pretrained("depth").to(_device)
_disk_model.eval()


@dataclass(frozen=True)
class TiltResult:
    median_shift_ratio: float  # raw score: median keypoint displacement / frame diagonal
    confidence: float
    is_candidate: bool
    reliable: bool


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(_device)


def extract_features(frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DISK on one frame; returns (keypoints, descriptors)."""
    tensor = _frame_to_tensor(frame)
    with torch.no_grad():
        features = _disk_model(tensor, TILT_MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
    return features.keypoints, features.descriptors


def _mad_inlier_mask(displacements: np.ndarray) -> np.ndarray:
    """Flag inlier displacements using median absolute deviation (MAD).

    MAD is a robust measure of spread: unlike standard deviation, a
    handful of extreme outliers can't drag it around, because it's
    built from medians rather than means. This makes it a good fit
    here specifically because it requires no geometric model of the
    scene (unlike RANSAC+homography) -- it looks only at the spread
    of the displacement values themselves.

    The 1.4826 scale factor is the standard correction that makes MAD
    comparable to a standard deviation under normal (Gaussian) noise,
    so `threshold` behaves like a "number of standard deviations" cutoff.
    A threshold of 3 (the "X84 rule") is a well-established default in
    robust-statistics and feature-tracking literature.
    """
    median = np.median(displacements)
    abs_deviations = np.abs(displacements - median)
    mad = 1.4826 * np.median(abs_deviations)

    if mad < TILT_MAD_EPSILON:
        # All displacements are already nearly identical (e.g. a static
        # scene) -- nothing to reject, and dividing by a near-zero MAD
        # would falsely flag normal variation as outliers.
        return np.ones_like(displacements, dtype=bool)

    z_scores = abs_deviations / mad
    return z_scores <= TILT_MAD_REJECTION_THRESHOLD


def compute_median_shift_ratio(
    baseline_keypoints: torch.Tensor,
    baseline_descriptors: torch.Tensor,
    current_frame: np.ndarray,
) -> tuple[float, bool]:
    """Measure how much matched keypoints have moved between baseline
    and current frame. Returns (shift_ratio, reliable).

    Uses median displacement of MAD-filtered matches (robust to
    outlier matches without fitting any geometric model), normalized
    by frame diagonal (scale-independent across resolutions).
    """
    height, width = current_frame.shape[:2]
    current_keypoints, current_descriptors = extract_features(current_frame)

    with torch.no_grad():
        _distances, match_idxs = KF.match_smnn(
            baseline_descriptors, current_descriptors, th=TILT_MATCH_RATIO_THRESHOLD,
        )

    if len(match_idxs) < TILT_MIN_RELIABLE_MATCHES:
        logger.warning(
            "Only %d matches found (need %d); tilt estimate skipped as unreliable.",
            len(match_idxs), TILT_MIN_RELIABLE_MATCHES,
        )
        return 0.0, False

    points_baseline = baseline_keypoints[match_idxs[:, 0]].cpu().numpy()
    points_current = current_keypoints[match_idxs[:, 1]].cpu().numpy()

    displacements = np.linalg.norm(points_current - points_baseline, axis=1)
    inliers = _mad_inlier_mask(displacements)

    if inliers.sum() < TILT_MIN_RELIABLE_MATCHES:
        logger.warning(
            "Only %d MAD-filtered inliers (need %d); tilt estimate skipped as unreliable.",
            inliers.sum(), TILT_MIN_RELIABLE_MATCHES,
        )
        return 0.0, False

    # Median displacement of inlier-filtered matched points,
    # normalized by frame diagonal for scale independence.
    frame_diagonal = np.hypot(width, height)
    shift_ratio = float(np.median(displacements[inliers]) / frame_diagonal)

    return shift_ratio, True


def evaluate(
    current_frame: np.ndarray,
    baseline_keypoints: torch.Tensor,
    baseline_descriptors: torch.Tensor,
) -> TiltResult:
    """Score a frame's framing shift relative to a camera's precomputed baseline features."""
    shift_ratio, reliable = compute_median_shift_ratio(baseline_keypoints, baseline_descriptors, current_frame)

    if not reliable:
        return TiltResult(median_shift_ratio=0.0, confidence=0.0, is_candidate=False, reliable=False)

    is_candidate = shift_ratio >= TILT_MEDIAN_SHIFT_THRESHOLD_RATIO
    confidence = float(np.clip(shift_ratio / TILT_SHIFT_CONFIDENCE_CEILING_RATIO, 0.0, 1.0))

    return TiltResult(
        median_shift_ratio=shift_ratio,
        confidence=confidence,
        is_candidate=is_candidate,
        reliable=True,
    )
