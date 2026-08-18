"""Blur / dirty-lens fault detector.

Detects loss of image sharpness (out-of-focus, smudged or dirty lens)
via Variance of Laplacian — a frame's Laplacian-filtered variance is
high when abundant sharp edges are present, and drops when fine detail
has been smoothed away by blur. Baseline-relative, since a naturally
low-texture scene has lower Laplacian variance even in perfect focus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import BLUR_SHARPNESS_DROP_RATIO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlurResult:
    sharpness: float       # raw score: Laplacian variance
    confidence: float      # normalized 0-1 score
    is_candidate: bool     # single-frame flag; temporal confirmation happens upstream


def compute_sharpness(frame: np.ndarray) -> float:
    """Laplacian variance of a frame — higher means sharper/more in-focus."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def evaluate(frame: np.ndarray, baseline_sharpness: float) -> BlurResult:
    """Score a frame's blur level relative to a camera's baseline sharpness."""
    sharpness = compute_sharpness(frame)

    safe_baseline = max(baseline_sharpness, 1e-6)  # guard against a degenerate all-flat baseline
    sharpness_ratio = sharpness / safe_baseline

    is_candidate = sharpness_ratio <= BLUR_SHARPNESS_DROP_RATIO

    # Min-max scaled: 1.0 confidence at baseline-drop-ratio or below,
    # 0.0 confidence at full baseline sharpness or above — matches the
    # same normalization principle used by the low-light detector.
    scale_range = max(1.0 - BLUR_SHARPNESS_DROP_RATIO, 1e-6)
    confidence = float(np.clip((1.0 - sharpness_ratio) / scale_range, 0.0, 1.0))

    return BlurResult(
        sharpness=sharpness,
        confidence=confidence,
        is_candidate=is_candidate,
    )