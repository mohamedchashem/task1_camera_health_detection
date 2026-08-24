"""Low-light (brightness) fault detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import LOWLIGHT_DARK_PIXEL_THRESHOLD, LOWLIGHT_BASELINE_DROP_RATIO


@dataclass(frozen=True)
class BrightnessResult:
    dark_pixel_ratio: float  # raw score
    confidence: float        # normalized 0-1 score
    is_candidate: bool       # single-frame flag; temporal confirmation happens upstream
    relative_increase: float = 0.0  # (current_ratio - baseline) / baseline


def compute_dark_pixel_ratio(frame: np.ndarray) -> float:
    """Fraction of pixels below the dark-pixel threshold, measured on HSV V-channel."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    dark_pixels = np.count_nonzero(v_channel < LOWLIGHT_DARK_PIXEL_THRESHOLD)
    return dark_pixels / v_channel.size


def evaluate(frame: np.ndarray, baseline_dark_ratio: float) -> BrightnessResult:
    """Score a frame's brightness relative to a camera's baseline."""
    current_ratio = compute_dark_pixel_ratio(frame)

    safe_baseline = max(baseline_dark_ratio, 1e-6)
    relative_increase = (current_ratio - safe_baseline) / safe_baseline
    is_candidate = relative_increase >= LOWLIGHT_BASELINE_DROP_RATIO

    scale_range = max(1.0 - baseline_dark_ratio, 1e-6)  # guard a baseline of 1.0
    confidence = float(np.clip((current_ratio - baseline_dark_ratio) / scale_range, 0.0, 1.0))

    return BrightnessResult(
        dark_pixel_ratio=current_ratio,
        confidence=confidence,
        is_candidate=is_candidate,
        relative_increase=relative_increase,
    )