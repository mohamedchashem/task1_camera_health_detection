"""Manual per-camera baseline capture.

Run against a clean, fault-free segment of footage for a given camera.
Produces saved reference data every detector compares live frames
against: brightness, stable edge structure, and sharpness. Generic to
any camera/footage — nothing here is specific to a particular video.

Accepts either a live RTSP URL or a local file path as the source.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from config import (
    BASELINE_CAPTURE_SECONDS,
    BASELINE_MAX_DARK_RATIO,
    BASELINE_MIN_SHARPNESS,
    BASELINES_DIR,
    TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO,
    TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION,
    validate_config,
)
from detectors.blur import compute_sharpness
from detectors.brightness import compute_dark_pixel_ratio
from detectors.tampering import compute_edge_map, meaningful_block_fraction
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id
from pipeline.stream_reader import read_frames

logger = logging.getLogger(__name__)


def _windowed_frames(source: str) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, elapsed_seconds_within_window) from either a live
    RTSP stream or a local file, using whichever timing source is
    accurate for that source type.
    """
    is_live_stream = source.lower().startswith("rtsp://")

    if is_live_stream:
        capture_start: float | None = None
        for _frame_number, _stream_elapsed, frame in read_frames(source):
            if capture_start is None:
                capture_start = time.monotonic()
            yield frame, time.monotonic() - capture_start
    else:
        for _frame_number, video_time_s, frame in read_frames_from_file(Path(source)):
            yield frame, video_time_s


def capture_baseline(camera_id: str, source: str) -> dict:
    """Capture and persist a baseline for one camera.

    Returns the baseline record that was saved, for logging/confirmation.
    """
    camera_id = validate_camera_id(camera_id)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    dark_ratios: list[float] = []
    sharpness_values: list[float] = []
    edge_accumulator: np.ndarray | None = None
    frame_count = 0
    reference_frame = None

    for frame, window_elapsed in _windowed_frames(source):
        dark_ratios.append(compute_dark_pixel_ratio(frame))
        sharpness_values.append(compute_sharpness(frame))

        edges = compute_edge_map(frame)
        if edge_accumulator is None:
            edge_accumulator = np.zeros(edges.shape, dtype=np.float64)
        edge_accumulator += (edges > 0)
        frame_count += 1

        if reference_frame is None and window_elapsed >= BASELINE_CAPTURE_SECONDS / 2:
            reference_frame = frame

        if window_elapsed >= BASELINE_CAPTURE_SECONDS:
            break

    if not dark_ratios or edge_accumulator is None:
        raise RuntimeError(f"No frames captured for camera {camera_id!r}; check source.")

    edge_persistence = edge_accumulator / frame_count
    stable_edges = (edge_persistence >= TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO).astype(np.uint8) * 255

    mean_dark_ratio = sum(dark_ratios) / len(dark_ratios)
    mean_sharpness = sum(sharpness_values) / len(sharpness_values)
    structure_fraction = meaningful_block_fraction(stable_edges)

    # Capture-time quality gates. These are warnings (not hard failures):
    # a genuinely dark/blurry deployment camera still gets a baseline, but
    # the record is marked so operators see the capture was improper and the
    # runtime guards (e.g. tampering's degraded-baseline gate) stay honest.
    quality_warnings: list[str] = []
    if mean_dark_ratio > BASELINE_MAX_DARK_RATIO:
        quality_warnings.append(
            f"mean dark-pixel ratio {mean_dark_ratio:.3f} is above "
            f"BASELINE_MAX_DARK_RATIO ({BASELINE_MAX_DARK_RATIO}); capture looks too dark"
        )
    if mean_sharpness < BASELINE_MIN_SHARPNESS:
        quality_warnings.append(
            f"mean Laplacian sharpness {mean_sharpness:.1f} is below "
            f"BASELINE_MIN_SHARPNESS ({BASELINE_MIN_SHARPNESS}); capture looks blurry"
        )
    if structure_fraction < TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION:
        quality_warnings.append(
            f"tampering meaningful-block fraction {structure_fraction:.3f} is below "
            f"TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION "
            f"({TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION}); structure-loss detection is unreliable"
        )
    for warning in quality_warnings:
        logger.warning("Baseline quality for %s: %s", camera_id, warning)

    baseline_record = {
        "camera_id": camera_id,
        "lowlight_dark_pixel_ratio": mean_dark_ratio,
        "blur_baseline_sharpness": mean_sharpness,
        "tampering_meaningful_block_fraction": structure_fraction,
        "quality_warnings": quality_warnings,
        "frames_averaged": frame_count,
        "captured_at": time.time(),
    }

    image_path = BASELINES_DIR / f"{camera_id}.jpg"
    json_path = BASELINES_DIR / f"{camera_id}.json"
    edges_path = BASELINES_DIR / f"{camera_id}_edges.png"

    cv2.imwrite(str(image_path), reference_frame)
    cv2.imwrite(str(edges_path), stable_edges)
    json_path.write_text(json.dumps(baseline_record, indent=2))

    return baseline_record


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_config()

    if len(sys.argv) != 3:
        logger.error("Usage: python -m pipeline.capture_baseline <camera_id> <source>")
        sys.exit(1)

    result = capture_baseline(sys.argv[1], sys.argv[2])
    logger.info("Baseline saved for %s: %s", result["camera_id"], result)