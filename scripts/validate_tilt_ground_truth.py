"""Validate the tilt detector against known fault timestamps in a
recorded file. File-based (not RTSP) so frame position maps to real
video time exactly — needed for ground-truth comparison.

The detector itself remains blind: it is given no fault timing. This
script only uses timing afterward, to grade the blind output against
what a human confirmed actually happens in the footage.

Usage:
    python -m scripts.validate_tilt_ground_truth cam1 data\\test_footage\\test_video.mp4
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import cv2

from config import BASELINES_DIR, TEST_RUNS_DIR, validate_config
from detectors.tilt import evaluate, extract_features
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id

logger = logging.getLogger(__name__)


def _load_baseline_frame(camera_id: str):
    image_path = BASELINES_DIR / f"{camera_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(
            f"No baseline reference frame found for {camera_id!r} at {image_path}. "
            "Run capture_baseline first."
        )
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to load baseline reference frame at {image_path}.")
    return frame


def validate_against_file(camera_id: str, video_path: Path) -> Path:
    camera_id = validate_camera_id(camera_id)
    baseline_frame = _load_baseline_frame(camera_id)

    logger.info("Computing baseline features (once)...")
    baseline_keypoints, baseline_descriptors = extract_features(baseline_frame)

    TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = TEST_RUNS_DIR / f"{camera_id}_tilt_ground_truth.csv"

    frame_number = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
                        ["frame_number", "video_time_s", "median_shift_ratio", "confidence", "is_candidate", "reliable"]
        )

        for frame_number, video_time_s, frame in read_frames_from_file(video_path):
            result = evaluate(frame, baseline_keypoints, baseline_descriptors, baseline_frame.shape[:2])
            writer.writerow(
                [
                    frame_number,
                    round(video_time_s, 3),
                    round(result.median_shift_ratio, 6),
                    round(result.confidence, 6),
                    result.is_candidate,
                    result.reliable,
                ]
            )

            if frame_number % 100 == 0:
                logger.info("Processed %d frames...", frame_number)

    logger.info("Wrote %d scored frames to %s", frame_number, output_csv)
    return output_csv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_config()

    if len(sys.argv) != 3:
        logger.error("Usage: python -m scripts.validate_tilt_ground_truth <camera_id> <video_path>")
        sys.exit(1)

    validate_against_file(sys.argv[1], Path(sys.argv[2]))