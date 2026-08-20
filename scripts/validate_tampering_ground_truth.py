"""Validate the tampering detector against known fault timestamps in a
recorded file. File-based (not RTSP) so frame position maps to real
video time exactly — needed for ground-truth comparison.

The detector itself remains blind: it is given no fault timing. This
script only uses timing afterward, to grade the blind output against
what a human confirmed actually happens in the footage.

Usage:
    python -m scripts.validate_tampering_ground_truth cam1 data\\test_footage\\test_video.mp4
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import cv2

from config import BASELINES_DIR, TEST_RUNS_DIR, validate_config
from detectors.tampering import evaluate
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id

logger = logging.getLogger(__name__)


def _load_baseline_edges(camera_id: str):
    edges_path = BASELINES_DIR / f"{camera_id}_edges.png"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"No edge baseline found for {camera_id!r} at {edges_path}. "
            "Run capture_baseline first."
        )
    edges = cv2.imread(str(edges_path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise RuntimeError(f"Failed to load edge baseline image at {edges_path}.")
    return edges


def validate_against_file(camera_id: str, video_path: Path) -> Path:
    camera_id = validate_camera_id(camera_id)
    baseline_edges = _load_baseline_edges(camera_id)

    TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = TEST_RUNS_DIR / f"{camera_id}_tampering_ground_truth.csv"

    frame_number = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_number", "video_time_s", "edge_disappearance_ratio", "confidence", "is_candidate"]
        )

        for frame_number, video_time_s, frame in read_frames_from_file(video_path):
            result = evaluate(frame, baseline_edges)
            writer.writerow(
                [
                    frame_number,
                    round(video_time_s, 3),
                    round(result.largest_contiguous_loss_fraction, 6),
                    round(result.confidence, 6),
                    result.is_candidate,
                ]
            )

    logger.info("Wrote %d scored frames to %s", frame_number, output_csv)
    return output_csv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_config()

    if len(sys.argv) != 3:
        logger.error("Usage: python -m scripts.validate_tampering_ground_truth <camera_id> <video_path>")
        sys.exit(1)

    validate_against_file(sys.argv[1], Path(sys.argv[2]))