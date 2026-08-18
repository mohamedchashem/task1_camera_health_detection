"""Validate the low-light detector against known fault timestamps in a
recorded file. Kept as a manual/exploratory tool; the automated,
repeatable version of this check lives in tests/test_lowlight_detector.py.

Usage:
    python -m scripts.validate_lowlight_ground_truth cam1 data\\test_footage\\test_video.mp4
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from config import BASELINES_DIR, TEST_RUNS_DIR
from detectors.brightness import evaluate
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id


def _load_baseline_dark_ratio(camera_id: str) -> float:
    baseline_path = BASELINES_DIR / f"{camera_id}.json"
    record = json.loads(baseline_path.read_text())
    return record["lowlight_dark_pixel_ratio"]


def validate_against_file(camera_id: str, video_path: Path) -> Path:
    camera_id = validate_camera_id(camera_id)
    baseline_dark_ratio = _load_baseline_dark_ratio(camera_id)

    TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = TEST_RUNS_DIR / f"{camera_id}_lowlight_ground_truth.csv"

    frame_number = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "video_time_s", "dark_pixel_ratio", "confidence", "is_candidate"])

        for frame_number, video_time_s, frame in read_frames_from_file(video_path):
            result = evaluate(frame, baseline_dark_ratio)
            writer.writerow(
                [
                    frame_number,
                    round(video_time_s, 3),
                    round(result.dark_pixel_ratio, 6),
                    round(result.confidence, 6),
                    result.is_candidate,
                ]
            )

    print(f"Wrote {frame_number} scored frames to {output_csv}")
    return output_csv


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.validate_lowlight_ground_truth <camera_id> <video_path>")
        sys.exit(1)

    validate_against_file(sys.argv[1], Path(sys.argv[2]))