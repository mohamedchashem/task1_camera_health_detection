"""Blind validation run: score every frame from a live stream and log
results to CSV, with no foreknowledge of where faults occur.

Usage:
    python -m scripts.run_blind_test cam1 rtsp://localhost:8554/camtest
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2

from config import BASELINES_DIR, DEBUG_FRAMES_DIR, TEST_RUNS_DIR, TEST_RUN_DURATION_SECONDS
from detectors.brightness import evaluate
from pipeline.paths import validate_camera_id
from pipeline.stream_reader import read_frames


def _load_baseline_dark_ratio(camera_id: str) -> float:
    baseline_path = BASELINES_DIR / f"{camera_id}.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"No baseline found for {camera_id!r} at {baseline_path}. "
            "Run capture_baseline first."
        )
    record = json.loads(baseline_path.read_text())
    return record["lowlight_dark_pixel_ratio"]


def run_blind_test(camera_id: str, stream_url: str) -> None:
    camera_id = validate_camera_id(camera_id)
    baseline_dark_ratio = _load_baseline_dark_ratio(camera_id)

    TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = TEST_RUNS_DIR / f"{camera_id}_lowlight_blind_test.csv"

    run_start = time.monotonic()

    # Rows are written as they arrive rather than collected in memory
    # and written at the end, so memory use stays flat regardless of
    # run length.
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_number", "stream_elapsed_s", "dark_pixel_ratio", "confidence", "is_candidate"]
        )

        for frame_number, stream_elapsed, frame in read_frames(stream_url):
            result = evaluate(frame, baseline_dark_ratio)

            if result.is_candidate:
                debug_path = DEBUG_FRAMES_DIR / f"{camera_id}_frame_{frame_number}.jpg"
                cv2.imwrite(str(debug_path), frame)

            writer.writerow(
                [
                    frame_number,
                    round(stream_elapsed, 3),
                    round(result.dark_pixel_ratio, 6),
                    round(result.confidence, 6),
                    result.is_candidate,
                ]
            )

            if time.monotonic() - run_start >= TEST_RUN_DURATION_SECONDS:
                break

    print(f"Wrote {frame_number} scored frames to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.run_blind_test <camera_id> <stream_url>")
        sys.exit(1)

    run_blind_test(sys.argv[1], sys.argv[2])