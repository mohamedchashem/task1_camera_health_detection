"""Diagnostic tool: visualize edge disappearance/appearance against a
camera's baseline at regular time intervals across a video.

Produces one composite image per sample point:
  - white  = edge present in both baseline and current frame (stable)
  - red    = edge present in baseline but missing in current frame (disappeared)
  - green  = edge present in current frame but not in baseline (new)

Diagnostic/manual-review tool only — not part of the production
pipeline or the automated test suite.

Usage:
    python -m scripts.debug_tampering_edges cam1 data\\test_footage\\test_video.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from config import BASELINES_DIR, DEBUG_FRAMES_DIR, DEBUG_FRAME_SAMPLE_INTERVAL_SECONDS
from detectors.tampering import compute_edge_map
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id


def _load_baseline_edges(camera_id: str) -> np.ndarray:
    edges_path = BASELINES_DIR / f"{camera_id}_edges.png"
    edges = cv2.imread(str(edges_path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise RuntimeError(f"Failed to load edge baseline image at {edges_path}.")
    return edges


def dump_edge_comparisons(camera_id: str, video_path: Path) -> None:
    camera_id = validate_camera_id(camera_id)
    baseline_edges = _load_baseline_edges(camera_id)
    DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    next_sample_time = 0.0
    saved_count = 0

    for _frame_number, video_time_s, frame in read_frames_from_file(video_path):
        if video_time_s < next_sample_time:
            continue
        next_sample_time += DEBUG_FRAME_SAMPLE_INTERVAL_SECONDS

        current_edges = compute_edge_map(frame)

        composite = np.zeros((*baseline_edges.shape, 3), dtype=np.uint8)
        stable = (baseline_edges > 0) & (current_edges > 0)
        disappeared = (baseline_edges > 0) & (current_edges == 0)
        new = (baseline_edges == 0) & (current_edges > 0)

        composite[stable] = (255, 255, 255)
        composite[disappeared] = (0, 0, 255)   # red, BGR order
        composite[new] = (0, 255, 0)           # green

        out_path = DEBUG_FRAMES_DIR / f"{camera_id}_edges_t{video_time_s:06.2f}.jpg"
        cv2.imwrite(str(out_path), composite)
        saved_count += 1

    print(f"Wrote {saved_count} debug edge-comparison images to {DEBUG_FRAMES_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.debug_tampering_edges <camera_id> <video_path>")
        sys.exit(1)

    dump_edge_comparisons(sys.argv[1], Path(sys.argv[2]))