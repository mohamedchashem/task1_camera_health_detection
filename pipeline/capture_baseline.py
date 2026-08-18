"""Manual per-camera baseline capture.

Run against a clean, fault-free segment of footage for a given camera.
Produces saved reference data every detector compares live frames
against: a brightness baseline (low-light detector) and a stable edge
map (tampering detector). Generic to any camera/footage — nothing
here is specific to a particular video.

Accepts either a live RTSP URL or a local file path as the source.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from config import (
    BASELINE_CAPTURE_SECONDS,
    BASELINES_DIR,
    TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO,
)
from detectors.brightness import compute_dark_pixel_ratio
from detectors.tampering import compute_edge_map
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id
from pipeline.stream_reader import read_frames


def _windowed_frames(source: str) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, elapsed_seconds_within_window) from either a live
    RTSP stream or a local file, using whichever timing source is
    accurate for that source type.
    """
    is_live_stream = source.lower().startswith("rtsp://")

    if is_live_stream:
        # No reliable per-frame timing metadata on a live stream;
        # wall-clock time is the correct proxy for video time here,
        # since a real stream is naturally paced in real time.
        capture_start: float | None = None
        for _frame_number, _stream_elapsed, frame in read_frames(source):
            if capture_start is None:
                capture_start = time.monotonic()
            yield frame, time.monotonic() - capture_start
    else:
        # A file has reliable frame-count/fps metadata and is read as
        # fast as the disk allows (not real-time paced), so wall-clock
        # time would be meaningless here — video-content time is the
        # correct, deterministic measure instead.
        for _frame_number, video_time_s, frame in read_frames_from_file(Path(source)):
            yield frame, video_time_s


def capture_baseline(camera_id: str, source: str) -> dict:
    """Capture and persist a baseline for one camera.

    Returns the baseline record that was saved, for logging/confirmation.
    """
    camera_id = validate_camera_id(camera_id)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    dark_ratios: list[float] = []
    edge_accumulator: np.ndarray | None = None
    frame_count = 0
    reference_frame = None

    for frame, window_elapsed in _windowed_frames(source):
        dark_ratios.append(compute_dark_pixel_ratio(frame))

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

    baseline_record = {
        "camera_id": camera_id,
        "lowlight_dark_pixel_ratio": sum(dark_ratios) / len(dark_ratios),
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

    if len(sys.argv) != 3:
        print("Usage: python -m pipeline.capture_baseline <camera_id> <source>")
        sys.exit(1)

    result = capture_baseline(sys.argv[1], sys.argv[2])
    print(f"Baseline saved for {result['camera_id']}: {result}")