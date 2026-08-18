"""File-based frame reader for deterministic ground-truth testing.

Unlike a live RTSP stream (see pipeline/stream_reader.py), a file has
reliable frame-rate metadata, so video time is derived from frame
count / fps rather than wall-clock time. Used for testing correctness
against known timestamps, not for production streaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def read_frames_from_file(video_path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_number, video_time_s, frame) from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read a valid frame rate from {video_path}")

    frame_number = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            yield frame_number, frame_number / fps, frame
    finally:
        cap.release()