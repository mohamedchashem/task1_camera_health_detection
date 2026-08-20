"""Annotated frame snapshots for confirmed faults.

Draws fault labels onto a frame and saves snapshots to EVENT_FRAMES_DIR
under a bounded ring: at most EVENT_FRAMES_MAX_TOTAL files are kept,
evicting the oldest (by file modification time) first, so disk usage
stays bounded regardless of event volume.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import EVENT_FRAMES_DIR, EVENT_FRAMES_MAX_TOTAL

_LINE_COLOR = (0, 0, 255)  # BGR red
_LINE_HEIGHT = 30
_FIRST_LINE_Y = 30


def annotate_frame(
    frame: np.ndarray,
    primary_fault: str | None,
    confidence: float,
    secondary_symptoms: Iterable[str] = (),
    suppressed_faults: Iterable[str] = (),
    video_time_s: float | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with fault labels drawn on it.

    The input frame is not modified.
    """
    annotated = frame.copy()

    lines: list[str] = []
    if primary_fault is not None:
        lines.append(f"FAULT: {primary_fault} (conf={confidence:.2f})")
    secondary = tuple(secondary_symptoms)
    suppressed = tuple(suppressed_faults)
    if secondary:
        lines.append("secondary: " + ", ".join(secondary))
    if suppressed:
        lines.append("suppressed: " + ", ".join(suppressed))
    if video_time_s is not None:
        lines.append(f"t={video_time_s:.2f}s")

    y = _FIRST_LINE_Y
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _LINE_COLOR,
            2,
            cv2.LINE_AA,
        )
        y += _LINE_HEIGHT
    return annotated


def save_annotated_frame(
    frame: np.ndarray,
    camera_id: str,
    fault_type: str,
    frame_number: int,
    video_time_s: float,
    event_frames_dir: str | Path = EVENT_FRAMES_DIR,
    max_total: int = EVENT_FRAMES_MAX_TOTAL,
) -> Path:
    """Save an annotated snapshot and enforce the bounded ring.

    The caller is expected to have drawn anything meaningful on ``frame``
    beforehand (e.g. via ``annotate_frame``). Returns the saved path.
    """
    directory = Path(event_frames_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{camera_id}_{fault_type}_f{frame_number:06d}_{time.time_ns()}.jpg"
    out_path = directory / filename
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"Failed to write annotated frame to {out_path}.")
    _enforce_ring_budget(directory, max_total)
    return out_path


def _enforce_ring_budget(directory: Path, max_total: int) -> None:
    """Evict oldest annotated frames (by file mtime) until <= max_total remain."""
    files = [p for p in directory.iterdir() if p.is_file()]
    if len(files) <= max_total:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    for stale in files[: len(files) - max_total]:
        stale.unlink(missing_ok=True)
