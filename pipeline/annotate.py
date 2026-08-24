"""Annotated frame snapshots for confirmed faults.

Draws fault labels onto a frame and saves snapshots to EVENT_FRAMES_DIR
under a bounded ring: at most EVENT_FRAMES_MAX_TOTAL files are kept,
evicting the oldest (by file modification time) first, so disk usage
stays bounded regardless of event volume.

Multi-label rendering: when a frame carries multiple active faults
(``DecisionFrame.faults``), each fault is rendered as its own ``FAULT:``
line, stacked top-to-bottom with dynamic Y-offsets derived from the
measured rendered height of the preceding label so labels never overlap.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import EVENT_FRAMES_DIR, EVENT_FRAMES_MAX_TOTAL
from pipeline.decision_engine import Fault

_LINE_COLOR = (0, 0, 255)  # BGR red
_FONT_SCALE = 0.7
_FONT_THICKNESS = 2
_LINE_PADDING = 8
_FIRST_LINE_Y = 30


def build_annotation_lines(
    primary_fault: str | None,
    confidence: float,
    faults: Iterable[Fault] = (),
    secondary_symptoms: Iterable[str] = (),
    suppressed_faults: Iterable[str] = (),
    video_time_s: float | None = None,
) -> list[str]:
    """Build the stacked text lines for one annotated snapshot.

    When ``faults`` is non-empty, every active fault is rendered as its own
    ``FAULT:`` line (in the order provided -- callers pass
    ``DecisionFrame.faults``, which is already ``DECISION_PRECEDENCE``
    order). The ``secondary`` summary is skipped in that path because the
    stacked fault lines already show every survivor. When ``faults`` is
    empty the single-fault fallback renders ``primary_fault``/``confidence``
    with the ``secondary`` summary.
    """
    lines: list[str] = []
    active = tuple(faults)
    if active:
        lines.extend(
            f"FAULT: {fault.fault_type} (conf={fault.confidence:.2f})"
            for fault in active
        )
    elif primary_fault is not None:
        lines.append(f"FAULT: {primary_fault} (conf={confidence:.2f})")
    secondary = tuple(secondary_symptoms)
    suppressed = tuple(suppressed_faults)
    if secondary and not active:
        lines.append("secondary: " + ", ".join(secondary))
    if suppressed:
        lines.append("suppressed: " + ", ".join(suppressed))
    if video_time_s is not None:
        lines.append(f"t={video_time_s:.2f}s")
    return lines


def _line_offset(line: str) -> int:
    """Vertical advance for one rendered label.

    Measured from the text actually drawn (height + baseline) plus padding,
    so stacked labels use dynamic Y-offsets and never overlap regardless of
    the label content.
    """
    (_, text_height), baseline = cv2.getTextSize(
        line, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _FONT_THICKNESS
    )
    return text_height + baseline + _LINE_PADDING


def annotate_frame(
    frame: np.ndarray,
    primary_fault: str | None,
    confidence: float,
    secondary_symptoms: Iterable[str] = (),
    suppressed_faults: Iterable[str] = (),
    video_time_s: float | None = None,
    faults: Iterable[Fault] = (),
) -> np.ndarray:
    """Return a copy of ``frame`` with fault labels drawn on it.

    The input frame is not modified. When ``faults`` carries multiple active
    faults, each fault label is drawn on its own line stacked vertically,
    with the Y-offset of each label derived from the rendered height of the
    preceding label.
    """
    annotated = frame.copy()

    y = _FIRST_LINE_Y
    for line in build_annotation_lines(
        primary_fault,
        confidence,
        faults=faults,
        secondary_symptoms=secondary_symptoms,
        suppressed_faults=suppressed_faults,
        video_time_s=video_time_s,
    ):
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            _FONT_SCALE,
            _LINE_COLOR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y += _line_offset(line)
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
