"""Annotated frame snapshots for confirmed faults.

Draws fault labels onto a frame and saves snapshots to EVENT_FRAMES_DIR
under a bounded ring: at most EVENT_FRAMES_MAX_TOTAL files are kept,
evicting the oldest (by file modification time) first, so disk usage
stays bounded regardless of event volume.

Multi-label rendering: when a frame carries multiple active faults
(``DecisionFrame.faults``), each fault is rendered as its own ``FAULT:``
line, stacked top-to-bottom with dynamic Y-offsets derived from the
measured rendered height of the preceding label so labels never overlap.

Banner-fix (Part A) rendering: ``build_annotation_lines`` additionally
accepts the confirmed/pending/below-floor/unmeasurable inputs the decision
engine now produces (``DecisionFrame.confirmed_faults``,
``below_floor_faults``, gate-skip status). When any of them is provided,
the five-section banner is rendered (confirmed active faults, pending
survivors, causally-suppressed, below-floor "too weak", and gate-skipped
"unmeasurable" detectors). When none is provided, the legacy banner
renders exactly as before, so current callers are unchanged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import DECISION_PRECEDENCE, EVENT_FRAMES_DIR, EVENT_FRAMES_MAX_TOTAL
from pipeline.decision_engine import Fault

logger = logging.getLogger(__name__)

_LINE_COLOR = (0, 0, 255)  # BGR red
_FONT_SCALE = 0.7
_FONT_THICKNESS = 2
_LINE_PADDING = 8
_FIRST_LINE_Y = 30

# Banner-fix sections render in DECISION_PRECEDENCE order. Fault types
# outside DECISION_PRECEDENCE (a caller error upstream) sort last, stably.
_PRECEDENCE_RANK = {fault: rank for rank, fault in enumerate(DECISION_PRECEDENCE)}
_UNKNOWN_RANK = len(DECISION_PRECEDENCE)


def build_annotation_lines(
    primary_fault: str | None,
    confidence: float,
    faults: Iterable[Fault] = (),
    secondary_symptoms: Iterable[str] = (),
    suppressed_faults: Iterable[str] = (),
    video_time_s: float | None = None,
    confirmed_faults: Iterable[Fault] = (),
    pending_faults: Iterable[Fault] = (),
    below_floor_faults: Iterable[str] = (),
    unmeasurable_faults: Iterable[str] = (),
) -> list[str]:
    """Build the stacked text lines for one annotated snapshot.

    Legacy path (default): when ``confirmed_faults``, ``pending_faults``,
    ``below_floor_faults``, and ``unmeasurable_faults`` are all empty, the
    banner renders exactly as before -- every active fault in ``faults`` as
    its own ``FAULT:`` line (order as provided), or the single-fault
    fallback from ``primary_fault``/``confidence``, plus the ``secondary``
    and ``suppressed`` summaries and the timestamp.

    Banner-fix path (any of the four new inputs non-empty): the banner
    renders five sections top-to-bottom, each omitted when empty, each
    ordered by ``DECISION_PRECEDENCE``:

    1. ``FAULT: {type} (conf={peak:.2f})`` per confirmed fault. A confirmed
       fault whose detector was gate-skipped this exact frame gets
       `` [unmeasurable]`` appended to its line (instead of being listed
       again in section 5).
    2. ``pending: {type} (conf={frame_conf:.2f})`` per current fusion
       survivor that is NOT yet confirmed.
    3. ``suppressed: {type}[, {type}...]`` -- causally-suppressed
       candidates (not confirmed), one comma-joined line.
    4. ``too weak: {type}[, {type}...]`` -- below-floor candidates, one
       comma-joined line (never called "suppressed").
    5. ``unmeasurable: {type}[, {type}...]`` -- gate-skipped detectors this
       frame that are not already shown via the ``[unmeasurable]`` marker,
       one comma-joined line.

    followed by the timestamp ``t={video_time_s:.2f}s`` when provided. In
    this path the legacy ``faults``/``primary_fault``/``confidence``/
    ``secondary_symptoms`` inputs are superseded and not rendered; only
    ``suppressed_faults`` is shared (it feeds section 3).

    Exclusivity invariant: a fault type must appear in exactly ONE of the
    five sections. The only exception is the ``[unmeasurable]`` marker,
    which annotates an existing confirmed line instead of creating a
    duplicate listing. Passing the same fault type in several of the new
    lists (or in ``suppressed_faults``) is a caller error: it is resolved
    by the priority order confirmed > pending > suppressed > too weak >
    unmeasurable, and a warning is logged for every dropped duplicate.
    """
    confirmed = tuple(confirmed_faults)
    pending = tuple(pending_faults)
    below_floor = tuple(below_floor_faults)
    gate_skipped = tuple(unmeasurable_faults)

    if not (confirmed or pending or below_floor or gate_skipped):
        return _legacy_banner(
            primary_fault,
            confidence,
            faults,
            secondary_symptoms,
            suppressed_faults,
            video_time_s,
        )

    lines: list[str] = []
    claimed: set[str] = set()

    confirmed_map = {fault.fault_type: fault for fault in confirmed}
    pending_map = {fault.fault_type: fault for fault in pending}

    confirmed_names = _claim_exclusive(
        (fault.fault_type for fault in confirmed), claimed, "confirmed"
    )
    pending_names = _claim_exclusive(
        (fault.fault_type for fault in pending), claimed, "pending"
    )
    suppressed = _claim_exclusive(tuple(suppressed_faults), claimed, "suppressed")
    too_weak = _claim_exclusive(below_floor, claimed, "too weak")
    # A confirmed fault that is also gate-skipped is marked on its FAULT
    # line (documented exception); only the remaining gate-skipped
    # detectors can appear in section 5.
    unmeasurable = _claim_exclusive(
        (fault for fault in gate_skipped if fault not in confirmed_names),
        claimed,
        "unmeasurable",
    )

    for name in _precedence_sorted(confirmed_names):
        marker = " [unmeasurable]" if name in gate_skipped else ""
        lines.append(f"FAULT: {name} (conf={confirmed_map[name].confidence:.2f}){marker}")
    for name in _precedence_sorted(pending_names):
        lines.append(f"pending: {name} (conf={pending_map[name].confidence:.2f})")
    if suppressed:
        lines.append("suppressed: " + ", ".join(_precedence_sorted(suppressed)))
    if too_weak:
        lines.append("too weak: " + ", ".join(_precedence_sorted(too_weak)))
    if unmeasurable:
        lines.append("unmeasurable: " + ", ".join(_precedence_sorted(unmeasurable)))
    if video_time_s is not None:
        lines.append(f"t={video_time_s:.2f}s")
    return lines


def _legacy_banner(
    primary_fault: str | None,
    confidence: float,
    faults: Iterable[Fault],
    secondary_symptoms: Iterable[str],
    suppressed_faults: Iterable[str],
    video_time_s: float | None,
) -> list[str]:
    """Render the pre-banner-fix annotation lines (unchanged behavior)."""
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


def _precedence_sorted(fault_types: Iterable[str]) -> list[str]:
    """Order fault names by DECISION_PRECEDENCE; unknown names sort last."""
    return sorted(
        fault_types,
        key=lambda fault: _PRECEDENCE_RANK.get(fault, _UNKNOWN_RANK),
    )


def _claim_exclusive(
    fault_types: Iterable[str],
    claimed: set[str],
    section: str,
) -> list[str]:
    """Return faults not yet rendered in a higher-priority section.

    Each returned fault is added to ``claimed`` so a later section cannot
    also render it (the banner exclusivity invariant). A fault passed in
    several sections is a caller error: the highest-priority section wins
    and a warning is logged for every dropped duplicate.
    """
    kept: list[str] = []
    for fault in fault_types:
        if fault in claimed:
            logger.warning(
                "annotate banner: fault %r already rendered in a "
                "higher-priority section; dropping it from the %r section "
                "(priority: confirmed > pending > suppressed > too weak > "
                "unmeasurable).",
                fault,
                section,
            )
        else:
            kept.append(fault)
            claimed.add(fault)
    return kept


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
    confirmed_faults: Iterable[Fault] = (),
    pending_faults: Iterable[Fault] = (),
    below_floor_faults: Iterable[str] = (),
    unmeasurable_faults: Iterable[str] = (),
) -> np.ndarray:
    """Return a copy of ``frame`` with fault labels drawn on it.

    The input frame is not modified. When ``faults`` carries multiple active
    faults, each fault label is drawn on its own line stacked vertically,
    with the Y-offset of each label derived from the rendered height of the
    preceding label. The banner-fix inputs (``confirmed_faults``,
    ``pending_faults``, ``below_floor_faults``, ``unmeasurable_faults``) are
    forwarded to ``build_annotation_lines`` unchanged.
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
        confirmed_faults=confirmed_faults,
        pending_faults=pending_faults,
        below_floor_faults=below_floor_faults,
        unmeasurable_faults=unmeasurable_faults,
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
