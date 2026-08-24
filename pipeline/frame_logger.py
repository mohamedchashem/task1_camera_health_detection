"""JSONL frame-level evaluation logging.

``FrameLogger`` appends one JSON object per logged frame to a JSONL
file, flushing periodically so a crash loses at most one buffer of
lines. Log files roll by size (FRAME_LOG_MAX_BYTES) so a long run never
grows one file without bound, and rotated backups are pruned by age
(FRAME_LOG_RETENTION_DAYS).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any

from config import (
    FRAME_LOG_INTERVAL_FRAMES,
    FRAME_LOG_MAX_BYTES,
    FRAME_LOG_RETENTION_DAYS,
)
from pipeline.decision_engine import (
    DETECTOR_STATUS_ERROR,
    DecisionFrame,
)

_SCHEMA_VERSION = 2


class FrameLogger:
    """Append-only JSONL writer for per-frame evaluation records.

    One JSON object per logged frame (see ``write_frame``). Frames with a
    detector in ``error`` state are always logged, regardless of sampling.
    Routine ``skipped`` observations (e.g. tilt sub-sampling) must not
    force a frame to be logged. The file is flushed every
    ``flush_interval_frames`` frames and on close.
    """

    def __init__(
        self,
        path: str | Path,
        interval_frames: int = FRAME_LOG_INTERVAL_FRAMES,
        flush_interval_frames: int = 100,
        retention_days: int = FRAME_LOG_RETENTION_DAYS,
        max_bytes: int | None = FRAME_LOG_MAX_BYTES,
    ) -> None:
        self._path = Path(path)
        self._interval_frames = interval_frames
        self._flush_interval_frames = flush_interval_frames
        self._retention_days = retention_days
        self._max_bytes = max_bytes
        self._file: Any = None
        self._frames_since_flush = 0
        self._frame_count = 0
        # Running count of bytes written to the active file (accurate even
        # before a flush because it is tracked from the strings we write).
        self._bytes_written = 0

    def _rotated_path(self) -> Path:
        """Unique backup name for a rolled file (nanosecond timestamp).

        Windows ``os.rename`` refuses to overwrite an existing file, so a
        second-resolution name can collide when rolls happen within the same
        second; nanosecond resolution makes that practically impossible.
        """
        return self._path.with_name(f"{self._path.name}.{time.time_ns()}")

    def open(self) -> None:
        """Open the log file, rotating expired logs per the retention policy."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        cutoff = self._retention_cutoff()
        if self._path.exists() and self._path.stat().st_mtime < cutoff:
            self._path.rename(self._rotated_path())
        self._prune_rotated()
        self._file = self._path.open("a", encoding="utf-8")
        self._bytes_written = self._path.stat().st_size

    def _retention_cutoff(self) -> float:
        """Unix timestamp before which rotated backups are considered expired."""
        return time.time() - self._retention_days * 86_400

    def _prune_rotated(self) -> None:
        """Delete rotated backups older than the retention window."""
        cutoff = self._retention_cutoff()
        for rotated in self._path.parent.glob(f"{self._path.name}.*"):
            if rotated.is_file() and rotated.stat().st_mtime < cutoff:
                rotated.unlink()

    def _roll(self) -> None:
        """Roll the active file to a timestamped backup and start a fresh one.

        Called when the active file has reached ``max_bytes`` so one long
        run cannot grow a single JSONL file without bound. Expired backups
        are pruned immediately, keeping disk usage bounded within the run.
        """
        self.flush()
        self._file.close()
        self._path.rename(self._rotated_path())
        self._file = self._path.open("a", encoding="utf-8")
        self._bytes_written = 0
        self._frames_since_flush = 0
        self._prune_rotated()

    def write_frame(self, frame: DecisionFrame, latency_ms: float | None = None) -> None:
        """Append one frame record, honoring the sampling interval."""
        if self._file is None:
            raise RuntimeError("FrameLogger not opened; call open() first.")

        always_log = any(
            obs.status == DETECTOR_STATUS_ERROR for obs in frame.detectors
        )
        self._frame_count += 1
        # 1-based sampling: with interval N, frames 1, 1+N, 1+2N, ... are logged.
        if not always_log and (self._frame_count - 1) % self._interval_frames != 0:
            return

        record = json.dumps(self._serialize(frame, latency_ms), separators=(",", ":"))
        # Bytes this line will occupy on disk: JSON output is ASCII
        # (ensure_ascii=True) and text mode translates the trailing newline
        # to os.linesep.
        record_bytes = len(record.encode("utf-8")) + len(os.linesep.encode("utf-8"))
        if (
            self._max_bytes is not None
            and self._bytes_written > 0
            and self._bytes_written + record_bytes > self._max_bytes
        ):
            self._roll()
        self._file.write(record + "\n")
        self._bytes_written += record_bytes
        self._frames_since_flush += 1
        if self._frames_since_flush >= self._flush_interval_frames:
            self.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._frames_since_flush = 0

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def _serialize(self, frame: DecisionFrame, latency_ms: float | None) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
            "camera_id": frame.camera_id,
            "frame_number": frame.frame_number,
            "video_time_s": round(frame.video_time_s, 6),
            "primary_fault": frame.primary_fault,
            "confidence": round(frame.confidence, 6),
            "secondary_symptoms": list(frame.secondary_symptoms),
            "suppressed_faults": list(frame.suppressed_faults),
            # Multi-label survivors (schema v2): each active fault and its
            # confidence, in DECISION_PRECEDENCE order.
            "faults": [
                {
                    "fault_type": fault.fault_type,
                    "confidence": round(fault.confidence, 6),
                }
                for fault in frame.faults
            ],
            "temporal_status": dict(frame.temporal_confirmation_status),
            "detectors": {
                obs.detector: {
                    "status": obs.status,
                    "is_candidate": obs.is_candidate,
                    "confidence": round(obs.confidence, 6),
                    "reason": obs.reason,
                    "error_message": obs.error_message,
                }
                for obs in frame.detectors
            },
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
        }


