"""Phase 3 end-to-end operational pipeline (Step 2A: CLI + device resolution).

The operational entry point: CLI parsing, compute-device resolution, and
the per-camera worker pipeline (CameraWorker). ``main()`` applies
``--config`` overrides, validates the configuration, resolves the compute
device (fail-fast, no silent CPU fallback), starts one CameraWorker per
camera, and blocks until the workers finish. Signal handling requests a
cooperative shutdown, and a metrics loop periodically reports per-camera
throughput and faults to the system log.

CLI example::

    python main.py --camera cam1=rtsp://user:pass@host/stream \\
                   --camera cam2=env:RTSP_URL

Device policy: ``--device`` > ``TILT_DEVICE`` > ``DEFAULT_DEVICE``
(``"cuda"``). A CUDA device that is unavailable aborts startup unless
``--allow-cpu-fallback`` (or ``ALLOW_CPU_FALLBACK``) explicitly opts into
CPU. The resolved device is applied via ``tilt.configure(device=...)``;
there is no mid-run device switching. Camera credentials passed through
``env:VAR`` / ``${VAR}`` are never logged.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

import config
import detectors.tilt as tilt
from config import validate_config
from detectors import blur as blur_detector, brightness, tampering
from pipeline.annotate import annotate_frame, save_annotated_frame
from pipeline.decision_engine import (
    DETECTOR_STATUS_ERROR,
    DETECTOR_STATUS_OK,
    DETECTOR_STATUS_SKIPPED,
    EVENT_STATUS_CONFIRMED,
    ConfirmedFault,
    DecisionEngine,
    DecisionFrame,
    split_candidates_for_primary,
)
from pipeline.event_store import EventStore
from pipeline.file_reader import read_frames_from_file
from pipeline.frame_logger import FrameLogger
from pipeline.paths import validate_camera_id
from pipeline.stream_reader import read_frames

logger = logging.getLogger(__name__)

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the operational CLI (see the module docstring for the device policy)."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Camera health monitoring: run the four fault detectors against one "
            "or more cameras through the full fault-detection pipeline."
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=SOURCE",
        help=(
            "Camera to monitor; repeatable. NAME must be a safe id "
            "([A-Za-z0-9_-]+). SOURCE is a live RTSP URL or a local video "
            "file. Credentials may be injected via env:VAR or ${VAR} and "
            "are never logged."
        ),
    )
    parser.add_argument("--session-id", default="default",
                        help="Stream session id; part of the event idempotency key.")
    parser.add_argument("--device", default=None,
                        help="Explicit torch device ('cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--tilt-interval", type=float, default=None,
                        help="Override TILT_SAMPLE_INTERVAL_SECONDS.")
    parser.add_argument("--db", type=Path, default=None, help="Override EVENTS_DB_PATH.")
    parser.add_argument("--frame-log", type=Path, default=None, help="Override FRAME_LOG_PATH.")
    parser.add_argument("--system-log", type=Path, default=None, help="Override SYSTEM_LOG_PATH.")
    parser.add_argument("--event-frames", type=Path, default=None, help="Override EVENT_FRAMES_DIR.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config.py constant before validate_config(); repeatable.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Fall back to CPU when the resolved CUDA device is unavailable.",
    )
    parser.add_argument("--log-level", choices=_LOG_LEVELS, default="INFO",
                        help="Console log level.")
    return parser.parse_args(argv)


def _is_live_source(source: str) -> bool:
    """True for live RTSP streams; everything else is treated as a file."""
    return source.lower().startswith("rtsp://")


def _resolve_credentials(source: str) -> str:
    """Resolve env:VAR / ${VAR} credential references in a camera source.

    The resolved value is used as-is and must never be logged.
    """
    if source.startswith("env:"):
        var = source[len("env:"):]
        if not var:
            raise ValueError("Invalid env: credential reference: empty variable name.")
        if var not in os.environ:
            raise ValueError(f"Environment variable {var!r} for a camera source is not set.")
        return os.environ[var]
    if source.startswith("${"):
        if not source.endswith("}"):
            raise ValueError("Invalid ${VAR} credential reference in a camera source.")
        var = source[2:-1]
        if not var:
            raise ValueError("Invalid ${VAR} credential reference: empty variable name.")
        if var not in os.environ:
            raise ValueError(f"Environment variable {var!r} for a camera source is not set.")
        return os.environ[var]
    return source


def _parse_camera_spec(spec: str) -> tuple[str, str]:
    """Split a 'NAME=SOURCE' camera spec into (camera_id, resolved_source).

    Splits on the FIRST '=' so URLs that contain '=' keep their query
    strings intact. Error messages never echo SOURCE (it may contain
    credentials).
    """
    if "=" not in spec:
        raise ValueError("Invalid --camera spec: expected NAME=SOURCE.")
    name, source = spec.split("=", 1)
    camera_id = validate_camera_id(name)
    if not source:
        raise ValueError("Invalid --camera spec: SOURCE must not be empty.")
    return camera_id, _resolve_credentials(source)


def _coerce_value(raw: str) -> object:
    """Best-effort scalar conversion for --config KEY=VALUE values."""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Map CLI flags onto config.py constants; explicit flags beat --config."""
    overrides: dict[str, object] = {}
    for kv in args.config:
        if "=" not in kv:
            raise ValueError(f"Invalid --config override {kv!r}: expected KEY=VALUE.")
        key, raw = kv.split("=", 1)
        if not key:
            raise ValueError("Invalid --config override: empty KEY.")
        if key in overrides:
            raise ValueError(f"Duplicate --config override for {key!r}.")
        overrides[key] = _coerce_value(raw)
    if args.tilt_interval is not None:
        overrides["TILT_SAMPLE_INTERVAL_SECONDS"] = args.tilt_interval
    if args.db is not None:
        overrides["EVENTS_DB_PATH"] = args.db
    if args.frame_log is not None:
        overrides["FRAME_LOG_PATH"] = args.frame_log
    if args.system_log is not None:
        overrides["SYSTEM_LOG_PATH"] = args.system_log
    if args.event_frames is not None:
        overrides["EVENT_FRAMES_DIR"] = args.event_frames
    if args.allow_cpu_fallback:
        overrides["ALLOW_CPU_FALLBACK"] = True
    return overrides


def _apply_overrides(overrides: Mapping[str, object]) -> None:
    """Apply config overrides to the config module; unknown keys fail fast."""
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config key {key!r} in --config override.")
        setattr(config, key, value)


def _load_baseline_record(camera_id: str, baselines_dir: Path) -> dict:
    """Load the JSON baseline record (low-light + blur values) for a camera."""
    json_path = baselines_dir / f"{camera_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"No baseline record for {camera_id!r} at {json_path}. "
            "Run pipeline.capture_baseline first."
        )
    record = json.loads(json_path.read_text(encoding="utf-8"))
    required = ("lowlight_dark_pixel_ratio", "blur_baseline_sharpness")
    missing = set(required) - set(record)
    if missing:
        raise ValueError(f"Baseline record {json_path} is missing keys: {sorted(missing)}.")
    return record


def _load_baseline_edges(camera_id: str, baselines_dir: Path) -> np.ndarray:
    """Load the stable edge baseline image used by the tampering detector."""
    edges_path = baselines_dir / f"{camera_id}_edges.png"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"No edge baseline for {camera_id!r} at {edges_path}. "
            "Run pipeline.capture_baseline first."
        )
    edges = cv2.imread(str(edges_path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise RuntimeError(f"Failed to load edge baseline image at {edges_path}.")
    return edges


def _load_baseline_frame(camera_id: str, baselines_dir: Path) -> np.ndarray:
    """Load the baseline reference frame (tilt features derive from it)."""
    image_path = baselines_dir / f"{camera_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(
            f"No baseline reference frame for {camera_id!r} at {image_path}. "
            "Run pipeline.capture_baseline first."
        )
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to load baseline reference frame at {image_path}.")
    return frame


def _build_detectors(
    camera_id: str, baselines_dir: str | Path
) -> dict[str, Callable[[np.ndarray], object]]:
    """Bind each detector to a camera's saved baseline data.

    The tilt detector's DISK baseline features are computed once per
    camera at startup (the one-time cost). The returned callables match
    the DecisionEngine contract: ``callable(frame)`` returns an object
    with ``is_candidate`` and ``confidence``.
    """
    baselines_dir = Path(baselines_dir)
    record = _load_baseline_record(camera_id, baselines_dir)
    edges = _load_baseline_edges(camera_id, baselines_dir)
    baseline_frame = _load_baseline_frame(camera_id, baselines_dir)

    baseline_keypoints, baseline_descriptors = tilt.extract_features(baseline_frame)
    baseline_shape = baseline_frame.shape[:2]

    dark_ratio = float(record["lowlight_dark_pixel_ratio"])
    sharpness = float(record["blur_baseline_sharpness"])

    return {
        "low_light": lambda f: brightness.evaluate(f, dark_ratio),
        "tampering": lambda f: tampering.evaluate(f, edges),
        "blur": lambda f: blur_detector.evaluate(f, sharpness),
        "tilt": lambda f: tilt.evaluate(f, baseline_keypoints, baseline_descriptors, baseline_shape),
    }


def _make_frame_source(source: str) -> Iterable:
    """Pick the reader for a source: live RTSP streams vs local video files."""
    if _is_live_source(source):
        return read_frames(source)
    return read_frames_from_file(Path(source))


def _frame_log_path(camera_id: str, base: str | Path) -> Path:
    """Per-camera frame log file so multiple cameras never share one JSONL."""
    base = Path(base)
    return base.with_name(f"{base.stem}_{camera_id}{base.suffix}")


def _ensure_output_directories() -> None:
    """Create parent directories for the persistent outputs before workers start."""
    for raw in (
        config.EVENTS_DB_PATH,
        config.FRAME_LOG_PATH,
        config.SYSTEM_LOG_PATH,
        config.EVENT_FRAMES_DIR,
    ):
        Path(raw).parent.mkdir(parents=True, exist_ok=True)


_LOG_HANDLERS: list[logging.Handler] = []


def _setup_logging(level: int, log_file: str | Path | None = None) -> None:
    """Configure the root logger: stderr stream plus a rotating file handler.

    Replaces ``logging.basicConfig`` in ``main()`` so the operational entry
    point also keeps an on-disk application log (config.APP_LOG_FILE). Only
    handlers installed by this function are removed on re-entry, so repeated
    ``main()`` calls do not stack duplicate handlers and pytest's own log
    capture handlers are left untouched.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in _LOG_HANDLERS:
        root.removeHandler(handler)
    _LOG_HANDLERS.clear()

    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    _LOG_HANDLERS.append(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.APP_LOG_MAX_BYTES,
            backupCount=config.APP_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        _LOG_HANDLERS.append(file_handler)


def _rotate_jsonl(path: str | Path, max_bytes: int | None, backup_count: int) -> None:
    """Roll ``path`` by size using RotatingFileHandler-style numbered backups.

    When the active file has reached ``max_bytes`` it becomes ``path.1``
    (newest), previous backups shift up (``.1`` -> ``.2``, ...), and backups
    beyond ``backup_count`` are deleted. The caller then appends to a fresh
    active file. ``max_bytes=None`` disables rotation.
    """
    log_path = Path(path)
    if max_bytes is None or not log_path.exists():
        return
    try:
        size = log_path.stat().st_size
    except OSError:
        return
    if size < max_bytes:
        return
    for index in range(backup_count - 1, 0, -1):
        older = log_path.with_name(f"{log_path.name}.{index + 1}")
        current = log_path.with_name(f"{log_path.name}.{index}")
        if older.exists():
            older.unlink()
        if current.exists():
            current.rename(older)
    newest = log_path.with_name(f"{log_path.name}.1")
    if newest.exists():
        newest.unlink()
    log_path.rename(newest)


class CameraMetrics:
    """Interval-based metrics accumulator for one camera.

    The reader and worker threads record counters; the main thread
    snapshots them (snapshot-and-reset per interval, with cumulative
    totals kept alongside) and emits them to system.jsonl. All record
    and snapshot operations are lock-protected.
    """
    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._interval_started = time.monotonic()
        self._processed = 0
        self._total_processed = 0
        self._dropped_late = 0
        self._dropped_overflow = 0
        self._total_dropped = 0
        self._latency_sum_ms = 0.0
        self._candidates: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._skipped: Counter[str] = Counter()
        self._events_persisted = 0
        self._total_events_persisted = 0

    def record_processed(self, latency_ms: float) -> None:
        """Record one processed frame with its engine latency in ms."""
        with self._lock:
            self._processed += 1
            self._total_processed += 1
            self._latency_sum_ms += latency_ms

    def record_dropped(self, reason: str) -> None:
        """Record a dropped frame: 'late' (stale live frame) or 'overflow'."""
        with self._lock:
            if reason == "late":
                self._dropped_late += 1
            elif reason == "overflow":
                self._dropped_overflow += 1
            else:
                raise ValueError(f"Unknown drop reason {reason!r}.")
            self._total_dropped += 1

    def record_observations(self, decision: DecisionFrame) -> None:
        """Tally per-detector candidate/error/skipped statuses."""
        with self._lock:
            for obs in decision.detectors:
                if obs.status == DETECTOR_STATUS_OK and obs.is_candidate:
                    self._candidates[obs.detector] += 1
                elif obs.status == DETECTOR_STATUS_ERROR:
                    self._errors[obs.detector] += 1
                elif obs.status == DETECTOR_STATUS_SKIPPED:
                    self._skipped[obs.detector] += 1

    def record_event_persisted(self) -> None:
        with self._lock:
            self._events_persisted += 1
            self._total_events_persisted += 1

    @property
    def total_processed(self) -> int:
        with self._lock:
            return self._total_processed

    @property
    def total_dropped(self) -> int:
        with self._lock:
            return self._total_dropped

    def snapshot(self) -> dict:
        """Return the interval counters (resetting them) plus cumulative totals."""
        now = time.monotonic()
        with self._lock:
            interval = now - self._interval_started
            processed = self._processed
            record = {
                "schema_version": 1,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                "camera_id": self._camera_id,
                "interval_seconds": round(interval, 3),
                "fps": round(processed / interval, 3) if processed and interval > 0 else 0.0,
                "avg_latency_ms": round(self._latency_sum_ms / processed, 3) if processed else None,
                "processed_frames": processed,
                "dropped_late": self._dropped_late,
                "dropped_overflow": self._dropped_overflow,
                "dropped_total": self._dropped_late + self._dropped_overflow,
                "candidate_counts": dict(self._candidates),
                "error_counts": dict(self._errors),
                "skipped_counts": dict(self._skipped),
                "events_persisted": self._events_persisted,
                "total_processed": self._total_processed,
                "total_dropped": self._total_dropped,
                "total_events_persisted": self._total_events_persisted,
            }
            self._interval_started = now
            self._processed = 0
            self._dropped_late = 0
            self._dropped_overflow = 0
            self._latency_sum_ms = 0.0
            self._candidates.clear()
            self._errors.clear()
            self._skipped.clear()
            self._events_persisted = 0
            return record

    def emit_to(self, path: str | Path) -> None:
        """Append one JSON snapshot line to ``path`` (caller is the main thread).

        Rolls ``path`` by size first (SYSTEM_LOG_MAX_BYTES /
        SYSTEM_LOG_BACKUP_COUNT) so long runs cannot grow system.jsonl
        without bound.
        """
        record = self.snapshot()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_jsonl(path, config.SYSTEM_LOG_MAX_BYTES, config.SYSTEM_LOG_BACKUP_COUNT)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


class CameraWorker(threading.Thread):
    """One camera, one thread: a reader sub-thread feeds a bounded queue
    that this worker drains through the decision engine.

    Frame flow::

        source reader (sub-thread) -> bounded queue (capacity
        FRAME_QUEUE_CAPACITY) -> worker loop -> DecisionEngine ->
        EventStore / FrameLogger / annotated snapshots

    Backpressure: live streams use drop-oldest eviction (bounded memory,
    stays current); a live frame dequeued after more than
    MAX_PROCESSING_LAG_SECONDS is dropped as stale. File sources are
    lossless: the reader blocks for queue space instead of evicting, so
    every frame is processed exactly once. Drops are counted in
    ``dropped_frames`` (0 for file sources).
    """

    # Bounds how quickly the loop notices the stop event during shutdown.
    _POLL_INTERVAL_SECONDS = 0.25

    def __init__(
        self,
        camera_id: str,
        source: str,
        *,
        detectors: Mapping[str, Callable[[np.ndarray], object]] | None = None,
        session_id: str = "default",
        baselines_dir: str | Path | None = None,
        is_live: bool | None = None,
        queue_capacity: int | None = None,
        max_lag_seconds: float | None = None,
        tilt_sample_interval_seconds: float | None = None,
        db_path: str | Path | None = None,
        frame_logger: FrameLogger | None = None,
        event_frames_dir: str | Path | None = None,
        metrics: CameraMetrics | None = None,
    ) -> None:
        super().__init__(name=f"worker-{camera_id}", daemon=True)
        self.camera_id = validate_camera_id(camera_id)
        self._source = source
        self._is_live = _is_live_source(source) if is_live is None else bool(is_live)
        self._baselines_dir = (
            Path(baselines_dir) if baselines_dir is not None else config.BASELINES_DIR
        )
        self._queue_capacity = (
            queue_capacity if queue_capacity is not None else config.FRAME_QUEUE_CAPACITY
        )
        self._max_lag_seconds = (
            max_lag_seconds if max_lag_seconds is not None else config.MAX_PROCESSING_LAG_SECONDS
        )
        self._tilt_sample_interval_seconds = (
            tilt_sample_interval_seconds
            if tilt_sample_interval_seconds is not None
            else config.TILT_SAMPLE_INTERVAL_SECONDS
        )
        self._db_path = Path(db_path) if db_path is not None else None
        self._frame_logger = frame_logger
        self._event_frames_dir = (
            Path(event_frames_dir) if event_frames_dir is not None else None
        )
        self._event_store: EventStore | None = None

        # Baseline binding happens here (main thread) so baseline/DISK load
        # failures fail fast before any thread starts.
        if detectors is None:
            detectors = _build_detectors(camera_id, self._baselines_dir)
        self._detectors = dict(detectors)
        self._engine = DecisionEngine(camera_id, self._detectors, session_id=session_id)

        self._queue: queue.Queue[tuple | None] = queue.Queue(
            maxsize=max(self._queue_capacity, 1)
        )
        self._stop = threading.Event()
        self._non_tilt_faults = set(self._detectors) - {"tilt"}
        self._last_tilt_run = float("-inf")

        # Metrics are recorded by the reader (overflow evictions) and the
        # worker (processed/late/observations); the main thread emits them.
        self.metrics = metrics or CameraMetrics(camera_id)

    @property
    def dropped_frames(self) -> int:
        """Total frames dropped by overflow eviction or live-stream lateness."""
        return self.metrics.total_dropped

    @property
    def processed_frames(self) -> int:
        """Total frames successfully evaluated by the decision engine."""
        return self.metrics.total_processed

    def stop(self) -> None:
        """Request cooperative shutdown; the worker stops after the current frame."""
        self._stop.set()

    def run(self) -> None:
        reader = None
        try:
            if self._frame_logger is not None:
                self._frame_logger.open()
            if self._db_path is not None:
                self._event_store = EventStore(self._db_path)
            reader = threading.Thread(
                target=self._reader_loop, name=f"reader-{self.camera_id}", daemon=True
            )
            reader.start()
            self._process_loop()
        except Exception:
            logger.exception("CameraWorker %r crashed", self.camera_id)
        finally:
            if reader is not None:
                reader.join(timeout=config.SHUTDOWN_TIMEOUT_SECONDS)
            self._flush_pending_events()
            if self._frame_logger is not None:
                self._frame_logger.close()
            if self._event_store is not None:
                self._event_store.close()


    def _reader_loop(self) -> None:
        """Feed the bounded queue from the frame source; evict-oldest on overflow."""
        frames = _make_frame_source(self._source)
        try:
            for item in frames:
                if self._stop.is_set():
                    break
                frame_number, video_time_s, frame, captured_at = self._unpack_frame_item(item)
                self._push_frame((frame_number, video_time_s, frame, captured_at))
        except Exception:
            logger.exception("Reader for camera %r failed; ending the stream", self.camera_id)
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
            # End-of-stream marker so the worker loop finishes naturally.
            while not self._stop.is_set():
                try:
                    self._queue.put(None, timeout=0.5)
                    break
                except queue.Full:
                    continue

    @staticmethod
    def _unpack_frame_item(item: object) -> tuple[int, float, np.ndarray, float]:
        """Accept (n, t, frame) or (n, t, frame, captured_at) items.

        The optional 4th element is the wall-clock capture time (monotonic)
        used by the live-stream lateness check; tests inject it to make the
        check deterministic. Without it, time.monotonic() at read time is
        used.
        """
        if isinstance(item, tuple) and len(item) == 4:
            frame_number, video_time_s, frame, captured_at = item
        else:
            frame_number, video_time_s, frame = item
            captured_at = time.monotonic()
        return frame_number, video_time_s, frame, captured_at

    def _push_frame(self, item: tuple[int, float, np.ndarray, float]) -> None:
        """Enqueue one frame, applying the source-appropriate full-queue policy.

        Live streams stay real-time (drop-oldest, bounded latency); file
        sources are processed losslessly (block until the worker frees a
        slot, so no frame is ever evicted).
        """
        if self._is_live:
            self._push_frame_drop_oldest(item)
        else:
            self._push_frame_blocking(item)

    def _push_frame_blocking(self, item: tuple[int, float, np.ndarray, float]) -> None:
        """Lossless file ingestion: wait for queue space instead of dropping.

        The bounded queue still caps memory; detector throughput becomes the
        natural rate limiter and every frame is processed exactly once. The
        put polls the stop event so shutdown is not blocked by a full queue.
        """
        while True:
            try:
                self._queue.put(item, timeout=self._POLL_INTERVAL_SECONDS)
                return
            except queue.Full:
                if self._stop.is_set():
                    return

    def _push_frame_drop_oldest(self, item: tuple[int, float, np.ndarray, float]) -> None:
        """Live-stream backpressure: evict the oldest frame to stay current."""
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                if self._stop.is_set():
                    return
                try:
                    self._queue.get_nowait()  # evict the oldest frame
                except queue.Empty:
                    continue
                self.metrics.record_dropped("overflow")

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self._POLL_INTERVAL_SECONDS)
            except queue.Empty:
                continue
            try:
                if item is None:
                    break
                self._process_item(item)
            except Exception:
                logger.exception("Error processing frame for camera %r", self.camera_id)
            finally:
                self._queue.task_done()

    def _process_item(self, item: tuple[int, float, np.ndarray, float]) -> None:
        frame_number, video_time_s, frame, captured_at = item

        # Lateness drop (live streams): the frame waited too long in the queue.
        if self._is_live and (time.monotonic() - captured_at) > self._max_lag_seconds:
            self.metrics.record_dropped("late")
            logger.warning(
                "Dropping late frame %d (t=%.3f) for camera %r: queued %.2fs",
                frame_number, video_time_s, self.camera_id,
                time.monotonic() - captured_at,
            )
            return

        enabled_faults = self._enabled_faults_for(video_time_s)
        started = time.monotonic()
        decision = self._engine.process_frame(
            frame, frame_number, video_time_s, enabled_faults=enabled_faults
        )
        latency_ms = (time.monotonic() - started) * 1000.0
        self.metrics.record_processed(latency_ms)
        self.metrics.record_observations(decision)

        if self._frame_logger is not None:
            self._frame_logger.write_frame(decision, latency_ms)
        self._persist_events(decision, frame)

    def _enabled_faults_for(self, video_time_s: float) -> set[str] | None:
        """Tilt sub-sampling: with an interval > 0 tilt runs at most once per
        interval; the rest yield 'skipped' tilt observations that the engine
        excludes from the confirmation tracker."""
        if self._tilt_sample_interval_seconds <= 0:
            return None  # every frame, all detectors (the GPU policy)
        if video_time_s - self._last_tilt_run >= self._tilt_sample_interval_seconds:
            self._last_tilt_run = video_time_s
            return None
        return self._non_tilt_faults


    def _persist_events(self, decision: DecisionFrame, frame: np.ndarray) -> None:
        for event in self._engine.drain_events():
            self._persist_event(event, decision, frame)

    def _flush_pending_events(self) -> None:
        """Persist any events still pending when the loop exits (defensive)."""
        for event in self._engine.drain_events():
            self._persist_event(event, None, None)

    def _persist_event(
        self,
        event: ConfirmedFault,
        decision: DecisionFrame | None,
        frame: np.ndarray | None,
    ) -> None:
        if self._event_store is not None:
            try:
                if self._event_store.insert_confirmed_fault(event):
                    self.metrics.record_event_persisted()
            except Exception:
                logger.exception("Failed to persist event for camera %r", self.camera_id)
        if (
            decision is not None
            and frame is not None
            and event.status == EVENT_STATUS_CONFIRMED
            and self._event_frames_dir is not None
        ):
            try:
                self._save_annotated_frame(event, decision, frame)
            except Exception:
                logger.exception("Failed to save annotated frame for camera %r", self.camera_id)

    def _save_annotated_frame(
        self, event: ConfirmedFault, decision: DecisionFrame, frame: np.ndarray
    ) -> None:
        # Multiple detectors can confirm on the same frame. The banner must
        # describe the event being written (event.fault_type), not the frame's
        # fused primary, which is shared by every event of that frame.
        candidates = [
            obs.detector
            for obs in decision.detectors
            if obs.status == DETECTOR_STATUS_OK and obs.is_candidate
        ]
        secondary, suppressed = split_candidates_for_primary(event.fault_type, candidates)
        annotated = annotate_frame(
            frame,
            event.fault_type,
            event.peak_confidence,
            secondary_symptoms=secondary,
            suppressed_faults=suppressed,
            video_time_s=decision.video_time_s,
            # Multi-label: render every active fault on this frame stacked
            # vertically (each with its own confidence); the event's own
            # fault is one of them.
            faults=decision.faults,
        )
        save_annotated_frame(
            annotated,
            self.camera_id,
            event.fault_type,
            decision.frame_number,
            decision.video_time_s,
            event_frames_dir=self._event_frames_dir,
        )


def resolve_device(args: argparse.Namespace) -> str:
    """Resolve the compute device and apply it to the tilt detector.

    Precedence: ``--device`` > ``TILT_DEVICE`` > ``DEFAULT_DEVICE``
    (``"cuda"``). A CUDA device that is unavailable aborts startup
    (RuntimeError) unless fallback to CPU is explicitly allowed via
    ``--allow-cpu-fallback`` or ``ALLOW_CPU_FALLBACK``. An explicit
    ``cpu`` device is always honored: it is an explicit opt-in, not a
    silent fallback. Returns the resolved device string.
    """
    resolved = args.device or config.TILT_DEVICE or config.DEFAULT_DEVICE
    try:
        device = torch.device(resolved)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device {resolved!r}.") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device {resolved!r}: expected 'cpu' or a CUDA device.")

    allow_fallback = bool(args.allow_cpu_fallback) or bool(config.ALLOW_CPU_FALLBACK)

    if device.type == "cuda" and not torch.cuda.is_available():
        if allow_fallback:
            resolved = "cpu"
            logger.warning(
                "CUDA device %r is unavailable; falling back to CPU "
                "(--allow-cpu-fallback / ALLOW_CPU_FALLBACK).",
                device,
            )
        else:
            raise RuntimeError(
                f"Device {resolved!r} requested but CUDA is unavailable. "
                "Install a CUDA-enabled torch build, or pass "
                "--allow-cpu-fallback to explicitly run on CPU."
            )

    tilt.configure(device=resolved)
    logger.info("Tilt detector configured for device: %s", resolved)
    return resolved


def _log_resolved_configuration(
    args: argparse.Namespace, cameras: list[tuple[str, str]], device: str
) -> None:
    """Log the resolved configuration. Sources are never printed raw, so
    credentials resolved from env:VAR / ${VAR} are never logged."""
    descriptions = [
        f"{camera_id} ({'live stream' if _is_live_source(source) else 'video file'})"
        for camera_id, source in cameras
    ]
    logger.info("Resolved configuration:")
    logger.info("  cameras: %s", ", ".join(descriptions))
    logger.info("  session_id: %s", args.session_id)
    logger.info("  device: %s", device)
    logger.info("  allow_cpu_fallback: %s", bool(config.ALLOW_CPU_FALLBACK))
    logger.info("  events db: %s", config.EVENTS_DB_PATH)
    logger.info("  frame log: %s", config.FRAME_LOG_PATH)
    logger.info("  system log: %s", config.SYSTEM_LOG_PATH)
    logger.info("  app log: %s", config.APP_LOG_FILE)
    logger.info("  event frames dir: %s", config.EVENT_FRAMES_DIR)
    logger.info("  frame queue capacity: %s", config.FRAME_QUEUE_CAPACITY)
    logger.info("  max processing lag s: %s", config.MAX_PROCESSING_LAG_SECONDS)
    logger.info("  tilt sample interval s: %s", config.TILT_SAMPLE_INTERVAL_SECONDS)
    logger.info("  metrics log interval s: %s", config.METRICS_LOG_INTERVAL_SECONDS)
    logger.info("  shutdown timeout s: %s", config.SHUTDOWN_TIMEOUT_SECONDS)


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """Install SIGINT/SIGTERM handlers that request a cooperative shutdown.

    The handlers only set the master stop event; all cleanup runs in the
    main thread afterwards. Signals a platform does not support are logged
    and skipped (never fatal).
    """

    def _on_signal(signum: int, _frame: object) -> None:
        logger.warning("Received signal %d; initiating graceful shutdown.", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError) as exc:
            logger.warning("Cannot install handler for signal %d: %s", sig, exc)


def _run_metrics_loop(
    workers: Sequence[CameraWorker],
    metrics_path: str | Path,
    stop_event: threading.Event,
) -> None:
    """Emit a CameraMetrics snapshot to ``metrics_path`` every
    METRICS_LOG_INTERVAL_SECONDS until the stop event is set or every
    worker has finished (natural end of a file source)."""
    next_emit = time.monotonic() + config.METRICS_LOG_INTERVAL_SECONDS
    while not stop_event.is_set():
        if stop_event.wait(0.5):
            break
        if all(not w.is_alive() for w in workers):
            break
        now = time.monotonic()
        if now >= next_emit:
            for worker in workers:
                worker.metrics.emit_to(metrics_path)
            if torch.cuda.is_available():
                # VRAM discipline: after each metrics interval, return any
                # CACHED (unused) CUDA allocation blocks to the driver so long
                # runs keep VRAM compact. Live tensors are never affected --
                # empty_cache only frees blocks the allocator is no longer
                # using.
                torch.cuda.empty_cache()
            next_emit = now + config.METRICS_LOG_INTERVAL_SECONDS


def _graceful_shutdown(
    workers: Sequence[CameraWorker], stop_event: threading.Event
) -> None:
    """Stop every worker cooperatively and wait up to
    SHUTDOWN_TIMEOUT_SECONDS for them to release resources.

    Each worker's run() finally closes its FrameLogger (which flushes it)
    and its EventStore, and drains any pending ConfirmedFault events;
    joining with the deadline enforces the timeout here. Workers that miss
    the deadline are logged as needing cleanup (degraded, not fatal).
    """
    logger.info("Graceful shutdown: requesting workers to stop.")
    stop_event.set()
    deadline = time.monotonic() + config.SHUTDOWN_TIMEOUT_SECONDS
    for worker in workers:
        worker.stop()
    for worker in workers:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            worker.join(timeout=remaining)
        if worker.is_alive():
            logger.warning(
                "CameraWorker %r did not stop within %ss; resources may need cleanup.",
                worker.camera_id, config.SHUTDOWN_TIMEOUT_SECONDS,
            )
        else:
            logger.info(
                "Camera worker stopped: %s (processed=%d, dropped=%d)",
                worker.camera_id, worker.processed_frames, worker.dropped_frames,
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse CLI, apply overrides, validate, resolve device, log."""
    args = parse_args(argv)

    try:
        _apply_overrides(_cli_overrides(args))
        validate_config()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _setup_logging(
        getattr(logging, args.log_level, logging.INFO),
        config.APP_LOG_FILE,
    )

    try:
        cameras = [_parse_camera_spec(spec) for spec in (args.camera or [])]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not cameras:
        print("error: at least one --camera NAME=SOURCE is required", file=sys.stderr)
        return 2

    try:
        device = resolve_device(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Fail fast on missing local sources before any thread starts.
    for camera_id, source in cameras:
        if not _is_live_source(source) and not Path(source).exists():
            print(f"error: camera {camera_id!r}: video file source not found.", file=sys.stderr)
            return 2

    _log_resolved_configuration(args, cameras, device)
    _ensure_output_directories()

    # Register signal handlers BEFORE any heavy startup work: CameraWorker
    # construction loads each camera's baselines and runs the DISK model, so
    # Ctrl+C / SIGTERM during that load must request a cooperative shutdown
    # (via the stop event) instead of interrupting startup with a bare
    # KeyboardInterrupt and leaking half-built state.
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    workers: list[CameraWorker] = []
    try:
        for camera_id, source in cameras:
            workers.append(
                CameraWorker(
                    camera_id,
                    source,
                    session_id=args.session_id,
                    db_path=config.EVENTS_DB_PATH,
                    frame_logger=FrameLogger(
                        _frame_log_path(camera_id, config.FRAME_LOG_PATH)
                    ),
                    event_frames_dir=config.EVENT_FRAMES_DIR,
                )
            )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    started_workers: list[CameraWorker] = []
    try:
        for worker in workers:
            worker.start()
            started_workers.append(worker)
            logger.info("Camera worker started: %s", worker.camera_id)
    except Exception:
        # A worker failed to launch mid-loop (e.g. its reader thread died
        # immediately or resource allocation failed). Gracefully stop any
        # workers that already started so no thread is left orphaned, then
        # re-raise for the caller to handle.
        logger.exception(
            "Camera worker startup failed; stopping the %d worker(s) already started.",
            len(started_workers),
        )
        _graceful_shutdown(started_workers, stop_event)
        raise

    try:
        _run_metrics_loop(workers, config.SYSTEM_LOG_PATH, stop_event)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received; shutting down.")
        stop_event.set()
    finally:
        _graceful_shutdown(workers, stop_event)

    # Final metrics snapshot per camera so short runs are fully accounted for.
    for worker in workers:
        worker.metrics.emit_to(config.SYSTEM_LOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

