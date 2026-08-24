"""Integration tests for the Phase 3 operational pipeline (main.py).

Covers:
- CLI parsing (parse_args) and the --config / flag override hierarchy.
- resolve_device(): precedence, CUDA fail-fast vs --allow-cpu-fallback.
- CameraWorker: lifecycle, bounded-queue backpressure (drop-oldest),
  live-stream lateness drops, tilt sub-sampling, event persistence,
  annotated snapshots, and graceful shutdown.
- CameraMetrics: snapshot/emit behavior.
- main() end-to-end (signal handlers and baseline loading are stubbed).

All CameraWorker tests use synthetic frames and mock detectors; no video
or baseline data is required. Output artifacts are written to pytest
``tmp_path`` only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import config
import detectors.tilt as tilt_module
import main as main_module
from main import (
    CameraMetrics,
    CameraWorker,
    _apply_overrides,
    _cli_overrides,
    _coerce_value,
    _log_resolved_configuration,
    _parse_camera_spec,
    parse_args,
    resolve_device,
)
from pipeline.event_store import EventStore
from pipeline.frame_logger import FrameLogger


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class _Result:
    def __init__(self, is_candidate: bool, confidence: float = 1.0, **metrics: float) -> None:
        self.is_candidate = is_candidate
        self.confidence = confidence
        # Raw measurands (e.g. total_loss_fraction, sharpness_ratio) so
        # _extract_detector_metrics can populate DetectorObservation.metrics
        # and the conditional co-occurrence predicates can read them (same
        # pattern as test_decision_engine._Result).
        for field, value in metrics.items():
            setattr(self, field, value)


class _Detector:
    def __init__(self, result: _Result | None = None, delay_s: float = 0.0) -> None:
        self.result = result
        self.delay_s = delay_s
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> object:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.result


def _detectors(**results: _Result) -> dict[str, _Detector]:
    """Detector map for the four valid names; pass keyword overrides."""
    defaults = {
        "low_light": _Result(False, 0.0),
        "tampering": _Result(False, 0.0),
        "blur": _Result(False, 0.0),
        "tilt": _Result(False, 0.0),
    }
    defaults.update(results)
    return {name: _Detector(result) for name, result in defaults.items()}


def _inject_frames(monkeypatch: pytest.MonkeyPatch, frames: list) -> None:
    """Route the worker's reader sub-thread to synthetic frames."""
    monkeypatch.setattr(main_module, "_make_frame_source", lambda src: list(frames))


@pytest.fixture
def tilt_state() -> Iterator[None]:
    """Save/restore detectors.tilt lazy-loading state across resolve_device tests."""
    saved = (tilt_module._device, tilt_module._disk_model, tilt_module._weights_path)
    yield
    tilt_module._device, tilt_module._disk_model, tilt_module._weights_path = saved

# --- CLI parsing & config overrides ------------------------------------------


def test_parse_args_all_flags() -> None:
    args = parse_args([
        "--camera", "cam1=rtsp://a", "--camera", "cam2=env:URL",
        "--session-id", "s1", "--device", "cuda:0",
        "--tilt-interval", "1.5", "--db", "x.db", "--frame-log", "fl.jsonl",
        "--system-log", "sl.jsonl", "--event-frames", "ef",
        "--config", "FRAME_QUEUE_CAPACITY=7", "--allow-cpu-fallback",
        "--log-level", "DEBUG",
    ])
    assert args.camera == ["cam1=rtsp://a", "cam2=env:URL"]
    assert args.session_id == "s1"
    assert args.device == "cuda:0"
    assert args.tilt_interval == 1.5
    assert args.db == Path("x.db")
    assert args.frame_log == Path("fl.jsonl")
    assert args.system_log == Path("sl.jsonl")
    assert args.event_frames == Path("ef")
    assert args.config == ["FRAME_QUEUE_CAPACITY=7"]
    assert args.allow_cpu_fallback is True
    assert args.log_level == "DEBUG"


def test_parse_args_defaults() -> None:
    args = parse_args(["--camera", "cam1=rtsp://a"])
    assert args.session_id == "default"
    assert args.device is None
    assert args.tilt_interval is None
    assert args.db is None
    assert args.frame_log is None
    assert args.system_log is None
    assert args.event_frames is None
    assert args.config == []
    assert args.allow_cpu_fallback is False
    assert args.log_level == "INFO"


def test_parse_camera_spec_credentials(monkeypatch) -> None:
    monkeypatch.setenv("RTSP_URL", "rtsp://user:secret@host/live")
    assert _parse_camera_spec("cam1=env:RTSP_URL") == ("cam1", "rtsp://user:secret@host/live")
    assert _parse_camera_spec("cam2=${RTSP_URL}") == ("cam2", "rtsp://user:secret@host/live")


def test_parse_camera_spec_keeps_url_query_string() -> None:
    assert _parse_camera_spec("cam1=rtsp://host/live?token=abc=1") == (
        "cam1", "rtsp://host/live?token=abc=1",
    )


def test_parse_camera_spec_errors(monkeypatch) -> None:
    with pytest.raises(ValueError, match="NAME=SOURCE"):
        _parse_camera_spec("no-equals")
    with pytest.raises(ValueError):
        _parse_camera_spec("bad..id=rtsp://x")
    with pytest.raises(ValueError):
        _parse_camera_spec("cam=")
    monkeypatch.delenv("MISSING_CAM_VAR", raising=False)
    with pytest.raises(ValueError, match="not set"):
        _parse_camera_spec("cam=env:MISSING_CAM_VAR")
    with pytest.raises(ValueError, match="not set"):
        _parse_camera_spec("cam=${MISSING_CAM_VAR}")


def test_coerce_value_types() -> None:
    assert _coerce_value("true") is True
    assert _coerce_value("FALSE") is False
    assert _coerce_value("42") == 42
    assert _coerce_value("0") == 0
    assert _coerce_value("1.5") == 1.5
    assert _coerce_value("rtsp://host/live") == "rtsp://host/live"


def test_cli_overrides_mapping_and_flags_beat_config() -> None:
    args = parse_args([
        "--config", "FRAME_QUEUE_CAPACITY=5",
        "--config", "TILT_SAMPLE_INTERVAL_SECONDS=9",
        "--tilt-interval", "2.0",
        "--db", "custom.db",
        "--allow-cpu-fallback",
    ])
    overrides = _cli_overrides(args)
    assert overrides["FRAME_QUEUE_CAPACITY"] == 5
    assert overrides["TILT_SAMPLE_INTERVAL_SECONDS"] == 2.0  # explicit flag wins
    assert overrides["EVENTS_DB_PATH"] == Path("custom.db")
    assert overrides["ALLOW_CPU_FALLBACK"] is True


def test_cli_overrides_reject_malformed_and_duplicate() -> None:
    bad = parse_args(["--camera", "cam1=rtsp://a", "--config", "NOVALUE"])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _cli_overrides(bad)
    dup = parse_args([
        "--camera", "cam1=rtsp://a",
        "--config", "FRAME_QUEUE_CAPACITY=1",
        "--config", "FRAME_QUEUE_CAPACITY=2",
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        _cli_overrides(dup)


def test_apply_overrides_mutates_config_and_rejects_unknown(monkeypatch) -> None:
    monkeypatch.setattr(config, "FRAME_QUEUE_CAPACITY", 3)
    _apply_overrides({"FRAME_QUEUE_CAPACITY": 7})
    assert config.FRAME_QUEUE_CAPACITY == 7
    with pytest.raises(ValueError, match="Unknown config key"):
        _apply_overrides({"BOGUS_KEY": 1})


def test_resolved_configuration_log_never_leaks_credentials(caplog) -> None:
    args = SimpleNamespace(session_id="default")
    with caplog.at_level(logging.INFO):
        _log_resolved_configuration(
            args, [("cam1", "rtsp://user:secret@host/live")], "cuda"
        )
    assert "secret" not in caplog.text


# --- resolve_device ----------------------------------------------------------


def test_resolve_device_precedence_cli_wins(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", "cpu")
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cpu")
    args = SimpleNamespace(device="cuda", allow_cpu_fallback=False)
    assert resolve_device(args) == "cuda"


def test_resolve_device_precedence_tilt_device_over_default(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", "cpu")
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cuda")
    args = SimpleNamespace(device=None, allow_cpu_fallback=False)
    assert resolve_device(args) == "cpu"
    assert tilt_module._device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resolve_device_defaults_to_cuda(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", None)
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cuda")
    args = SimpleNamespace(device=None, allow_cpu_fallback=False)
    assert resolve_device(args) == "cuda"
    assert tilt_module._device == torch.device("cuda")


def test_resolve_device_fails_fast_without_cuda(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", None)
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cuda")
    monkeypatch.setattr(config, "ALLOW_CPU_FALLBACK", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(device=None, allow_cpu_fallback=False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device(args)


def test_resolve_device_falls_back_to_cpu(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", None)
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cuda")
    monkeypatch.setattr(config, "ALLOW_CPU_FALLBACK", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(device=None, allow_cpu_fallback=True)
    assert resolve_device(args) == "cpu"
    assert tilt_module._device == torch.device("cpu")


def test_resolve_device_config_fallback_flag_also_works(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", None)
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cuda")
    monkeypatch.setattr(config, "ALLOW_CPU_FALLBACK", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(device=None, allow_cpu_fallback=False)
    assert resolve_device(args) == "cpu"


def test_resolve_device_rejects_invalid_device(monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(config, "TILT_DEVICE", None)
    monkeypatch.setattr(config, "DEFAULT_DEVICE", "cuda")
    args = SimpleNamespace(device="bogus", allow_cpu_fallback=False)
    with pytest.raises(ValueError, match="Invalid torch device"):
        resolve_device(args)


# --- CameraMetrics -----------------------------------------------------------


def test_metrics_snapshot_resets_interval_keeps_totals() -> None:
    metrics = CameraMetrics("cam1")
    metrics.record_processed(10.0)
    metrics.record_processed(20.0)
    metrics.record_dropped("late")
    metrics.record_dropped("overflow")
    metrics.record_event_persisted()
    snap = metrics.snapshot()
    assert snap["camera_id"] == "cam1"
    assert snap["processed_frames"] == 2
    assert snap["total_processed"] == 2
    assert snap["avg_latency_ms"] == 15.0
    assert snap["dropped_late"] == 1
    assert snap["dropped_overflow"] == 1
    assert snap["dropped_total"] == 2
    assert snap["events_persisted"] == 1
    assert snap["total_events_persisted"] == 1
    # Second snapshot: interval counters reset, totals persist.
    snap2 = metrics.snapshot()
    assert snap2["processed_frames"] == 0
    assert snap2["dropped_total"] == 0
    assert snap2["total_processed"] == 2
    assert snap2["avg_latency_ms"] is None


def test_metrics_emit_writes_jsonl(tmp_path: Path) -> None:
    metrics = CameraMetrics("cam1")
    metrics.record_processed(12.5)
    out = tmp_path / "system.jsonl"
    metrics.emit_to(out)
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["camera_id"] == "cam1"
    assert record["processed_frames"] == 1
    assert record["avg_latency_ms"] == 12.5
    assert record["schema_version"] == 1


def test_rotate_jsonl_rolls_and_keeps_bounded_backups(tmp_path: Path) -> None:
    path = tmp_path / "system.jsonl"
    # Four rollovers with backup_count=3 must leave .1 (newest) .. .3 and
    # drop the oldest generation entirely.
    for generation in range(4):
        path.write_text(f"generation-{generation}\n", encoding="utf-8")
        main_module._rotate_jsonl(path, max_bytes=10, backup_count=3)

    assert not path.exists()  # the active file was rolled away, not truncated
    backups = sorted(p.name for p in tmp_path.glob("system.jsonl.*"))
    assert backups == ["system.jsonl.1", "system.jsonl.2", "system.jsonl.3"]
    assert (tmp_path / "system.jsonl.1").read_text(encoding="utf-8") == "generation-3\n"
    assert (tmp_path / "system.jsonl.2").read_text(encoding="utf-8") == "generation-2\n"
    assert (tmp_path / "system.jsonl.3").read_text(encoding="utf-8") == "generation-1\n"


def test_rotate_jsonl_noop_below_cap(tmp_path: Path) -> None:
    path = tmp_path / "system.jsonl"
    path.write_text("small\n", encoding="utf-8")
    main_module._rotate_jsonl(path, max_bytes=10_000, backup_count=3)
    assert path.read_text(encoding="utf-8") == "small\n"  # untouched
    assert not list(tmp_path.glob("system.jsonl.*"))


def test_emit_to_rotates_system_log_at_size_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SYSTEM_LOG_MAX_BYTES", 120)
    monkeypatch.setattr(config, "SYSTEM_LOG_BACKUP_COUNT", 3)
    metrics = CameraMetrics("cam1")
    metrics.record_processed(1.0)
    path = tmp_path / "system.jsonl"
    for _ in range(10):
        metrics.emit_to(path)
    assert (tmp_path / "system.jsonl.1").exists()  # at least one roll
    assert path.exists()  # fresh active file continues the log
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["camera_id"] == "cam1"


def test_metrics_unknown_drop_reason_rejected() -> None:
    metrics = CameraMetrics("cam1")
    with pytest.raises(ValueError, match="Unknown drop reason"):
        metrics.record_dropped("bogus")


def test_metrics_observations_tally_per_detector() -> None:
    from pipeline.decision_engine import DetectorObservation

    metrics = CameraMetrics("cam1")
    decision = SimpleNamespace(detectors=(
        DetectorObservation("low_light", "ok", True, 0.7),
        DetectorObservation("tampering", "ok", False, 0.1),
        DetectorObservation("blur", "error", False, 0.0, "boom"),
        DetectorObservation("tilt", "skipped", False, 0.0),
    ))
    metrics.record_observations(decision)
    snap = metrics.snapshot()
    assert snap["candidate_counts"] == {"low_light": 1}
    assert snap["error_counts"] == {"blur": 1}
    assert snap["skipped_counts"] == {"tilt": 1}


# --- CameraWorker ------------------------------------------------------------


def test_worker_processes_frames_and_persists_events(tmp_path: Path, monkeypatch) -> None:
    frames = [(1, 0.0, _frame()), (2, 1.0, _frame()), (3, 2.0, _frame())]
    _inject_frames(monkeypatch, frames)
    dets = _detectors(low_light=_Result(True, 0.7))
    logger = FrameLogger(tmp_path / "frame_log.jsonl", interval_frames=1)
    db_path = tmp_path / "events.db"
    worker = CameraWorker(
        "cam1", "missing.mp4", detectors=dets, db_path=db_path, frame_logger=logger
    )
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert worker.processed_frames == 3
    assert worker.dropped_frames == 0

    # 3 consecutive candidate frames within the window -> one confirmed event.
    store = EventStore(db_path)
    rows = store.query_events()
    store.close()
    assert len(rows) == 1
    assert rows[0]["camera_id"] == "cam1"
    assert rows[0]["session_id"] == "default"
    assert rows[0]["fault_type"] == "low_light"
    assert rows[0]["status"] == "confirmed"

    records = [
        json.loads(line)
        for line in (tmp_path / "frame_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3
    assert records[0]["frame_number"] == 1
    assert records[0]["latency_ms"] is not None


def test_worker_records_annotated_snapshot_on_confirmed_event(tmp_path: Path, monkeypatch) -> None:
    frames = [(1, 0.0, _frame()), (2, 1.0, _frame()), (3, 2.0, _frame())]
    _inject_frames(monkeypatch, frames)
    dets = _detectors(low_light=_Result(True, 0.7))
    event_frames = tmp_path / "event_frames"
    worker = CameraWorker(
        "cam1", "missing.mp4", detectors=dets, event_frames_dir=event_frames
    )
    worker.start()
    worker.join(timeout=10)
    snapshots = list(event_frames.glob("*.jpg"))
    assert len(snapshots) == 1


def test_worker_annotates_each_event_with_its_own_fault(tmp_path: Path, monkeypatch) -> None:
    # Each saved snapshot must carry that event's own detector in the banner,
    # not the frame's fused primary (regression: every snapshot used to show
    # the same fused label).
    #
    # Scripted physical narrative. The mocks carry the raw measurands the real
    # detectors expose, so the conditional co-occurrence predicates actually
    # run end-to-end (metrics -> _pair_should_suppress -> suppression ->
    # tracker credit) instead of falling back to "unverifiable -> co-survive":
    # - tampering (frames 1-3): a large obstruction (total_loss_fraction 0.9)
    #   causally explains the simultaneous severe sharpness loss at frame 3
    #   (sharpness_ratio 0.05; the tampering->blur area-conservation predicate
    #   holds), so tampering suppresses blur there and blur gets no tracker
    #   credit; tilt is skipped by the blur execution gate (blur 0.9 >= 0.9).
    #   Tampering confirms on frame 3.
    # - blur (frames 5-7): with frame 3 not crediting, blur confirms on frame
    #   7 (3 of the 6 observed window frames at/above its floor), where
    #   tampering fires at 0.1 below its emission floor, so it stays a
    #   secondary symptom in blur's banner; tilt is again skipped by the blur
    #   gate.
    # - tilt (frames 11-13): blur fires at frame 13 just below its gate floor
    #   (0.8 < 0.9) so tilt remains measurable, and tilt suppresses blur via
    #   the always-margin relation (0.7 * DECISION_SUPPRESSION_MARGIN >= 0.8);
    #   tampering fires below its emission floor.
    # On every confirmation frame the other detectors still fire as
    # candidates, so the per-event secondary/suppressed banner lists below
    # are exercised.
    def _marker_frame(frame_number: int) -> np.ndarray:
        frame = _frame()
        frame[0, 0, 0] = frame_number
        return frame

    class _ScriptedDetector:
        """Returns a scripted result keyed by the frame-number marker."""

        def __init__(self, by_frame: dict[int, _Result]) -> None:
            self.by_frame = by_frame

        def __call__(self, frame: np.ndarray) -> _Result:
            return self.by_frame.get(int(frame[0, 0, 0]), _Result(False, 0.0))

    frames = [
        (1, 1.0, _marker_frame(1)),
        (2, 2.0, _marker_frame(2)),
        (3, 3.0, _marker_frame(3)),
        (4, 3.5, _marker_frame(4)),
        (5, 4.0, _marker_frame(5)),
        (6, 4.5, _marker_frame(6)),
        (7, 5.0, _marker_frame(7)),
        (8, 6.5, _marker_frame(8)),
        (9, 7.0, _marker_frame(9)),
        (10, 7.5, _marker_frame(10)),
        (11, 8.0, _marker_frame(11)),
        (12, 8.5, _marker_frame(12)),
        (13, 9.0, _marker_frame(13)),
    ]
    _inject_frames(monkeypatch, frames)
    dets = {
        "tampering": _ScriptedDetector({
            # Real obstruction: a large lost-structure area (total_loss_fraction)
            # that the conditional tampering->blur predicate reads to decide
            # whether the obstruction causally explains a simultaneous blur signal.
            1: _Result(True, 0.8, meaningful_block_fraction=0.9, total_loss_fraction=0.9),
            2: _Result(True, 0.8, meaningful_block_fraction=0.9, total_loss_fraction=0.9),
            3: _Result(True, 0.8, meaningful_block_fraction=0.9, total_loss_fraction=0.9),
            7: _Result(True, 0.1, meaningful_block_fraction=0.9, total_loss_fraction=0.2),  # below gate: secondary candidate, no suppression
            13: _Result(True, 0.1, meaningful_block_fraction=0.9, total_loss_fraction=0.2),
        }),
        "blur": _ScriptedDetector({
            # sharpness_ratio is the retention ratio (1.0 = unchanged). Frame 3's
            # near-full obstruction (loss 0.9) conservatively explains the severe
            # sharpness loss (1 - 0.05 = 0.95 <= 0.9 + TAMPERING_BLUR_AREA_SLACK)
            # so the predicate suppresses blur -> no tracker credit.
            3: _Result(True, 0.9, sharpness=150.0, sharpness_ratio=0.05),
            5: _Result(True, 0.9, sharpness=450.0, sharpness_ratio=0.12),
            6: _Result(True, 0.9, sharpness=450.0, sharpness_ratio=0.12),
            7: _Result(True, 0.9, sharpness=450.0, sharpness_ratio=0.12),
            13: _Result(True, 0.8, sharpness=600.0, sharpness_ratio=0.25),  # below blur gate floor (0.9): tilt measurable; suppressed by tilt -> no tracker credit
        }),
        "tilt": _ScriptedDetector({
            3: _Result(True, 0.7, median_shift_ratio=0.05, match_count=150),  # skipped by blur gate (blur 0.9 >= 0.9) -> no tracker credit
            7: _Result(True, 0.1, median_shift_ratio=0.01, match_count=40),  # unreachable while the blur gate (>= 0.9) skips tilt at frame 7
            11: _Result(True, 0.7, median_shift_ratio=0.05, match_count=150),
            12: _Result(True, 0.7, median_shift_ratio=0.05, match_count=150),
            13: _Result(True, 0.7, median_shift_ratio=0.05, match_count=150),
        }),
        "low_light": _Detector(_Result(False, 0.0)),
    }
    event_frames = tmp_path / "event_frames"

    annotations = []
    saved = []

    def _fake_annotate(frame, primary_fault, confidence,
                       secondary_symptoms=(), suppressed_faults=(), video_time_s=None,
                       faults=()):
        annotations.append(
            (primary_fault, confidence, tuple(secondary_symptoms), tuple(suppressed_faults))
        )
        return frame

    def _fake_save(frame, camera_id, fault_type, frame_number, video_time_s,
                   event_frames_dir=event_frames, max_total=config.EVENT_FRAMES_MAX_TOTAL):
        saved.append((fault_type, frame_number, video_time_s))
        return event_frames / f"{camera_id}_{fault_type}_{frame_number}.jpg"

    monkeypatch.setattr(main_module, "annotate_frame", _fake_annotate)
    monkeypatch.setattr(main_module, "save_annotated_frame", _fake_save)

    worker = CameraWorker(
        "cam1", "missing.mp4", detectors=dets, event_frames_dir=event_frames
    )
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()

    assert len(annotations) == 3
    assert len(saved) == 3
    assert sorted(fault for fault, _, _ in saved) == ["blur", "tampering", "tilt"]

    banner = {
        fault: (primary, confidence, secondary, suppressed)
        for (primary, confidence, secondary, suppressed), (fault, _, _)
        in zip(annotations, saved)
    }

    # The banner primary matches the event's own detector for every file.
    assert banner["blur"][0] == "blur"
    assert banner["tampering"][0] == "tampering"
    assert banner["tilt"][0] == "tilt"

    # Confidence is the event's own peak confidence, not the fused primary's.
    assert banner["blur"][1] == pytest.approx(0.9)
    assert banner["tampering"][1] == pytest.approx(0.8)
    assert banner["tilt"][1] == pytest.approx(0.7)

    # Secondary/suppressed lists are relative to each event's detector.
    assert banner["tampering"][2] == ()                     # secondary
    assert banner["tampering"][3] == ("blur",)              # suppressed (tilt was gated/skipped)
    assert banner["blur"][2] == ("tampering",)              # secondary (tampering fires below its floor at frame 7; tilt gated/skipped)
    assert banner["blur"][3] == ()
    assert banner["tilt"][2] == ("tampering",)              # secondary
    assert banner["tilt"][3] == ("blur",)                   # suppressed


def test_worker_backpressure_drops_oldest(monkeypatch) -> None:
    # Drop-oldest is the live-stream policy: the reader bursts ahead of the
    # slow detector and evicts the oldest queued frame to stay current.
    frames = [(i, float(i - 1), _frame()) for i in range(1, 101)]
    _inject_frames(monkeypatch, frames)
    dets = {"blur": _Detector(_Result(False, 0.0), delay_s=0.02)}
    worker = CameraWorker(
        "cam1", "rtsp://fake", detectors=dets, queue_capacity=3,
        is_live=True, max_lag_seconds=1_000_000,  # isolate overflow drops
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert worker.dropped_frames > 0
    # Every frame is accounted for: processed or evicted (drop-oldest).
    assert worker.processed_frames + worker.dropped_frames == 100


def test_worker_file_source_processes_every_frame_without_drops(monkeypatch) -> None:
    # File ingestion is lossless: the reader blocks for queue space instead
    # of evicting, so even a detector slower than the file's decode rate
    # still sees every frame (regression for the live-only drop policy).
    frames = [(i, float(i - 1), _frame()) for i in range(1, 101)]
    _inject_frames(monkeypatch, frames)
    dets = {"blur": _Detector(_Result(False, 0.0), delay_s=0.02)}
    worker = CameraWorker(
        "cam1", "missing.mp4", detectors=dets, queue_capacity=3
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert worker.processed_frames == 100
    assert worker.dropped_frames == 0


def test_worker_live_stream_drops_late_frames(monkeypatch) -> None:
    stale = time.monotonic() - 10.0
    fresh = time.monotonic()
    frames = [
        (1, 0.0, _frame(), stale),
        (2, 1.0, _frame(), fresh),
        (3, 2.0, _frame(), stale),
        (4, 3.0, _frame(), fresh),
    ]
    _inject_frames(monkeypatch, frames)
    dets = {"blur": _Detector(_Result(False, 0.0))}
    worker = CameraWorker(
        "cam1", "rtsp://fake", detectors=dets, is_live=True, max_lag_seconds=2.0
    )
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert worker.processed_frames == 2
    assert worker.dropped_frames == 2


def test_worker_tilt_subsampling_skips_tilt(monkeypatch) -> None:
    # Exactly 3 frames (queue capacity is 3) so none can be evicted by
    # drop-oldest backpressure: the test stays deterministic.
    frames = [(1, 0.0, _frame()), (2, 1.0, _frame()), (3, 2.0, _frame())]
    _inject_frames(monkeypatch, frames)
    tilt_det = _Detector(_Result(False, 0.0))
    dets = {"low_light": _Detector(_Result(False, 0.0)), "tilt": tilt_det}
    worker = CameraWorker(
        "cam1", "missing.mp4", detectors=dets, tilt_sample_interval_seconds=2.0
    )
    worker.start()
    worker.join(timeout=10)
    assert worker.processed_frames == 3
    assert tilt_det.calls == 2  # runs at t=0.0 and t=2.0
    snap = worker.metrics.snapshot()
    assert snap["skipped_counts"] == {"tilt": 1}  # t=1.0 skipped, not observed



def test_worker_graceful_shutdown_stops_promptly(tmp_path: Path, monkeypatch) -> None:
    def infinite_source():
        i = 0
        while True:
            yield i, float(i), _frame()
            i += 1

    monkeypatch.setattr(main_module, "_make_frame_source", lambda src: infinite_source())
    dets = {"blur": _Detector(_Result(False, 0.0), delay_s=0.01)}
    logger = FrameLogger(tmp_path / "frame_log.jsonl", interval_frames=10)
    db_path = tmp_path / "events.db"
    worker = CameraWorker(
        "cam1", "rtsp://fake", detectors=dets, is_live=True,
        db_path=db_path, frame_logger=logger,
    )
    worker.start()
    time.sleep(0.3)
    started = time.monotonic()
    worker.stop()
    worker.join(timeout=config.SHUTDOWN_TIMEOUT_SECONDS + 2)
    elapsed = time.monotonic() - started
    assert not worker.is_alive()
    assert elapsed < config.SHUTDOWN_TIMEOUT_SECONDS
    # run() finally flushed and closed the logger and event store.
    assert logger._file is None
    with pytest.raises(sqlite3.ProgrammingError):
        worker._event_store._conn.execute("SELECT 1")


def test_metrics_loop_exits_when_all_workers_finish(tmp_path: Path, monkeypatch) -> None:
    _inject_frames(monkeypatch, [(1, 0.0, _frame())])
    worker = CameraWorker(
        "cam1", "missing.mp4", detectors={"blur": _Detector(_Result(False, 0.0))}
    )
    worker.start()
    worker.join(timeout=10)
    stop_event = threading.Event()
    main_module._run_metrics_loop([worker], tmp_path / "system.jsonl", stop_event)
    assert not stop_event.is_set()  # loop returned on its own once workers finished


def test_main_end_to_end(tmp_path: Path, monkeypatch, tilt_state) -> None:
    monkeypatch.setattr(main_module, "_setup_logging", lambda level, log_file=None: None)
    monkeypatch.setattr(main_module, "_install_signal_handlers", lambda stop_event: None)
    monkeypatch.setattr(
        main_module, "_build_detectors",
        lambda cid, bdir: {"blur": _Detector(_Result(False, 0.0))},
    )
    _inject_frames(monkeypatch, [(1, 0.0, _frame()), (2, 1.0, _frame()), (3, 2.0, _frame())])
    monkeypatch.setattr(config, "METRICS_LOG_INTERVAL_SECONDS", 0.1)

    source = tmp_path / "input.mp4"
    source.write_bytes(b"not-a-real-video")  # exists -> passes the pre-flight check
    exit_code = main_module.main([
        "--camera", f"cam1={source}",
        "--session-id", "itest",
        "--db", str(tmp_path / "events.db"),
        "--frame-log", str(tmp_path / "frame_log.jsonl"),
        "--system-log", str(tmp_path / "system.jsonl"),
        "--event-frames", str(tmp_path / "event_frames"),
    ])
    assert exit_code == 0

    # Final metrics snapshots are always emitted.
    metrics_lines = [
        json.loads(line)
        for line in (tmp_path / "system.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert metrics_lines
    assert metrics_lines[-1]["camera_id"] == "cam1"
    assert metrics_lines[-1]["total_processed"] == 3

    # Per-camera frame log and event DB were produced.
    assert (tmp_path / "frame_log_cam1.jsonl").exists()
    assert (tmp_path / "events.db").exists()




