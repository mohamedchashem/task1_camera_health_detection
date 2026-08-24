"""Unit tests for Phase 2 persistence: EventStore, FrameLogger, annotate.

No video or baseline data is required. All artifacts are written to
pytest ``tmp_path`` directories, never to the repository's data dirs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from config import EVENT_FRAMES_MAX_TOTAL
from pipeline.annotate import annotate_frame, build_annotation_lines, save_annotated_frame
from pipeline.decision_engine import (
    DETECTOR_STATUS_ERROR,
    DETECTOR_STATUS_OK,
    DETECTOR_STATUS_SKIPPED,
    ConfirmedFault,
    DecisionFrame,
    DetectorObservation,
    Fault,
)
from pipeline.event_store import EventStore, build_event_key
from pipeline.frame_logger import FrameLogger


def _event(
    *,
    camera_id: str = "cam1",
    session_id: str = "default",
    fault_type: str = "blur",
    status: str = "confirmed",
    started_frame: int = 0,
    started_time_s: float = 0.0,
    ended_frame: int | None = None,
    ended_time_s: float | None = None,
    peak: float = 0.8,
    rate: float = 1.0,
) -> ConfirmedFault:
    return ConfirmedFault(
        camera_id=camera_id,
        fault_type=fault_type,
        status=status,
        started_frame=started_frame,
        started_time_s=started_time_s,
        ended_frame=ended_frame,
        ended_time_s=ended_time_s,
        peak_confidence=peak,
        window_positive_rate=rate,
        session_id=session_id,
    )


def _decision_frame(
    number: int,
    *,
    primary: str | None = None,
    error_detector: bool = False,
    skipped_detector: bool = False,
    secondary: tuple[str, ...] = (),
    suppressed: tuple[str, ...] = (),
    faults: tuple[Fault, ...] = (),
) -> DecisionFrame:
    observations = [DetectorObservation("low_light", DETECTOR_STATUS_OK, False, 0.1)]
    if error_detector:
        observations.append(
            DetectorObservation("tilt", DETECTOR_STATUS_ERROR, False, 0.0, "boom")
        )
    elif skipped_detector:
        observations.append(
            DetectorObservation("tilt", DETECTOR_STATUS_SKIPPED, False, 0.0)
        )
    observations.append(
        DetectorObservation(
            "blur", DETECTOR_STATUS_OK, primary == "blur",
            0.8 if primary == "blur" else 0.1,
        )
    )
    return DecisionFrame(
        camera_id="cam1",
        frame_number=number,
        video_time_s=float(number),
        primary_fault=primary,
        confidence=0.8 if primary else 0.0,
        secondary_symptoms=secondary,
        suppressed_faults=suppressed,
        temporal_confirmation_status={"blur": "confirmed"} if primary == "blur" else {},
        detectors=tuple(observations),
        faults=faults,
    )


# --- EventStore --------------------------------------------------------------


def test_event_store_creates_schema_and_version(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    tables = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='confirmed_faults'"
    ).fetchall()
    assert len(tables) == 1
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 1
    store.close()


def test_event_store_parameterized_insert_stores_fields(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = _event(
        session_id="sess-9",
        fault_type="tampering",
        status="confirmed",
        started_frame=100,
        started_time_s=10.0,
        peak=0.6,
        rate=1.0,
    )
    assert store.insert_confirmed_fault(event) is True

    row = store.query_events()[0]
    assert row["camera_id"] == "cam1"
    assert row["session_id"] == "sess-9"
    assert row["fault_type"] == "tampering"
    assert row["status"] == "confirmed"
    assert row["started_frame"] == 100
    assert row["started_time_s"] == pytest.approx(10.0)
    assert row["ended_frame"] is None
    assert row["ended_time_s"] is None
    assert row["peak_confidence"] == pytest.approx(0.6)
    assert row["window_positive_rate"] == pytest.approx(1.0)
    assert row["event_key"] == "cam1|sess-9|tampering|100|confirmed"
    store.close()


def test_event_store_insert_is_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    event = _event()
    assert store.insert_confirmed_fault(event) is True
    assert store.insert_confirmed_fault(event) is False  # duplicate -> ignored
    assert len(store.query_events()) == 1
    store.close()


def test_session_id_participates_in_idempotency_key(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    first = _event(session_id="sess-a")
    second = _event(session_id="sess-b")  # same camera/fault/frame/status, new session
    assert store.insert_confirmed_fault(first) is True
    assert store.insert_confirmed_fault(second) is True  # distinct key
    assert len(store.query_events()) == 2
    store.close()


def test_event_store_multi_fault_same_frame_do_not_collide(tmp_path: Path) -> None:
    # Multi-label: two faults confirmed on the exact same frame timestamp
    # must not collide in SQLite. fault_type is part of the composite
    # idempotency key, so each event gets its own row (first write wins per
    # key, but the keys differ).
    store = EventStore(tmp_path / "events.db")
    blur = _event(fault_type="blur", started_frame=7, started_time_s=1.0)
    tilt = _event(fault_type="tilt", started_frame=7, started_time_s=1.0)
    assert blur.started_frame == tilt.started_frame
    assert blur.started_time_s == tilt.started_time_s
    assert build_event_key(blur) != build_event_key(tilt)
    assert store.insert_confirmed_fault(blur) is True
    assert store.insert_confirmed_fault(tilt) is True
    assert store.insert_confirmed_fault(blur) is False  # same fault -> idempotent
    assert len(store.query_events()) == 2
    store.close()


def test_confirmed_and_cleared_rows_coexist(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    assert store.insert_confirmed_fault(_event(status="confirmed")) is True
    assert (
        store.insert_confirmed_fault(
            _event(status="cleared", ended_frame=50, ended_time_s=5.0)
        )
        is True
    )
    assert len(store.query_events()) == 2
    store.close()


def test_event_store_query_filters_and_ordering(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.insert_confirmed_fault(_event(fault_type="blur", started_time_s=1.0))
    store.insert_confirmed_fault(_event(fault_type="tampering", started_time_s=2.0))
    store.insert_confirmed_fault(_event(camera_id="cam2", fault_type="blur", started_time_s=3.0))

    assert len(store.query_events(camera_id="cam1")) == 2
    assert len(store.query_events(fault_type="blur")) == 2
    since = store.query_events(since_time_s=2.5)
    assert len(since) == 1
    assert since[0]["camera_id"] == "cam2"
    assert len(store.query_events(limit=1)) == 1
    assert [r["started_time_s"] for r in store.query_events()] == [1.0, 2.0, 3.0]
    store.close()

# --- FrameLogger -------------------------------------------------------------


def test_frame_logger_writes_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "frame_log.jsonl"
    logger = FrameLogger(path, interval_frames=1, flush_interval_frames=1, retention_days=7)
    logger.open()
    logger.write_frame(_decision_frame(1, primary="blur"), latency_ms=3.5)
    logger.close()

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == 2
    assert record["camera_id"] == "cam1"
    assert record["frame_number"] == 1
    assert record["primary_fault"] == "blur"
    assert record["faults"] == []  # schema v2: no active faults on this frame
    assert record["confidence"] == pytest.approx(0.8)
    assert record["latency_ms"] == pytest.approx(3.5)
    assert record["detectors"]["blur"]["is_candidate"] is True


def test_frame_logger_serializes_multi_fault_view(tmp_path: Path) -> None:
    # Multi-label: primary is the top-precedence survivor, secondary are the
    # remaining survivors, suppressed are the candidates that lost fusion,
    # and faults is the explicit multi-label active tuple (schema v2).
    path = tmp_path / "frame_log.jsonl"
    logger = FrameLogger(path, interval_frames=1, flush_interval_frames=1, retention_days=7)
    logger.open()
    logger.write_frame(
        _decision_frame(
            2,
            primary="tampering",
            secondary=("blur",),
            suppressed=("low_light",),
            faults=(Fault("tampering", 0.85), Fault("blur", 0.62)),
        )
    )
    logger.close()

    record = json.loads(
        [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][0]
    )
    assert record["primary_fault"] == "tampering"
    assert record["confidence"] == pytest.approx(0.8)
    assert record["secondary_symptoms"] == ["blur"]
    assert record["suppressed_faults"] == ["low_light"]
    # Each surviving fault is serialized with fault_type + confidence, in
    # DECISION_PRECEDENCE order (tampering ranks above blur).
    assert record["faults"] == [
        {"fault_type": "tampering", "confidence": pytest.approx(0.85)},
        {"fault_type": "blur", "confidence": pytest.approx(0.62)},
    ]


def test_frame_logger_sampling_logs_only_interval_frames(tmp_path: Path) -> None:
    path = tmp_path / "frame_log.jsonl"
    logger = FrameLogger(path, interval_frames=2, flush_interval_frames=100, retention_days=7)
    logger.open()
    for number in range(1, 6):
        logger.write_frame(_decision_frame(number))
    logger.close()

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["frame_number"] for r in records] == [1, 3, 5]


def test_frame_logger_error_always_logged_skipped_not(tmp_path: Path) -> None:
    path = tmp_path / "frame_log.jsonl"
    logger = FrameLogger(path, interval_frames=5, flush_interval_frames=100, retention_days=7)
    logger.open()
    logger.write_frame(_decision_frame(1))  # 1st call -> sampled
    # 2nd call: tilt skipped (sub-sampling) but off the interval -> NOT always logged
    logger.write_frame(_decision_frame(2, skipped_detector=True))
    logger.write_frame(_decision_frame(3, error_detector=True))  # error -> always logged
    logger.close()

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["frame_number"] for r in records] == [1, 3]
    assert records[1]["detectors"]["tilt"]["status"] == "error"


def test_frame_logger_periodic_flush_reaches_disk(tmp_path: Path) -> None:
    path = tmp_path / "frame_log.jsonl"
    logger = FrameLogger(path, interval_frames=1, flush_interval_frames=2, retention_days=7)
    logger.open()
    logger.write_frame(_decision_frame(1))
    logger.write_frame(_decision_frame(2))  # second frame triggers the periodic flush

    # Without closing, the two buffered lines must already be on disk.
    with path.open(encoding="utf-8") as fh:
        assert len([line for line in fh if line.strip()]) == 2

    logger.write_frame(_decision_frame(3))
    logger.close()
    with path.open(encoding="utf-8") as fh:
        assert len([line for line in fh if line.strip()]) == 3


def test_frame_logger_rotates_when_size_limit_reached(tmp_path: Path) -> None:
    path = tmp_path / "frame_log.jsonl"
    max_bytes = 2048
    logger = FrameLogger(
        path,
        interval_frames=1,
        flush_interval_frames=100,
        retention_days=7,
        max_bytes=max_bytes,
    )
    logger.open()
    for i in range(1, 61):
        logger.write_frame(_decision_frame(i))
    logger.close()

    rotated = list(tmp_path.glob("frame_log.jsonl.*"))
    assert rotated  # size-based rolling triggered within the run

    # The active file never exceeds the cap: each roll starts a fresh file
    # before the record that would cross the limit is written.
    assert path.exists()
    assert path.stat().st_size <= max_bytes

    # Every line, active or rolled, is still a complete JSON record — a
    # rotation never splits a line.
    for log_file in [path, *rotated]:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            assert line.strip()
            json.loads(line)


# --- annotate -----------------------------------------------------------------


def test_annotate_frame_returns_overlaid_copy() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = annotate_frame(
        frame,
        "tampering",
        0.6,
        secondary_symptoms=("low_light",),
        suppressed_faults=("blur",),
        video_time_s=1.5,
    )
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert annotated.sum() > 0  # text was actually drawn
    assert frame.sum() == 0  # the input frame is untouched

    no_fault = annotate_frame(frame, None, 0.0)
    assert no_fault.shape == frame.shape


def test_annotate_frame_stacks_multiple_active_faults() -> None:
    # Multi-label: each active fault in DecisionFrame.faults renders as its
    # own stacked FAULT line (precedence order), each with its confidence.
    faults = (Fault("tampering", 0.85), Fault("blur", 0.62))
    lines = build_annotation_lines(
        "tampering",
        0.85,
        faults=faults,
        secondary_symptoms=("blur",),
        suppressed_faults=("low_light",),
        video_time_s=1.5,
    )
    assert lines == [
        "FAULT: tampering (conf=0.85)",
        "FAULT: blur (conf=0.62)",
        "suppressed: low_light",
        "t=1.50s",
    ]

    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    annotated = annotate_frame(
        frame,
        "tampering",
        0.85,
        faults=faults,
        suppressed_faults=("low_light",),
        video_time_s=1.5,
    )
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert annotated.sum() > 0  # text was actually drawn
    assert frame.sum() == 0  # the input frame is untouched


def test_save_annotated_frame_writes_snapshot(tmp_path: Path) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    out_path = save_annotated_frame(
        frame, "cam1", "blur", 10, 1.0, event_frames_dir=tmp_path
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_bounded_ring_evicts_oldest_first(tmp_path: Path) -> None:
    budget = 3
    for i in range(1, 6):
        out_path = save_annotated_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "cam1", "blur", i, float(i),
            event_frames_dir=tmp_path, max_total=budget,
        )
        if i == 1:
            old = 1_000_000_000
            os.utime(out_path, (old, old))  # make the first snapshot the oldest

    names = {p.name for p in tmp_path.iterdir()}
    assert len(names) == budget
    assert not any("_f000001_" in name for name in names)  # oldest evicted
    assert not any("_f000002_" in name for name in names)
    assert any("_f000003_" in name for name in names)
    assert any("_f000004_" in name for name in names)
    assert any("_f000005_" in name for name in names)


def test_bounded_ring_caps_total_at_config_max(tmp_path: Path) -> None:
    assert EVENT_FRAMES_MAX_TOTAL == 200
    for i in range(205):  # exceed the ring budget
        save_annotated_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "cam1", "blur", i, float(i),
            event_frames_dir=tmp_path,  # max_total defaults to the config value
        )
    assert len(list(tmp_path.iterdir())) == 200

