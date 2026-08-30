"""Tests for the capture-time degraded-baseline gate (Approach A).

A baseline whose computed ``quality_warnings`` list is non-empty (dark,
blurry, or lacking structure) must not be silently persisted: the
tampering detector's ``evaluate()`` would then report ``degraded_baseline``
(confidence 0.0) on every frame until a usable baseline is captured — the
recurring silent-failure bug this gate eliminates. These tests pin the
three behaviors:

* warnings present, no acknowledgment -> refusal, nothing written;
* warnings present, acknowledged -> persisted with the acknowledgment
  recorded on the JSON record;
* no warnings -> unchanged clean-capture behavior, no new fields.

Frames are injected by monkeypatching ``_windowed_frames`` so the tests
never touch real video; the baseline output directory is redirected to a
pytest tmp dir so nothing leaks into ``data/baselines/``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import pipeline.capture_baseline as capture_module
from pipeline.capture_baseline import DegradedBaselineRefusedError, capture_baseline

# The reference frame is grabbed at BASELINE_CAPTURE_SECONDS / 2 (>= 1.5 s)
# and the loop stops at BASELINE_CAPTURE_SECONDS (3.0 s), so the fake
# window must span at least that far in "elapsed" time for a write to have
# a reference frame to persist.
_ELAPSED_SECONDS = [0.0, 1.0, 2.0, 3.0, 4.0]


def _fake_dark_window():
    """Yield black frames: triggers dark + blur + structure warnings."""
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    for elapsed in _ELAPSED_SECONDS:
        yield frame.copy(), float(elapsed)


def _fake_clean_window():
    """Yield bright, high-detail frames: passes all quality gates."""
    rng = np.random.default_rng(0)
    frame = rng.integers(140, 256, size=(96, 128, 3), dtype=np.uint8).astype(np.uint8)
    for elapsed in _ELAPSED_SECONDS:
        yield frame.copy(), float(elapsed)


def _monkeypatch_window(monkeypatch: pytest.MonkeyPatch, window) -> None:
    monkeypatch.setattr(capture_module, "_windowed_frames", lambda source: iter(window()))


@pytest.fixture
def isolated_baselines_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the capture output directory to a per-test tmp path."""
    target = tmp_path / "baselines"
    monkeypatch.setattr(capture_module, "BASELINES_DIR", target)
    return target

def test_degraded_without_acknowledgment_refuses_and_writes_nothing(
    isolated_baselines_dir, monkeypatch
) -> None:
    _monkeypatch_window(monkeypatch, _fake_dark_window)

    with pytest.raises(DegradedBaselineRefusedError) as excinfo:
        capture_baseline("cam_test", "fake://source")

    message = str(excinfo.value)
    assert "cam_test" in message
    # Every warning is listed verbatim in the message.
    assert excinfo.value.quality_warnings
    for warning in excinfo.value.quality_warnings:
        assert warning in message
    # The message explains the consequence for tampering detection.
    assert "tampering detection inoperative" in message
    assert "degraded_baseline" in message

    # Nothing was persisted (the dir may exist because the function mkdirs
    # it before computing quality, but it must be empty).
    assert list(isolated_baselines_dir.glob("*")) == []


def test_degraded_with_acknowledgment_persists_with_new_field(
    isolated_baselines_dir, monkeypatch
) -> None:
    _monkeypatch_window(monkeypatch, _fake_dark_window)

    record = capture_baseline("cam_test", "fake://source", acknowledge_degraded=True)

    assert record["quality_warnings"]
    assert record["degraded_acknowledged"] is True
    assert isinstance(record["degraded_acknowledged_at"], (int, float))

    json_record = json.loads((isolated_baselines_dir / "cam_test.json").read_text())
    assert json_record["degraded_acknowledged"] is True
    assert json_record["degraded_acknowledged_at"] == json_record["captured_at"]
    assert (isolated_baselines_dir / "cam_test.jpg").exists()
    assert (isolated_baselines_dir / "cam_test_edges.png").exists()


def test_clean_capture_persists_unchanged_without_acknowledgment_fields(
    isolated_baselines_dir, monkeypatch
) -> None:
    _monkeypatch_window(monkeypatch, _fake_clean_window)

    record = capture_baseline("cam_test", "fake://source")

    assert record["quality_warnings"] == []
    assert "degraded_acknowledged" not in record
    assert "degraded_acknowledged_at" not in record

    json_record = json.loads((isolated_baselines_dir / "cam_test.json").read_text())
    assert json_record["quality_warnings"] == []
    assert "degraded_acknowledged" not in json_record
    assert "degraded_acknowledged_at" not in json_record


def test_cli_degraded_without_acknowledgment_exits_nonzero_and_writes_nothing(
    isolated_baselines_dir, monkeypatch, caplog
) -> None:
    _monkeypatch_window(monkeypatch, _fake_dark_window)

    with caplog.at_level(logging.ERROR, logger="pipeline.capture_baseline"):
        code = capture_module.main(["cam_test", "fake://source"])

    assert code == 1
    assert list(isolated_baselines_dir.glob("*")) == []
    assert any(
        "Refusing to persist degraded baseline" in record.message
        for record in caplog.records
    )


def test_cli_degraded_with_acknowledgment_exits_zero_and_persists(
    isolated_baselines_dir, monkeypatch
) -> None:
    _monkeypatch_window(monkeypatch, _fake_dark_window)

    code = capture_module.main(["--acknowledge-degraded", "cam_test", "fake://source"])

    assert code == 0
    json_record = json.loads((isolated_baselines_dir / "cam_test.json").read_text())
    assert json_record["degraded_acknowledged"] is True


def test_cli_clean_capture_exits_zero_and_persists_unchanged(
    isolated_baselines_dir, monkeypatch
) -> None:
    _monkeypatch_window(monkeypatch, _fake_clean_window)

    code = capture_module.main(["cam_test", "fake://source"])

    assert code == 0
    json_record = json.loads((isolated_baselines_dir / "cam_test.json").read_text())
    assert json_record["quality_warnings"] == []
    assert "degraded_acknowledged" not in json_record

