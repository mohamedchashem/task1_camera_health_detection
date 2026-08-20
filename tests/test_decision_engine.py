"""Unit tests for the decision layer.

Covers the four approved contracts:
1. Fusion & precedence: type-precedence ranking and cross-trigger suppression.
2. Confidence gating: sub-0.3 suppressors cannot override lower-priority faults.
3. Temporal confirmation: pending -> confirmed -> cleared transitions,
   minimum frame thresholds, and the event-gap spam guard.
4. Fault isolation: a raising detector becomes an "error" observation,
   triggers backoff, and never crashes the remaining detectors or fusion.

No video or baseline data is required -- detectors are mocked.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DECISION_SUPPRESSOR_MIN_CONFIDENCE
from pipeline.decision_engine import (
    DETECTOR_STATUS_ERROR,
    DETECTOR_STATUS_OK,
    DETECTOR_STATUS_SKIPPED,
    EVENT_STATUS_CLEARED,
    EVENT_STATUS_CONFIRMED,
    TEMPORAL_STATUS_CONFIRMED,
    TEMPORAL_STATUS_PENDING,
    ConfirmationTracker,
    DecisionEngine,
    DetectorObservation,
    fuse_observations,
    resolve_primary_fault,
    split_candidates_for_primary,
)

GATE = DECISION_SUPPRESSOR_MIN_CONFIDENCE


def _obs(
    detector: str,
    *,
    status: str = DETECTOR_STATUS_OK,
    is_candidate: bool = False,
    confidence: float = 0.0,
) -> DetectorObservation:
    return DetectorObservation(detector, status, is_candidate, confidence)


class _Result:
    def __init__(self, is_candidate: bool, confidence: float = 1.0) -> None:
        self.is_candidate = is_candidate
        self.confidence = confidence


class _Detector:
    def __init__(self, result: object | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> object:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _make_tracker(**overrides) -> ConfirmationTracker:
    defaults = dict(
        window_seconds=3.0,
        min_positive_ratio=0.5,
        min_window_frames=3,
        min_event_gap_seconds=10.0,
    )
    defaults.update(overrides)
    return ConfirmationTracker(camera_id="cam1", **defaults)


def _feed(tracker: ConfirmationTracker, frames: list[tuple[int, float, bool]]) -> list:
    """Feed (frame_number, video_time_s, is_candidate) frames for low_light."""
    events = []
    for frame_number, t, candidate in frames:
        candidates = {"low_light": 0.7} if candidate else {}
        events.extend(tracker.update(candidates, ["low_light"], frame_number, t))
    return events


# --- 1. Fusion & precedence ------------------------------------------------


def test_no_candidates_fuses_to_no_primary() -> None:
    obs = (_obs("low_light"), _obs("tampering"), _obs("blur"), _obs("tilt"))
    assert fuse_observations(obs) == (None, (), (), 0.0)


def test_tampering_suppresses_low_light_blur_tilt() -> None:
    obs = (
        _obs("tampering", is_candidate=True, confidence=0.4),
        _obs("low_light", is_candidate=True, confidence=0.9),
        _obs("blur", is_candidate=True, confidence=0.8),
        _obs("tilt", is_candidate=True, confidence=0.5),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "tampering"
    assert secondary == ()
    assert suppressed == ("low_light", "tilt", "blur")
    assert confidence == pytest.approx(0.4)


def test_low_light_suppresses_blur_and_tilt() -> None:
    obs = (
        _obs("low_light", is_candidate=True, confidence=0.6),
        _obs("blur", is_candidate=True, confidence=0.9),
        _obs("tilt", is_candidate=True, confidence=0.5),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "low_light"
    assert secondary == ()
    assert suppressed == ("tilt", "blur")


@pytest.mark.parametrize("other", ["low_light", "blur", "tilt"])
def test_tampering_suppresses_other_fault(other: str) -> None:
    primary, secondary, suppressed = resolve_primary_fault(
        {"tampering": GATE + 0.1, other: 0.9}
    )
    assert primary == "tampering"
    assert secondary == ()
    assert suppressed == (other,)


@pytest.mark.parametrize("other", ["blur", "tilt"])
def test_low_light_suppresses_other_fault(other: str) -> None:
    primary, secondary, suppressed = resolve_primary_fault(
        {"low_light": GATE + 0.1, other: 0.9}
    )
    assert primary == "low_light"
    assert secondary == ()
    assert suppressed == (other,)


def test_precedence_beats_confidence() -> None:
    # Tilt now outranks blur: a detected tilt causally explains blur's
    # apparent sharpness loss, so tilt wins even when blur's confidence is
    # higher (precedence beats confidence).
    obs = (
        _obs("tilt", is_candidate=True, confidence=0.5),
        _obs("blur", is_candidate=True, confidence=0.9),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "tilt"
    assert secondary == ()
    assert suppressed == ("blur",)
    assert confidence == pytest.approx(0.5)


def test_tilt_suppresses_blur_when_both_fire() -> None:
    # Rotation resampling lowers Laplacian sharpness, so a real tilt event
    # also trips the blur detector; the causal cause must win the primary.
    obs = (
        _obs("blur", is_candidate=True, confidence=1.0),
        _obs("tilt", is_candidate=True, confidence=0.32),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "tilt"
    assert secondary == ()
    assert suppressed == ("blur",)
    assert confidence == pytest.approx(0.32)


def test_weak_candidate_alone_still_becomes_primary() -> None:
    weak = max(GATE - 0.1, 0.0)
    primary, secondary, suppressed, confidence = fuse_observations(
        (_obs("low_light", is_candidate=True, confidence=weak),)
    )
    assert primary == "low_light"
    assert secondary == ()
    assert suppressed == ()
    assert confidence == pytest.approx(weak)


def test_error_detector_never_contributes_candidate() -> None:
    obs = (
        _obs("blur", is_candidate=True, confidence=0.8),
        _obs("low_light", status=DETECTOR_STATUS_ERROR, is_candidate=True, confidence=0.9),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "blur"
    assert secondary == ()
    assert suppressed == ()


def test_split_candidates_for_primary_partitions_around_fixed_primary() -> None:
    # Three detectors firing on the same frame, in precedence order.
    candidates = ("tampering", "blur", "tilt")

    # tampering causally explains blur + tilt (per DECISION_SUPPRESSION_MAP).
    secondary, suppressed = split_candidates_for_primary("tampering", candidates)
    assert secondary == ()
    assert suppressed == ("blur", "tilt")

    # blur and tilt explain nothing: all others become secondary symptoms.
    secondary, suppressed = split_candidates_for_primary("blur", candidates)
    assert secondary == ("tampering", "tilt")
    assert suppressed == ()

    # tilt now causally explains blur (rotation resampling reduces sharpness).
    secondary, suppressed = split_candidates_for_primary("tilt", candidates)
    assert secondary == ("tampering",)
    assert suppressed == ("blur",)

    # A primary not among the candidates still excludes itself and applies
    # its own suppression map to the remaining candidates.
    secondary, suppressed = split_candidates_for_primary("low_light", candidates)
    assert secondary == ("tampering",)
    assert suppressed == ("blur", "tilt")

# --- 2. Confidence gating --------------------------------------------------


def test_weak_low_light_does_not_suppress_blur() -> None:
    weak = max(GATE - 0.1, 0.0)
    obs = (
        _obs("low_light", is_candidate=True, confidence=weak),
        _obs("blur", is_candidate=True, confidence=0.9),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "blur"
    assert secondary == ("low_light",)
    assert suppressed == ()
    assert confidence == pytest.approx(0.9)


def test_weak_tampering_does_not_suppress_strong_low_light() -> None:
    weak = max(GATE - 0.1, 0.0)
    obs = (
        _obs("tampering", is_candidate=True, confidence=weak),
        _obs("low_light", is_candidate=True, confidence=0.9),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "low_light"
    assert secondary == ("tampering",)
    assert suppressed == ()


def test_suppressor_at_exact_gate_is_active() -> None:
    # The gate is inclusive: confidence == GATE counts as an active suppressor.
    obs = (
        _obs("low_light", is_candidate=True, confidence=GATE),
        _obs("blur", is_candidate=True, confidence=0.9),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "low_light"
    assert secondary == ()
    assert suppressed == ("blur",)


# --- 3. Temporal confirmation ----------------------------------------------


def test_pending_to_confirmed_to_cleared() -> None:
    tracker = _make_tracker()

    # Two candidate frames: not enough observed frames yet -> pending.
    events = _feed(tracker, [(0, 0.0, True), (1, 1.0, True)])
    assert events == []
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING

    # Third consecutive candidate frame confirms (rate 1.0, 3 frames).
    events = _feed(tracker, [(2, 2.0, True)])
    assert len(events) == 1
    confirmed = events[0]
    assert confirmed.status == EVENT_STATUS_CONFIRMED
    assert confirmed.fault_type == "low_light"
    assert confirmed.started_frame == 0
    assert confirmed.started_time_s == 0.0
    assert confirmed.ended_frame is None
    assert confirmed.window_positive_rate == pytest.approx(1.0)
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_CONFIRMED

    # Rate stays at/above 0.5 in the sliding window -> still confirmed.
    events = _feed(tracker, [(3, 3.0, False), (4, 4.0, False)])
    assert events == []
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_CONFIRMED

    # Rate drops below 0.5 -> cleared.
    events = _feed(tracker, [(5, 5.0, False)])
    assert len(events) == 1
    cleared = events[0]
    assert cleared.status == EVENT_STATUS_CLEARED
    assert cleared.started_frame == 0
    assert cleared.started_time_s == 0.0
    assert cleared.ended_frame == 5
    assert cleared.ended_time_s == 5.0
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING


def test_min_window_frames_required() -> None:
    tracker = _make_tracker()
    events = _feed(tracker, [(0, 0.0, True), (1, 1.0, True)])
    assert events == []
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING

    events = _feed(tracker, [(2, 2.0, True)])
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED


def test_confirmation_at_exact_positive_ratio() -> None:
    tracker = _make_tracker()
    events = _feed(tracker, [(0, 0.0, True), (1, 1.0, False), (2, 2.0, False), (3, 3.0, True)])
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED
    assert events[0].window_positive_rate == pytest.approx(0.5)


def test_positive_rate_below_threshold_stays_pending() -> None:
    tracker = _make_tracker()
    events = _feed(tracker, [(0, 0.0, True), (1, 1.0, False), (2, 2.0, False)])
    assert events == []
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING


def test_unobserved_fault_clears_as_window_ages() -> None:
    tracker = _make_tracker()
    events = _feed(tracker, [(0, 0.0, True), (1, 1.0, True), (2, 2.0, True)])
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED

    # From here on low_light is no longer observed; its window shrinks in time.
    events = tracker.update({}, ["blur"], 3, 3.0)
    assert events == []
    events = tracker.update({}, ["blur"], 4, 4.0)
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CLEARED
    assert events[0].ended_frame == 4
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING


def test_event_gap_blocks_immediate_reconfirmation() -> None:
    tracker = _make_tracker()
    _feed(tracker, [(0, 0.0, True), (1, 1.0, True), (2, 2.0, True)])  # confirmed
    _feed(tracker, [(3, 3.0, False), (4, 4.0, False), (5, 5.0, False)])  # cleared at t=5

    # Re-candidates inside the 10s gap stay pending (no new confirmed event).
    events = _feed(tracker, [(6, 6.0, True), (7, 7.0, True), (8, 8.0, True)])
    assert events == []
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING

    # After the gap (>= 15.0) the fault confirms again with a fresh event.
    events = _feed(tracker, [(15, 15.0, True), (16, 16.0, True), (17, 17.0, True)])
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED
    assert events[0].started_frame == 15
    assert events[0].started_time_s == 15.0


def test_faults_confirm_independently() -> None:
    tracker = _make_tracker()
    for frame_number, t, ll_candidate, blur_candidate in [
        (0, 0.0, True, False),
        (1, 1.0, False, False),
        (2, 2.0, False, False),
        (3, 3.0, True, False),
    ]:
        candidates = {}
        if ll_candidate:
            candidates["low_light"] = 0.7
        if blur_candidate:
            candidates["blur"] = 0.7
        tracker.update(candidates, ["low_light", "blur"], frame_number, t)

    status = tracker.confirmation_status()
    assert status["low_light"] == TEMPORAL_STATUS_CONFIRMED
    assert status["blur"] == TEMPORAL_STATUS_PENDING


# --- 4. Fault isolation ------------------------------------------------------


def test_detector_exception_does_not_crash_fusion() -> None:
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(exc=RuntimeError("boom")),
            "blur": _Detector(result=_Result(is_candidate=True, confidence=0.8)),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)

    statuses = {obs.detector: obs for obs in result.detectors}
    assert statuses["low_light"].status == DETECTOR_STATUS_ERROR
    assert statuses["low_light"].error_message == "boom"
    assert statuses["blur"].status == DETECTOR_STATUS_OK
    assert result.primary_fault == "blur"
    assert result.confidence == pytest.approx(0.8)
    assert engine.drain_events() == []  # single frame: temporal still pending


def test_all_detectors_failing_yields_no_primary() -> None:
    engine = DecisionEngine(
        "cam1",
        {name: _Detector(exc=RuntimeError("fail")) for name in ("low_light", "tampering", "blur", "tilt")},
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert result.primary_fault is None
    assert result.confidence == 0.0
    assert all(obs.status == DETECTOR_STATUS_ERROR for obs in result.detectors)


def test_malformed_detector_result_is_error() -> None:
    class _Broken:
        pass

    engine = DecisionEngine("cam1", {"blur": _Detector(result=_Broken())})
    result = engine.process_frame(_frame(), 0, 0.0)
    obs = result.detectors[0]
    assert obs.status == DETECTOR_STATUS_ERROR
    assert "is_candidate" in (obs.error_message or "")


def test_unknown_detector_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        DecisionEngine("cam1", {"bogus": _Detector(result=_Result(False))})


def test_consecutive_errors_trigger_backoff() -> None:
    engine = DecisionEngine(
        "cam1",
        {"low_light": _Detector(exc=RuntimeError("fail"))},
        max_consecutive_errors=2,
        error_backoff_seconds=5.0,
    )

    assert engine.process_frame(_frame(), 0, 0.0).detectors[0].status == DETECTOR_STATUS_ERROR  # count 1
    assert engine.process_frame(_frame(), 1, 1.0).detectors[0].status == DETECTOR_STATUS_ERROR  # count 2 -> backoff
    assert engine.process_frame(_frame(), 2, 2.0).detectors[0].status == DETECTOR_STATUS_SKIPPED
    assert engine.process_frame(_frame(), 3, 3.0).detectors[0].status == DETECTOR_STATUS_SKIPPED
    # Backoff expired (t >= 6.0): the detector runs again (and fails once more).
    assert engine.process_frame(_frame(), 7, 7.0).detectors[0].status == DETECTOR_STATUS_ERROR


def test_success_resets_error_counter() -> None:
    calls = {"n": 0}

    def flaky(frame: np.ndarray) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("fail")
        return _Result(is_candidate=False)

    engine = DecisionEngine("cam1", {"blur": flaky}, max_consecutive_errors=2, error_backoff_seconds=5.0)

    assert engine.process_frame(_frame(), 0, 0.0).detectors[0].status == DETECTOR_STATUS_ERROR  # count 1
    assert engine.process_frame(_frame(), 1, 1.0).detectors[0].status == DETECTOR_STATUS_OK  # success resets count

    # Two more consecutive failures must again trigger backoff -- proving the reset.
    engine._detectors["blur"] = _Detector(exc=RuntimeError("fail"))
    assert engine.process_frame(_frame(), 2, 2.0).detectors[0].status == DETECTOR_STATUS_ERROR  # count 1
    assert engine.process_frame(_frame(), 3, 3.0).detectors[0].status == DETECTOR_STATUS_ERROR  # count 2 -> backoff
    assert engine.process_frame(_frame(), 4, 4.0).detectors[0].status == DETECTOR_STATUS_SKIPPED


def test_engine_confirms_persistent_fault_end_to_end() -> None:
    engine = DecisionEngine(
        "cam1",
        {"low_light": _Detector(result=_Result(is_candidate=True, confidence=0.7))},
    )
    for frame_number, t in [(0, 0.0), (1, 1.0), (2, 2.0)]:
        engine.process_frame(_frame(), frame_number, t)

    events = engine.drain_events()
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED
    assert events[0].fault_type == "low_light"
    assert events[0].started_frame == 0
    assert engine.drain_events() == []

