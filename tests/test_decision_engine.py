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

from config import (
    DECISION_GATE_CONFIDENCE,
    DECISION_GATE_CONFIDENCE_BY_GATE,
    DECISION_SUPPRESSOR_MIN_CONFIDENCE,
)
from pipeline.decision_engine import (
    DETECTOR_STATUS_ERROR,
    DETECTOR_STATUS_OK,
    DETECTOR_STATUS_SKIPPED,
    DETECTOR_STATUS_UNAVAILABLE,
    DEGRADED_AMBIENT_EXPLAINED_REASON,
    DEGRADED_AMBIENT_UNEXPLAINED_REASON,
    EVENT_STATUS_CLEARED,
    EVENT_STATUS_CONFIRMED,
    TEMPORAL_STATUS_CONFIRMED,
    TEMPORAL_STATUS_PENDING,
    CandidateClassification,
    ConfirmationTracker,
    DecisionEngine,
    DetectorObservation,
    Fault,
    classify_candidates,
    fuse_observations,
    resolve_primary_fault,
    split_candidates_for_primary,
    _resolve_degraded_ambient,
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
    def __init__(self, is_candidate: bool, confidence: float = 1.0, **metrics: float) -> None:
        self.is_candidate = is_candidate
        self.confidence = confidence
        # Raw measurands (e.g. total_loss_fraction, sharpness_ratio) so
        # _extract_detector_metrics can populate DetectorObservation.metrics
        # and the co-occurrence predicates can read them.
        for field, value in metrics.items():
            setattr(self, field, value)


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
    # tampering at 0.6 is an active suppressor (>= GATE) and clears the
    # relative margin against every symptom (0.6 * MARGIN = 0.9).
    obs = (
        _obs("tampering", is_candidate=True, confidence=0.6),
        _obs("low_light", is_candidate=True, confidence=0.9),
        _obs("blur", is_candidate=True, confidence=0.8),
        _obs("tilt", is_candidate=True, confidence=0.5),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "tampering"
    assert secondary == ()
    assert suppressed == ("low_light", "tilt", "blur")
    assert confidence == pytest.approx(0.6)


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
    # higher (precedence beats confidence) -- as long as the suppressor
    # clears the relative margin (0.6 * MARGIN = 0.9 >= 0.9).
    obs = (
        _obs("tilt", is_candidate=True, confidence=0.6),
        _obs("blur", is_candidate=True, confidence=0.9),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "tilt"
    assert secondary == ()
    assert suppressed == ("blur",)
    assert confidence == pytest.approx(0.6)


def test_tilt_suppresses_blur_when_both_fire() -> None:
    # Rotation resampling lowers Laplacian sharpness, so a real tilt event
    # also trips the blur detector; the causal cause must win the primary.
    # tilt 0.7 clears the margin against blur 1.0 (0.7 * MARGIN = 1.05).
    obs = (
        _obs("blur", is_candidate=True, confidence=1.0),
        _obs("tilt", is_candidate=True, confidence=0.7),
    )
    primary, secondary, suppressed, confidence = fuse_observations(obs)
    assert primary == "tilt"
    assert secondary == ()
    assert suppressed == ("blur",)
    assert confidence == pytest.approx(0.7)


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
    # Three detectors firing on the same frame. The input arrives in
    # detector execution order (DECISION_EXECUTION_ORDER), which is NOT the
    # fusion precedence order -- output must still be precedence-ordered.
    candidates = ("blur", "tampering", "tilt")

    # tampering causally explains blur + tilt (per DECISION_SUPPRESSION_MAP).
    secondary, suppressed = split_candidates_for_primary("tampering", candidates)
    assert secondary == ()
    assert suppressed == ("tilt", "blur")

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
    assert suppressed == ("tilt", "blur")

# --- Multi-label fusion (Phase 1) ------------------------------------------


def test_single_fault_no_regression() -> None:
    # Single-fault frames must produce identical engine output to the V1
    # engine: exactly one survivor, same primary/confidence/symptoms.
    engine = DecisionEngine(
        "cam1",
        {"tampering": _Detector(result=_Result(is_candidate=True, confidence=0.85))},
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert result.faults == (Fault(fault_type="tampering", confidence=0.85),)
    assert result.primary_fault == "tampering"
    assert result.confidence == pytest.approx(0.85)
    assert result.secondary_symptoms == ()
    assert result.suppressed_faults == ()
    assert engine.drain_events() == []  # single frame: temporal still pending


def test_suppression_still_works() -> None:
    # The tampering->low_light conditional predicate now decides: the relative
    # margin clears (0.85 * MARGIN >= 0.75) AND the dark region is
    # conservatively explained by the obstruction area (relative_increase
    # <= total_loss_fraction + TAMPERING_LOW_LIGHT_AREA_SLACK), so low_light
    # is removed from the survivor list exactly as V1 suppressed it.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(
                result=_Result(is_candidate=True, confidence=0.85, total_loss_fraction=0.9)
            ),
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=0.75, relative_increase=0.3)
            ),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert result.faults == (Fault(fault_type="tampering", confidence=0.85),)
    assert result.primary_fault == "tampering"
    assert all(fault.fault_type != "low_light" for fault in result.faults)
    assert result.suppressed_faults == ("low_light",)
    # Causally-suppressed (floor-clearing) candidates are NOT below-floor.
    assert result.below_floor_faults == ()


def test_multi_fault_independent() -> None:
    # Multi-label happy path: tampering->tilt is intentionally absent from
    # DECISION_SUPPRESSION_RULES (independent -- no physical pathway from
    # structure loss to geometric displacement), so the two faults always
    # co-survive once each clears its emission floor.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(result=_Result(is_candidate=True, confidence=0.5)),
            "tilt": _Detector(result=_Result(is_candidate=True, confidence=0.9)),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert result.faults == (
        Fault(fault_type="tampering", confidence=0.5),
        Fault(fault_type="tilt", confidence=0.9),
    )
    assert result.primary_fault == "tampering"
    assert {fault.fault_type for fault in result.faults} == {"tampering", "tilt"}



def test_tampering_and_tilt_co_survive_when_independent() -> None:
    # tampering->tilt is intentionally absent from DECISION_SUPPRESSION_RULES
    # (no physical pathway: obstruction is a structure-loss symptom, tilt is a
    # geometric displacement), so the pair is independent: both faults always
    # co-survive once each clears its emission floor -- here even though the
    # suppressor gate and relative margin are both met (0.9 >= GATE, 0.9 *
    # MARGIN = 1.35 >= 0.8).
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(
                result=_Result(is_candidate=True, confidence=0.9, total_loss_fraction=0.9)
            ),
            "tilt": _Detector(result=_Result(is_candidate=True, confidence=0.8)),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["tampering", "tilt"]
    assert result.suppressed_faults == ()


def test_low_light_and_tilt_co_survive_when_independent() -> None:
    # low_light->tilt is likewise absent from DECISION_SUPPRESSION_RULES:
    # darkness cannot causally explain a geometric displacement, so the two
    # always co-survive. low_light stays below its gate floor so tilt is not
    # skipped; the predicate's independence decision still wins over the met
    # margin (0.75 * MARGIN = 1.125 >= 0.8).
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=0.75, dark_pixel_ratio=0.99)
            ),
            "tilt": _Detector(result=_Result(is_candidate=True, confidence=0.8)),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["low_light", "tilt"]
    assert result.suppressed_faults == ()


def test_tampering_low_light_co_survive_when_area_not_conserved() -> None:
    # The conditional tampering->low_light predicate: the margin clears
    # (0.9 * MARGIN >= 0.6) but the dark region is far larger than the
    # obstruction area can explain (relative_increase 0.9 > total_loss_fraction
    # 0.3 + TAMPERING_LOW_LIGHT_AREA_SLACK), so low_light is NOT a symptom of
    # tampering and both faults genuinely co-occur.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(
                result=_Result(is_candidate=True, confidence=0.9, total_loss_fraction=0.3)
            ),
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=0.6, relative_increase=0.9)
            ),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["tampering", "low_light"]
    assert result.suppressed_faults == ()


def test_tampering_blur_co_survive_when_area_not_conserved() -> None:
    # The conditional tampering->blur predicate: the obstruction's 0.3 lost
    # structure cannot explain blur's 0.8 sharpness loss (sharpness_ratio 0.2
    # -> 1 - 0.2 = 0.8 > 0.3 + TAMPERING_BLUR_AREA_SLACK), so blur survives
    # as an independent fault alongside tampering.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(
                result=_Result(is_candidate=True, confidence=0.9, total_loss_fraction=0.3)
            ),
            "blur": _Detector(
                result=_Result(is_candidate=True, confidence=0.6, sharpness_ratio=0.2)
            ),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["tampering", "blur"]
    assert result.suppressed_faults == ()


def test_low_light_blur_co_survive_when_not_near_black() -> None:
    # The conditional low_light->blur predicate: below the near-black floor a
    # dim-but-measurable scene cannot explain the blur signal, so both faults
    # survive even though the relative margin is met (0.6 * MARGIN = 0.9
    # >= 0.5).
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=0.6, dark_pixel_ratio=0.8)
            ),
            "blur": _Detector(result=_Result(is_candidate=True, confidence=0.5)),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["low_light", "blur"]
    assert result.suppressed_faults == ()


def test_tilt_suppresses_blur_always_relation_unchanged() -> None:
    # tilt->blur stays the "always" relation: the margin alone decides and raw
    # metrics are irrelevant. tilt 0.8 clears the margin against blur 0.6
    # (0.8 * MARGIN = 1.2 >= 0.6), so blur is suppressed exactly as before.
    # blur stays below its own gate floor (0.9) so tilt is not skipped.
    engine = DecisionEngine(
        "cam1",
        {
            "tilt": _Detector(result=_Result(is_candidate=True, confidence=0.8)),
            "blur": _Detector(
                result=_Result(is_candidate=True, confidence=0.6, sharpness_ratio=0.1)
            ),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["tilt"]
    assert result.suppressed_faults == ("blur",)


def test_low_light_suppresses_blur_when_near_black() -> None:
    # At/above the near-black floor (LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR) the whole
    # frame is unmeasurable, so low_light fully explains any blur signal: the
    # conditional predicate passes and blur is suppressed.
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=0.85, dark_pixel_ratio=0.99)
            ),
            "blur": _Detector(result=_Result(is_candidate=True, confidence=0.5)),
        },
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    assert [fault.fault_type for fault in result.faults] == ["low_light"]
    assert result.suppressed_faults == ("blur",)


# --- classify_candidates (Part B: suppressed vs below-floor split) ---------


def _candidate(
    detector: str,
    confidence: float,
    **metrics: float,
) -> DetectorObservation:
    """Candidate observation carrying raw metrics for conditional predicates."""
    return DetectorObservation(
        detector, DETECTOR_STATUS_OK, True, confidence, metrics=dict(metrics)
    )


def test_classify_candidates_below_floor_only() -> None:
    # A candidate below its own emission floor never had a chance to be
    # suppressed -- it lands in below_floor, never in suppressed.
    candidates = {"low_light": _candidate("low_light", confidence=0.2)}
    result = classify_candidates(candidates)
    assert result.survivors == frozenset()
    assert result.below_floor == frozenset({"low_light"})
    assert result.suppressed == frozenset()


def test_classify_candidates_causally_suppressed_only() -> None:
    # tilt (survivor) suppresses blur via the "always" relation: blur clears
    # its own floor but is removed by a surviving higher-precedence fault --
    # real causal suppression only, not a below-floor candidate.
    candidates = {
        "tilt": _candidate("tilt", confidence=0.8),
        "blur": _candidate("blur", confidence=0.6),
    }
    result = classify_candidates(candidates)
    assert result.survivors == frozenset({"tilt"})
    assert result.below_floor == frozenset()
    assert result.suppressed == frozenset({"blur"})


def test_classify_candidates_both_buckets() -> None:
    # tampering (survivor) suppresses low_light through the conditional
    # area-conservation predicate, while tilt is simply too weak (below its
    # own floor). Both non-survivor kinds are present and stay separate.
    candidates = {
        "tampering": _candidate("tampering", confidence=0.85, total_loss_fraction=0.9),
        "low_light": _candidate("low_light", confidence=0.75, relative_increase=0.3),
        "tilt": _candidate("tilt", confidence=0.2),
    }
    result = classify_candidates(candidates)
    assert result.survivors == frozenset({"tampering"})
    assert result.below_floor == frozenset({"tilt"})
    assert result.suppressed == frozenset({"low_light"})


def test_classify_candidates_clean_survivor() -> None:
    # A single candidate at or above its own floor with no suppressor:
    # survivor only -- neither below-floor nor suppressed.
    candidates = {"tampering": _candidate("tampering", confidence=0.85)}
    result = classify_candidates(candidates)
    assert result.survivors == frozenset({"tampering"})
    assert result.below_floor == frozenset()
    assert result.suppressed == frozenset()


def test_classify_candidates_empty_candidates() -> None:
    # No candidates: every bucket is empty and the buckets stay disjoint.
    result = classify_candidates({})
    assert result == CandidateClassification(
        survivors=frozenset(),
        below_floor=frozenset(),
        suppressed=frozenset(),
    )


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
    # The gate is inclusive: confidence == GATE counts as an active
    # suppressor, and it clears the margin against a 0.75 candidate
    # (GATE * DECISION_SUPPRESSION_MARGIN = 0.75).
    obs = (
        _obs("low_light", is_candidate=True, confidence=GATE),
        _obs("blur", is_candidate=True, confidence=0.75),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "low_light"
    assert secondary == ()
    assert suppressed == ("blur",)


def test_margin_rule_blocks_gate_passing_suppressor_over_strong_candidate() -> None:
    # The relative-margin rule: tampering at 0.6 passes the confidence gate
    # but its effective strength (0.6 * DECISION_SUPPRESSION_MARGIN = 0.9) is
    # below the 1.0 tilt signal, so it must NOT hijack the stronger
    # lower-precedence candidate. This is the mechanism that stops weak
    # tampering noise from overriding a strong tilt.
    obs = (
        _obs("tampering", is_candidate=True, confidence=0.6),
        _obs("tilt", is_candidate=True, confidence=1.0),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "tilt"
    assert secondary == ("tampering",)
    assert suppressed == ()


def test_margin_rule_keeps_strong_suppressor() -> None:
    # tampering at 0.7 clears the margin against tilt 1.0 (0.7 * MARGIN
    # = 1.05): a genuinely strong cause still wins the primary label.
    obs = (
        _obs("tampering", is_candidate=True, confidence=0.7),
        _obs("tilt", is_candidate=True, confidence=1.0),
    )
    primary, secondary, suppressed, _ = fuse_observations(obs)
    assert primary == "tampering"
    assert secondary == ()
    assert suppressed == ("tilt",)


# --- Execution gating (gate-skip) -------------------------------------------


def test_high_confidence_low_light_gate_skips_tampering_and_tilt() -> None:
    calls = {"tampering": 0, "tilt": 0}

    def counting_detector(name: str):
        def detect(frame: np.ndarray) -> object:
            calls[name] += 1
            return _Result(is_candidate=False)
        return detect

    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=DECISION_GATE_CONFIDENCE + 0.1)
            ),
            "blur": _Detector(result=_Result(is_candidate=True, confidence=0.9)),
            "tampering": counting_detector("tampering"),
            "tilt": counting_detector("tilt"),
        },
    )
    decision = engine.process_frame(_frame(), 0, 0.0)
    by_name = {obs.detector: obs for obs in decision.detectors}

    assert by_name["tampering"].status == DETECTOR_STATUS_SKIPPED
    assert by_name["tampering"].reason == "suppressed_by_gate"
    assert by_name["tilt"].status == DETECTOR_STATUS_SKIPPED
    assert by_name["tilt"].reason == "suppressed_by_gate"
    # blur is the primary optical signal during low-light transitions and
    # must stay active.
    assert by_name["blur"].status == DETECTOR_STATUS_OK
    # The expensive structural detectors never ran: no wasted DISK work.
    assert calls["tampering"] == 0
    assert calls["tilt"] == 0


def test_low_light_below_gate_keeps_structural_detectors_active() -> None:
    calls = {"tampering": 0, "tilt": 0}

    def counting_detector(name: str):
        def detect(frame: np.ndarray) -> object:
            calls[name] += 1
            return _Result(is_candidate=False)
        return detect

    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(
                result=_Result(is_candidate=True, confidence=DECISION_GATE_CONFIDENCE - 0.1)
            ),
            "tampering": counting_detector("tampering"),
            "tilt": counting_detector("tilt"),
        },
    )
    engine.process_frame(_frame(), 0, 0.0)
    # Mild dimming is not near-black: structural detectors still run so a
    # real tilt during measurable light is not missed.
    assert calls["tampering"] == 1
    assert calls["tilt"] == 1


def test_high_confidence_blur_gate_skips_tilt() -> None:
    # Regression: severe blur (test_video2, cam_02) made DISK emit a handful
    # of spurious correspondences whose median displacement read as a huge
    # false tilt (confidence 1.0 at blur ~0.99), which even got temporally
    # confirmed during the blur window. A blur gate at >= 0.90 must skip the
    # unmeasurable tilt detector so its confirmation window ages out.
    calls = {"tilt": 0}

    def counting_tilt(frame: np.ndarray) -> object:
        calls["tilt"] += 1
        return _Result(is_candidate=False)

    engine = DecisionEngine(
        "cam1",
        {
            "blur": _Detector(result=_Result(is_candidate=True, confidence=0.95)),
            "tilt": counting_tilt,
        },
    )
    decision = engine.process_frame(_frame(), 0, 0.0)
    by_name = {obs.detector: obs for obs in decision.detectors}

    assert by_name["tilt"].status == DETECTOR_STATUS_SKIPPED
    assert by_name["tilt"].reason == "suppressed_by_gate"
    # The expensive structural detector never ran: no wasted DISK work.
    assert calls["tilt"] == 0


def test_blur_below_gate_keeps_tilt_active() -> None:
    # Moderate blur is still measurable for keypoint matching: a real tilt
    # during it must not be missed.
    calls = {"tilt": 0}

    def counting_tilt(frame: np.ndarray) -> object:
        calls["tilt"] += 1
        return _Result(is_candidate=False)

    engine = DecisionEngine(
        "cam1",
        {
            "blur": _Detector(
                result=_Result(
                    is_candidate=True,
                    confidence=DECISION_GATE_CONFIDENCE_BY_GATE["blur"] - 0.1,
                )
            ),
            "tilt": counting_tilt,
        },
    )
    engine.process_frame(_frame(), 0, 0.0)
    assert calls["tilt"] == 1


# --- Degraded-baseline non-measurements (Approach C) -------------------------


def test_degraded_baseline_observation_is_unavailable_not_ok() -> None:
    # A degraded-baseline tampering result means "ran but could not measure":
    # it must NOT be packaged as a genuine OK negative observation. It gets a
    # distinct status (unavailable) while is_candidate/confidence/raw_confidence
    # keep their correct False/0.0 semantics.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(
                result=_Result(
                    is_candidate=False,
                    confidence=0.0,
                    reason="degraded_baseline",
                )
            )
        },
    )
    decision = engine.process_frame(_frame(), 0, 0.0)
    obs = decision.detectors[0]
    assert obs.status == DETECTOR_STATUS_UNAVAILABLE
    assert obs.reason == "degraded_baseline"
    assert obs.is_candidate is False
    assert obs.confidence == 0.0
    assert obs.raw_confidence == 0.0
    # It is not a candidate and never enters fusion.
    assert decision.faults == ()
    assert decision.primary_fault is None


def test_degraded_baseline_frames_do_not_dilute_positive_rate() -> None:
    # Degraded-baseline frames are non-measurements, not negative samples:
    # they must not be counted in the tracker's observed set, so a confirmed
    # tampering window keeps its 3/3 positive rate instead of being diluted
    # toward zero and spuriously clearing.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(result=_Result(is_candidate=True, confidence=0.7)),
        },
    )
    # Three real candidate frames confirm tampering (rate 1.0, 3 observed).
    for frame_number, t in [(0, 0.0), (1, 0.5), (2, 1.0)]:
        engine.process_frame(_frame(), frame_number, t)
    events = engine.drain_events()
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED
    assert events[0].window_positive_rate == pytest.approx(1.0)

    # Switch the detector to a degraded baseline (never a measurement). Four
    # non-measurement frames follow at t=1.5..3.0. If they counted as observed
    # negatives the window would be 3/7 = 0.43 < 0.5 and tampering would clear;
    # excluded, the window keeps its three positives and stays confirmed.
    engine._detectors["tampering"] = _Detector(
        result=_Result(
            is_candidate=False,
            confidence=0.0,
            reason="degraded_baseline",
        )
    )
    for frame_number, t in [(3, 1.5), (4, 2.0), (5, 2.5), (6, 3.0)]:
        engine.process_frame(_frame(), frame_number, t)

    assert engine.drain_events() == []  # no spurious clear
    assert engine._tracker.confirmation_status()["tampering"] == TEMPORAL_STATUS_CONFIRMED
    # The tracker window holds only the three real observations.
    assert len(engine._tracker._windows["tampering"]) == 3


def test_all_frames_degraded_baseline_never_observed_no_crash() -> None:
    # A camera whose tampering baseline is degraded for its ENTIRE lifetime:
    # tampering is never observed, so its tracker window is never created.
    # _window_rate must handle the zero-observed case without a divide-by-zero
    # and the engine must never confirm or clear a phantom tampering event.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(
                result=_Result(
                    is_candidate=False,
                    confidence=0.0,
                    reason="degraded_baseline",
                )
            ),
        },
    )
    for frame_number in range(20):
        decision = engine.process_frame(_frame(), frame_number, float(frame_number))
        obs = decision.detectors[0]
        assert obs.status == DETECTOR_STATUS_UNAVAILABLE
        assert obs.reason == "degraded_baseline"
        assert decision.confirmed_faults == ()

    assert engine.drain_events() == []
    # Never observed -> tampering never appears in temporal status (no window).
    assert "tampering" not in engine._tracker.confirmation_status()
    assert engine._tracker._windows.get("tampering") is None


# --- Degraded-ambient collapse (Approach C, subtask 8) -----------------------


def test_degraded_ambient_explained_by_severe_blur_is_unavailable_not_ok() -> None:
    # Tampering ran but the frame's ambient retention collapsed below the
    # degeneracy floor AND the same frame's blur is independently severe
    # (0.95 >= the contamination floor). The collapse is explained by
    # contamination: tampering's contribution is "unverifiable, explained"
    # (status unavailable, reason degraded_ambient_explained) -- never a
    # measured negative, never a candidate.
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "blur": _Detector(result=_Result(is_candidate=True, confidence=0.95)),
            "tampering": _Detector(
                result=_Result(
                    is_candidate=False, confidence=0.0, reason="degraded_ambient"
                )
            ),
        },
    )
    decision = engine.process_frame(_frame(), 0, 0.0)
    obs = {o.detector: o for o in decision.detectors}["tampering"]
    assert obs.status == DETECTOR_STATUS_UNAVAILABLE
    assert obs.reason == DEGRADED_AMBIENT_EXPLAINED_REASON
    assert obs.is_candidate is False
    assert obs.confidence == 0.0
    # The severe blur is its own genuine fault this frame; tampering is not.
    assert decision.primary_fault == "blur"
    # Excluded from the tracker's observed set: no window is ever created.
    assert engine._tracker._windows.get("tampering") is None
    assert "tampering" not in engine._tracker.confirmation_status()


def test_degraded_ambient_unexplained_is_still_not_a_clean_negative() -> None:
    # Ambient retention collapsed but NO other detector is severely degraded
    # at that frame. Per the subtask-8 decision, this unexplained case keeps
    # the same practical handling as the explained one -- excluded from the
    # observed set, never a false clean negative -- but is distinguished for
    # diagnostics via the reason string.
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "blur": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "tampering": _Detector(
                result=_Result(
                    is_candidate=False, confidence=0.0, reason="degraded_ambient"
                )
            ),
        },
    )
    decision = engine.process_frame(_frame(), 0, 0.0)
    obs = {o.detector: o for o in decision.detectors}["tampering"]
    assert obs.status == DETECTOR_STATUS_UNAVAILABLE
    assert obs.reason == DEGRADED_AMBIENT_UNEXPLAINED_REASON
    assert obs.is_candidate is False
    assert decision.primary_fault is None
    assert engine._tracker._windows.get("tampering") is None
    assert "tampering" not in engine._tracker.confirmation_status()


def test_degraded_ambient_explained_by_severe_low_light() -> None:
    # The low_light branch of the contamination resolution: low_light at/above
    # its near-black floor independently explains the ambient collapse. This
    # path is defensive in the current gate configuration (low_light >= 0.8
    # gate-skips tampering before it can produce a degraded_ambient result),
    # so it is exercised directly on the observation list via the
    # gate-independent helper rather than through process_frame.
    observations = [
        DetectorObservation("low_light", DETECTOR_STATUS_OK, True, 0.85),
        DetectorObservation("blur", DETECTOR_STATUS_OK, False, 0.0),
        DetectorObservation(
            "tampering", DETECTOR_STATUS_UNAVAILABLE, False, 0.0,
            reason="degraded_ambient",
        ),
    ]
    resolved = _resolve_degraded_ambient(observations)
    by_name = {o.detector: o for o in resolved}
    tampering = by_name["tampering"]
    assert tampering.status == DETECTOR_STATUS_UNAVAILABLE
    assert tampering.reason == DEGRADED_AMBIENT_EXPLAINED_REASON
    # Non-tampering observations pass through untouched.
    assert by_name["low_light"].confidence == 0.85
    assert by_name["blur"].confidence == 0.0


def test_degraded_ambient_frames_do_not_dilute_positive_rate() -> None:
    # Mirror of the degraded_baseline dilution regression: degraded_ambient
    # frames are non-measurements, not negative samples. They must not count
    # in the tracker's observed set, so a confirmed tampering window keeps its
    # 3/3 positive rate instead of being diluted toward zero and clearing.
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "blur": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "tampering": _Detector(result=_Result(is_candidate=True, confidence=0.7)),
        },
    )
    # Three real candidate frames confirm tampering (rate 1.0, 3 observed).
    for frame_number, t in [(0, 0.0), (1, 0.5), (2, 1.0)]:
        engine.process_frame(_frame(), frame_number, t)
    events = engine.drain_events()
    assert len(events) == 1
    assert events[0].status == EVENT_STATUS_CONFIRMED
    assert events[0].window_positive_rate == pytest.approx(1.0)

    # Switch tampering to an unexplained degraded-ambient collapse. Four
    # non-measurement frames follow at t=1.5..3.0. If they counted as observed
    # negatives the window would be 3/7 = 0.43 < 0.5 and tampering would
    # clear; excluded, the window keeps its three positives and stays
    # confirmed.
    engine._detectors["tampering"] = _Detector(
        result=_Result(
            is_candidate=False, confidence=0.0, reason="degraded_ambient"
        )
    )
    for frame_number, t in [(3, 1.5), (4, 2.0), (5, 2.5), (6, 3.0)]:
        engine.process_frame(_frame(), frame_number, t)

    assert engine.drain_events() == []  # no spurious clear
    assert engine._tracker.confirmation_status()["tampering"] == TEMPORAL_STATUS_CONFIRMED
    # The tracker window holds only the three real observations.
    assert len(engine._tracker._windows["tampering"]) == 3


def test_all_frames_degraded_ambient_never_observed_no_crash() -> None:
    # A camera whose frames' ambient retention collapses for its ENTIRE
    # lifetime with nothing explaining it (e.g. full-lens obstruction or
    # total sensor failure): tampering is never observed, its tracker window
    # is never created, and the engine must neither confirm nor clear a
    # phantom tampering event -- the same no-crash/no-misbehavior guarantee
    # subtask 3 established for all-frames-degraded-baseline.
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "blur": _Detector(result=_Result(is_candidate=False, confidence=0.0)),
            "tampering": _Detector(
                result=_Result(
                    is_candidate=False, confidence=0.0, reason="degraded_ambient"
                )
            ),
        },
    )
    for frame_number in range(20):
        decision = engine.process_frame(_frame(), frame_number, float(frame_number))
        obs = {o.detector: o for o in decision.detectors}["tampering"]
        assert obs.status == DETECTOR_STATUS_UNAVAILABLE
        assert obs.reason == DEGRADED_AMBIENT_UNEXPLAINED_REASON
        assert decision.confirmed_faults == ()

    assert engine.drain_events() == []
    # Never observed -> tampering never appears in temporal status (no window).
    assert "tampering" not in engine._tracker.confirmation_status()
    assert engine._tracker._windows.get("tampering") is None


def test_gate_skipped_observation_stays_out_of_tracker_window() -> None:
    # Regression: a gate-skipped detector (status "skipped", reason
    # "suppressed_by_gate") is excluded from the confirmation tracker's
    # observed set exactly as before this subtask -- its window never gains
    # entries and it never confirms.
    engine = DecisionEngine(
        "cam1",
        {
            "low_light": _Detector(
                result=_Result(
                    is_candidate=True,
                    confidence=DECISION_GATE_CONFIDENCE + 0.1,
                )
            ),
            "tampering": _Detector(result=_Result(is_candidate=False)),
        },
    )
    decision = engine.process_frame(_frame(), 0, 0.0)
    by_name = {obs.detector: obs for obs in decision.detectors}
    assert by_name["tampering"].status == DETECTOR_STATUS_SKIPPED
    assert by_name["tampering"].reason == "suppressed_by_gate"
    # The skipped detector never ran and never entered the tracker.
    assert engine._detectors["tampering"].calls == 0
    assert engine._tracker._windows.get("tampering") is None
    assert "tampering" not in engine._tracker.confirmation_status()
    assert engine.drain_events() == []


# --- Emission gating (confirmation confidence floor) ------------------------


def test_confirmation_requires_emission_confidence_floor() -> None:
    tracker = _make_tracker()
    # Candidates below the per-fault floor (0.5) never count as positives:
    # three consecutive weak candidates must NOT confirm an event.
    for frame_number, t in [(0, 0.0), (1, 1.0), (2, 2.0)]:
        events = tracker.update({"low_light": 0.2}, ["low_light"], frame_number, t)
        assert events == []
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_PENDING

    # The same detector above the floor confirms normally.
    for frame_number, t in [(3, 3.0), (4, 4.0), (5, 5.0)]:
        tracker.update({"low_light": 0.7}, ["low_light"], frame_number, t)
    assert tracker.confirmation_status()["low_light"] == TEMPORAL_STATUS_CONFIRMED


def test_engine_emission_floor_prevents_weak_candidate_event() -> None:
    # Regression for the low-confidence tampering false positive (t=47.49s of
    # test_video2): a persistent 0.2-confidence candidate must never log.
    engine = DecisionEngine(
        "cam1",
        {"tampering": _Detector(result=_Result(is_candidate=True, confidence=0.2))},
    )
    # A sub-floor candidate is bucketed as below_floor_faults (never
    # suppressed_faults) and never logs.
    result = engine.process_frame(_frame(), 0, 0.0)
    assert result.faults == ()
    assert result.suppressed_faults == ()
    assert result.below_floor_faults == ("tampering",)
    for frame_number, t in [(1, 1.0), (2, 2.0)]:
        engine.process_frame(_frame(), frame_number, t)
    assert engine.drain_events() == []


def test_engine_emission_floor_still_confirms_strong_candidate() -> None:
    engine = DecisionEngine(
        "cam1",
        {"tampering": _Detector(result=_Result(is_candidate=True, confidence=0.7))},
    )
    for frame_number, t in [(0, 0.0), (1, 1.0), (2, 2.0)]:
        engine.process_frame(_frame(), frame_number, t)
    events = engine.drain_events()
    assert len(events) == 1
    assert events[0].fault_type == "tampering"
    assert events[0].status == EVENT_STATUS_CONFIRMED
    assert events[0].peak_confidence == pytest.approx(0.7)


def test_tilt_reason_surfaces_in_observation() -> None:
    class _TiltResult:
        is_candidate = False
        confidence = 0.0
        reason = "insufficient_matches"

    engine = DecisionEngine("cam1", {"tilt": _Detector(result=_TiltResult())})
    decision = engine.process_frame(_frame(), 0, 0.0)
    obs = decision.detectors[0]
    assert obs.status == DETECTOR_STATUS_OK
    assert obs.is_candidate is False
    assert obs.reason == "insufficient_matches"


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


def test_confirmed_peaks_empty_when_nothing_confirmed() -> None:
    tracker = _make_tracker()
    assert tracker.confirmed_peaks() == {}

    # A pending (observed but not yet confirmed) fault is not reported.
    _feed(tracker, [(0, 0.0, True), (1, 1.0, True)])
    assert tracker.confirmed_peaks() == {}


def test_confirmed_peaks_single_confirmed_fault() -> None:
    tracker = _make_tracker()
    _feed(tracker, [(0, 0.0, True), (1, 1.0, True), (2, 2.0, True)])
    assert tracker.confirmed_peaks() == {"low_light": 0.7}

    # The reported value is the tracker's running peak over the episode: a
    # stronger candidate while confirmed raises it, a weaker one does not.
    tracker.update({"low_light": 0.9}, ["low_light"], 3, 3.0)
    assert tracker.confirmed_peaks() == {"low_light": 0.9}
    tracker.update({"low_light": 0.6}, ["low_light"], 4, 4.0)
    assert tracker.confirmed_peaks() == {"low_light": 0.9}


def test_confirmed_peaks_multiple_faults_overlapping_windows() -> None:
    # cam_05 scenario: two faults' confirmed windows overlap in time. Both
    # must be reported simultaneously, each with its own peak confidence.
    tracker = _make_tracker()
    for frame_number, t in [(0, 0.0), (1, 1.0), (2, 2.0)]:
        tracker.update(
            {"low_light": 0.7, "blur": 0.8},
            ["low_light", "blur"],
            frame_number,
            t,
        )
    assert tracker.confirmed_peaks() == {"low_light": 0.7, "blur": 0.8}

    # The overlap persists while both faults are still observed...
    tracker.update({"low_light": 0.7, "blur": 0.8}, ["low_light", "blur"], 3, 3.0)
    assert tracker.confirmed_peaks() == {"low_light": 0.7, "blur": 0.8}

    # ...and when low_light's window ages out, only low_light clears.
    for frame_number, t in [(4, 4.0), (5, 5.0)]:
        tracker.update({"blur": 0.8}, ["blur"], frame_number, t)
    assert tracker.confirmed_peaks() == {"blur": 0.8}


def test_engine_frame_exposes_confirmed_faults() -> None:
    # Two faults confirmed on the same frames: process_frame must surface
    # BOTH in confirmed_faults (DECISION_PRECEDENCE order) with their peak
    # confidences. Low-light stays below its 0.8 gate floor so tampering is
    # never execution-gated out.
    engine = DecisionEngine(
        "cam1",
        {
            "tampering": _Detector(result=_Result(is_candidate=True, confidence=0.7)),
            "low_light": _Detector(result=_Result(is_candidate=True, confidence=0.75)),
        },
    )
    # Nothing confirmed yet on the first frame.
    assert engine.process_frame(_frame(), 0, 0.0).confirmed_faults == ()
    engine.process_frame(_frame(), 1, 1.0)
    # After the third observed frame both faults confirm together.
    assert engine.process_frame(_frame(), 2, 2.0).confirmed_faults == (
        Fault(fault_type="tampering", confidence=0.7),
        Fault(fault_type="low_light", confidence=0.75),
    )
    result = engine.process_frame(_frame(), 3, 3.0)
    assert result.confirmed_faults == (
        Fault(fault_type="tampering", confidence=0.7),
        Fault(fault_type="low_light", confidence=0.75),
    )

    # confirmed_faults is the temporal truth, not this frame's survivors:
    # when tampering's detector stops flagging a candidate, it still stays
    # listed because it remains temporally confirmed.
    engine._detectors["tampering"] = _Detector(result=_Result(is_candidate=False))
    frame4 = engine.process_frame(_frame(), 4, 4.0)
    assert [fault.fault_type for fault in frame4.faults] == ["low_light"]
    assert frame4.confirmed_faults == (
        Fault(fault_type="tampering", confidence=0.7),
        Fault(fault_type="low_light", confidence=0.75),
    )


# --- 4. Fault isolation ------------------------------------------------------


def test_non_candidate_observation_zeroes_reported_confidence() -> None:
    # Regression: a detector can emit a high normalized confidence while
    # declaring is_candidate=False (observed: tampering confidence 1.0 with
    # is_candidate False when blur coexistence pushed total_loss_fraction
    # over its ceiling). _run_detector must package this honestly: the
    # reportable confidence is zeroed while raw_confidence preserves the
    # original value for debugging.
    engine = DecisionEngine(
        "cam1",
        {"tampering": _Detector(result=_Result(is_candidate=False, confidence=1.0))},
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    obs = result.detectors[0]
    assert obs.status == DETECTOR_STATUS_OK
    assert obs.is_candidate is False
    assert obs.confidence == 0.0
    assert obs.raw_confidence == pytest.approx(1.0)
    # The non-candidate stays out of every decision path regardless.
    assert result.faults == ()
    assert result.primary_fault is None


def test_candidate_observation_keeps_confidence_unchanged() -> None:
    # Normal case: a real candidate's reportable confidence equals its raw
    # value -- zeroing applies only to non-candidates.
    engine = DecisionEngine(
        "cam1",
        {"blur": _Detector(result=_Result(is_candidate=True, confidence=0.8))},
    )
    result = engine.process_frame(_frame(), 0, 0.0)
    obs = result.detectors[0]
    assert obs.status == DETECTOR_STATUS_OK
    assert obs.is_candidate is True
    assert obs.confidence == pytest.approx(0.8)
    assert obs.raw_confidence == pytest.approx(0.8)


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

