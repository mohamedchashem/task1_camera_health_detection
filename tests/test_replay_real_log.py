"""Replay tests: prove the banner fix against the preserved real frame log.

The schema-v2 frame log ``data/logs/frame_log_cam_06.jsonl`` (copied verbatim
to ``tests/fixtures/frame_log_cam_06.jsonl`` so the regression test is
hermetic) contains the actual detector observations from the cam_06 run that
originally exposed the active/suppressed contradiction and the missing-fault
bugs. These tests reload six specific frames referenced in the root-cause
report, reconstruct each frame's observations, run them through the
NOW-FIXED pipeline (``classify_candidates`` -> the new ``DecisionFrame``
fields -> ``build_annotation_lines`` with the new parameters), and assert the
banner that would be rendered today -- from real historical data, not fresh
synthetic inputs.

Schema-v2 data limitations (handled explicitly, never silently):
- ``raw_confidence`` did not exist yet, so the logged ``confidence`` IS the
  detector's raw output: it becomes ``raw_confidence``, and the reportable
  ``confidence`` follows the fixed engine convention (zeroed for
  non-candidates).
- the raw measurands (``metrics``) were not persisted. Only frame 646 needs
  them (the tampering->blur conditional predicate), and both sides are
  recovered from the logged confidences via the detectors' own documented
  formulas plus a physical invariant -- see ``_reconstructed_metrics``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from config import (
    DECISION_CONFIRM_MIN_CONFIDENCE,
    DECISION_PRECEDENCE,
    TAMPERING_BLUR_AREA_SLACK,
    TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION,
)
from pipeline.annotate import build_annotation_lines
from pipeline.decision_engine import (
    DETECTOR_STATUS_OK,
    DETECTOR_STATUS_SKIPPED,
    DetectorObservation,
    Fault,
    classify_candidates,
)

_LOG_PATH = Path(__file__).resolve().parent / "fixtures" / "frame_log_cam_06.jsonl"

_REPLAY_FRAMES = (196, 346, 496, 646, 946, 1096)


def _load_frame(frame_number: int) -> dict:
    """Return the raw JSONL record for ``frame_number`` from the preserved log."""
    for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["frame_number"] == frame_number:
            return record
    raise AssertionError(f"frame {frame_number} not found in {_LOG_PATH}")


def _reconstructed_metrics(record: dict, detector: str, raw: float) -> dict[str, float]:
    """Recover the raw measurands a conditional predicate needs from logged
    data, using only the detectors' documented formulas and physical
    invariants -- never synthetic inputs.

    Schema v2 did not persist ``metrics``; only frame 646's tampering->blur
    predicate requires measurands, and both sides are recoverable:

    - blur: ``confidence == clip(1 - sharpness_ratio)`` (linear detector
      formula, exactly invertible) -> ``sharpness_ratio = 1 - confidence``.
    - tampering: ``confidence`` is a normalization of
      ``largest_contiguous_loss_fraction`` over
      ``[TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION, 1.0]`` (invertible), and
      ``total_loss_fraction >= largest_contiguous_loss_fraction`` (a single
      connected cluster can never contain more blocks than the total loss).
      The inverted largest fraction is therefore a conservative LOWER bound
      on ``total_loss_fraction``; the area-conservation predicate is asserted
      to hold even at that minimum, so it holds for every physically
      consistent observation.
    """
    if record["frame_number"] == 646:
        if detector == "blur":
            return {"sharpness_ratio": 1.0 - raw}
        if detector == "tampering":
            span = 1.0 - TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION
            largest = TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION + raw * span
            return {"total_loss_fraction": largest}
    return {}


def _replay(record: dict) -> dict:
    """Reconstruct the fixed pipeline's per-frame fields from a log record."""
    observations: dict[str, DetectorObservation] = {}
    for name, entry in record["detectors"].items():
        raw = float(entry["confidence"])
        observations[name] = DetectorObservation(
            detector=name,
            status=entry["status"],
            is_candidate=entry["is_candidate"],
            confidence=raw if entry["is_candidate"] else 0.0,
            error_message=entry["error_message"],
            reason=entry["reason"],
            metrics=_reconstructed_metrics(record, name, raw),
            raw_confidence=raw,
        )
    candidates = {
        name: obs
        for name, obs in observations.items()
        if obs.status == DETECTOR_STATUS_OK and obs.is_candidate
    }
    classification = classify_candidates(candidates)

    rank = {fault: i for i, fault in enumerate(DECISION_PRECEDENCE)}
    ordered = lambda names: tuple(sorted(names, key=lambda fault: rank[fault]))
    survivors = ordered(classification.survivors)
    below_floor = ordered(classification.below_floor)
    suppressed = ordered(classification.suppressed)
    unmeasurable = tuple(
        name
        for name, obs in observations.items()
        if obs.status == DETECTOR_STATUS_SKIPPED and obs.reason == "suppressed_by_gate"
    )
    # The log's temporal_status records which faults were temporally confirmed
    # on this frame (the tracker state was not itself persisted).
    confirmed_names = {
        fault for fault, status in record["temporal_status"].items() if status == "confirmed"
    }
    confirmed_faults = tuple(
        Fault(fault_type=fault, confidence=round(candidates[fault].confidence, 6))
        for fault in survivors
        if fault in confirmed_names
    )
    pending_faults = tuple(
        Fault(fault_type=fault, confidence=round(candidates[fault].confidence, 6))
        for fault in survivors
        if fault not in confirmed_names
    )
    return {
        "frame_number": record["frame_number"],
        "video_time_s": record["video_time_s"],
        "primary": record["primary_fault"],
        "confidence": record["confidence"],
        "candidates": candidates,
        "survivors": survivors,
        "below_floor": below_floor,
        "suppressed": suppressed,
        "unmeasurable": unmeasurable,
        "confirmed_faults": confirmed_faults,
        "pending_faults": pending_faults,
        "faults": confirmed_faults + pending_faults,
    }


def _render(replay: dict) -> list[str]:
    """Render the banner exactly as ``_save_annotated_frame`` does today."""
    return build_annotation_lines(
        replay["primary"],
        replay["confidence"],
        suppressed_faults=replay["suppressed"],
        video_time_s=replay["video_time_s"],
        faults=replay["faults"],
        confirmed_faults=replay["confirmed_faults"],
        pending_faults=replay["pending_faults"],
        below_floor_faults=replay["below_floor"],
        unmeasurable_faults=replay["unmeasurable"],
    )


_SECTION_PREFIXES = ("FAULT:", "pending:", "suppressed:", "too weak:", "unmeasurable:")


def _section_names(line: str, prefix: str) -> set[str]:
    body = line[len(prefix):].strip()
    if not body:
        return set()
    return {part.strip().split()[0] for part in body.split(",")}


def _sections(lines: list[str]) -> dict[str, set[str]]:
    """Partition a rendered banner into its five fault sections.

    The ``[unmeasurable]`` marker lives ON a FAULT line (a legitimate
    annotation, not a second listing), so it never creates a separate
    section entry here.
    """
    sections = {prefix[:-1]: set() for prefix in _SECTION_PREFIXES}
    for line in lines:
        for prefix in _SECTION_PREFIXES:
            if line.startswith(prefix):
                sections[prefix[:-1]] = _section_names(line, prefix)
                break
    return sections


def _assert_exclusivity(lines: list[str]) -> None:
    """A fault type must never appear in more than one banner section."""
    sections = _sections(lines)
    for name in set().union(*sections.values()):
        present = [section for section, names in sections.items() if name in names]
        assert len(present) <= 1, (
            f"{name!r} rendered in multiple sections: {present} (lines={lines!r})"
        )


# --- replay tests against the preserved real log ------------------------------


def test_replay_frame_196_too_weak_not_missing() -> None:
    # Plan Part A.3 Example 5: tilt+tampering-small. The old banner silently
    # dropped tampering; today tampering must appear as "too weak" (it was a
    # real candidate at 0.033, below its 0.5 emission floor), never absent.
    replay = _replay(_load_frame(196))
    lines = _render(replay)
    assert lines == [
        "FAULT: tilt (conf=0.74)",
        "too weak: tampering",
        "t=6.53s",
    ]
    assert replay["below_floor"] == ("tampering",)
    _assert_exclusivity(lines)


def test_replay_frame_346_no_contradiction() -> None:
    # One of the two literal original contradiction cases: the old output was
    # "FAULT: low_light / FAULT: tilt / suppressed: tilt, blur" -- tilt listed
    # as both active and suppressed. Today tilt is a survivor (confirmed) and
    # must NOT appear in any suppressed/too-weak/unmeasurable line.
    replay = _replay(_load_frame(346))
    lines = _render(replay)
    # Actual new output (no pre-specified banner in the plan):
    assert lines == [
        "FAULT: low_light (conf=0.65)",
        "FAULT: tilt (conf=0.70)",
        "suppressed: blur",
        "t=11.53s",
    ]
    sections = _sections(lines)
    assert "tilt" in sections["FAULT"]
    assert not (sections["suppressed"] | sections["too weak"] | sections["unmeasurable"]) & {"tilt"}
    _assert_exclusivity(lines)


def test_replay_frame_496_internally_consistent() -> None:
    # No pre-specified expected banner for this frame (tampering-small +
    # low_light). Report the actual output and confirm internal consistency:
    # blur survives (0.507 >= 0.5), tampering and low_light were below their
    # floors (0.408 / 0.069 < 0.5) -> "too weak", not silently dropped.
    replay = _replay(_load_frame(496))
    lines = _render(replay)
    # Actual new output (no pre-specified banner in the plan):
    assert lines == [
        "FAULT: blur (conf=0.51)",
        "too weak: tampering, low_light",
        "t=16.53s",
    ]
    assert replay["below_floor"] == ("tampering", "low_light")
    assert replay["suppressed"] == ()
    _assert_exclusivity(lines)


def test_replay_frame_646_suppressed_is_causal_not_mislabeled() -> None:
    # Plan Part A.3 Example 3: tampering-large alone. Same visible output as
    # before the fix, but "suppressed: blur" must now mean ONLY causal
    # suppression: blur genuinely cleared its emission floor (0.6185 >= 0.5)
    # and was removed by the tampering->blur area-conservation predicate.
    record = _load_frame(646)
    replay = _replay(record)
    lines = _render(replay)
    assert lines == [
        "FAULT: tampering (conf=0.55)",
        "suppressed: blur",
        "t=21.53s",
    ]
    # 1) blur cleared its own floor: it was eligible for suppression, so
    #    calling it "suppressed" is not the old "below-floor mislabeled" bug.
    blur = replay["candidates"]["blur"]
    assert blur.confidence >= DECISION_CONFIRM_MIN_CONFIDENCE["blur"]
    # 2) The suppression is predicate-gated (conditional relation), not the
    #    V1 DECISION_SUPPRESSION_MAP: with the reconstructed measurands the
    #    area-conservation predicate holds even at the minimum physically
    #    possible total_loss_fraction.
    tampering = replay["candidates"]["tampering"]
    sharpness_ratio = blur.metrics["sharpness_ratio"]
    total_loss_fraction = tampering.metrics["total_loss_fraction"]
    assert 1.0 - sharpness_ratio <= total_loss_fraction + TAMPERING_BLUR_AREA_SLACK
    # 3) Proof it is predicate-gated, not unconditional: the SAME logged
    #    candidates with empty metrics (metrics were not persisted in schema
    #    v2) co-survive -- "unverifiable -> never suppress".
    bare = {name: replace(obs, metrics={}) for name, obs in replay["candidates"].items()}
    assert "blur" in classify_candidates(bare).survivors
    _assert_exclusivity(lines)


def test_replay_frame_946_gate_skip_visible_not_dropped() -> None:
    # Plan Part A.3 Example 2: tampering-large+blur. tilt was gate-skipped
    # (observation status "skipped", reason "suppressed_by_gate"); it must be
    # rendered as "unmeasurable" instead of silently vanishing.
    replay = _replay(_load_frame(946))
    lines = _render(replay)
    assert lines == [
        "FAULT: blur (conf=1.00)",
        "unmeasurable: tilt",
        "t=31.53s",
    ]
    assert replay["unmeasurable"] == ("tilt",)
    _assert_exclusivity(lines)


def test_replay_frame_1096_exclusivity_on_original_contradiction() -> None:
    # Plan Part A.3 Example 4 -- the DIRECT fix of the original contradiction:
    # the old output was "FAULT: low_light / FAULT: blur / suppressed: blur"
    # (blur listed as both active and suppressed). Explicitly assert the word
    # "blur" does NOT appear in any suppressed/too-weak/unmeasurable line
    # while it is also in a FAULT line -- the exclusivity invariant holding
    # on the literal original bug evidence.
    replay = _replay(_load_frame(1096))
    lines = _render(replay)
    assert lines == [
        "FAULT: low_light (conf=0.69)",
        "FAULT: blur (conf=1.00)",
        "unmeasurable: tilt",
        "t=36.53s",
    ]
    sections = _sections(lines)
    assert "blur" in sections["FAULT"]
    assert not (sections["suppressed"] | sections["too weak"] | sections["unmeasurable"]) & {"blur"}
    _assert_exclusivity(lines)


@pytest.mark.parametrize("frame_number", _REPLAY_FRAMES)
def test_replay_all_frames_respect_exclusivity_invariant(frame_number: int) -> None:
    # Cross-cutting invariant across every replayed real frame: no fault type
    # appears in more than one of {FAULT/confirmed, pending, suppressed, too
    # weak, unmeasurable} except via the legitimate [unmeasurable] marker.
    replay = _replay(_load_frame(frame_number))
    _assert_exclusivity(_render(replay))


@pytest.mark.parametrize(
    "confirmed,pending,suppressed,below_floor,unmeasurable",
    [
        # The same fault fed to every section at once -> rendered exactly once.
        (
            (Fault("tilt", 0.70),),
            (Fault("tilt", 0.60),),
            ("tilt",),
            ("tilt",),
            ("tilt",),
        ),
        # Confirmed + gate-skipped overlap: the legitimate [unmeasurable]
        # marker on the FAULT line, never a second listing.
        ((Fault("tampering", 0.80),), (), (), (), ("tampering",)),
        # Distinct faults spread across all five sections.
        (
            (Fault("tampering", 0.80),),
            (Fault("low_light", 0.60),),
            ("blur",),
            ("tilt",),
            ("low_light",),
        ),
        # A pending survivor duplicated in suppressed/too-weak/unmeasurable.
        (
            (),
            (Fault("blur", 0.60),),
            ("blur",),
            ("blur",),
            ("blur",),
        ),
    ],
)
def test_banner_never_contradicts_regardless_of_inputs(
    confirmed, pending, suppressed, below_floor, unmeasurable
) -> None:
    # General cross-cutting exclusivity regression: for any constructed
    # DecisionFrame-equivalent input, the rendered banner never contains a
    # contradiction (priority resolution + gate-skip marker handle overlap).
    lines = build_annotation_lines(
        None,
        0.0,
        confirmed_faults=confirmed,
        pending_faults=pending,
        suppressed_faults=suppressed,
        below_floor_faults=below_floor,
        unmeasurable_faults=unmeasurable,
        video_time_s=1.0,
    )
    _assert_exclusivity(lines)



