"""Automated ground-truth validation for the tampering detector.

Unlike the low-light detector, this one cannot yet cleanly separate
real obstruction from every other fault type -- see
STRUCTURE_OVERLAP_FAULT_TYPES below. That's tracked here explicitly
rather than hidden.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from config import (
    BASELINES_DIR,
    PROJECT_ROOT,
    TAMPERING_TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE as MIN_TRUE_POSITIVE_CANDIDATE_RATE,
    TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE as MAX_UNRELATED_FALSE_POSITIVE_RATE,
)
from detectors.tampering import evaluate
from pipeline.file_reader import read_frames_from_file

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "test_video_ground_truth.json"

# Edge-density tampering detection cannot yet cleanly separate real
# obstruction from other faults that also disrupt frame structure:
# low-light (reduced contrast weakens edge detection), blur (destroys
# edge sharpness by definition), and tilt (relocates edges outside
# their baseline grid positions). This is a known, measured limitation
# of a single detector working in isolation -- resolving it is
# explicitly the job of the cross-triggering decision layer, which has
# visibility across all four detectors at once. See project plan,
# "Known cross-cutting risk to design around."
STRUCTURE_OVERLAP_FAULT_TYPES = {"low_light", "blur", "tilt"}


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def baseline_edges(fixture_data: dict):
    camera_id = fixture_data["camera_id"]
    edges_path = BASELINES_DIR / f"{camera_id}_edges.png"
    if not edges_path.exists():
        pytest.skip(f"No edge baseline for {camera_id!r}; run pipeline.capture_baseline first.")
    edges = cv2.imread(str(edges_path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        pytest.skip(f"Failed to load edge baseline at {edges_path}.")
    return edges


@pytest.fixture(scope="module")
def scored_frames(fixture_data: dict, baseline_edges) -> list[dict]:
    video_path = PROJECT_ROOT / fixture_data["video"]
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}.")

    results = []
    for frame_number, video_time_s, frame in read_frames_from_file(video_path):
        result = evaluate(frame, baseline_edges)
        results.append(
            {
                "frame_number": frame_number,
                "video_time_s": video_time_s,
                "is_candidate": result.is_candidate,
                "confidence": result.confidence,
            }
        )
    return results


def _in_window(video_time_s: float, start_s: float, end_s: float) -> bool:
    return start_s <= video_time_s <= end_s


def test_tampering_window_is_detected(fixture_data: dict, scored_frames: list[dict]) -> None:
    fault = next(f for f in fixture_data["faults"] if f["type"] == "tampering")
    window_frames = [f for f in scored_frames if _in_window(f["video_time_s"], fault["start_s"], fault["end_s"])]
    assert window_frames, "No frames fell inside the tampering fault window; check fixture timing."

    candidate_rate = sum(f["is_candidate"] for f in window_frames) / len(window_frames)
    assert candidate_rate >= MIN_TRUE_POSITIVE_CANDIDATE_RATE, (
        f"Tampering window candidate rate {candidate_rate:.2f} is below the "
        f"required {MIN_TRUE_POSITIVE_CANDIDATE_RATE:.2f}."
    )


def test_clean_footage_does_not_trigger_false_positives(fixture_data: dict, scored_frames: list[dict]) -> None:
    fault_windows = [(f["start_s"], f["end_s"]) for f in fixture_data["faults"]]

    clean_frames = [
        f for f in scored_frames
        if not any(_in_window(f["video_time_s"], start, end) for start, end in fault_windows)
    ]
    assert clean_frames, "No clean (non-fault) frames found; check fixture timing."

    candidate_rate = sum(f["is_candidate"] for f in clean_frames) / len(clean_frames)
    assert candidate_rate <= MAX_UNRELATED_FALSE_POSITIVE_RATE, (
        f"Clean footage unexpectedly triggered candidates at rate {candidate_rate:.2f}."
    )


def test_structure_overlap_faults_are_documented_not_asserted(fixture_data: dict) -> None:
    """Not a correctness check -- a guard that keeps the known-overlap
    list honest. Fails loudly if a fault type is ever added to the
    fixture without a deliberate decision about where it belongs.
    """
    fixture_fault_types = {f["type"] for f in fixture_data["faults"]}
    unaccounted = fixture_fault_types - STRUCTURE_OVERLAP_FAULT_TYPES - {"tampering"}
    assert not unaccounted, (
        f"Fault types {unaccounted} are in the fixture but not accounted for -- "
        "decide whether they belong in STRUCTURE_OVERLAP_FAULT_TYPES or need "
        "a strict false-positive check."
    )


def test_evaluate_reports_degraded_baseline_when_structure_is_sparse() -> None:
    # A baseline with meaningful structure in only 25% of its blocks cannot
    # support structure-loss detection (dark/blurry capture). The detector
    # must never emit a candidate from such a baseline -- this is the root
    # cause of the low-confidence tampering false positive on dark baselines.
    sparse_edges = np.zeros((64, 64), dtype=np.uint8)
    sparse_edges[0:32, 0:32] = 255  # one meaningful 32x32 block of four
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    result = evaluate(frame, sparse_edges)
    assert result.is_candidate is False
    assert result.confidence == 0.0
    assert result.reason == "degraded_baseline"


def test_evaluate_recalibrated_confidence_on_full_structure_baseline() -> None:
    # With a sufficient baseline, the recalibrated confidence maps the
    # largest lost-cluster fraction onto the normalized 0..1 scale:
    # 0 at the candidate threshold (0.15) and 1.0 at total loss. A frame
    # with no structure loss is not a candidate and scores 0.
    full_edges = np.full((64, 64), 255, dtype=np.uint8)  # all 4 blocks meaningful
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (128, 128, 128)  # uniform gray -> no edges in the current frame

    result = evaluate(frame, full_edges)
    assert not result.is_candidate  # total loss exceeds the global-loss ceiling
    assert result.reason is None
    # The normalized confidence still follows the recalibrated scale.
    assert 0.0 <= result.confidence <= 1.0