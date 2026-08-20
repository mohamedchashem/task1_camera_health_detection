"""Automated ground-truth validation for the blur detector.

Like the tampering detector, this cannot yet cleanly separate real
blur from every other fault type -- see STRUCTURE_OVERLAP_FAULT_TYPES
below. That's tracked here explicitly rather than hidden.
"""

from __future__ import annotations

import json

import pytest

from config import (
    BASELINES_DIR,
    PROJECT_ROOT,
    TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE as MAX_UNRELATED_FALSE_POSITIVE_RATE,
    TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE as MIN_TRUE_POSITIVE_CANDIDATE_RATE,
)
from detectors.blur import evaluate
from pipeline.file_reader import read_frames_from_file

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "test_video_ground_truth.json"

# Laplacian-variance blur detection cannot yet cleanly separate real
# out-of-focus/dirty-lens blur from other faults that also reduce
# measurable edge sharpness: tampering (a textureless obstruction has
# near-zero edge content, indistinguishable from blur by this signal),
# low-light (reduced contrast lowers Laplacian variance even on an
# otherwise-sharp scene), and tilt to a lesser degree (25.6% candidate
# rate measured -- meaningfully above the ~3% clean-footage baseline,
# so not claimed as silent even though it's the least-affected of the
# three). This is a known, measured limitation of a single detector
# working in isolation -- resolving it is explicitly the job of the
# cross-triggering decision layer, which has visibility across all
# four detectors at once. See project plan, "Known cross-cutting risk
# to design around."
STRUCTURE_OVERLAP_FAULT_TYPES = {"tampering", "low_light", "tilt"}


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def baseline_sharpness(fixture_data: dict) -> float:
    camera_id = fixture_data["camera_id"]
    baseline_path = BASELINES_DIR / f"{camera_id}.json"
    if not baseline_path.exists():
        pytest.skip(f"No baseline captured for {camera_id!r}; run pipeline.capture_baseline first.")
    return json.loads(baseline_path.read_text())["blur_baseline_sharpness"]


@pytest.fixture(scope="module")
def scored_frames(fixture_data: dict, baseline_sharpness: float) -> list[dict]:
    video_path = PROJECT_ROOT / fixture_data["video"]
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}.")

    results = []
    for frame_number, video_time_s, frame in read_frames_from_file(video_path):
        result = evaluate(frame, baseline_sharpness)
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


def test_blur_window_is_detected(fixture_data: dict, scored_frames: list[dict]) -> None:
    fault = next(f for f in fixture_data["faults"] if f["type"] == "blur")
    window_frames = [f for f in scored_frames if _in_window(f["video_time_s"], fault["start_s"], fault["end_s"])]
    assert window_frames, "No frames fell inside the blur fault window; check fixture timing."

    candidate_rate = sum(f["is_candidate"] for f in window_frames) / len(window_frames)
    assert candidate_rate >= MIN_TRUE_POSITIVE_CANDIDATE_RATE, (
        f"Blur window candidate rate {candidate_rate:.2f} is below the "
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
    unaccounted = fixture_fault_types - STRUCTURE_OVERLAP_FAULT_TYPES - {"blur"}
    assert not unaccounted, (
        f"Fault types {unaccounted} are in the fixture but not accounted for -- "
        "decide whether they belong in STRUCTURE_OVERLAP_FAULT_TYPES or need "
        "a strict false-positive check."
    )