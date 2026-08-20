"""Automated ground-truth validation for the low-light detector.

Replaces the manual CSV-eyeballing check with repeatable assertions.
Requires a captured baseline for the fixture's camera_id
(run pipeline.capture_baseline first) and the fixture's video file
to exist on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import (
    BASELINES_DIR,
    PROJECT_ROOT,
    TEST_MAX_UNRELATED_FALSE_POSITIVE_RATE as MAX_UNRELATED_FALSE_POSITIVE_RATE,
    TEST_MIN_TRUE_POSITIVE_CANDIDATE_RATE as MIN_TRUE_POSITIVE_CANDIDATE_RATE,
)
from detectors.brightness import evaluate
from pipeline.file_reader import read_frames_from_file

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "test_video_ground_truth.json"

# A single low-light detector isn't expected to fully ignore every
# other fault type — tampering can also darken a frame, a known,
# accepted overlap the plan defers to the decision layer. Only faults
# unrelated to brightness (blur, tilt) are held to a strict silence
# requirement here.
BRIGHTNESS_ADJACENT_FAULT_TYPES = {"tampering"}


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def baseline_dark_ratio(fixture_data: dict) -> float:
    camera_id = fixture_data["camera_id"]
    baseline_path = BASELINES_DIR / f"{camera_id}.json"
    if not baseline_path.exists():
        pytest.skip(f"No baseline captured for {camera_id!r}; run pipeline.capture_baseline first.")
    return json.loads(baseline_path.read_text())["lowlight_dark_pixel_ratio"]


@pytest.fixture(scope="module")
def scored_frames(fixture_data: dict, baseline_dark_ratio: float) -> list[dict]:
    video_path = PROJECT_ROOT / fixture_data["video"]
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}.")

    results = []
    for frame_number, video_time_s, frame in read_frames_from_file(video_path):
        result = evaluate(frame, baseline_dark_ratio)
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


def test_low_light_window_is_detected(fixture_data: dict, scored_frames: list[dict]) -> None:
    fault = next(f for f in fixture_data["faults"] if f["type"] == "low_light")
    window_frames = [
        f for f in scored_frames if _in_window(f["video_time_s"], fault["start_s"], fault["end_s"])
    ]
    assert window_frames, "No frames fell inside the low-light fault window; check fixture timing."

    candidate_rate = sum(f["is_candidate"] for f in window_frames) / len(window_frames)
    assert candidate_rate >= MIN_TRUE_POSITIVE_CANDIDATE_RATE, (
        f"Low-light window candidate rate {candidate_rate:.2f} is below the "
        f"required {MIN_TRUE_POSITIVE_CANDIDATE_RATE:.2f}."
    )


def test_unrelated_faults_do_not_trigger_false_positives(
    fixture_data: dict, scored_frames: list[dict]
) -> None:
    for fault in fixture_data["faults"]:
        if fault["type"] == "low_light" or fault["type"] in BRIGHTNESS_ADJACENT_FAULT_TYPES:
            continue

        window_frames = [
            f for f in scored_frames if _in_window(f["video_time_s"], fault["start_s"], fault["end_s"])
        ]
        assert window_frames, f"No frames fell inside the {fault['type']} window; check fixture timing."

        candidate_rate = sum(f["is_candidate"] for f in window_frames) / len(window_frames)
        assert candidate_rate <= MAX_UNRELATED_FALSE_POSITIVE_RATE, (
            f"{fault['type']} window unexpectedly triggered low-light candidates "
            f"at rate {candidate_rate:.2f}."
        )


def test_clean_footage_does_not_trigger_false_positives(
    fixture_data: dict, scored_frames: list[dict]
) -> None:
    fault_windows = [(f["start_s"], f["end_s"]) for f in fixture_data["faults"]]

    clean_frames = [
        f
        for f in scored_frames
        if not any(_in_window(f["video_time_s"], start, end) for start, end in fault_windows)
    ]
    assert clean_frames, "No clean (non-fault) frames found; check fixture timing."

    candidate_rate = sum(f["is_candidate"] for f in clean_frames) / len(clean_frames)
    assert candidate_rate <= MAX_UNRELATED_FALSE_POSITIVE_RATE, (
        f"Clean footage unexpectedly triggered candidates at rate {candidate_rate:.2f}."
    )