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
from detectors.blur import compute_sharpness
from detectors.blur import evaluate as evaluate_blur
from detectors.tampering import compute_edge_map, compute_loss_fractions, evaluate
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
    # A uniform-gray current frame has no edges at all, so every block's
    # retention ratio is 0 and the frame's ambient level itself collapses to
    # zero. Under the ambient-retention model this is the DEGENERATE case:
    # the frame demonstrably retains no structure anywhere, so no obstruction
    # claim is meaningful and the detector must not confirm (the fully
    # ambiguous case is handled by Approach C in a later subtask). The reason
    # surfaces that diagnostic, where the legacy model reported a plain
    # non-candidate with no reason at all.
    full_edges = np.full((64, 64), 255, dtype=np.uint8)  # all 4 blocks meaningful
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (128, 128, 128)  # uniform gray -> no edges in the current frame

    result = evaluate(frame, full_edges)
    assert not result.is_candidate
    assert result.reason == "degraded_ambient"
    assert result.confidence == 0.0  # no measurable obstruction magnitude


def _bug_case_scene() -> np.ndarray:
    """Synthetic scene reproducing the cam_08 bug-case physics.

    Strong COARSE structure (large checkerboard squares) gives the ambient
    region edges that survive a moderate defocus, while strong FINE detail
    (dense 4px lines) gives a high baseline Laplacian sharpness that the
    defocus destroys -- so a blurred frame reads as severe blur (high blur
    confidence) yet still retains measurable ambient edge structure (the new
    model's ambient floor is 0.1, and the blurred ambient retains ~0.17).
    """
    height, width = 384, 384  # 12 x 12 grid of 32px blocks
    scene = np.full((height, width, 3), 120, dtype=np.uint8)
    step = 16
    for y in range(0, height, step):
        for x in range(0, width, step):
            color = (235, 235, 235) if ((x // step) + (y // step)) % 2 == 0 else (15, 15, 15)
            scene[y : y + 8, x : x + 8] = color
    scene[::4, :] = 200  # fine detail: high baseline sharpness
    scene[:, ::4] = 60
    return scene


def test_large_obstruction_with_severe_blur_is_detected_end_to_end() -> None:
    # Real cam_08 evidence reproduction through the ACTUAL evaluate() path:
    # a large obstruction (120 of 144 blocks, ~83%) co-occurring with severe
    # global blur. The legacy absolute total-loss ceiling rejected this exact
    # case (total loss > 0.75). The ambient-relative model must flag it: the
    # obstruction stays a deep statistical outlier against the collapsed
    # (blurred) ambient level.
    scene = _bug_case_scene()
    baseline_edges = compute_edge_map(scene)

    frame = cv2.GaussianBlur(scene, (0, 0), 1.5)  # severe global defocus
    frame[0:320, :] = (15, 15, 15)                 # large obstruction: ~83% of blocks

    result = evaluate(frame, baseline_edges)
    assert result.is_candidate is True  # the bug case is now detected
    assert result.reason is None
    # raw_confidence ~0.8, matching the cam_08 evidence: confidence stays on
    # the legacy magnitude scale (largest obstruction cluster as a fraction
    # of meaningful blocks), so ~83% coverage maps to ~0.8.
    assert result.confidence == pytest.approx(0.80, abs=0.05)
    # The obstruction coverage exceeds the old 0.75 hard ceiling, i.e. the
    # legacy gate rejected exactly this frame.
    assert result.total_loss_fraction > 0.75
    assert result.largest_contiguous_loss_fraction > 0.75
    # Direct proof the frame is the bug case: the LEGACY binary structure-loss
    # model computes a total disappeared fraction above the old ceiling too.
    legacy_total = compute_loss_fractions(baseline_edges, compute_edge_map(frame))[1]
    assert legacy_total > 0.75

    # Sanity: the co-occurring degradation genuinely reads as severe blur.
    blur_result = evaluate_blur(frame, compute_sharpness(scene))
    assert bool(blur_result.is_candidate)
    assert blur_result.confidence >= 0.90


def test_evaluate_maps_reject_reason_to_diagnostic_reason() -> None:
    # Two large but DISCONNECTED obstructions (3 block rows each): each passes
    # ambient/depth/size, but together they hold only 50% of all obstruction
    # blocks, so the compactness gate rejects. The assessment's reject reason
    # "compactness" must surface on the result as a diagnostic reason (status
    # OK -- a measured negative, not a non-measurement).
    scene = _bug_case_scene()
    baseline_edges = compute_edge_map(scene)
    frame = scene.copy()
    frame[0:96, :] = (15, 15, 15)     # block rows 0-2
    frame[224:320, :] = (15, 15, 15)  # block rows 7-9, disconnected from the first
    result = evaluate(frame, baseline_edges)
    assert result.is_candidate is False
    assert result.reason == "tampering_scattered"

    # A clean frame (scene unchanged) is a plain negative: no reason diagnostic.
    clean = evaluate(scene, baseline_edges)
    assert clean.is_candidate is False
    assert clean.reason is None
