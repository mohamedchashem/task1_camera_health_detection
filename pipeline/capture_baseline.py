"""Manual per-camera baseline capture.

Run against a clean, fault-free segment of footage for a given camera.
Produces saved reference data every detector compares live frames
against: brightness, stable edge structure, and sharpness. Generic to
any camera/footage — nothing here is specific to a particular video.

Accepts either a live RTSP URL or a local file path as the source.

A capture whose computed ``quality_warnings`` list is non-empty is
REFUSED by default: a dark/blurry/unstructured baseline makes the
tampering detector's ``evaluate()`` report ``degraded_baseline`` with
confidence 0.0 on every frame (and never a candidate) until a usable
baseline is captured — with no visible runtime failure. Refusing the
write forces the operator to either recapture from a genuinely clean
segment or explicitly acknowledge the degraded capture with
``--acknowledge-degraded``, which persists the baseline and records the
acknowledgment on the saved JSON record.

CLI:
    python -m pipeline.capture_baseline <camera_id> <source>
    python -m pipeline.capture_baseline <camera_id> <source> --acknowledge-degraded
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from config import (
    BASELINE_CAPTURE_SECONDS,
    BASELINE_MAX_DARK_RATIO,
    BASELINE_MIN_SHARPNESS,
    BASELINES_DIR,
    TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO,
    validate_config,
)
from detectors.blur import compute_sharpness
from detectors.brightness import compute_dark_pixel_ratio
from detectors.tampering import baseline_degradation_info, compute_edge_map
from pipeline.file_reader import read_frames_from_file
from pipeline.paths import validate_camera_id
from pipeline.stream_reader import read_frames

logger = logging.getLogger(__name__)


class DegradedBaselineRefusedError(RuntimeError):
    """A capture produced quality warnings and was not acknowledged, so
    nothing was persisted.

    Carries the offending warnings as structured data (``quality_warnings``)
    so callers can log or relay them without re-parsing the message text.
    """

    def __init__(self, camera_id: str, quality_warnings: list[str]) -> None:
        self.camera_id = camera_id
        self.quality_warnings = list(quality_warnings)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [
            f"Refusing to persist degraded baseline for camera {self.camera_id!r}: "
            f"{len(self.quality_warnings)} quality warning(s)."
        ]
        lines.extend(f"  - {warning}" for warning in self.quality_warnings)
        lines.append(
            "Persisting this baseline would make tampering detection inoperative "
            "for this camera until a usable baseline is captured: "
            "detectors.tampering.evaluate() reports 'degraded_baseline' "
            "(confidence 0.0, never a candidate) on every frame."
        )
        lines.append(
            "Recapture from a clean, well-lit, in-focus segment, or re-run with "
            "--acknowledge-degraded to explicitly accept the degraded baseline."
        )
        return "\n".join(lines)


def _windowed_frames(source: str) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, elapsed_seconds_within_window) from either a live
    RTSP stream or a local file, using whichever timing source is
    accurate for that source type.
    """
    is_live_stream = source.lower().startswith("rtsp://")

    if is_live_stream:
        capture_start: float | None = None
        for _frame_number, _stream_elapsed, frame in read_frames(source):
            if capture_start is None:
                capture_start = time.monotonic()
            yield frame, time.monotonic() - capture_start
    else:
        for _frame_number, video_time_s, frame in read_frames_from_file(Path(source)):
            yield frame, video_time_s


def capture_baseline(camera_id: str, source: str, acknowledge_degraded: bool = False) -> dict:
    """Capture and persist a baseline for one camera.

    A capture whose ``quality_warnings`` list is non-empty is refused by
    default: raises ``DegradedBaselineRefusedError`` and persists nothing.
    Pass ``acknowledge_degraded=True`` to persist anyway; the
    acknowledgment is recorded on the saved record as
    ``degraded_acknowledged`` (plus a timestamp). A clean capture is
    persisted normally and carries no acknowledgment fields.

    Returns the baseline record that was saved, for logging/confirmation.
    """
    camera_id = validate_camera_id(camera_id)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    dark_ratios: list[float] = []
    sharpness_values: list[float] = []
    edge_accumulator: np.ndarray | None = None
    frame_count = 0
    reference_frame = None

    for frame, window_elapsed in _windowed_frames(source):
        dark_ratios.append(compute_dark_pixel_ratio(frame))
        sharpness_values.append(compute_sharpness(frame))

        edges = compute_edge_map(frame)
        if edge_accumulator is None:
            edge_accumulator = np.zeros(edges.shape, dtype=np.float64)
        edge_accumulator += (edges > 0)
        frame_count += 1

        if reference_frame is None and window_elapsed >= BASELINE_CAPTURE_SECONDS / 2:
            reference_frame = frame

        if window_elapsed >= BASELINE_CAPTURE_SECONDS:
            break

    if not dark_ratios or edge_accumulator is None:
        raise RuntimeError(f"No frames captured for camera {camera_id!r}; check source.")

    edge_persistence = edge_accumulator / frame_count
    stable_edges = (edge_persistence >= TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO).astype(np.uint8) * 255

    mean_dark_ratio = sum(dark_ratios) / len(dark_ratios)
    mean_sharpness = sum(sharpness_values) / len(sharpness_values)
    structure_fraction, _structure_degraded, tampering_warning = baseline_degradation_info(
        stable_edges
    )

    # Capture-time quality gates. A capture that trips any warning is
    # degraded: a dark/blurry/unstructured baseline makes the tampering
    # detector's evaluate() report "degraded_baseline" (confidence 0.0)
    # on every frame until a usable baseline is captured, with no visible
    # runtime failure. Such a baseline must therefore not be silently
    # persisted — it is refused unless the operator explicitly
    # acknowledges it (--acknowledge-degraded), and the acknowledgment is
    # recorded on the saved JSON record.
    quality_warnings: list[str] = []
    if mean_dark_ratio > BASELINE_MAX_DARK_RATIO:
        quality_warnings.append(
            f"mean dark-pixel ratio {mean_dark_ratio:.3f} is above "
            f"BASELINE_MAX_DARK_RATIO ({BASELINE_MAX_DARK_RATIO}); capture looks too dark"
        )
    if mean_sharpness < BASELINE_MIN_SHARPNESS:
        quality_warnings.append(
            f"mean Laplacian sharpness {mean_sharpness:.1f} is below "
            f"BASELINE_MIN_SHARPNESS ({BASELINE_MIN_SHARPNESS}); capture looks blurry"
        )
    if tampering_warning is not None:
        quality_warnings.append(tampering_warning)
    for warning in quality_warnings:
        logger.warning("Baseline quality for %s: %s", camera_id, warning)

    # Refuse the degraded capture before anything is written. A
    # previously existing baseline for this camera is intentionally left
    # untouched: the operator may have a valid older capture and merely
    # re-captured from a bad segment.
    if quality_warnings and not acknowledge_degraded:
        raise DegradedBaselineRefusedError(camera_id, quality_warnings)

    captured_at = time.time()
    baseline_record = {
        "camera_id": camera_id,
        "lowlight_dark_pixel_ratio": mean_dark_ratio,
        "blur_baseline_sharpness": mean_sharpness,
        "tampering_meaningful_block_fraction": structure_fraction,
        "quality_warnings": quality_warnings,
        "frames_averaged": frame_count,
        "captured_at": captured_at,
    }
    if quality_warnings:  # acknowledged; an unacknowledged capture raised above
        baseline_record["degraded_acknowledged"] = True
        baseline_record["degraded_acknowledged_at"] = captured_at

    image_path = BASELINES_DIR / f"{camera_id}.jpg"
    json_path = BASELINES_DIR / f"{camera_id}.json"
    edges_path = BASELINES_DIR / f"{camera_id}_edges.png"

    cv2.imwrite(str(image_path), reference_frame)
    cv2.imwrite(str(edges_path), stable_edges)
    json_path.write_text(json.dumps(baseline_record, indent=2))

    return baseline_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the baseline-capture CLI."""
    parser = argparse.ArgumentParser(
        prog="pipeline.capture_baseline",
        description=(
            "Capture and persist a per-camera baseline. A capture whose "
            "quality_warnings list is non-empty (dark/blurry/unstructured) "
            "is refused unless --acknowledge-degraded is given."
        ),
    )
    parser.add_argument("camera_id", help="Camera id (letters, digits, '-' and '_').")
    parser.add_argument(
        "source",
        help="RTSP URL or local video file path to capture the baseline from.",
    )
    parser.add_argument(
        "--acknowledge-degraded",
        action="store_true",
        help="Persist a degraded baseline despite quality warnings. The "
             "acknowledgment is recorded on the saved JSON record "
             "(degraded_acknowledged / degraded_acknowledged_at).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 persisted, 1 degraded-refusal or failure."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_config()
    args = parse_args(argv)

    try:
        result = capture_baseline(
            args.camera_id,
            args.source,
            acknowledge_degraded=args.acknowledge_degraded,
        )
    except DegradedBaselineRefusedError as exc:
        logger.error("\n%s", exc)
        return 1

    logger.info("Baseline saved for %s: %s", result["camera_id"], result)
    return 0


if __name__ == "__main__":
    sys.exit(main())