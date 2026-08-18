"""Manual per-camera baseline capture.

Run against a clean, fault-free segment of footage for a given camera.
Produces a saved reference frame (for visual/manual audit) and the
numeric baseline values each detector compares live frames against.
"""

from __future__ import annotations

import json
import time

import cv2

from config import BASELINE_CAPTURE_SECONDS, BASELINES_DIR
from detectors.brightness import compute_dark_pixel_ratio
from pipeline.paths import validate_camera_id
from pipeline.stream_reader import read_frames


def capture_baseline(camera_id: str, stream_url: str) -> dict:
    """Capture and persist a baseline for one camera.

    Returns the baseline record that was saved, for logging/confirmation.
    """
    camera_id = validate_camera_id(camera_id)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    dark_ratios: list[float] = []
    reference_frame = None
    capture_start: float | None = None

    for frame_number, _stream_elapsed, frame in read_frames(stream_url):
        if capture_start is None:
            capture_start = time.monotonic()  # start the window on first frame received

        dark_ratios.append(compute_dark_pixel_ratio(frame))
        window_elapsed = time.monotonic() - capture_start

        # Keep a frame from partway through the window as the saved
        # visual reference, rather than the very first (possibly still
        # settling after connect) or very last frame.
        if reference_frame is None and window_elapsed >= BASELINE_CAPTURE_SECONDS / 2:
            reference_frame = frame

        if window_elapsed >= BASELINE_CAPTURE_SECONDS:
            break

    if not dark_ratios:
        raise RuntimeError(f"No frames captured for camera {camera_id!r}; check stream_url.")

    baseline_record = {
        "camera_id": camera_id,
        "lowlight_dark_pixel_ratio": sum(dark_ratios) / len(dark_ratios),
        "frames_averaged": len(dark_ratios),
        "captured_at": time.time(),
    }

    image_path = BASELINES_DIR / f"{camera_id}.jpg"
    json_path = BASELINES_DIR / f"{camera_id}.json"

    cv2.imwrite(str(image_path), reference_frame)
    json_path.write_text(json.dumps(baseline_record, indent=2))

    return baseline_record


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m pipeline.capture_baseline <camera_id> <stream_url>")
        sys.exit(1)

    result = capture_baseline(sys.argv[1], sys.argv[2])
    print(f"Baseline saved for {result['camera_id']}: {result}")