"""Deterministic synthetic CI fixture generator (Phase 4).

Materializes the media assets the ground-truth detector tests and the
end-to-end pipeline tests need, so a fresh or headless checkout can run the
full suite without committing binary files:

    data/test_footage/test_video.mp4   50 s, 30 FPS, 640x360 synthetic clip
    data/baselines/cam1.json           brightness + sharpness baseline record
    data/baselines/cam1.jpg            reference frame (DISK baseline)
    data/baselines/cam1_edges.png      stable-edge baseline (tampering)

The scene is procedural and fully deterministic (fixed seed): a multiscale
value-noise texture with structural shapes, a fine-detail noise layer, and a
small moving object. Faults are injected as physical pixel transforms in
exactly the time windows recorded in
``tests/fixtures/test_video_ground_truth.json``:

    tampering  14-18 s    dark occluding region over the lens
    low_light  26-29.9 s  HSV value-channel scaling
    blur       36-39 s    Gaussian defocus
    tilt       45-48.9 s  affine rotation (+ small translation)

Encoding uses a codec fallback loop (mp4v -> avc1 -> MJPG -> XVID) because
encoder availability differs across platforms and OpenCV builds, followed by
a smoke check that reopens the produced file and verifies the frame count and
that every frame decodes to non-empty, correctly sized content. Baselines are
captured from the generated clip with the production
``pipeline.capture_baseline`` code path, so they match the video by
construction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from config import BASELINES_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

# --- Scene / fixture constants ---------------------------------------------
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_FPS = 30
DEFAULT_DURATION_SECONDS = 50
DEFAULT_SEED = 20260101
DEFAULT_CAMERA_ID = "cam1"

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test_footage" / "test_video.mp4"
GROUND_TRUTH_PATH = PROJECT_ROOT / "tests" / "fixtures" / "test_video_ground_truth.json"

# Fault windows mirroring tests/fixtures/test_video_ground_truth.json:
# (start_s, end_s, fault_type). generate_fixtures() cross-checks these against
# the committed JSON when it is present, so a drift between the two fails
# loudly instead of producing a fixture the tests cannot use.
FAULT_WINDOWS: tuple[tuple[float, float, str], ...] = (
    (14.0, 18.0, "tampering"),
    (26.0, 29.9, "low_light"),
    (36.0, 39.0, "blur"),
    (45.0, 48.9, "tilt"),
)

# Tilt injection. The tilt detector flags a camera when the median
# matched-keypoint displacement reaches TILT_MEDIAN_SHIFT_THRESHOLD_RATIO
# (0.1) of the frame diagonal, so the synthetic event must move keypoints
# past that with margin while staying mild enough for DISK matching to stay
# reliable: a moderate rotation plus a small translation (a physically
# plausible "camera knocked" event). Provisional; validated by the
# automated ground-truth suite (tests/test_tilt_detector.py).
TILT_ROTATION_DEGREES = 22.0
TILT_TRANSLATION_PX = (80.0, 25.0)

# Fine-detail noise amplitude: the sharpness source the blur detector and the
# edge density the tampering detector measures both come from this layer.
DETAIL_NOISE_SIGMA = 13.0

# Codec fallback order (first that writes AND reads back cleanly wins).
_CODECS: tuple[tuple[str, str], ...] = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("MJPG", ".mp4"),
    ("XVID", ".mp4"),
)
_MAX_FRAME_COUNT_TOLERANCE = 2


def _in_window(t: float, start_s: float, end_s: float) -> bool:
    """Same inclusive window test the ground-truth tests use."""
    return start_s <= t <= end_s


def _faults_at(t: float) -> list[str]:
    return [name for start_s, end_s, name in FAULT_WINDOWS if _in_window(t, start_s, end_s)]


def _render_base_scene(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Static procedural background shared by every frame (BGR uint8)."""
    # Multiscale value noise gives an organic, texture-rich base.
    texture = np.zeros((height, width), np.float32)
    for grid, amplitude in ((6, 70.0), (12, 40.0), (24, 22.0), (48, 12.0)):
        field = rng.random((grid, grid), dtype=np.float32)
        upscaled = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
        texture += (upscaled - 0.5) * amplitude

    # Gentle vertical lighting gradient.
    gradient = np.linspace(-25.0, 35.0, height, dtype=np.float32)[:, None]
    texture += gradient

    # Shift to mid-gray: the octaves above are zero-centered, so without this
    # the scene would render near black and the low-light baseline would
    # leave no headroom for the detector's relative threshold.
    texture += 128.0

    # Fine-grain detail: the sharpness/edge source the detectors measure.
    detail = rng.standard_normal((height, width), dtype=np.float32) * DETAIL_NOISE_SIGMA
    base = np.clip(texture + detail, 0.0, 255.0).astype(np.uint8)
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Bright sun disc.
    cv2.circle(
        frame,
        (int(width * 0.82), int(height * 0.20)),
        int(height * 0.12),
        (200, 205, 220),
        -1,
    )

    # Vertical fence posts: strong, persistent edge structure.
    bar_width = max(6, int(width * 0.02))
    for i in range(6):
        x = int(width * (0.14 + i * 0.13))
        cv2.rectangle(
            frame,
            (x, int(height * 0.42)),
            (x + bar_width, int(height * 0.88)),
            (90, 95, 105),
            -1,
        )

    # Buildings with window grids: many corners for DISK keypoints.
    buildings = (
        (int(width * 0.05), int(height * 0.60), int(width * 0.28), height - 1, (110, 118, 130)),
        (int(width * 0.30), int(height * 0.55), int(width * 0.48), height - 1, (80, 86, 96)),
        (int(width * 0.72), int(height * 0.62), int(width * 0.95), height - 1, (100, 108, 118)),
    )
    for left, top, right, bottom, color in buildings:
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
        win_w, win_h = max(8, (right - left) // 10), max(6, (bottom - top) // 12)
        step_x, step_y = win_w + 4, win_h + 4
        for wx in range(left + 4, right - win_w - 2, step_x):
            for wy in range(top + 4, bottom - win_h - 2, step_y):
                cv2.rectangle(frame, (wx, wy), (wx + win_w, wy + win_h), (220, 224, 232), -1)

    return frame


def _overlay_motion(frame: np.ndarray, t: float) -> np.ndarray:
    """Draw one small moving object so the scene is not perfectly static.

    The object stays small and low-texture so it never forms a large
    connected structure-loss cluster (tampering), a measurable whole-frame
    brightness/sharpness change, or a majority of matched keypoints (tilt).
    """
    height, width = frame.shape[:2]
    out = frame.copy()
    vehicle_w, vehicle_h = 30, 20
    road_y = int(height * 0.90) - vehicle_h
    x = int((t * 8.0) % (width + vehicle_w * 2)) - vehicle_w
    cv2.rectangle(out, (x, road_y), (x + vehicle_w, road_y + vehicle_h), (60, 66, 76), -1)
    cv2.rectangle(out, (x + 6, road_y + 4), (x + 14, road_y + 10), (150, 158, 170), -1)
    cv2.circle(out, (x + 6, road_y + vehicle_h), 3, (30, 32, 36), -1)
    cv2.circle(out, (x + vehicle_w - 6, road_y + vehicle_h), 3, (30, 32, 36), -1)
    return out


def _apply_tampering(frame: np.ndarray) -> np.ndarray:
    """Localized dark occlusion: a flat region with soft (blurred) edges.

    The interior is texture-free, so the blocks underneath lose their edge
    structure -- the tampering detector's signature. The soft edge avoids a
    crisp ring that would register as a new edge rather than structure loss.
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.float32)
    top, bottom = int(height * 0.30), int(height * 0.72)
    left, right = int(width * 0.22), int(width * 0.82)
    cv2.rectangle(mask, (left, top), (right, bottom), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 8.0)
    occlusion = np.full((height, width, 3), (18, 20, 22), dtype=np.uint8)
    alpha = np.clip(mask, 0.0, 1.0)[:, :, None]
    return (frame * (1.0 - alpha) + occlusion * alpha).astype(np.uint8)


def _apply_low_light(frame: np.ndarray) -> np.ndarray:
    """Global darkening: scale the HSV value channel by 0.25."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    scaled = np.clip(hsv[:, :, 2].astype(np.float32) * 0.25, 0.0, 255.0).astype(np.uint8)
    hsv[:, :, 2] = scaled
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _apply_blur(frame: np.ndarray) -> np.ndarray:
    """Defocus: heavy Gaussian blur (sigma 7) kills the fine-detail edges."""
    return cv2.GaussianBlur(frame, (0, 0), 7.0)


def _apply_tilt(
    frame: np.ndarray,
    angle_degrees: float = TILT_ROTATION_DEGREES,
    translation_px: tuple[float, float] = TILT_TRANSLATION_PX,
) -> np.ndarray:
    """Affine rotation matrix (rotation + small translation) applied to the
    whole frame. BORDER_REPLICATE keeps the brightness signature unchanged so
    the low-light detector stays quiet during the tilt window.
    """
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    matrix[0, 2] += translation_px[0]
    matrix[1, 2] += translation_px[1]
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _render_frame(base: np.ndarray, t: float) -> np.ndarray:
    """Render one frame at video time ``t`` with any active fault transform."""
    frame = _overlay_motion(base, t)
    for fault in _faults_at(t):
        if fault == "tampering":
            frame = _apply_tampering(frame)
        elif fault == "low_light":
            frame = _apply_low_light(frame)
        elif fault == "blur":
            frame = _apply_blur(frame)
        elif fault == "tilt":
            frame = _apply_tilt(frame)
        else:
            raise ValueError(f"Unknown fault type {fault!r}.")
    return frame


def _frame_stream(
    height: int,
    width: int,
    fps: int,
    duration_seconds: float,
    seed: int,
) -> "callable[[], object]":
    """Return a factory producing a fresh, deterministic frame generator.

    A factory (not a plain generator) lets the codec fallback loop re-render
    identical frames for each codec attempt without holding 1500 frames in
    memory.
    """
    total_frames = int(round(duration_seconds * fps))

    def _frames() -> object:
        rng = np.random.default_rng(seed)
        base = _render_base_scene(rng, height, width)
        for i in range(total_frames):
            yield _render_frame(base, i / fps)

    return _frames


def _smoke_check(video_path: Path, expected_frames: int, expected_shape: tuple[int, int]) -> int:
    """Reopen the produced video and verify it decodes correctly.

    Returns the decoded frame count. Raises RuntimeError on any failure:
    unreadable frame rate, empty/None frames, wrong dimensions, no content,
    or a frame count that is off by more than the tolerance.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"No valid frame rate read back from {video_path}.")
        decoded = 0
        first_mean: float | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Frame {decoded} decoded as empty.")
            if tuple(frame.shape[:2]) != expected_shape:
                raise RuntimeError(
                    f"Frame {decoded} has shape {tuple(frame.shape[:2])}, "
                    f"expected {expected_shape}."
                )
            if first_mean is None:
                first_mean = float(frame.mean())
            decoded += 1
        if first_mean is None or first_mean < 1.0:
            raise RuntimeError(f"{video_path} contains no measurable content.")
        if abs(decoded - expected_frames) > _MAX_FRAME_COUNT_TOLERANCE:
            raise RuntimeError(
                f"Decoded {decoded} frames, expected ~{expected_frames} "
                f"(tolerance {_MAX_FRAME_COUNT_TOLERANCE})."
            )
        return decoded
    finally:
        cap.release()


def _write_video(
    video_path: Path,
    frame_stream_factory: "callable[[], object]",
    fps: int,
    resolution: tuple[int, int],
    expected_frames: int,
) -> tuple[str, int]:
    """Write the clip with codec fallback; return (fourcc, decoded_frames)."""
    width, height = resolution
    errors: list[str] = []
    for fourcc_name, _suffix in _CODECS:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        if video_path.exists():
            video_path.unlink()
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            errors.append(f"{fourcc_name}: writer failed to open")
            continue
        try:
            written = 0
            for frame in frame_stream_factory():
                writer.write(frame)
                written += 1
        except Exception as exc:  # noqa: BLE001 - report and fall through
            writer.release()
            errors.append(f"{fourcc_name}: write failed: {exc}")
            continue
        writer.release()
        try:
            decoded = _smoke_check(video_path, expected_frames, (height, width))
        except RuntimeError as exc:
            errors.append(f"{fourcc_name}: smoke check failed: {exc}")
            continue
        logger.info(
            "Encoded fixture with fourcc %r (%d/%d frames decoded, %.1f s).",
            fourcc_name, decoded, written, written / fps,
        )
        return fourcc_name, decoded
    raise RuntimeError(
        f"All video codecs failed for {video_path}: " + "; ".join(errors)
    )


def _capture_baselines(camera_id: str, video_path: Path) -> dict:
    """Capture cam1 baseline files with the production capture code path."""
    from pipeline.capture_baseline import capture_baseline

    return capture_baseline(camera_id, str(video_path))


def _cross_check_ground_truth() -> None:
    """Fail loudly if FAULT_WINDOWS drifts from the committed test JSON."""
    if not GROUND_TRUTH_PATH.exists():
        logger.warning(
            "Ground-truth file %s not found; skipping drift check.",
            GROUND_TRUTH_PATH,
        )
        return
    data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    recorded = {(f["start_s"], f["end_s"], f["type"]) for f in data["faults"]}
    ours = {(start_s, end_s, name) for start_s, end_s, name in FAULT_WINDOWS}
    if recorded != ours:
        raise RuntimeError(
            f"FAULT_WINDOWS {sorted(ours)} do not match committed ground truth "
            f"{sorted(recorded)} at {GROUND_TRUTH_PATH}."
        )


def _write_ground_truth(video_path: Path, camera_id: str) -> None:
    """Regenerate the ground-truth JSON (--write-ground-truth)."""
    try:
        relative = video_path.resolve().relative_to(PROJECT_ROOT.resolve())
        video_ref = str(relative).replace("\\", "/")
    except ValueError:
        video_ref = str(video_path)
    payload = {
        "video": video_ref,
        "camera_id": camera_id,
        "faults": [{"type": name, "start_s": start_s, "end_s": end_s} for start_s, end_s, name in FAULT_WINDOWS],
    }
    GROUND_TRUTH_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_fixtures(
    video_path: str | Path | None = None,
    camera_id: str = DEFAULT_CAMERA_ID,
    fps: int = DEFAULT_FPS,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    resolution: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    seed: int = DEFAULT_SEED,
    write_baseline: bool = True,
    write_ground_truth: bool = False,
) -> dict:
    """Generate the synthetic fixture video and (optionally) the baselines.

    Returns a metadata dict (paths, codec, frame counts, baseline record).
    Raises RuntimeError/ValueError on any failure so callers (the conftest,
    CI) fail loudly instead of silently running the suite without fixtures.
    """
    video_path = Path(video_path) if video_path is not None else DEFAULT_OUTPUT
    _cross_check_ground_truth()

    width, height = resolution
    if width <= 0 or height <= 0 or fps <= 0 or duration_seconds <= 0:
        raise ValueError(
            f"resolution, fps, and duration must be positive; got "
            f"{resolution}, {fps}, {duration_seconds}."
        )
    expected_frames = int(round(duration_seconds * fps))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    fourcc, decoded = _write_video(
        video_path,
        _frame_stream(height, width, fps, duration_seconds, seed),
        fps,
        (width, height),
        expected_frames,
    )

    baseline_record: dict | None = None
    if write_baseline:
        baseline_record = _capture_baselines(camera_id, video_path)
    if write_ground_truth:
        _write_ground_truth(video_path, camera_id)

    return {
        "video_path": str(video_path),
        "camera_id": camera_id,
        "fourcc": fourcc,
        "frames_written": expected_frames,
        "frames_decoded": decoded,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "resolution": list(resolution),
        "seed": seed,
        "baseline_record": baseline_record,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the generator CLI."""
    parser = argparse.ArgumentParser(
        prog="generate_test_fixtures",
        description="Generate the synthetic CI fixture video and cam1 baselines.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help="Output video path (default: data/test_footage/test_video.mp4).",
    )
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID,
                        help="Camera id used for the baseline files (default: cam1).")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS,
                        help="Clip duration in seconds (default: 50).")
    parser.add_argument("--resolution", default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
                        help="Frame resolution WIDTHxHEIGHT (default: 640x360).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Deterministic scene seed (default: %(default)s).")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip capturing the cam1 baselines.")
    parser.add_argument("--write-ground-truth", action="store_true",
                        help="Regenerate tests/fixtures/test_video_ground_truth.json.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 success, 1 generation failure, 2 usage error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        width_s, height_s = args.resolution.lower().split("x")
        resolution = (int(width_s), int(height_s))
    except ValueError:
        print(f"error: invalid --resolution {args.resolution!r}; expected WIDTHxHEIGHT",
              file=sys.stderr)
        return 2

    try:
        result = generate_fixtures(
            video_path=args.out,
            camera_id=args.camera_id,
            fps=args.fps,
            duration_seconds=args.duration,
            resolution=resolution,
            seed=args.seed,
            write_baseline=not args.no_baseline,
            write_ground_truth=args.write_ground_truth,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("Fixture generation failed: %s", exc)
        return 1
    logger.info("Fixture generated: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

