"""Tilt / angle-change fault detector.

Detects camera physical rotation via learned local feature matching
(DISK, accessed through Kornia) between the current frame and a
camera's baseline reference frame. Measures the median displacement
of matched keypoints directly, rather than fitting a full homography
or affine transform — geometric model fitting proved numerically
unstable on real footage where sparse/clustered matches during faults
produced degenerate fits (condition numbers in the millions), making
any transform-based metric unreliable, including using a fitted
model purely as a RANSAC outlier filter (RANSAC's internal fit is
the same degenerate model, so it silently discarded real tilt
matches along with the bad ones).

Outlier rejection instead uses median absolute deviation (MAD): the
median of matched-point displacements is inherently robust to
outliers, and MAD-based rejection (the "X84 rule") filters remaining
noise using only the displacement distribution itself, with no
geometric model of the scene involved at any point.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import kornia.feature as KF
import numpy as np
import torch

from config import (
    TILT_MAX_KEYPOINTS,
    TILT_MIN_RELIABLE_MATCHES,
    TILT_MIN_MATCH_RATIO,
    TILT_MIN_INLIER_RATIO,
    TILT_MEDIAN_SHIFT_THRESHOLD_RATIO,
    TILT_MAD_REJECTION_THRESHOLD,
    TILT_MAD_EPSILON,
    TILT_SHIFT_CONFIDENCE_CEILING_RATIO,
    TILT_MATCH_RATIO_THRESHOLD,
    TILT_MODEL_NAME,
    TILT_DEVICE,
)

logger = logging.getLogger(__name__)

# Tilt estimate statuses (see TiltResult.reason).
TILT_STATUS_OK = "ok"
TILT_STATUS_INSUFFICIENT_MATCHES = "insufficient_matches"
TILT_STATUS_NO_FEATURES = "no_features"

# The DISK model is deliberately NOT initialized at import time: importing
# this module must not download weights or construct the model. Loading
# happens lazily on the first extract_features() call. configure() can
# inject a pre-built model or point at local weights before that happens.
_lock = threading.Lock()
_device: torch.device | None = None
_disk_model: torch.nn.Module | None = None
_weights_path: Path | None = None


@dataclass(frozen=True)
class TiltResult:
    median_shift_ratio: float  # raw score: median keypoint displacement / frame diagonal
    confidence: float
    is_candidate: bool
    reliable: bool
    reason: str | None = None  # TILT_STATUS_*; None for pre-guard code paths
    match_count: int = 0       # correspondences surviving mutual-NN matching


def match_volume_sufficient(
    match_count: int,
    baseline_keypoint_count: int,
    current_keypoint_count: int,
) -> bool:
    """Guard that a rotation estimate has enough correspondence volume.

    Requires an absolute floor of matched pairs AND that the matches
    represent a meaningful fraction of the SMALLER of the two frames'
    keypoint sets. On severely blurred/obscured/near-black frames DISK can
    still emit a handful of spurious correspondences that pass the absolute
    floor; their median displacement is meaningless but would otherwise read
    as a huge false tilt. Pure function (no tensors) so it is unit-testable.
    """
    if match_count < TILT_MIN_RELIABLE_MATCHES:
        return False
    denominator = min(baseline_keypoint_count, current_keypoint_count)
    if denominator <= 0:
        return False
    return (match_count / denominator) >= TILT_MIN_MATCH_RATIO


def inlier_ratio_sufficient(inlier_count: int, match_count: int) -> bool:
    """Guard that most matches agree after MAD outlier rejection.

    A genuine camera displacement moves features coherently, so most matches
    are inliers; blur/obscuration garbage is scattered and MAD rejects most
    of it. Pure function (no tensors) so it is unit-testable.
    """
    if match_count <= 0:
        return False
    return (
        inlier_count >= TILT_MIN_RELIABLE_MATCHES
        and (inlier_count / match_count) >= TILT_MIN_INLIER_RATIO
    )


def _auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure(
    device: str | torch.device | None = None,
    model: torch.nn.Module | None = None,
    weights_path: str | Path | None = None,
) -> None:
    """Explicitly configure the DISK model and device used by extract_features().

    No model is loaded or downloaded here and none is loaded at import
    time: the model is constructed lazily on the first extract_features()
    call. Re-callable -- later calls replace the previous configuration.

    Args:
        device: Explicit torch device ("cpu", "cuda:0", ...). Defaults to
            config.TILT_DEVICE if set, otherwise auto-selects CUDA-if-
            available at first model load.
        model: A pre-built feature-extraction model with DISK's calling
            convention (``forward(images, n, pad_if_not_divisible=...)``
            returning per-image features with ``keypoints`` and
            ``descriptors``). Tests can inject a fake here to avoid
            loading the real DISK weights. The model is used as provided;
            the caller is responsible for placing it on ``device``.
        weights_path: Optional local DISK checkpoint file to load instead
            of kornia's pretrained download. Raises FileNotFoundError on
            first extraction if the file is missing.
    """
    global _device, _disk_model, _weights_path

    if device is not None:
        _device = torch.device(device)
    elif TILT_DEVICE is not None:
        _device = torch.device(TILT_DEVICE)

    if model is not None:
        _disk_model = model
        _weights_path = None
    elif weights_path is not None:
        _disk_model = None
        _weights_path = Path(weights_path)
    else:
        # No model/weights argument: reset to the default lazy behavior
        # (pretrained checkpoint from kornia/torch.hub).
        _disk_model = None
        _weights_path = None


def _load_disk_from_checkpoint(weights_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(weights_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Configured DISK weights file not found at {checkpoint_path}. "
            "Unset TILT_DISK_WEIGHTS_PATH (or configure without weights_path) "
            "to use kornia's pretrained download instead."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = KF.DISK()
    model.load_state_dict(checkpoint["extractor"])
    model.to(device)
    model.eval()
    return model


def _ensure_disk_model() -> torch.nn.Module:
    """Return the lazily-constructed DISK model, building it exactly once.

    Safe against concurrent first calls (single-flight under a lock).
    """
    global _device, _disk_model, _weights_path
    if _disk_model is None:
        with _lock:
            if _disk_model is None:
                _device = _device or _auto_device()
                if _weights_path is not None:
                    _disk_model = _load_disk_from_checkpoint(_weights_path, _device)
                else:
                    _disk_model = KF.DISK.from_pretrained(TILT_MODEL_NAME, device=_device)
                logger.info("Tilt detector loaded DISK model on device: %s", _device)
    return _disk_model


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    device = _device or _auto_device()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def extract_features(frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DISK on one frame; returns (keypoints, descriptors).

    The DISK model is constructed lazily on the first call (never at
    module import time). See configure() to inject a pre-built model or
    point at a local weights file before the first call.
    """
    model = _ensure_disk_model()
    tensor = None
    features = None
    try:
        tensor = _frame_to_tensor(frame)
        with torch.no_grad():
            features = model(tensor, TILT_MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
        return features.keypoints, features.descriptors
    finally:
        # Explicit local tensor cleanup: ``tensor`` (the device copy of the
        # frame) and ``features`` (the model's per-image output container)
        # own the bulk of this call's transient VRAM. Dropping them here,
        # rather than waiting for the GC, bounds peak VRAM across repeated
        # sub-sampled calls. The returned tensors survive: they are already
        # referenced by the return tuple before ``finally`` runs.
        del tensor, features


def _mad_inlier_mask(displacements: np.ndarray) -> np.ndarray:
    """Flag inlier displacements using median absolute deviation (MAD).

    MAD is a robust measure of spread: unlike standard deviation, a
    handful of extreme outliers can't drag it around, because it's
    built from medians rather than means. This makes it a good fit
    here specifically because it requires no geometric model of the
    scene (unlike RANSAC+homography) -- it looks only at the spread
    of the displacement values themselves.

    The 1.4826 scale factor is the standard correction that makes MAD
    comparable to a standard deviation under normal (Gaussian) noise,
    so `threshold` behaves like a "number of standard deviations" cutoff.
    A threshold of 3 (the "X84 rule") is a well-established default in
    robust-statistics and feature-tracking literature.
    """
    median = np.median(displacements)
    abs_deviations = np.abs(displacements - median)
    mad = 1.4826 * np.median(abs_deviations)

    if mad < TILT_MAD_EPSILON:
        # All displacements are already nearly identical (e.g. a static
        # scene) -- nothing to reject, and dividing by a near-zero MAD
        # would falsely flag normal variation as outliers.
        return np.ones_like(displacements, dtype=bool)

    z_scores = abs_deviations / mad
    return z_scores <= TILT_MAD_REJECTION_THRESHOLD


def compute_median_shift_ratio(
    baseline_keypoints: torch.Tensor,
    baseline_descriptors: torch.Tensor,
    current_frame: np.ndarray,
    baseline_frame_shape: tuple[int, int] | None = None,
) -> tuple[float, bool, str, int]:
    """Measure how much matched keypoints have moved between baseline
    and current frame. Returns (shift_ratio, reliable, reason, match_count).

    Uses median displacement of MAD-filtered matches (robust to
    outlier matches without fitting any geometric model), normalized
    by frame diagonal (scale-independent across resolutions).

    Reliability requires the match-volume guards (absolute count, relative
    match ratio, inlier ratio). When they fail, ``reliable`` is False and
    ``reason`` explains why (TILT_STATUS_*); ``match_count`` reports the
    number of mutual-NN correspondences for observability. All guard math
    runs on the on-device tensors -- tensors are pulled to CPU only after
    the guards pass, so degraded frames never pay the transfer cost.

    ``baseline_frame_shape`` is the (height, width) of the frame the
    baseline features were extracted from. When provided, a current
    frame with different dimensions raises ValueError: keypoint
    displacements are only meaningful when both frames share a single
    pixel coordinate space. Callers that know the baseline frame
    should always pass it.
    """
    if baseline_frame_shape is not None and tuple(current_frame.shape[:2]) != tuple(baseline_frame_shape):
        raise ValueError(
            f"Current frame shape {tuple(current_frame.shape[:2])} does not match "
            f"baseline frame shape {tuple(baseline_frame_shape)}. Re-capture the "
            "baseline at the camera's current resolution."
        )

    height, width = current_frame.shape[:2]

    # Explicit local tensor cleanup: the current-frame features produced here
    # and the SMNN matching output are this call's transient GPU tensors.
    # Dropping them in ``finally`` keeps VRAM bounded across repeated tilt
    # evaluations. The ``baseline_keypoints`` / ``baseline_descriptors``
    # tensors are owned by the caller (the camera baseline) and are
    # intentionally NOT deleted here.
    current_keypoints = None
    current_descriptors = None
    _distances = None
    match_idxs = None
    match_count = 0
    try:
        current_keypoints, current_descriptors = extract_features(current_frame)

        with torch.no_grad():
            _distances, match_idxs = KF.match_smnn(
                baseline_descriptors, current_descriptors, th=TILT_MATCH_RATIO_THRESHOLD,
            )

        # Shape metadata is available without transferring tensor data, so
        # all volume guards run before any .cpu() / .numpy() transfer.
        match_count = int(match_idxs.shape[0])
        baseline_kp_count = int(baseline_keypoints.shape[0])
        current_kp_count = int(current_keypoints.shape[0])

        if not match_volume_sufficient(match_count, baseline_kp_count, current_kp_count):
            logger.warning(
                "Only %d matches from %d/%d keypoints (need >= %d and >= %.3f of "
                "the smaller set); tilt estimate skipped as unreliable.",
                match_count, baseline_kp_count, current_kp_count,
                TILT_MIN_RELIABLE_MATCHES, TILT_MIN_MATCH_RATIO,
            )
            return 0.0, False, TILT_STATUS_INSUFFICIENT_MATCHES, match_count

        points_baseline = baseline_keypoints[match_idxs[:, 0]].cpu().numpy()
        points_current = current_keypoints[match_idxs[:, 1]].cpu().numpy()

        displacements = np.linalg.norm(points_current - points_baseline, axis=1)
        inliers = _mad_inlier_mask(displacements)
        inlier_count = int(inliers.sum())

        if not inlier_ratio_sufficient(inlier_count, match_count):
            logger.warning(
                "Only %d/%d MAD-filtered inliers (need >= %d and ratio >= %.3f); "
                "tilt estimate skipped as unreliable.",
                inlier_count, match_count,
                TILT_MIN_RELIABLE_MATCHES, TILT_MIN_INLIER_RATIO,
            )
            return 0.0, False, TILT_STATUS_INSUFFICIENT_MATCHES, match_count

        # Median displacement of inlier-filtered matched points,
        # normalized by frame diagonal for scale independence.
        frame_diagonal = np.hypot(width, height)
        shift_ratio = float(np.median(displacements[inliers]) / frame_diagonal)

        return shift_ratio, True, TILT_STATUS_OK, match_count
    finally:
        del current_keypoints, current_descriptors, _distances, match_idxs


def evaluate(
    current_frame: np.ndarray,
    baseline_keypoints: torch.Tensor,
    baseline_descriptors: torch.Tensor,
    baseline_frame_shape: tuple[int, int] | None = None,
) -> TiltResult:
    """Score a frame's framing shift relative to a camera's precomputed baseline features.

    ``baseline_frame_shape`` is the (height, width) of the frame the
    baseline features were extracted from; see compute_median_shift_ratio.
    """
    shift_ratio, reliable, reason, match_count = compute_median_shift_ratio(
        baseline_keypoints, baseline_descriptors, current_frame, baseline_frame_shape
    )

    if not reliable:
        return TiltResult(
            median_shift_ratio=0.0,
            confidence=0.0,
            is_candidate=False,
            reliable=False,
            reason=reason,
            match_count=match_count,
        )

    is_candidate = shift_ratio >= TILT_MEDIAN_SHIFT_THRESHOLD_RATIO
    # Linear severity map: confidence reaches 1.0 when the median keypoint
    # shift reaches TILT_SHIFT_CONFIDENCE_CEILING_RATIO of the frame
    # diagonal (a clearly moved camera), and scales proportionally below
    # that, so real rotations land in the upper half of the range instead
    # of being compressed near zero. Candidate gating is separate and stays
    # on the (lower) TILT_MEDIAN_SHIFT_THRESHOLD_RATIO.
    confidence = float(np.clip(shift_ratio / TILT_SHIFT_CONFIDENCE_CEILING_RATIO, 0.0, 1.0))

    return TiltResult(
        median_shift_ratio=shift_ratio,
        confidence=confidence,
        is_candidate=is_candidate,
        reliable=True,
        reason=reason,
        match_count=match_count,
    )
