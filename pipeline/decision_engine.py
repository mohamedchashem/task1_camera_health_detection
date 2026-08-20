"""Decision layer: per-frame fusion and temporal confirmation.

Consumes the four detectors' per-frame results and produces:
- a per-frame ``DecisionFrame`` (primary fault, symptoms, suppression,
  temporal status, per-detector status);
- ``ConfirmedFault`` events on temporal state transitions (Phase 2
  persists these to the event store).

Fault isolation guarantee: a detector that raises is contained per frame
(observation status ``error``) while the remaining detectors still run;
after repeated consecutive errors a detector is temporarily skipped
(backoff), then retried automatically.

Fusion contract (approved Phase 1 spec):
- candidates are ranked by ``DECISION_PRECEDENCE`` (higher severity
  first; confidence breaks only ties);
- the primary is the highest-ranked candidate not suppressed by a
  higher-ranked *active* suppressor (confidence >=
  ``DECISION_SUPPRESSOR_MIN_CONFIDENCE``);
- an active candidate beats a below-gate candidate of lower precedence;
  if no candidate is active, the top-ranked candidate still becomes
  primary (documented limitation of the V1 rules);
- ``suppressed_faults`` are the remaining candidates the primary's
  signal explains (``DECISION_SUPPRESSION_MAP``); everything else is a
  ``secondary_symptom``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from config import (
    DECISION_CONFIRMATION_MIN_POSITIVE_RATIO,
    DECISION_CONFIRMATION_MIN_WINDOW_FRAMES,
    DECISION_CONFIRMATION_WINDOW_SECONDS,
    DECISION_DETECTOR_ERROR_BACKOFF_SECONDS,
    DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS,
    DECISION_MIN_EVENT_GAP_SECONDS,
    DECISION_PRECEDENCE,
    DECISION_SUPPRESSION_MAP,
    DECISION_SUPPRESSOR_MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# Detector observation statuses
DETECTOR_STATUS_OK = "ok"
DETECTOR_STATUS_ERROR = "error"
DETECTOR_STATUS_SKIPPED = "skipped"

# Temporal confirmation statuses (per fault, per frame)
TEMPORAL_STATUS_PENDING = "pending"
TEMPORAL_STATUS_CONFIRMED = "confirmed"

# ConfirmedFault event statuses
EVENT_STATUS_CONFIRMED = "confirmed"
EVENT_STATUS_CLEARED = "cleared"

_FAULT_RANK = {fault: rank for rank, fault in enumerate(DECISION_PRECEDENCE)}


@dataclass(frozen=True)
class DetectorObservation:
    """Result of running one detector on one frame, after isolation."""

    detector: str
    status: str                # DETECTOR_STATUS_OK | ERROR | SKIPPED
    is_candidate: bool
    confidence: float
    error_message: str | None = None


@dataclass(frozen=True)
class DecisionFrame:
    """Per-frame fused decision for one camera."""

    camera_id: str
    frame_number: int
    video_time_s: float
    primary_fault: str | None
    confidence: float
    secondary_symptoms: tuple[str, ...]
    suppressed_faults: tuple[str, ...]
    temporal_confirmation_status: dict[str, str]
    detectors: tuple[DetectorObservation, ...]


@dataclass(frozen=True)
class ConfirmedFault:
    """Event emitted when a fault is confirmed or cleared temporally."""

    camera_id: str
    fault_type: str
    status: str                # EVENT_STATUS_CONFIRMED | CLEARED
    started_frame: int
    started_time_s: float
    ended_frame: int | None
    ended_time_s: float | None
    peak_confidence: float
    window_positive_rate: float
    session_id: str = "default"  # stream session; part of the idempotency key


def split_candidates_for_primary(
    primary: str,
    candidate_names: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition simultaneously-firing candidates around a fixed primary.

    Used for per-event annotations: when several detectors confirm on the
    same frame, each event's banner lists the other firing candidates as
    ``suppressed`` (signals ``primary`` causally explains, per
    ``DECISION_SUPPRESSION_MAP``) or as ``secondary`` symptoms. Input
    order (precedence) is preserved.
    """
    suppressed_set = DECISION_SUPPRESSION_MAP.get(primary, ())
    suppressed = tuple(f for f in candidate_names if f != primary and f in suppressed_set)
    secondary = tuple(f for f in candidate_names if f != primary and f not in suppressed_set)
    return secondary, suppressed


def resolve_primary_fault(
    candidates: Mapping[str, float],
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """Resolve the primary fault among per-frame candidates.

    Args:
        candidates: mapping fault_type -> confidence, for faults flagged
            as candidates by their detector this frame. Fault names must
            be in ``DECISION_PRECEDENCE``.

    Returns:
        (primary_fault, secondary_symptoms, suppressed_faults). The
        tuples preserve precedence order.
    """

    def sort_key(fault: str) -> tuple[int, float]:
        return _FAULT_RANK[fault], -candidates[fault]

    ranked = sorted(candidates, key=sort_key)

    primary: str | None = None
    fallback: str | None = None

    for i, fault in enumerate(ranked):
        higher = ranked[:i]
        suppressed_by_higher = any(
            candidates[g] >= DECISION_SUPPRESSOR_MIN_CONFIDENCE
            and fault in DECISION_SUPPRESSION_MAP.get(g, ())
            for g in higher
        )
        if suppressed_by_higher:
            continue
        if candidates[fault] >= DECISION_SUPPRESSOR_MIN_CONFIDENCE:
            primary = fault
            break
        if fallback is None:
            fallback = fault

    if primary is None:
        primary = fallback
    if primary is None:
        return None, (), ()

    secondary, suppressed = split_candidates_for_primary(primary, ranked)
    return primary, secondary, suppressed


def fuse_observations(
    observations: Sequence[DetectorObservation],
) -> tuple[str | None, tuple[str, ...], tuple[str, ...], float]:
    """Pure per-frame fusion of detector observations.

    Only observations with status ``ok`` participate: errored or skipped
    detectors never contribute candidates.

    Returns (primary_fault, secondary_symptoms, suppressed_faults,
    confidence). Confidence is the primary's detector confidence (0.0
    when there is no primary).
    """
    candidates = {
        obs.detector: obs.confidence
        for obs in observations
        if obs.status == DETECTOR_STATUS_OK and obs.is_candidate
    }
    primary, secondary, suppressed = resolve_primary_fault(candidates)
    confidence = candidates.get(primary, 0.0) if primary is not None else 0.0
    return primary, secondary, suppressed, confidence
class ConfirmationTracker:
    """Temporal confirmation state machine for one camera.

    Time-based sliding window per fault: a fault is *confirmed* while the
    fraction of candidate frames among observed frames in the last
    ``window_seconds`` is at least ``min_positive_ratio`` and at least
    ``min_window_frames`` frames were observed. Frames where a detector
    did not run are NOT counted as observations (they are simply not
    passed to ``update``); their old window entries age out naturally.

    Callers must provide monotonically increasing ``video_time_s``.
    """

    def __init__(
        self,
        camera_id: str,
        window_seconds: float = DECISION_CONFIRMATION_WINDOW_SECONDS,
        min_positive_ratio: float = DECISION_CONFIRMATION_MIN_POSITIVE_RATIO,
        min_window_frames: int = DECISION_CONFIRMATION_MIN_WINDOW_FRAMES,
        min_event_gap_seconds: float = DECISION_MIN_EVENT_GAP_SECONDS,
        session_id: str = "default",
    ) -> None:
        self._camera_id = camera_id
        self._session_id = session_id
        self._window_seconds = window_seconds
        self._min_positive_ratio = min_positive_ratio
        self._min_window_frames = min_window_frames
        self._min_event_gap_seconds = min_event_gap_seconds

        self._windows: dict[str, deque[tuple[int, float, bool, float]]] = {}
        self._confirmed: set[str] = set()
        self._confirmed_since: dict[str, tuple[int, float]] = {}
        self._peak_confidence: dict[str, float] = {}
        self._last_event_end_time: dict[str, float] = {}

    def update(
        self,
        fault_candidates: Mapping[str, float],
        observed_faults: Iterable[str],
        frame_number: int,
        video_time_s: float,
    ) -> list[ConfirmedFault]:
        """Advance the per-fault sliding windows for one frame.

        Args:
            fault_candidates: mapping fault_type -> confidence for the
                faults that were candidates this frame.
            observed_faults: faults whose detectors ran successfully this
                frame (candidate or not). Unlisted faults are treated as
                not observed.
            frame_number, video_time_s: identity/timestamp of the frame.

        Returns:
            The ``ConfirmedFault`` events emitted by state transitions
            (confirmations and clearances) for this frame.
        """
        events: list[ConfirmedFault] = []
        observed = set(observed_faults)

        for fault in observed:
            window = self._windows.setdefault(fault, deque())
            window.append(
                (frame_number, video_time_s, fault in fault_candidates, fault_candidates.get(fault, 0.0))
            )

        cutoff = video_time_s - self._window_seconds
        for fault, window in list(self._windows.items()):
            while window and window[0][1] < cutoff:
                window.popleft()

            rate, frame_count = self._window_rate(fault)
            confirmable = (
                frame_count >= self._min_window_frames
                and rate >= self._min_positive_ratio
            )

            if fault in self._confirmed:
                if confirmable:
                    self._update_peak(fault)
                else:
                    events.append(
                        self._make_event(fault, EVENT_STATUS_CLEARED, frame_number, video_time_s, rate)
                    )
                    self._unmark_confirmed(fault)
            elif confirmable and self._can_confirm_now(fault, video_time_s):
                self._mark_confirmed(fault, frame_number, video_time_s)
                events.append(
                    self._make_event(fault, EVENT_STATUS_CONFIRMED, frame_number, video_time_s, rate)
                )

        return events

    def confirmation_status(self) -> dict[str, str]:
        """Current per-fault temporal status for observed faults."""
        return {
            fault: TEMPORAL_STATUS_CONFIRMED if fault in self._confirmed else TEMPORAL_STATUS_PENDING
            for fault in self._windows
        }

    def _can_confirm_now(self, fault: str, video_time_s: float) -> bool:
        return video_time_s >= (
            self._last_event_end_time.get(fault, float("-inf")) + self._min_event_gap_seconds
        )

    def _window_rate(self, fault: str) -> tuple[float, int]:
        window = self._windows.get(fault, deque())
        count = len(window)
        if count == 0:
            return 0.0, 0
        positives = sum(1 for _frame, _t, is_candidate, _c in window if is_candidate)
        return positives / count, count

    def _mark_confirmed(self, fault: str, frame_number: int, video_time_s: float) -> None:
        window = self._windows.get(fault, deque())
        # The event starts at the first candidate frame in the window that
        # established the confirmation, not at the confirmation frame itself.
        started = next(
            ((f, t) for f, t, is_candidate, _c in window if is_candidate),
            (frame_number, video_time_s),
        )
        self._confirmed.add(fault)
        self._confirmed_since[fault] = started
        self._update_peak(fault)

    def _unmark_confirmed(self, fault: str) -> None:
        self._confirmed.discard(fault)
        self._confirmed_since.pop(fault, None)
        self._peak_confidence.pop(fault, None)

    def _update_peak(self, fault: str) -> None:
        window = self._windows.get(fault, deque())
        current = max((c for _frame, _t, is_candidate, c in window if is_candidate), default=0.0)
        self._peak_confidence[fault] = max(self._peak_confidence.get(fault, 0.0), current)

    def _make_event(
        self, fault: str, status: str, frame_number: int, video_time_s: float, rate: float
    ) -> ConfirmedFault:
        start_frame, start_time = self._confirmed_since[fault]
        if status == EVENT_STATUS_CLEARED:
            self._last_event_end_time[fault] = video_time_s
            peak = self._peak_confidence.get(fault, 0.0)
            self._peak_confidence.pop(fault, None)
            return ConfirmedFault(
                camera_id=self._camera_id,
                fault_type=fault,
                status=status,
                started_frame=start_frame,
                started_time_s=start_time,
                ended_frame=frame_number,
                ended_time_s=video_time_s,
                peak_confidence=peak,
                window_positive_rate=rate,
                session_id=self._session_id,
            )
        return ConfirmedFault(
            camera_id=self._camera_id,
            fault_type=fault,
            status=status,
            started_frame=start_frame,
            started_time_s=start_time,
            ended_frame=None,
            ended_time_s=None,
            peak_confidence=self._peak_confidence.get(fault, 0.0),
            window_positive_rate=rate,
            session_id=self._session_id,
        )

class DecisionEngine:
    """Per-camera frame processor: isolation + fusion + temporal confirmation.

    One instance per camera. ``detectors`` maps a fault type (a name in
    ``DECISION_PRECEDENCE``) to a callable ``callable(frame) -> result``
    where ``result`` exposes ``is_candidate`` and ``confidence``. The
    pipeline binds the camera's baselines into these callables (Phase 3).
    """

    def __init__(
        self,
        camera_id: str,
        detectors: Mapping[str, Callable[[np.ndarray], object]],
        window_seconds: float = DECISION_CONFIRMATION_WINDOW_SECONDS,
        min_positive_ratio: float = DECISION_CONFIRMATION_MIN_POSITIVE_RATIO,
        min_window_frames: int = DECISION_CONFIRMATION_MIN_WINDOW_FRAMES,
        min_event_gap_seconds: float = DECISION_MIN_EVENT_GAP_SECONDS,
        max_consecutive_errors: int = DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS,
        error_backoff_seconds: float = DECISION_DETECTOR_ERROR_BACKOFF_SECONDS,
        session_id: str = "default",
    ) -> None:
        unknown = set(detectors) - set(DECISION_PRECEDENCE)
        if unknown:
            raise ValueError(f"Unknown detector names: {sorted(unknown)}")

        self._camera_id = camera_id
        self._detectors = dict(detectors)
        self._tracker = ConfirmationTracker(
            camera_id=camera_id,
            session_id=session_id,
            window_seconds=window_seconds,
            min_positive_ratio=min_positive_ratio,
            min_window_frames=min_window_frames,
            min_event_gap_seconds=min_event_gap_seconds,
        )
        self._max_consecutive_errors = max_consecutive_errors
        self._error_backoff_seconds = error_backoff_seconds
        self._consecutive_errors: dict[str, int] = {}
        self._backoff_until: dict[str, float] = {}
        self._pending_events: list[ConfirmedFault] = []

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        video_time_s: float,
        enabled_faults: Iterable[str] | None = None,
    ) -> DecisionFrame:
        """Run enabled detectors (with isolation), fuse, and advance temporal state.

        Args:
            enabled_faults: fault types (names in ``DECISION_PRECEDENCE``) to
                run for this frame. ``None`` (default) runs every configured
                detector. A detector not enabled for a frame yields a
                ``status="skipped"`` observation that is NOT added to the
                tracker's observed set, so an unobserved frame cannot dilute
                the confirmation positive ratio.
        """
        if enabled_faults is None:
            enabled = set(self._detectors)
        else:
            enabled = set(enabled_faults)
            unknown = enabled - set(DECISION_PRECEDENCE)
            if unknown:
                raise ValueError(f"Unknown fault names in enabled_faults: {sorted(unknown)}")

        observations: list[DetectorObservation] = []
        for name in DECISION_PRECEDENCE:
            if name not in self._detectors:
                continue
            if name not in enabled:
                observations.append(
                    DetectorObservation(name, DETECTOR_STATUS_SKIPPED, False, 0.0)
                )
                continue
            observations.append(
                self._run_detector(name, self._detectors[name], frame, frame_number, video_time_s)
            )

        primary, secondary, suppressed, confidence = fuse_observations(observations)

        candidates = {
            obs.detector: obs.confidence
            for obs in observations
            if obs.status == DETECTOR_STATUS_OK and obs.is_candidate
        }
        # A candidate suppressed by the precedence rules (its signal is
        # explained by the primary's physical cause) is not independent
        # evidence: it must not register as a positive sample in its own
        # confirmation tracker for this frame.
        candidates = {
            fault: conf
            for fault, conf in candidates.items()
            if fault not in suppressed
        }
        # Only OK observations count as observed: skipped detectors (disabled,
        # sub-sampled, or in backoff) must not dilute the confirmation ratio.
        observed = [obs.detector for obs in observations if obs.status == DETECTOR_STATUS_OK]
        self._pending_events.extend(
            self._tracker.update(candidates, observed, frame_number, video_time_s)
        )

        return DecisionFrame(
            camera_id=self._camera_id,
            frame_number=frame_number,
            video_time_s=video_time_s,
            primary_fault=primary,
            confidence=confidence,
            secondary_symptoms=secondary,
            suppressed_faults=suppressed,
            temporal_confirmation_status=self._tracker.confirmation_status(),
            detectors=tuple(observations),
        )

    def drain_events(self) -> list[ConfirmedFault]:
        """Return and clear the ConfirmedFault events emitted so far."""
        events = self._pending_events
        self._pending_events = []
        return events

    def _run_detector(
        self,
        name: str,
        detector_call: Callable[[np.ndarray], object],
        frame: np.ndarray,
        frame_number: int,
        video_time_s: float,
    ) -> DetectorObservation:
        if video_time_s < self._backoff_until.get(name, float("-inf")):
            return DetectorObservation(name, DETECTOR_STATUS_SKIPPED, False, 0.0)

        try:
            result = detector_call(frame)
            is_candidate = bool(result.is_candidate)
            confidence = float(result.confidence)
        except Exception as exc:  # noqa: BLE001 - containment is the point
            self._consecutive_errors[name] = self._consecutive_errors.get(name, 0) + 1
            if self._consecutive_errors[name] >= self._max_consecutive_errors:
                self._backoff_until[name] = video_time_s + self._error_backoff_seconds
                self._consecutive_errors[name] = 0
            logger.error(
                "Detector %r failed for camera %r (frame %d, t=%.3f): %s",
                name, self._camera_id, frame_number, video_time_s, exc,
            )
            return DetectorObservation(name, DETECTOR_STATUS_ERROR, False, 0.0, str(exc))

        self._consecutive_errors[name] = 0
        return DetectorObservation(name, DETECTOR_STATUS_OK, is_candidate, confidence)

