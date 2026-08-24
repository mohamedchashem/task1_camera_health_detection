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

Fusion contract (multi-label revision, Phase 1):
- per-frame candidates are filtered to those reaching their per-fault
  emission floor (``DECISION_CONFIRM_MIN_CONFIDENCE``);
- a candidate is removed when a *surviving* higher-precedence active
  suppressor (confidence >= ``DECISION_SUPPRESSOR_MIN_CONFIDENCE``)
  suppresses it via ``_pair_should_suppress``, which routes the ordered
  pair through ``DECISION_SUPPRESSION_RULES``: the relative margin
  ``suppressor * DECISION_SUPPRESSION_MARGIN >= suppressed`` always, plus
  for ``conditional`` relations a physical predicate on the two
  candidates' raw metrics (area conservation, near-black floor). Pairs
  absent from the rules are independent and always co-survive;
- ``classify_candidates`` partitions each frame's candidates into three
  disjoint buckets: ``survivors`` (exactly the ``resolve_active_faults``
  output), ``below_floor`` (confidence below the candidate's own emission
  floor in ``DECISION_CONFIRM_MIN_CONFIDENCE`` -- too weak to ever be
  suppressed), and ``suppressed`` (cleared its own floor but removed by a
  surviving higher-precedence fault -- real causal suppression only);
- survivors become ``DecisionFrame.faults`` in ``DECISION_PRECEDENCE``
  order; ``primary_fault`` is the top-ranking survivor (backward
  compatible), the remaining survivors are ``secondary_symptoms``;
  removed floor-clearing candidates are ``suppressed_faults`` and
  sub-floor candidates are ``below_floor_faults``;
- ``DecisionFrame.confirmed_faults`` additionally carries every fault
  currently in the tracker's confirmed state (peak confidence per fault,
  ``DECISION_PRECEDENCE`` order) -- the camera's full set of
  currently-confirmed faults, independent of this frame's survivors;
- the V1 single-primary rules remain available as
  ``resolve_primary_fault`` / ``fuse_observations`` (unchanged behavior).

Execution gating: detectors run in ``DECISION_EXECUTION_ORDER`` (cheap
signal detectors first, expensive structural last). A gate detector that
fires a candidate at or above its gate floor (``DECISION_GATE_CONFIDENCE``,
overridden per gate by ``DECISION_GATE_CONFIDENCE_BY_GATE``) causes the
detectors listed under it in ``DECISION_GATE_SKIP_MAP`` to be skipped for
that frame (status ``skipped``, reason ``suppressed_by_gate``), so e.g. a
high-confidence low-light frame never runs DISK keypoint matching on
near-black pixels, and a severely blurred frame (blur >= 0.9) skips the
unmeasurable tilt detector instead of risking a spurious false tilt.
Skipped detectors are excluded from the confirmation tracker, so their
windows age out instead of being polluted by unmeasurable frames.

Emission gating: a candidate frame counts toward temporal confirmation
only when its confidence reaches the fault's floor in
``DECISION_CONFIRM_MIN_CONFIDENCE``, so weak noise is never logged as a
confirmed event.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from config import (
    DECISION_CONFIRMATION_MIN_POSITIVE_RATIO,
    DECISION_CONFIRMATION_MIN_WINDOW_FRAMES,
    DECISION_CONFIRMATION_WINDOW_SECONDS,
    DECISION_CONFIRM_MIN_CONFIDENCE,
    DECISION_DETECTOR_ERROR_BACKOFF_SECONDS,
    DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS,
    DECISION_EXECUTION_ORDER,
    DECISION_GATE_CONFIDENCE,
    DECISION_GATE_CONFIDENCE_BY_GATE,
    DECISION_GATE_SKIP_MAP,
    DECISION_MIN_EVENT_GAP_SECONDS,
    DECISION_PRECEDENCE,
    DECISION_SUPPRESSION_MAP,
    DECISION_SUPPRESSION_MARGIN,
    DECISION_SUPPRESSION_RULES,
    DECISION_SUPPRESSOR_MIN_CONFIDENCE,
    LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR,
    TAMPERING_BLUR_AREA_SLACK,
    TAMPERING_LOW_LIGHT_AREA_SLACK,
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

# Tolerance for the relative-margin comparisons: ``suppressor * MARGIN >=
# suppressed`` must not fail due to binary floating-point rounding.
_MARGIN_EPSILON = 1e-9


def _margin_met(suppressor_confidence: float, suppressed_confidence: float) -> bool:
    """True when an active suppressor clears the relative margin against the
    candidate it would suppress (approved formula: suppressor_confidence *
    DECISION_SUPPRESSION_MARGIN >= suppressed_confidence)."""
    return (
        suppressor_confidence * DECISION_SUPPRESSION_MARGIN + _MARGIN_EPSILON
        >= suppressed_confidence
    )


# Raw measurands each detector already computes and exposes on its result
# object, copied into DetectorObservation.metrics so downstream predicate
# logic can read physical evidence (obstruction coverage, dark ratio,
# sharpness ratio, tilt shift) without re-running detectors. Keys are fault
# names in DECISION_PRECEDENCE. No new calculations: only values the
# detectors already produce internally.
_DETECTOR_METRIC_FIELDS: Mapping[str, tuple[str, ...]] = {
    "low_light": ("dark_pixel_ratio", "relative_increase"),
    "tampering": ("meaningful_block_fraction", "total_loss_fraction"),
    "blur": ("sharpness", "sharpness_ratio"),
    "tilt": ("median_shift_ratio", "match_count"),
}


def _extract_detector_metrics(name: str, result: object) -> dict[str, float]:
    """Copy ``result``'s raw measurands for detector ``name`` into a metrics dict.

    Only numeric fields listed in ``_DETECTOR_METRIC_FIELDS`` for ``name`` are
    copied. A listed field the detector does not expose (e.g. a mocked result
    in tests) simply contributes no entry, so extraction never raises.
    """
    metrics: dict[str, float] = {}
    for field_name in _DETECTOR_METRIC_FIELDS.get(name, ()):
        value = getattr(result, field_name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[field_name] = float(value)
    return metrics


def _area_conserved(obstruction_coverage: float, measured_increase: float, slack: float) -> bool:
    """True when the suppressor's measured extent can plausibly account for
    the suppressed detector's observed increase.

    Area-conservation predicate for the ``conditional`` tampering relations:
    the obstruction area (structure-loss blocks) must be able to explain the
    full measured change -- ``measured_increase <= obstruction_coverage +
    slack``. The two signals are measured on different grids, so ``slack``
    absorbs the unavoidable extent mismatch (the ``*_AREA_SLACK`` config
    constants). Empirical starting points, not derived from any clip.
    """
    return measured_increase <= obstruction_coverage + slack


def _is_near_black(low_light_confidence: float, floor: float) -> bool:
    """True when low-light is strong enough that the frame is near-black.

    Near-black predicate for the ``conditional`` low_light->blur relation:
    at or above ``floor`` the whole frame is unmeasurable (the existing
    low-light gate floor), so low-light fully explains any blur signal and no
    area check is meaningful on a black frame. ``floor`` is passed in from
    ``LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR``.
    """
    return low_light_confidence >= floor


def _pair_should_suppress(
    suppressor: str,
    suppressed: str,
    suppressor_metrics: Mapping[str, float],
    suppressed_metrics: Mapping[str, float],
    suppressor_confidence: float,
    suppressed_confidence: float,
) -> bool:
    """Decide whether ``suppressor`` suppresses ``suppressed`` on one frame.

    Routes the pair through ``DECISION_SUPPRESSION_RULES``:
    - absent from the map -> independent, never suppress;
    - ``always`` -> the relative margin check alone decides
      (``_margin_met``), no physical predicate;
    - ``conditional`` -> the margin check AND the pair's specific predicate
      (area conservation for the tampering pairs, near-black for
      low_light->blur) must both hold before suppression applies.

    Metrics dicts may be empty (older observations, mocked tests): a
    conditional predicate that cannot read the measurand it needs returns
    False (not verifiable -> never suppress) rather than raising.
    """
    relation = DECISION_SUPPRESSION_RULES.get((suppressor, suppressed))
    if relation is None:
        return False
    if not _margin_met(suppressor_confidence, suppressed_confidence):
        return False
    if relation == "always":
        return True
    if relation == "conditional":
        if (suppressor, suppressed) == ("tampering", "low_light"):
            coverage = suppressor_metrics.get("total_loss_fraction")
            increase = suppressed_metrics.get("relative_increase")
            if coverage is None or increase is None:
                return False
            return _area_conserved(coverage, increase, TAMPERING_LOW_LIGHT_AREA_SLACK)
        if (suppressor, suppressed) == ("tampering", "blur"):
            coverage = suppressor_metrics.get("total_loss_fraction")
            sharpness_ratio = suppressed_metrics.get("sharpness_ratio")
            if coverage is None or sharpness_ratio is None:
                return False
            # Blur's measurand is a retention ratio (1.0 = unchanged), so the
            # observed increase in blur is the sharpness fraction lost.
            return _area_conserved(coverage, 1.0 - sharpness_ratio, TAMPERING_BLUR_AREA_SLACK)
        if (suppressor, suppressed) == ("low_light", "blur"):
            return _is_near_black(suppressor_confidence, LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR)
    return False


@dataclass(frozen=True)
class DetectorObservation:
    """Result of running one detector on one frame, after isolation."""

    detector: str
    status: str                # DETECTOR_STATUS_OK | ERROR | SKIPPED
    is_candidate: bool
    confidence: float
    error_message: str | None = None
    reason: str | None = None   # optional diagnostic for non-error states
                                # ("suppressed_by_gate", "degraded_baseline",
                                # tilt TILT_STATUS_*, ...)
    metrics: dict[str, float] = field(default_factory=dict)
                                # raw measurands the detector exposes on its
                                # result (see _DETECTOR_METRIC_FIELDS), for
                                # physical co-occurrence predicate evaluation
    raw_confidence: float = 0.0
                                # the detector's raw confidence kept verbatim,
                                # separate from ``confidence`` so a
                                # non-candidate's reportable confidence can be
                                # zeroed without losing the underlying value
                                # for debugging. The engine populates it on
                                # every OK observation; the default keeps
                                # direct constructions in tests working.


@dataclass(frozen=True)
class Fault:
    """A fault that survives per-frame multi-label fusion.

    ``fault_type`` is a name in ``DECISION_PRECEDENCE``; ``confidence`` is
    the detector's normalized confidence for that fault on this frame.
    """

    fault_type: str
    confidence: float


@dataclass(frozen=True)
class CandidateClassification:
    """One frame's candidates partitioned into three disjoint buckets.

    Returned by ``classify_candidates``. Every candidate lands in exactly
    one bucket:

    - ``survivors``: the fault names of the ``Fault`` objects
      ``resolve_active_faults`` returns for this frame (reused verbatim --
      the emission-floor and suppression decision logic lives only there).
    - ``below_floor``: candidates whose confidence is below their own
      emission floor in ``DECISION_CONFIRM_MIN_CONFIDENCE``. These never
      had a chance to be suppressed -- they were just too weak.
    - ``suppressed``: candidates that clear their own floor but did NOT
      survive fusion, i.e. were removed by ``_pair_should_suppress``
      against a surviving higher-precedence fault. Real causal
      suppression only -- never a sub-floor candidate.

    The three buckets are pairwise disjoint and cover every candidate.
    """

    survivors: frozenset[str]
    below_floor: frozenset[str]
    suppressed: frozenset[str]


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
    # Multi-label survivors in DECISION_PRECEDENCE order. The default keeps
    # callers that construct a frame from persisted data working; the engine
    # always populates this field.
    faults: tuple[Fault, ...] = ()
    # Sub-floor candidates (confidence below DECISION_CONFIRM_MIN_CONFIDENCE
    # for their own fault) -- too weak to even be considered for causal
    # suppression, unlike suppressed_faults. Default keeps callers that
    # construct a frame from persisted data working; the engine always
    # populates this field.
    below_floor_faults: tuple[str, ...] = ()
    # The full set of faults currently in the temporal tracker's confirmed
    # state (each with its peak confidence), in DECISION_PRECEDENCE order.
    # Unlike ``faults`` this is NOT limited to this frame's survivors: a
    # fault confirmed moments ago stays listed even when this frame's
    # detectors no longer flag it. Default keeps callers that construct a
    # frame from persisted data working; the engine always populates this
    # field. Not consumed by any renderer yet (wired in a later banner-fix
    # subtask).
    confirmed_faults: tuple[Fault, ...] = ()


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
    ``DECISION_SUPPRESSION_MAP``) or as ``secondary`` symptoms. Output is
    always ordered by ``DECISION_PRECEDENCE`` regardless of the order the
    candidate names arrive in (the engine executes detectors in
    ``DECISION_EXECUTION_ORDER``, which differs from precedence).
    """
    ranked = sorted(candidate_names, key=lambda f: _FAULT_RANK[f])
    suppressed_set = DECISION_SUPPRESSION_MAP.get(primary, ())
    suppressed = tuple(f for f in ranked if f != primary and f in suppressed_set)
    secondary = tuple(f for f in ranked if f != primary and f not in suppressed_set)
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
    strongest_active: str | None = None

    for i, fault in enumerate(ranked):
        higher = ranked[:i]
        suppressed_by_higher = any(
            candidates[g] >= DECISION_SUPPRESSOR_MIN_CONFIDENCE
            and _margin_met(candidates[g], candidates[fault])
            and fault in DECISION_SUPPRESSION_MAP.get(g, ())
            for g in higher
        )
        if suppressed_by_higher:
            continue
        if candidates[fault] >= DECISION_SUPPRESSOR_MIN_CONFIDENCE:
            if strongest_active is None or candidates[fault] > candidates[strongest_active]:
                strongest_active = fault
            # The relative-margin rule: an active candidate only takes the
            # primary over a lower-ranked candidate it would suppress when it
            # clears the margin against that candidate. A weak suppressor
            # (e.g. a 0.6 tampering vs a 1.0 tilt) yields to the stronger
            # lower-ranked signal instead of hijacking it.
            blocked = any(
                candidates[lower] >= DECISION_SUPPRESSOR_MIN_CONFIDENCE
                and not _margin_met(candidates[fault], candidates[lower])
                for lower in ranked[i + 1:]
                if lower in DECISION_SUPPRESSION_MAP.get(fault, ())
            )
            if not blocked:
                primary = fault
                break
            continue
        if fallback is None:
            fallback = fault

    if primary is None:
        # No active candidate cleared its margin: prefer the strongest active
        # candidate (by confidence), then the top-ranked below-gate candidate.
        primary = strongest_active if strongest_active is not None else fallback
    if primary is None:
        return None, (), ()

    secondary, suppressed = split_candidates_for_primary(primary, ranked)
    return primary, secondary, suppressed


def resolve_active_faults(
    candidates: Mapping[str, DetectorObservation],
) -> tuple[Fault, ...]:
    """Resolve the multi-label fault list for one frame.

    Args:
        candidates: mapping fault_type -> ``DetectorObservation`` for faults
            flagged as candidates by their detector this frame (status
            ``ok``). Each observation carries the candidate's confidence plus
            the raw metrics the co-occurrence predicates read (see
            ``_DETECTOR_METRIC_FIELDS``). Fault names must be in
            ``DECISION_PRECEDENCE``.

    Returns:
        A tuple of ``Fault`` survivors in ``DECISION_PRECEDENCE`` order. A
        candidate survives when:
        - its confidence reaches the fault's per-fault emission floor in
          ``DECISION_CONFIRM_MIN_CONFIDENCE``; and
        - no *surviving* higher-precedence candidate actively suppresses it
          (suppressor confidence >= ``DECISION_SUPPRESSOR_MIN_CONFIDENCE``
          and ``_pair_should_suppress`` returns True for the ordered pair,
          which combines the relative margin check with the pair's relation
          class in ``DECISION_SUPPRESSION_RULES``: ``always`` relations use
          the margin alone, ``conditional`` relations additionally require a
          physical predicate on the two observations' metrics, and pairs
          absent from the rules are independent and never suppress).

        A suppressed candidate never acts as a suppressor itself, so only
        higher-precedence survivors can remove lower-precedence candidates.
        Output is deterministic: candidates are visited in precedence order
        and survivors keep that order.
    """
    ranked = sorted(candidates, key=lambda fault: _FAULT_RANK[fault])
    survivors: list[Fault] = []
    for fault in ranked:
        observation = candidates[fault]
        confidence = observation.confidence
        if confidence < DECISION_CONFIRM_MIN_CONFIDENCE.get(fault, 0.0):
            continue
        if any(
            candidates[s.fault_type].confidence
            >= DECISION_SUPPRESSOR_MIN_CONFIDENCE
            and _pair_should_suppress(
                s.fault_type,
                fault,
                candidates[s.fault_type].metrics,
                observation.metrics,
                candidates[s.fault_type].confidence,
                confidence,
            )
            for s in survivors
        ):
            continue
        survivors.append(Fault(fault_type=fault, confidence=confidence))
    return tuple(survivors)


def classify_candidates(
    candidates: Mapping[str, DetectorObservation],
) -> CandidateClassification:
    """Partition one frame's candidates into survivors, below-floor, and
    causally-suppressed buckets.

    Pure function. Reuses ``resolve_active_faults`` for the survivor
    decision instead of reimplementing it: the ``survivors`` bucket is
    exactly the names of the ``Fault`` objects that function returns.

    Every candidate lands in exactly one bucket:

    - ``survivors``: cleared its own emission floor and survived fusion;
    - ``below_floor``: confidence below its own floor in
      ``DECISION_CONFIRM_MIN_CONFIDENCE`` (too weak to be considered for
      suppression at all -- it never had a chance to be suppressed);
    - ``suppressed``: cleared its own floor but was removed by a surviving
      higher-precedence fault via ``_pair_should_suppress`` (real causal
      suppression only).

    Args:
        candidates: mapping fault_type -> ``DetectorObservation`` for faults
            flagged as candidates by their detector this frame (status
            ``ok``). Fault names must be in ``DECISION_PRECEDENCE``.

    Returns:
        The three disjoint buckets as frozen sets of fault-type names.
    """
    survivors = frozenset(
        fault.fault_type for fault in resolve_active_faults(candidates)
    )
    below_floor = frozenset(
        fault
        for fault, observation in candidates.items()
        if observation.confidence < DECISION_CONFIRM_MIN_CONFIDENCE.get(fault, 0.0)
    )
    # Remaining candidates clear their own floor but are not among the
    # survivors, so ``resolve_active_faults`` must have removed them as
    # causally suppressed by a surviving higher-precedence fault.
    suppressed = frozenset(candidates) - survivors - below_floor
    return CandidateClassification(
        survivors=survivors,
        below_floor=below_floor,
        suppressed=suppressed,
    )


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
    fraction of positive frames among observed frames in the last
    ``window_seconds`` is at least ``min_positive_ratio`` and at least
    ``min_window_frames`` frames were observed. A frame counts as a
    *positive* only when its detector flagged it as a candidate AND its
    confidence reaches the fault's floor in ``min_confirm_confidence``
    (emission gating: weak noise never confirms an event). Frames where a
    detector did not run are NOT counted as observations (they are simply
    not passed to ``update``); their old window entries age out naturally.

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
        min_confirm_confidence: Mapping[str, float] | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._session_id = session_id
        self._window_seconds = window_seconds
        self._min_positive_ratio = min_positive_ratio
        self._min_window_frames = min_window_frames
        self._min_event_gap_seconds = min_event_gap_seconds
        self._min_confirm_confidence = (
            dict(min_confirm_confidence)
            if min_confirm_confidence is not None
            else dict(DECISION_CONFIRM_MIN_CONFIDENCE)
        )

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
            confidence = fault_candidates.get(fault, 0.0)
            floor = self._min_confirm_confidence.get(fault, 0.0)
            # Emission gating: only candidates at/above the fault's floor
            # count as positive samples toward confirmation.
            is_positive = fault in fault_candidates and confidence >= floor
            window.append((frame_number, video_time_s, is_positive, confidence))

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

    def confirmed_peaks(self) -> dict[str, float]:
        """Peak confidence of every fault currently in the confirmed state.

        Read-only snapshot of the tracker's existing state: returns
        ``{fault_type: peak_confidence}`` for each fault currently
        confirmed (``self._confirmed``), using the running peak the tracker
        already maintains in ``self._peak_confidence`` -- no recomputation
        and no change to the confirmation logic. Faults not currently
        confirmed (or never observed) are absent.
        """
        return {
            fault: self._peak_confidence.get(fault, 0.0)
            for fault in self._confirmed
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
        positives = sum(1 for _frame, _t, is_positive, _c in window if is_positive)
        return positives / count, count

    def _mark_confirmed(self, fault: str, frame_number: int, video_time_s: float) -> None:
        window = self._windows.get(fault, deque())
        # The event starts at the first positive frame in the window that
        # established the confirmation, not at the confirmation frame itself.
        started = next(
            ((f, t) for f, t, is_positive, _c in window if is_positive),
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
        current = max((c for _frame, _t, is_positive, c in window if is_positive), default=0.0)
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
        active_gates: dict[str, float] = {}
        # Execution order is DECISION_EXECUTION_ORDER (cheap signal detectors
        # first, expensive structural last) so a high-confidence gate can
        # short-circuit expensive work on frames it already explains. Fusion
        # ranking is unaffected: resolve_primary_fault re-sorts by
        # DECISION_PRECEDENCE.
        for name in DECISION_EXECUTION_ORDER:
            if name not in self._detectors:
                continue
            if name not in enabled:
                observations.append(
                    DetectorObservation(name, DETECTOR_STATUS_SKIPPED, False, 0.0)
                )
                continue
            if self._skipped_by_gate(name, active_gates):
                observations.append(
                    DetectorObservation(
                        name, DETECTOR_STATUS_SKIPPED, False, 0.0,
                        reason="suppressed_by_gate",
                    )
                )
                continue
            obs = self._run_detector(name, self._detectors[name], frame, frame_number, video_time_s)
            observations.append(obs)
            # A gate detector is only a gate when it actually fired at or
            # above its per-gate floor this frame.
            if (
                obs.status == DETECTOR_STATUS_OK
                and obs.is_candidate
                and obs.confidence
                >= DECISION_GATE_CONFIDENCE_BY_GATE.get(name, DECISION_GATE_CONFIDENCE)
            ):
                active_gates[name] = obs.confidence

        candidates = {
            obs.detector: obs
            for obs in observations
            if obs.status == DETECTOR_STATUS_OK and obs.is_candidate
        }
        # Multi-label fusion: classify_candidates partitions the frame's
        # candidates into survivors (resolve_active_faults output, reused
        # verbatim -- no reimplementation), sub-floor candidates (below their
        # own emission floor in DECISION_CONFIRM_MIN_CONFIDENCE), and
        # causally-suppressed candidates (cleared their floor but removed by
        # a surviving higher-precedence fault via _pair_should_suppress).
        classification = classify_candidates(candidates)
        # Survivors in DECISION_PRECEDENCE order: resolve_active_faults
        # visits candidates in precedence rank, so ranking the survivor names
        # reproduces its output order exactly.
        faults = tuple(
            Fault(fault_type=name, confidence=candidates[name].confidence)
            for name in sorted(
                classification.survivors, key=lambda fault: _FAULT_RANK[fault]
            )
        )

        # A candidate removed by the precedence rules (its signal is explained
        # by a surviving fault's physical cause) is not independent evidence:
        # it must not register as a positive sample in its own confirmation
        # tracker for this frame, so it can never confirm. Removed candidates
        # (suppressed or below-floor) stay *observed*: they still count in the
        # tracker's ratio denominator but never as positives.
        tracker_candidates = {
            fault.fault_type: fault.confidence for fault in faults
        }
        # Only OK observations count as observed: skipped detectors (disabled,
        # sub-sampled, or in backoff) must not dilute the confirmation ratio.
        observed = [obs.detector for obs in observations if obs.status == DETECTOR_STATUS_OK]
        self._pending_events.extend(
            self._tracker.update(tracker_candidates, observed, frame_number, video_time_s)
        )

        # Full set of currently-confirmed faults for this camera, read from
        # the tracker's confirmed state (existing peaks, no recomputation),
        # in DECISION_PRECEDENCE order. This deliberately includes faults
        # confirmed on earlier frames that this frame's detectors no longer
        # flag -- it is the temporal truth, not the per-frame survivor list.
        confirmed_faults = tuple(
            Fault(fault_type=name, confidence=peak)
            for name, peak in sorted(
                self._tracker.confirmed_peaks().items(),
                key=lambda item: _FAULT_RANK[item[0]],
            )
        )

        # Backward-compatible single-fault view of the multi-label result:
        # primary is the top-ranking survivor; the remaining survivors are
        # secondary symptoms; floor-clearing candidates removed by a survivor
        # are suppressed_faults and sub-floor candidates are below_floor_faults
        # (the two buckets are disjoint and together replace the old mixed
        # "did not survive fusion" bucket). Both keep precedence order.
        primary = faults[0].fault_type if faults else None
        confidence = faults[0].confidence if faults else 0.0
        secondary = tuple(fault.fault_type for fault in faults[1:])
        suppressed = tuple(
            fault
            for fault in sorted(
                classification.suppressed, key=lambda name: _FAULT_RANK[name]
            )
        )
        below_floor = tuple(
            fault
            for fault in sorted(
                classification.below_floor, key=lambda name: _FAULT_RANK[name]
            )
        )

        return DecisionFrame(
            camera_id=self._camera_id,
            frame_number=frame_number,
            video_time_s=video_time_s,
            primary_fault=primary,
            confidence=confidence,
            secondary_symptoms=secondary,
            suppressed_faults=suppressed,
            below_floor_faults=below_floor,
            confirmed_faults=confirmed_faults,
            temporal_confirmation_status=self._tracker.confirmation_status(),
            detectors=tuple(observations),
            faults=faults,
        )

    def drain_events(self) -> list[ConfirmedFault]:
        """Return and clear the ConfirmedFault events emitted so far."""
        events = self._pending_events
        self._pending_events = []
        return events

    def _skipped_by_gate(self, detector: str, active_gates: Mapping[str, float]) -> bool:
        """True when an already-executed gate detector this frame explains
        ``detector`` (per ``DECISION_GATE_SKIP_MAP``), so it is skipped."""
        return any(
            detector in DECISION_GATE_SKIP_MAP.get(gate, ())
            for gate in active_gates
        )

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
            raw_confidence = float(result.confidence)
            # Reportable confidence is zeroed when the detector itself says
            # this frame is not a valid candidate -- a non-candidate must not
            # present a high confidence that reads as a contradiction in logs
            # (observed: tampering confidence 1.0 with is_candidate False).
            # The raw value is preserved separately (raw_confidence) for
            # debugging. Inert for all decision paths, which only ever read
            # confidence of is_candidate observations (see step-3 audit).
            confidence = raw_confidence if is_candidate else 0.0
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
        return DetectorObservation(
            name,
            DETECTOR_STATUS_OK,
            is_candidate,
            confidence,
            reason=getattr(result, "reason", None),
            metrics=_extract_detector_metrics(name, result),
            raw_confidence=raw_confidence,
        )

