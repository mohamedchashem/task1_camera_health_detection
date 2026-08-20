"""Resilient RTSP frame source for production video pipelines."""

from __future__ import annotations

import logging
import random
import time
import urllib.parse
from typing import Iterator

import cv2
import numpy as np

from config import (
    STREAM_MAX_CONSECUTIVE_FAILURES,
    STREAM_RECONNECT_BACKOFF_FACTOR,
    STREAM_RECONNECT_BASE_SECONDS,
    STREAM_RECONNECT_MAX_SECONDS,
)

logger = logging.getLogger(__name__)

# Small multiplicative jitter (±10%) applied to each backoff delay. Cameras
# sharing one source would otherwise reconnect in perfect lockstep after an
# outage; the jitter desynchronizes them while staying small enough that the
# exponential schedule remains predictable.
_JITTER_FRACTION = 0.1


def _reconnect_delay_seconds(reconnect_attempt: int) -> float:
    """Exponential-backoff delay (seconds) before the Nth reconnect attempt.

    Attempt 1 waits ``STREAM_RECONNECT_BASE_SECONDS``; every further attempt
    multiplies by ``STREAM_RECONNECT_BACKOFF_FACTOR``, capped at
    ``STREAM_RECONNECT_MAX_SECONDS``. A dedicated reconnect-attempt counter
    drives the schedule (not the raw read-failure count), so the first
    backoff is the base delay regardless of how many reads were lost on the
    dead connection.
    """
    if reconnect_attempt < 1:
        raise ValueError(f"reconnect_attempt must be >= 1; got {reconnect_attempt}.")
    delay = STREAM_RECONNECT_BASE_SECONDS * (
        STREAM_RECONNECT_BACKOFF_FACTOR ** (reconnect_attempt - 1)
    )
    return min(delay, STREAM_RECONNECT_MAX_SECONDS)


def _jittered_delay(delay_seconds: float) -> float:
    """Apply the small multiplicative jitter to a backoff delay."""
    return max(
        0.0, delay_seconds * (1.0 + random.uniform(-_JITTER_FRACTION, _JITTER_FRACTION))
    )


def _open_capture(stream_url: str) -> cv2.VideoCapture | None:
    """Open ``stream_url`` or return None when the connection cannot be made.

    ``cv2.VideoCapture`` is lazy: constructing it can succeed even when the
    source is unreachable. ``isOpened()`` is the authoritative connection
    check, so a failed open returns None and is routed back through the
    reconnect/backoff path instead of being read as a (nonexistent) stream.

    A constructor that raises (e.g. the FFmpeg backend rejecting a malformed
    URL) is treated as the same failed connection: the exception is caught
    here so every construction attempt -- the initial open and each reconnect
    -- recovers through the caller's release/backoff logic instead of
    terminating the generator.
    """
    try:
        cap = cv2.VideoCapture(stream_url)
    except Exception:
        logger.warning(
            "stream connection failed to construct: %s",
            _redact_url_credentials(stream_url),
        )
        return None
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _validate_frame(frame: object) -> bool:
    """Return True when ``frame`` is safe for the detectors to consume.

    Detectors assume an HxWx3 ``uint8`` BGR array. Anything else (``None``
    from a decode failure, grayscale, float, or empty buffers) would crash
    downstream or silently produce wrong results, so it is rejected at the
    ingestion boundary.
    """
    if not isinstance(frame, np.ndarray):
        return False
    if frame.dtype != np.uint8:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.size == 0:
        return False
    return True


def _redact_url_credentials(stream_url: str) -> str:
    """Return ``stream_url`` with embedded basic-auth credentials removed.

    RTSP URLs commonly embed credentials as ``rtsp://user:pass@host/...``.
    Logging such URLs leaks secrets into log files, so the netloc is
    rebuilt without the userinfo portion before anything is logged. URLs
    without credentials are returned unchanged.
    """
    parsed = urllib.parse.urlsplit(stream_url)
    if parsed.username is None:
        return stream_url
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literals need brackets when rebuilt
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, host, parsed.path, parsed.query, parsed.fragment)
    )


def read_frames(stream_url: str) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_number, elapsed_seconds, frame) from a live stream.

    A live RTSP source has no reliable frame-position metadata (unlike a
    file), so elapsed time is tracked from wall-clock time instead.

    Robustness contract:
    - A capture that fails ``isOpened()``, a ``VideoCapture`` constructor
      that raises, or a read that fails, counts as a failed connection and
      is never forwarded.
    - After ``STREAM_MAX_CONSECUTIVE_FAILURES`` consecutive failures the
      connection is re-established with an exponentially growing wait (see
      ``_reconnect_delay_seconds``) plus small jitter, capped at
      ``STREAM_RECONNECT_MAX_SECONDS``.
    - The downtime spent reconnecting is accumulated into ``stream_start``,
      so ``elapsed_seconds`` keeps advancing from where it left off and never
      resets to ~0 after a reconnect (tilt sub-sampling and event timing
      depend on a monotonic video time).
    - Frames that fail ``_validate_frame`` (None, non-uint8, not 3-channel,
      or empty) are logged and counted, then skipped -- never forwarded.
    """
    cap: cv2.VideoCapture | None = None
    stream_start = time.monotonic()
    frame_number = 0
    consecutive_failures = 0
    reconnect_attempt = 0
    invalid_frames = 0

    try:
        # The initial construction lives inside the guarded block, not before
        # it: if it fails (see _open_capture) the None result flows through
        # the same consecutive-failure/backoff handling as any other failed
        # connection, and the finally block below always releases any capture
        # that was created.
        cap = _open_capture(stream_url)

        while True:
            if cap is None:
                consecutive_failures += 1
                logger.warning(
                    "stream connection failed to open (%d consecutive)", consecutive_failures
                )
                ret = False
            else:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning("frame read failed (%d consecutive)", consecutive_failures)

            if not ret:
                if consecutive_failures >= STREAM_MAX_CONSECUTIVE_FAILURES:
                    reconnect_attempt += 1
                    delay = _jittered_delay(_reconnect_delay_seconds(reconnect_attempt))
                    logger.warning(
                        "reconnecting after repeated failures (attempt %d, backoff %.2fs): %s",
                        reconnect_attempt,
                        delay,
                        _redact_url_credentials(stream_url),
                    )
                    downtime_start = time.monotonic()
                    if cap is not None:
                        cap.release()
                        cap = None
                    time.sleep(delay)
                    cap = _open_capture(stream_url)
                    # Keep the stream timeline continuous: the outage (release,
                    # sleep and reconnect handshake) is accumulated into
                    # stream_start instead of replacing it with the current
                    # wall-clock time, which would push elapsed_seconds
                    # backwards after a drop.
                    stream_start += time.monotonic() - downtime_start
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            reconnect_attempt = 0

            if not _validate_frame(frame):
                invalid_frames += 1
                logger.warning(
                    "dropping invalid frame (%d since start): expected uint8, "
                    "3-channel, non-empty array",
                    invalid_frames,
                )
                continue

            frame_number += 1
            elapsed = time.monotonic() - stream_start
            yield frame_number, elapsed, frame
    finally:
        if cap is not None:
            cap.release()