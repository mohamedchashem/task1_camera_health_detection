"""Resilient RTSP frame source for production video pipelines."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import cv2
import numpy as np

from config import STREAM_RECONNECT_DELAY_SECONDS, STREAM_MAX_CONSECUTIVE_FAILURES

logger = logging.getLogger(__name__)


def read_frames(stream_url: str) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_number, elapsed_seconds, frame) from a live stream.

    A live RTSP source has no reliable frame-position metadata (unlike a
    file), so elapsed time is tracked from wall-clock time instead.

    Transient read failures (e.g. decode errors right after connecting)
    are skipped rather than raised. If failures happen consecutively past
    a threshold, the connection is treated as dropped and re-established.
    """
    cap = cv2.VideoCapture(stream_url)
    stream_start = time.monotonic()
    frame_number = 0
    consecutive_failures = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning("frame read failed (%d consecutive)", consecutive_failures)

                if consecutive_failures >= STREAM_MAX_CONSECUTIVE_FAILURES:
                    logger.warning("reconnecting after repeated failures: %s", stream_url)
                    cap.release()
                    time.sleep(STREAM_RECONNECT_DELAY_SECONDS)
                    cap = cv2.VideoCapture(stream_url)
                    stream_start = time.monotonic()
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            frame_number += 1
            elapsed = time.monotonic() - stream_start
            yield frame_number, elapsed, frame
    finally:
        cap.release()