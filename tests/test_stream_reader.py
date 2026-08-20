"""Security and resilience regression tests for the live RTSP frame source.

Layers:

1. Pure unit tests for the credential-redaction helper
   (``pipeline.stream_reader._redact_url_credentials``).
2. Pure unit tests for the reconnect backoff schedule
   (``pipeline.stream_reader._reconnect_delay_seconds``) and the frame
   validation predicate (``pipeline.stream_reader._validate_frame``).
3. Integration tests that drive the real reconnect path through a mocked
   OpenCV capture: the reconnect warning must log a redacted URL, a failed
   ``isOpened()`` must route to backoff, corrupted frames must be dropped
   rather than forwarded, and ``video_time_s`` must stay strictly monotonic
   across reconnects (regression for the old ``stream_start =
   time.monotonic()`` reset).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pipeline import stream_reader


def _frame(*shape: int) -> np.ndarray:
    """Build a valid uint8 BGR frame of the given shape for fakes."""
    return np.zeros(shape, dtype=np.uint8)


# --- _redact_url_credentials -------------------------------------------------


def test_redact_removes_user_and_password() -> None:
    assert (
        stream_reader._redact_url_credentials(
            "rtsp://alice:s3cr3t@192.168.1.10:554/stream1"
        )
        == "rtsp://192.168.1.10:554/stream1"
    )


def test_redact_removes_username_only() -> None:
    assert (
        stream_reader._redact_url_credentials("rtsp://alice@host/live")
        == "rtsp://host/live"
    )


def test_redact_returns_credential_free_url_unchanged() -> None:
    url = "rtsp://host:554/live?ch=1#frag"
    assert stream_reader._redact_url_credentials(url) == url


def test_redact_preserves_path_query_and_fragment() -> None:
    assert (
        stream_reader._redact_url_credentials(
            "rtsp://alice:pw@host:554/live?ch=1#frag"
        )
        == "rtsp://host:554/live?ch=1#frag"
    )


def test_redact_handles_password_containing_at_sign() -> None:
    assert (
        stream_reader._redact_url_credentials("rtsp://alice:p@ss@host/live")
        == "rtsp://host/live"
    )


def test_redact_handles_ipv6_literal() -> None:
    assert (
        stream_reader._redact_url_credentials("rtsp://alice:pw@[fe80::1]:554/live")
        == "rtsp://[fe80::1]:554/live"
    )


# --- reconnect log path ------------------------------------------------------


def test_reconnect_log_never_contains_credentials(monkeypatch, caplog) -> None:
    """The reconnect warning must log a redacted URL, never the raw one."""
    stream_url = "rtsp://alice:s3cr3t@192.168.1.10:554/stream1"
    monkeypatch.setattr(stream_reader, "STREAM_MAX_CONSECUTIVE_FAILURES", 2)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BASE_SECONDS", 0.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_MAX_SECONDS", 30.0)
    monkeypatch.setattr(stream_reader, "_jittered_delay", lambda delay: delay)

    class _FakeCapture:
        def __init__(self, fails_before_success: int) -> None:
            self._fails_before_success = fails_before_success
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            if self._fails_before_success > 0:
                self._fails_before_success -= 1
                return False, None
            return True, _frame(16, 16, 3)

        def release(self) -> None:
            self.released = True

    created: list[_FakeCapture] = []

    def _fake_video_capture(*_args: object, **_kwargs: object) -> _FakeCapture:
        # The first connection must fail MAX+1 times to force a reconnect;
        # the re-created connection succeeds immediately so the generator
        # reaches a yield and next() returns instead of looping forever.
        fails = (
            stream_reader.STREAM_MAX_CONSECUTIVE_FAILURES + 1
            if not created
            else 0
        )
        cap = _FakeCapture(fails_before_success=fails)
        created.append(cap)
        return cap

    monkeypatch.setattr(stream_reader.cv2, "VideoCapture", _fake_video_capture)

    with caplog.at_level(logging.WARNING, logger="pipeline.stream_reader"):
        frames = stream_reader.read_frames(stream_url)
        frame_number, _elapsed, frame_data = next(frames)
        frames.close()

    assert frame_number == 1
    assert len(created) == 2  # initial connection + one reconnect
    assert "reconnecting after repeated failures" in caplog.text
    assert "rtsp://192.168.1.10:554/stream1" in caplog.text
    assert "alice" not in caplog.text
    assert "s3cr3t" not in caplog.text


# --- reconnect backoff schedule ----------------------------------------------


def test_reconnect_delay_grows_exponentially(monkeypatch) -> None:
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BASE_SECONDS", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_MAX_SECONDS", 30.0)

    assert stream_reader._reconnect_delay_seconds(1) == pytest.approx(2.0)
    assert stream_reader._reconnect_delay_seconds(2) == pytest.approx(4.0)
    assert stream_reader._reconnect_delay_seconds(3) == pytest.approx(8.0)


def test_reconnect_delay_caps_at_max(monkeypatch) -> None:
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BASE_SECONDS", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_MAX_SECONDS", 30.0)

    # 2.0 * 2.0**4 == 32.0, which must be capped down to MAX.
    assert stream_reader._reconnect_delay_seconds(5) == pytest.approx(30.0)
    assert stream_reader._reconnect_delay_seconds(100) == pytest.approx(30.0)


def test_reconnect_delay_rejects_invalid_attempt() -> None:
    with pytest.raises(ValueError):
        stream_reader._reconnect_delay_seconds(0)
    with pytest.raises(ValueError):
        stream_reader._reconnect_delay_seconds(-1)


# --- _validate_frame ---------------------------------------------------------


def test_validate_frame_accepts_uint8_bgr() -> None:
    assert stream_reader._validate_frame(_frame(16, 16, 3))


def test_validate_frame_rejects_none() -> None:
    assert not stream_reader._validate_frame(None)


def test_validate_frame_rejects_non_array() -> None:
    assert not stream_reader._validate_frame("not-a-frame")


def test_validate_frame_rejects_wrong_dtype() -> None:
    assert not stream_reader._validate_frame(np.zeros((16, 16, 3), dtype=np.float32))


def test_validate_frame_rejects_wrong_channel_count() -> None:
    assert not stream_reader._validate_frame(_frame(16, 16, 1))  # grayscale
    assert not stream_reader._validate_frame(_frame(16, 16, 4))  # BGRA


def test_validate_frame_rejects_non_image_shape() -> None:
    assert not stream_reader._validate_frame(_frame(16))  # 1-D vector
    assert not stream_reader._validate_frame(_frame(16, 16))  # 2-D matrix


def test_validate_frame_rejects_empty_buffer() -> None:
    assert not stream_reader._validate_frame(np.empty((0, 16, 3), dtype=np.uint8))


# --- reconnect integration paths ---------------------------------------------


def test_unopened_capture_routes_to_backoff(monkeypatch) -> None:
    """A connection that fails ``isOpened()`` must retry with backoff."""
    monkeypatch.setattr(stream_reader, "STREAM_MAX_CONSECUTIVE_FAILURES", 1)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BASE_SECONDS", 1.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_MAX_SECONDS", 30.0)
    monkeypatch.setattr(stream_reader, "_jittered_delay", lambda delay: delay)

    clock = {"now": 0.0}

    def _monotonic() -> float:
        return clock["now"]

    def _sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(stream_reader.time, "monotonic", _monotonic)
    monkeypatch.setattr(stream_reader.time, "sleep", _sleep)

    class _DeadCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    class _GoodCapture:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            return True, _frame(16, 16, 3)

        def release(self) -> None:
            pass

    created = {"count": 0}

    def _factory(*_args: object, **_kwargs: object) -> object:
        created["count"] += 1
        return _DeadCapture() if created["count"] == 1 else _GoodCapture()

    monkeypatch.setattr(stream_reader.cv2, "VideoCapture", _factory)

    frames = stream_reader.read_frames("rtsp://host/live")
    frame_number, elapsed, _frame_data = next(frames)
    frames.close()

    assert frame_number == 1
    assert created["count"] == 2  # failed open + one backoff retry
    assert clock["now"] == pytest.approx(1.0)  # slept the full base backoff
    # The outage was folded into stream_start, so elapsed starts at 0
    # instead of counting the 1s downtime as stream time.
    assert elapsed == pytest.approx(0.0)


def test_invalid_frames_are_logged_counted_and_skipped(monkeypatch, caplog) -> None:
    """Corrupted frames must never reach the consumer: logged, counted, skipped."""
    monkeypatch.setattr(stream_reader, "STREAM_MAX_CONSECUTIVE_FAILURES", 2)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BASE_SECONDS", 0.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_MAX_SECONDS", 30.0)

    invalid_payloads: list[object] = [
        None,  # decode-adjacent None frame
        np.zeros((8, 8, 1), dtype=np.uint8),  # grayscale, not BGR
        np.zeros((8, 8, 3), dtype=np.float32),  # wrong dtype
        np.empty((0, 8, 3), dtype=np.uint8),  # empty buffer
        np.zeros((8, 8), dtype=np.uint8),  # 2-D, not an image
    ]

    class _FakeCapture:
        def __init__(self) -> None:
            self._items = list(invalid_payloads)
            self._items.append(_frame(8, 8, 3))  # one valid frame at the end
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            if self._items:
                return True, self._items.pop(0)
            return False, None

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(stream_reader.cv2, "VideoCapture", lambda *_a, **_k: _FakeCapture())

    with caplog.at_level(logging.WARNING, logger="pipeline.stream_reader"):
        frames = stream_reader.read_frames("rtsp://host/live")
        frame_number, _elapsed, frame = next(frames)
        frames.close()

    assert frame_number == 1  # invalid frames never consumed a frame number
    assert frame.shape == (8, 8, 3)
    assert caplog.text.count("dropping invalid frame") == len(invalid_payloads)
    assert "dropping invalid frame (5 since start)" in caplog.text


def test_video_time_stays_strictly_monotonic_across_reconnects(monkeypatch, caplog) -> None:
    """``video_time_s`` must never reset when the connection is re-established.

    Regression for the reconnect path that used to do
    ``stream_start = time.monotonic()`` after each reconnect, which sent
    ``elapsed_seconds`` back to ~0 after every drop and broke downstream
    consumers (e.g. the tilt sub-sampling window in main.py). The outage
    must be accumulated into ``stream_start`` so the stream timeline
    continues, while the backoff delay still grows per consecutive failed
    reconnect attempt.
    """
    monkeypatch.setattr(stream_reader, "STREAM_MAX_CONSECUTIVE_FAILURES", 1)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BASE_SECONDS", 1.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(stream_reader, "STREAM_RECONNECT_MAX_SECONDS", 30.0)
    monkeypatch.setattr(stream_reader, "_jittered_delay", lambda delay: delay)

    frame_interval = 1 / 30  # simulated 30fps pacing

    class _FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = _FakeClock()
    monkeypatch.setattr(stream_reader.time, "monotonic", clock)

    def _fake_sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(stream_reader.time, "sleep", _fake_sleep)

    created: list[str] = []

    class _FakeCapture:
        """good-once: one frame then drops; dead: cannot open; good: keeps going."""

        def __init__(self, kind: str) -> None:
            self.kind = kind
            self._reads = 0
            self.released = False

        def isOpened(self) -> bool:
            return self.kind != "dead"

        def read(self) -> tuple[bool, object]:
            self._reads += 1
            clock.now += frame_interval
            if self.kind == "good-once" and self._reads == 1:
                return True, _frame(16, 16, 3)
            if self.kind == "good":
                return True, _frame(16, 16, 3)
            return False, None

        def release(self) -> None:
            self.released = True

    kinds = iter(["good-once", "dead", "dead", "good-once", "dead", "good"])

    def _factory(*_args: object, **_kwargs: object) -> _FakeCapture:
        kind = next(kinds)
        created.append(kind)
        return _FakeCapture(kind)

    monkeypatch.setattr(stream_reader.cv2, "VideoCapture", _factory)

    elapsed_values: list[float] = []
    with caplog.at_level(logging.WARNING, logger="pipeline.stream_reader"):
        frames = stream_reader.read_frames("rtsp://host/live")
        for _frame_number, elapsed, _frame_data in frames:
            elapsed_values.append(elapsed)
            if len(elapsed_values) >= 3:
                break
        frames.close()

    assert created == ["good-once", "dead", "dead", "good-once", "dead", "good"]
    assert all(
        later > earlier for earlier, later in zip(elapsed_values, elapsed_values[1:])
    ), f"video_time_s went backwards across reconnects: {elapsed_values}"
    # The exponential backoff really grew per consecutive dead reconnect
    # (1s -> 2s -> 4s in the first burst, then 1s -> 2s after the stream
    # came back and failed again).
    assert "attempt 1, backoff 1.00s" in caplog.text
    assert "attempt 2, backoff 2.00s" in caplog.text
    assert "attempt 3, backoff 4.00s" in caplog.text
    # ... and all ~10s of downtime (1+2+4+1+2) was folded into stream_start:
    # the last frame's elapsed time is well under a second, so the outage was
    # neither counted as stream time nor reset the timeline to ~0.
    assert clock.now >= 10.0
    assert elapsed_values[-1] < 1.0
