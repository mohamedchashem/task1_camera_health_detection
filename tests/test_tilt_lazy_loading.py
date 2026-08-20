"""Unit tests for lazy DISK model loading and injection in the tilt detector.

These tests verify the model-loading contract only:

- importing detectors.tilt never builds or downloads the DISK model;
- the model is constructed lazily on the first extract_features() call;
- configure() can inject a model or point at a local checkpoint so tests
  never have to load the real pretrained DISK weights;
- the constructed model is built at most once and then reused.

Detection quality against real footage is covered by test_tilt_detector.py.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

import kornia.feature as KF

import detectors.tilt as tilt


class _FakeFeatures:
    """Minimal stand-in for kornia's DISKFeatures object."""

    def __init__(self, keypoints: torch.Tensor, descriptors: torch.Tensor) -> None:
        self.keypoints = keypoints
        self.descriptors = descriptors


class _FakeDiskModel:
    """Fake DISK matching the calling convention used by extract_features().

    ``forward(images, n=..., pad_if_not_divisible=...)`` must return a
    list with one features object per batch element.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        images: torch.Tensor,
        n: int | None = None,
        pad_if_not_divisible: bool = False,
    ) -> list[_FakeFeatures]:
        self.calls += 1
        batch = images.shape[0]
        return [
            _FakeFeatures(
                keypoints=torch.zeros((batch, 2)),
                descriptors=torch.zeros((batch, 8)),
            )
        ]


@pytest.fixture(autouse=True)
def _reset_tilt_lazy_state() -> Iterator[None]:
    """Reset the module's lazy-loading state before/after every test here."""
    module = importlib.import_module("detectors.tilt")
    module._disk_model = None
    module._device = None
    module._weights_path = None
    yield
    module = importlib.import_module("detectors.tilt")
    module._disk_model = None
    module._device = None
    module._weights_path = None


def test_module_import_does_not_build_or_download_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def _fake_from_pretrained(*args: object, **kwargs: object):
        from_pretrained_calls.append((args, kwargs))
        return _FakeDiskModel()

    monkeypatch.setattr(KF.DISK, "from_pretrained", _fake_from_pretrained)

    importlib.reload(tilt)

    assert from_pretrained_calls == [], (
        "DISK.from_pretrained was called at module import time; "
        "the DISK model must be lazy-loaded."
    )
    assert tilt._disk_model is None
    assert tilt._device is None


def test_first_extract_features_loads_model_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def _fake_from_pretrained(*args: object, **kwargs: object):
        from_pretrained_calls.append((args, kwargs))
        return _FakeDiskModel()

    monkeypatch.setattr(KF.DISK, "from_pretrained", _fake_from_pretrained)

    tilt.configure(device="cuda")
    assert tilt._disk_model is None  # configure() itself loads nothing

    keypoints, descriptors = tilt.extract_features(_frame())

    assert len(from_pretrained_calls) == 1
    assert from_pretrained_calls[0][0] == (tilt.TILT_MODEL_NAME,)
    assert from_pretrained_calls[0][1]["device"] == torch.device("cuda")
    assert tilt._disk_model is not None
    assert keypoints.shape == (1, 2)
    assert descriptors.shape == (1, 8)

    # The loaded model is reused: a second extraction does not rebuild.
    tilt.extract_features(_frame())
    assert len(from_pretrained_calls) == 1


def test_configure_injects_model_without_loading_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_from_pretrained(*args: object, **kwargs: object):
        raise AssertionError("DISK.from_pretrained must not run when a model is injected")

    monkeypatch.setattr(KF.DISK, "from_pretrained", _fail_from_pretrained)

    fake = _FakeDiskModel()
    tilt.configure(device="cuda", model=fake)

    keypoints, descriptors = tilt.extract_features(_frame())

    assert tilt._disk_model is fake
    assert fake.calls == 1
    assert keypoints.shape == (1, 2)
    assert descriptors.shape == (1, 8)

    tilt.extract_features(_frame())
    assert fake.calls == 2  # same injected instance reused


def test_configure_without_model_resets_to_default_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    tilt.configure(device="cuda", model=_FakeDiskModel())
    assert tilt._disk_model is not None

    tilt.configure(device="cuda")  # no model/weights argument -> reset to default

    assert tilt._disk_model is None

    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def _fake_from_pretrained(*args: object, **kwargs: object):
        from_pretrained_calls.append((args, kwargs))
        return _FakeDiskModel()

    monkeypatch.setattr(KF.DISK, "from_pretrained", _fake_from_pretrained)

    tilt.extract_features(_frame())
    assert len(from_pretrained_calls) == 1


def test_configure_weights_path_missing_raises_on_first_extract(tmp_path: Path) -> None:
    tilt.configure(device="cuda", weights_path=tmp_path / "missing-disk.pth")
    with pytest.raises(FileNotFoundError):
        tilt.extract_features(_frame())


def test_configure_weights_path_loads_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_from_pretrained(*args: object, **kwargs: object):
        raise AssertionError("DISK.from_pretrained must not run when a local checkpoint is configured")

    monkeypatch.setattr(KF.DISK, "from_pretrained", _fail_from_pretrained)

    # A checkpoint in the same format kornia's DISK.from_pretrained expects:
    # a dict with an "extractor" state-dict key. Building the DISK
    # architecture requires no pretrained weights and no network.
    checkpoint_path = tmp_path / "disk-depth.pth"
    architecture = KF.DISK()
    torch.save({"extractor": architecture.state_dict()}, checkpoint_path)

    tilt.configure(device="cuda", weights_path=checkpoint_path)

    keypoints, descriptors = tilt.extract_features(_noisy_frame())

    assert isinstance(tilt._disk_model, KF.DISK)
    assert keypoints.shape[1] == 2
    assert descriptors.shape[1] == 128


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _noisy_frame() -> np.ndarray:
    """Fixed-seed random frame.

    Used for the real-DISK extraction test: a zero frame through an
    untrained (randomly initialized) DISK produces a constant heatmap
    that can have no detections at all, which makes kornia's
    heatmap_to_keypoints fail on an empty tensor. Random input produces
    varied heatmap values, so positive local maxima are effectively
    guaranteed and the test stays deterministic.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
