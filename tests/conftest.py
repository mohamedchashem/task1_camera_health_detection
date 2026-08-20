"""Session-level materialization of the CI media fixtures (Phase 4).

A fresh or headless checkout has no binary media assets: the real footage
and captured baselines are machine-local, and committing them would bloat the
repository. This conftest makes the ground-truth detector tests and the
end-to-end pipeline tests self-sufficient by generating deterministic
synthetic assets in the standard locations before any test runs when either
the fixture video or the cam1 baseline is missing:

    data/test_footage/test_video.mp4
    data/baselines/cam1.json
    data/baselines/cam1.jpg
    data/baselines/cam1_edges.png

When both assets already exist they are left untouched (no unnecessary
regeneration). A generation failure raises, so CI fails loudly instead of
silently running a suite whose ground-truth tests would skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config import BASELINES_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

_VIDEO_PATH = PROJECT_ROOT / "data" / "test_footage" / "test_video.mp4"
_BASELINE_JSON_PATH = BASELINES_DIR / "cam1.json"
_CAMERA_ID = "cam1"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--device`` flag used to enforce the GPU mandate.

    ``python -m pytest tests/ --device cuda:0`` is the supported way to run
    the suite: the resolved device is applied to the tilt detector's
    ``configure()`` before any test runs. A CUDA device that is unavailable
    fails the run loudly instead of silently falling back to CPU (matching
    the project's fail-fast device policy in ``config.py``).
    """
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Explicit torch device for detector execution (e.g. 'cuda', "
        "'cuda:0'). Enforced as a GPU by default; a requested CUDA device "
        "that is unavailable aborts the session.",
    )


@pytest.fixture(scope="session", autouse=True)
def _apply_device_option(pytestconfig: pytest.Config) -> None:
    """Apply ``--device`` to the tilt detector before the first test runs.

    Runs lazily on the first test, i.e. after all test modules have been
    imported, so it takes precedence over any module-level ``configure()``
    call such as the CUDA mandate in ``test_tilt_detector.py``.
    """
    device = pytestconfig.getoption("device")
    if not device:
        return
    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        pytest.fail(
            f"--device {device} requested but CUDA is not available; "
            "refusing to fall back to CPU."
        )
    from detectors import tilt

    tilt.configure(device=device)


@pytest.fixture(scope="session", autouse=True)
def _materialize_ci_fixtures() -> None:
    """Generate the fixture video and baselines when either asset is missing."""
    if _VIDEO_PATH.exists() and _BASELINE_JSON_PATH.exists():
        return
    logger.warning(
        "Fixture assets missing (video=%s, baseline=%s); generating synthetic fixtures.",
        _VIDEO_PATH.exists(),
        _BASELINE_JSON_PATH.exists(),
    )
    from scripts.generate_test_fixtures import generate_fixtures

    generate_fixtures(video_path=_VIDEO_PATH, camera_id=_CAMERA_ID)
    logger.info("Synthetic fixture video and cam1 baselines materialized.")
