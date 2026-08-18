"""Shared, safe path-handling helpers."""

from __future__ import annotations

import re

_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_camera_id(camera_id: str) -> str:
    """Reject anything that isn't a safe filename component.

    camera_id ends up directly in file paths across this project, so
    this guards against path traversal (e.g. '../../etc') or invalid
    filename characters.
    """
    if not _CAMERA_ID_PATTERN.match(camera_id):
        raise ValueError(
            f"Invalid camera_id {camera_id!r}: only letters, digits, "
            "'-' and '_' are allowed."
        )
    return camera_id