"""Pytest configuration for holo-shed."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
