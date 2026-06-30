"""UI package (MainWindow loaded lazily to avoid Qt at import time)."""

from __future__ import annotations

__all__ = ["MainWindow"]


def __getattr__(name: str):
    if name == "MainWindow":
        from holoshed.ui.main_window import MainWindow as _MainWindow

        return _MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
