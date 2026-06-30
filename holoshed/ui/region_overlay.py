"""Pop-out region overlay window."""

from __future__ import annotations

from holoshed.ui.qt import QMainWindow

class RegionOverlayWindow(QMainWindow):
    """
    Small pop-out window used for showing an R-Z cut overlay on a 2D colormap.
    """

    def __init__(self, title: str, *, on_close=None):
        super().__init__()
        self._on_close = on_close
        try:
            self.setWindowTitle(title)
        except Exception:
            pass

    def closeEvent(self, event):  # type: ignore[override]
        try:
            if callable(self._on_close):
                self._on_close()
        except Exception:
            pass
        return super().closeEvent(event)
