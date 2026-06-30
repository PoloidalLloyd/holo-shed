"""Qt and Matplotlib theme helpers."""

from __future__ import annotations

from matplotlib import rcParams

from holoshed.ui.qt import QApplication, QColor, QPalette

def apply_mpl_light_theme() -> None:
    """
    Force Matplotlib to a light theme (white backgrounds / dark text).

    Matplotlib itself is not "aware" of the OS theme, but on macOS + Qt backends
    it can look dark if facecolors are not explicit.
    """
    rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222",
            "axes.labelcolor": "#111",
            "text.color": "#111",
            "xtick.color": "#111",
            "ytick.color": "#111",
            "grid.color": "#dddddd",
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
        }
    )


def apply_qt_light_theme(app: "QApplication") -> None:
    """
    Force the Qt application to a light palette (so the UI doesn't inherit macOS dark mode).
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    p = QPalette()
    # Light palette (Qt docs style)
    p.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    p.setColor(QPalette.ColorRole.WindowText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Text, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Button, QColor(245, 245, 245))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.BrightText, QColor(180, 0, 0))
    p.setColor(QPalette.ColorRole.Link, QColor(0, 90, 180))
    p.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    app.setPalette(p)
