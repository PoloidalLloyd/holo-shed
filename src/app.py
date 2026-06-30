"""Application entry point."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from matplotlib import rcParams

from src.theme import apply_mpl_light_theme, apply_qt_light_theme
from src.ui.main_window import MainWindow
from src.ui.qt import QApplication, _QT_API


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes-3 1D GUI (PyQt + embedded Matplotlib).")
    parser.add_argument(
        "casepath",
        nargs="?",
        default=None,
        help="Path to Hermes-3 case directory (contains BOUT.dmp.*.nc and BOUT.inp).",
    )
    parser.add_argument(
        "--spatial-dim",
        type=str,
        default=None,
        help="Force the spatial dimension name (default: infer, usually 'pos').",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=("system", "light", "dark"),
        help="GUI theme override.",
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    if args.theme == "light":
        apply_qt_light_theme(app)
        apply_mpl_light_theme()
    elif args.theme == "dark":
        rcParams.update(
            {
                "figure.facecolor": "#111",
                "savefig.facecolor": "#111",
                "axes.facecolor": "#111",
                "axes.edgecolor": "#ddd",
                "axes.labelcolor": "#eee",
                "text.color": "#eee",
                "xtick.color": "#eee",
                "ytick.color": "#eee",
                "grid.color": "#444",
                "legend.facecolor": "#111",
                "legend.edgecolor": "#444",
            }
        )
    win = MainWindow(initial_case_path=args.casepath, spatial_dim=args.spatial_dim)
    win.setWindowTitle(f"Hermes-3 GUI - Qt ({_QT_API})")
    win.resize(1400, 850)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
