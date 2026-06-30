"""Qt imports (PyQt6 preferred, PySide6 fallback)."""

from __future__ import annotations

# ---- Qt imports (PyQt6 preferred; fall back to PySide6) ----
try:
    from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer  # type: ignore
    from PyQt6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut  # type: ignore
    from PyQt6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QPushButton,
        QSpinBox,
        QSlider,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    _QT_API = "PyQt6"

    def _qt_checked() -> "Qt.CheckState":
        return Qt.CheckState.Checked

    def _qt_unchecked() -> "Qt.CheckState":
        return Qt.CheckState.Unchecked

except Exception:  # pragma: no cover
    from PySide6.QtCore import QEvent, QPoint, Qt, QTimer  # type: ignore
    from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QPushButton,
        QSpinBox,
        QSlider,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    _QT_API = "PySide6"

    def _qt_checked():
        return Qt.Checked

    def _qt_unchecked():
        return Qt.Unchecked



qt_checked = _qt_checked
qt_unchecked = _qt_unchecked
