"""
Run:

```bash
python hermes3_gui_pyqt.py /path/to/case_dir
```
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure we use a Qt backend for embedded Matplotlib.
import matplotlib

matplotlib.use("QtAgg", force=True)
from matplotlib import rcParams  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar  # noqa: E402


def _infer_time_dim(ds) -> Optional[str]:
    for cand in ("t", "time"):
        if cand in ds.dims:
            return cand
    # fall back: any 1D dimension with monotonic coordinate
    for d in ds.dims:
        if d in ds.coords and ds[d].ndim == 1:
            return d
    return None


def _infer_spatial_dim(ds) -> str:
    # Hermes-3 1D typically uses "pos"
    for cand in ("pos", "y", "x", "s"):
        if cand in ds.dims:
            return cand
    # fall back: choose a non-time dimension
    tdim = _infer_time_dim(ds)
    for d in ds.dims:
        if d != tdim:
            return d
    # last resort
    return list(ds.dims)[0]


def _is_plottable_1d_var(da, spatial_dim: str, time_dim: Optional[str]) -> bool:
    dims = tuple(da.dims)
    if spatial_dim not in dims:
        return False
    if len(dims) == 1 and dims[0] == spatial_dim:
        return True
    if time_dim is None:
        return False
    if len(dims) == 2 and set(dims) == {time_dim, spatial_dim}:
        return True
    return False


def _list_plottable_vars(ds, spatial_dim: str, time_dim: Optional[str]) -> List[str]:
    out: List[str] = []
    for name, da in ds.data_vars.items():
        try:
            if _is_plottable_1d_var(da, spatial_dim=spatial_dim, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(out)


def _is_plottable_2d_var(da, time_dim: Optional[str]) -> bool:
    """
    Heuristic for Hermes-3 2D fields.

    sdtools uses dims ('x', 'theta') (optionally with 't') for 2D tokamak data.
    """
    dims = tuple(getattr(da, "dims", ()))
    if "x" not in dims or "theta" not in dims:
        return False
    if time_dim is None:
        # allow static fields
        return set(dims) >= {"x", "theta"}
    # common cases: (x,theta) or (t,x,theta)
    if set(dims) >= {"x", "theta"} and (time_dim in dims or time_dim not in dims):
        return True
    return False


def _list_plottable_vars_2d(ds, time_dim: Optional[str]) -> List[str]:
    out: List[str] = []
    for name, da in ds.data_vars.items():
        try:
            if _is_plottable_2d_var(da, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(out)


def _guard_replace_1d_profile_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Guard-replace for *profiles* (x,y arrays).

    Behavior (as requested):
    - Strip the *outer* guard cells (drop first and last points)
    - Replace the remaining inner guard cell values by averaging with the adjacent
      last/first real cell, so the endpoints represent the last real face values.

    Assumed indexing (common in Hermes-3 1D outputs):
    - index 0 and -1 are the unused "outer" guards
    - index 1 is the inlet-side inner guard, index 2 is first real cell
    - index -2 is the target-side inner guard, index -3 is last real cell
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 4:
        return x, y

    # Strip outer guard cells
    xx = x[1:-1].copy()
    yy = y[1:-1].copy()

    try:
        # Inlet face: average (inner guard, first real)
        yy[0] = 0.5 * (y[1] + y[2])
        xx[0] = 0.5 * (x[1] + x[2])

        # Target face: average (inner guard, last real)
        yy[-1] = 0.5 * (y[-2] + y[-3])
        xx[-1] = 0.5 * (x[-2] + x[-3])
    except Exception:
        return x, y

    return xx, yy


def _format_case_label(case_path: str) -> str:
    p = Path(case_path).expanduser().resolve()
    return p.name or str(p)


@dataclass
class _LoadedCase:
    label: str
    case_path: str
    ds: "object"  # xarray.Dataset (kept generic)
    n_time: int = 1
    is_2d: bool = False


def _pick_bout_output_for_probe(case_dir: Path) -> Path:
    """
    Pick a representative BOUT output file for cheap metadata probing.

    Prefer squashed output (single file) when available.
    """
    # Prefer squashed output if present
    squash = case_dir / "BOUT.squash.nc"
    if squash.exists():
        return squash

    # Prefer the first dump file if present (common naming)
    d0 = case_dir / "BOUT.dmp.0.nc"
    if d0.exists():
        return d0

    # Fall back to any dump file
    dmps = sorted(case_dir.glob("BOUT.dmp.*.nc"))
    if dmps:
        return dmps[0]

    raise FileNotFoundError(
        f"Could not find BOUT output in {case_dir} (expected BOUT.squash.nc or BOUT.dmp.*.nc)."
    )


def _probe_is_2d_case(case_dir: Path) -> bool:
    """
    Fast 1D vs 2D probe using only netCDF metadata.

    Hermes-3 2D tokamak outputs are typically (x, y, t) in raw BOUT files, and
    become (x, theta, t) after xHermes processing.

    Some "1D" cases may still include an `x` dimension of length 1, so we require:
      - `x` AND (`theta` OR `y`) AND len(x) > 1
    """
    probe_path = _pick_bout_output_for_probe(case_dir)
    try:
        # netCDF4 is commonly available via xbout/boutdata dependencies.
        from netCDF4 import Dataset  # type: ignore

        with Dataset(str(probe_path), mode="r") as ds:
            dims = ds.dimensions
            if "x" not in dims:
                return False
            nx = int(len(dims["x"]))
            has_pol = ("theta" in dims) or ("y" in dims)
            if not has_pol:
                return False
            return nx > 1
    except Exception:
        # Fallback: a lightweight xarray open also tends to only read metadata
        # (but requires an engine; if that fails too, we'll raise a clear error).
        try:
            import xarray as xr  # type: ignore

            with xr.open_dataset(str(probe_path), decode_cf=False, mask_and_scale=False) as ds:
                if "x" not in ds.dims:
                    return False
                nx = int(ds.sizes.get("x", 0))
                has_pol = ("theta" in ds.dims) or ("y" in ds.dims)
                if not has_pol:
                    return False
                return nx > 1
        except Exception as e:
            raise RuntimeError(
                f"Failed to probe case dimensionality from {probe_path}.\n"
                "Tried netCDF4 and xarray. Original error:\n"
                f"{e}"
            ) from e


def _parse_mesh_grid_filename_from_bout_inp(inp_path: Path) -> Optional[str]:
    """
    Parse BOUT.inp for:

      [mesh]
      file = "grid.nc"

    Returns the filename (not a path) if found, else None.
    """
    try:
        txt = inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    section = None
    for raw in txt:
        # Strip comments (BOUT inputs commonly use '#')
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        m = re.match(r"^\[(.+?)\]\s*$", line)
        if m:
            section = m.group(1).strip().lower()
            continue

        if section != "mesh":
            continue

        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip().lower()
        if key != "file":
            continue

        val = val.strip().strip('"').strip("'").strip()
        if not val:
            return None
        return val

    return None


def _ensure_sdtools_2d_metadata(ds) -> None:
    """
    sdtools' 2D selectors expect certain derived keys in `ds.metadata`
    (e.g. omp_a/omp_b/imp_a/imp_b). When loading via xHermes these may be missing.

    This function back-fills those keys from the base separatrix indices + guard sizes.
    """
    try:
        m = ds.metadata
        if not isinstance(m, dict):
            return
    except Exception:
        return

    # Only meaningful for 2D tokamak datasets
    try:
        dims = getattr(ds, "dims", {})
        if "x" not in dims or "theta" not in dims:
            return
    except Exception:
        pass

    need_midplanes = not all(k in m for k in ("omp_a", "omp_b", "imp_a", "imp_b"))

    try:
        topology = str(m.get("topology", "")).lower()
        MYG = int(m.get("MYG", 0))
        MXG = int(m.get("MXG", 0))

        j1_1 = int(m["jyseps1_1"])
        j1_2 = int(m["jyseps1_2"])
        j2_1 = int(m["jyseps2_1"])
        j2_2 = int(m["jyseps2_2"])

        ixseps1 = int(m.get("ixseps1", 0))
        ixseps2 = int(m.get("ixseps2", ixseps1))
    except Exception:
        # Missing base metadata; cannot derive midplane indices,
        # but we can still provide geometry variable aliases below.
        topology = str(m.get("topology", "")).lower()
        MYG = int(m.get("MYG", 0) or 0)
        MXG = int(m.get("MXG", 0) or 0)
        j1_1 = j1_2 = j2_1 = j2_2 = 0
        ixseps1 = int(m.get("ixseps1", 0) or 0)
        ixseps2 = int(m.get("ixseps2", ixseps1) or ixseps1)

    # Targets list mirrors sdtools' expectations
    if "single-null" in topology:
        targets = ["inner_lower", "outer_lower"]
    elif "double-null" in topology:
        targets = ["inner_lower", "outer_lower", "inner_upper", "outer_upper"]
    else:
        # Unknown topology; still allow aliasing below
        targets = list(m.get("targets") or [])

    num_targets = len(targets) if targets else 0

    # Guard-adjusted separatrix indices
    m.setdefault("ixseps1g", ixseps1 - MXG)
    m.setdefault("ixseps2g", ixseps2 - MXG)

    if need_midplanes and num_targets:
        # jyseps accounting for y-guards
        j1_1g = j1_1 + MYG
        j2_1g = j2_1 + MYG
        # second divertor leg is offset by (num_targets-1) guard blocks
        j1_2g = j1_2 + MYG * (num_targets - 1)
        j2_2g = j2_2 + MYG * (num_targets - 1)

        m.setdefault("j1_1g", j1_1g)
        m.setdefault("j2_1g", j2_1g)
        m.setdefault("j1_2g", j1_2g)
        m.setdefault("j2_2g", j2_2g)

        # Midplane indices used by sdtools selectors
        # (matches sdtools hermes3/load.py extract_2d_tokamak_geometry)
        try:
            omp_a = int((j2_2g - j1_2g) / 2) + j1_2g
            omp_b = omp_a + 1
            imp_a = int((j2_1g - j1_1g) / 2) + j1_1g + 1
            imp_b = int((j2_1g - j1_1g) / 2) + j1_1g
            m.setdefault("omp_a", omp_a)
            m.setdefault("omp_b", omp_b)
            m.setdefault("imp_a", imp_a)
            m.setdefault("imp_b", imp_b)
        except Exception:
            pass

    # Keep a consistent targets list for downstream logic
    if targets:
        m.setdefault("targets", targets)

    # sdtools selectors expect some geometry variable names which may differ depending on loader.
    # xHermes commonly uses "dl" for poloidal arc length; sdtools expects "dpol".
    try:
        if "dpol" not in ds and "dl" in ds:
            ds["dpol"] = ds["dl"]
    except Exception:
        pass


def _ensure_sdtools_on_path():
    """
    Make a best-effort attempt to ensure sdtools is importable.

    Preferred (self-contained) layout:
      - repo_root/external/sdtools  (git submodule)

    Legacy layout:
      - repo_root/analysis/sdtools
    """
    here = Path(__file__).resolve()

    # Preferred: sdtools is vendored as a submodule under this repo.
    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "external" / "sdtools"
        if sdtools_dir.exists():
            sp = str(sdtools_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return

    # Legacy: some repos keep sdtools under analysis/sdtools.
    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "analysis" / "sdtools"
        if sdtools_dir.exists():
            sp = str(sdtools_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return
    for parent in [here.parent, *here.parents]:
        if parent.name == "analysis":
            sdtools_dir = parent / "sdtools"
            if sdtools_dir.exists():
                sp = str(sdtools_dir)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


# ---- Qt imports (PyQt6 preferred; fall back to PySide6) ----
try:
    from PyQt6.QtCore import QEvent, Qt, QTimer  # type: ignore
    from PyQt6.QtGui import QAction, QColor, QPalette  # type: ignore
    from PyQt6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
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
    from PySide6.QtCore import QEvent, Qt, QTimer  # type: ignore
    from PySide6.QtGui import QAction, QColor, QPalette  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
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


def _apply_mpl_light_theme() -> None:
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


def _apply_qt_light_theme(app: "QApplication") -> None:
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


class Hermes3QtMainWindow(QMainWindow):

    def __init__(self, *, initial_case_path: Optional[str], spatial_dim: Optional[str]):
        super().__init__()

        _ensure_sdtools_on_path()
        try:
            from hermes3.load import Load  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import `hermes3.load.Load`.\n"
                "Fix by either:\n"
                "- initialising the bundled sdtools submodule:\n"
                "    git submodule update --init --recursive\n"
                "- OR setting PYTHONPATH to include a local sdtools checkout (repo root), e.g.\n"
                "    export PYTHONPATH=/path/to/sdtools:$PYTHONPATH\n"
                "\n"
                "Expected locations:\n"
                "- ./external/sdtools   (recommended)\n"
                "- ./analysis/sdtools   (legacy)\n"
                f"Original error: {e}"
            ) from e

        # Ensure sdtools xarray accessors are registered.
        # (Without this, datasets won't have `.hermesm` and 2D mode will crash.)
        try:
            import hermes3.accessors  # type: ignore  # noqa: F401
        except Exception:
            # Not fatal for 1D mode; 2D features will raise clearer errors later.
            pass

        self.Load = Load

        self.setWindowTitle(f"Hermes-3 GUI (1D) - Qt ({_QT_API})")

        self.cases: Dict[str, _LoadedCase] = {}
        self.spatial_dim_forced = spatial_dim
        self.state = dict(spatial_dim=None, time_dim=None, vars=[], t_values=None)
        self._mode_is_2d = False

        self.selected_vars: List[str] = []  # preserve selection order
        self._selected_set: set[str] = set()
        self._yscale_by_var: Dict[str, str] = {}  # var -> {"linear","log","symlog"}
        self._ylim_mode_by_var: Dict[str, str] = {}  # var -> {"auto","final","global"}
        self._var_filter: str = ""

        self._build_ui()

        # Overlay controls (Qt buttons positioned on top of each subplot).
        # var -> (ylim_button, yscale_button)
        self._overlay_buttons: Dict[str, Tuple["QPushButton", "QPushButton"]] = {}
        # var -> matplotlib Axes (for positioning)
        self._overlay_axes_by_var: Dict[str, "object"] = {}
        # Geometry constants (pixels, in canvas coordinates)
        self._overlay_btn_h = 22
        self._overlay_btn_w_yscale = 56
        self._overlay_btn_w_ylim = 72
        self._overlay_pad = 6

        # Keep overlay buttons positioned correctly on draw + resize.
        self.canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons())
        self.canvas.installEventFilter(self)

        # Time-history performance helpers:
        # - debounce redraws (avoid redrawing many times while user drags/scrolls)
        # - cache extracted time series per case/var/index (avoid repeated xarray slicing)
        self._hist_redraw_timer = QTimer(self)
        self._hist_redraw_timer.setSingleShot(True)
        self._hist_redraw_timer.timeout.connect(self._do_redraw_time_history)
        self._hist_cache: Dict[tuple, tuple] = {}
        self._hist_max_points = 2000  # downsample long traces for responsiveness
        self._mon_cache: Dict[tuple, tuple] = {}

        if initial_case_path:
            self.path_edit.setText(str(initial_case_path))
            self.load_dataset(replace=True)
        else:
            self.set_status("Enter a case directory path and click 'Load dataset'.")
            self.redraw()
            self.request_time_history_redraw()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter()

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Dataset path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("dataset path"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/case_dir")
        path_row.addWidget(self.path_edit, 1)
        left_layout.addLayout(path_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load dataset")
        self.add_btn = QPushButton("Load additional")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.add_btn)
        left_layout.addLayout(btn_row)

        # Status + datasets
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        self.datasets_label = QLabel("Loaded datasets: (none)")
        self.datasets_label.setWordWrap(True)
        left_layout.addWidget(self.datasets_label)

        # Shared variable list (used for both Profiles and Time history)
        left_layout.addWidget(QLabel("Search variables"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("search variables…")
        left_layout.addWidget(self.search_edit)

        left_layout.addWidget(QLabel("Variables (check/double-click to plot; right-click for options)"))
        self.vars_list = QListWidget()
        self.vars_list.setUniformItemSizes(True)
        self.vars_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # Make sure double-click doesn't try to edit labels
        try:
            self.vars_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        except Exception:
            self.vars_list.setEditTriggers(QAbstractItemView.NoEditTriggers)  # type: ignore[attr-defined]
        left_layout.addWidget(self.vars_list, 1)

        self.deselect_btn = QPushButton("Deselect All")
        left_layout.addWidget(self.deselect_btn)

        # 2D-only time slider (kept out of the 1D plot layout so 1D UI stays unchanged)
        self.time2d_widget = QWidget()
        time2d_layout = QVBoxLayout(self.time2d_widget)
        time2d_layout.setContentsMargins(0, 0, 0, 0)
        time2d_layout.setSpacing(4)
        time2d_layout.addWidget(QLabel("2D time index"))
        time2d_row = QHBoxLayout()
        self.time_slider_2d = QSlider(Qt.Orientation.Horizontal)
        self.time_slider_2d.setMinimum(0)
        self.time_slider_2d.setMaximum(0)
        self.time_slider_2d.setSingleStep(1)
        self.time_slider_2d.setPageStep(1)
        self.time_slider_2d.setValue(0)
        self.time_readout_2d = QLabel("time index = 0")
        time2d_row.addWidget(self.time_slider_2d, 1)
        time2d_row.addWidget(self.time_readout_2d)
        time2d_layout.addLayout(time2d_row)
        self.time2d_widget.setVisible(False)
        left_layout.addWidget(self.time2d_widget)

        # Per-view controls (selection is shared)
        self.controls_tabs = QTabWidget()

        # Profiles controls (informational only; buttons live on plots)
        prof_ctrl_tab = QWidget()
        prof_ctrl_layout = QVBoxLayout(prof_ctrl_tab)
        prof_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        prof_ctrl_layout.setSpacing(6)
        prof_ctrl_layout.addWidget(QLabel("Profiles controls:\n- Use the overlay buttons on each plot for y-scale and y-limits.\n- Or right-click a variable to set modes."))
        self.guard_replace_check = QCheckBox("Replace guard cells (1D only)")
        self.guard_replace_check.setChecked(True)
        prof_ctrl_layout.addWidget(self.guard_replace_check)
        prof_ctrl_layout.addStretch(1)

        # Time history controls
        hist_ctrl_tab = QWidget()
        hist_ctrl_layout = QVBoxLayout(hist_ctrl_tab)
        hist_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        hist_ctrl_layout.setSpacing(6)
        hist_ctrl_layout.addWidget(QLabel("Time history controls:"))

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("upstream idx"))
        self.hist_upstream_spin = QSpinBox()
        self.hist_upstream_spin.setRange(-1000000, 1000000)
        self.hist_upstream_spin.setValue(2)
        ctrl_row.addWidget(self.hist_upstream_spin)

        ctrl_row.addWidget(QLabel("target idx"))
        self.hist_target_spin = QSpinBox()
        self.hist_target_spin.setRange(-1000000, 1000000)
        self.hist_target_spin.setValue(-2)
        ctrl_row.addWidget(self.hist_target_spin)

        ctrl_row.addWidget(QLabel("time slices"))
        self.hist_time_slices_spin = QSpinBox()
        self.hist_time_slices_spin.setRange(10, 1000000)
        self.hist_time_slices_spin.setValue(800)
        ctrl_row.addWidget(self.hist_time_slices_spin)
        ctrl_row.addStretch(1)
        hist_ctrl_layout.addLayout(ctrl_row)
        hist_ctrl_layout.addStretch(1)

        self._ctrl1d_profiles = prof_ctrl_tab
        self._ctrl1d_history = hist_ctrl_tab
        left_layout.addWidget(self.controls_tabs)

        # --- 2D controls tabs (created once; added/removed depending on mode) ---
        self._pol_ctrl_tab = QWidget()
        pol_ctrl_layout = QVBoxLayout(self._pol_ctrl_tab)
        pol_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        pol_ctrl_layout.setSpacing(6)
        pol_ctrl_layout.addWidget(QLabel("Poloidal 1D (SOL ring)"))
        pol_region_row = QHBoxLayout()
        pol_region_row.addWidget(QLabel("region"))
        self.pol_region_combo = QComboBox()
        self.pol_region_combo.addItems(["outer_lower", "outer_upper", "inner_lower", "inner_upper"])
        pol_region_row.addWidget(self.pol_region_combo, 1)
        pol_ctrl_layout.addLayout(pol_region_row)

        pol_sepadd_row = QHBoxLayout()
        pol_sepadd_row.addWidget(QLabel("sepadd (SOL ring index)"))
        self.pol_sepadd_spin = QSpinBox()
        self.pol_sepadd_spin.setRange(0, 1000000)
        self.pol_sepadd_spin.setValue(0)
        pol_sepadd_row.addWidget(self.pol_sepadd_spin)
        pol_sepadd_row.addStretch(1)
        pol_ctrl_layout.addLayout(pol_sepadd_row)
        pol_ctrl_layout.addStretch(1)

        self._rad_ctrl_tab = QWidget()
        rad_ctrl_layout = QVBoxLayout(self._rad_ctrl_tab)
        rad_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        rad_ctrl_layout.setSpacing(6)
        rad_ctrl_layout.addWidget(QLabel("Radial 1D"))
        rad_region_row = QHBoxLayout()
        rad_region_row.addWidget(QLabel("region"))
        self.rad_region_combo = QComboBox()
        self.rad_region_combo.addItems(
            [
                "omp",
                "imp",
                "outer_lower_target",
                "outer_upper_target",
                "inner_lower_target",
                "inner_upper_target",
            ]
        )
        rad_region_row.addWidget(self.rad_region_combo, 1)
        rad_ctrl_layout.addLayout(rad_region_row)
        rad_ctrl_layout.addStretch(1)

        self._poly_ctrl_tab = QWidget()
        poly_ctrl_layout = QVBoxLayout(self._poly_ctrl_tab)
        poly_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        poly_ctrl_layout.setSpacing(6)
        poly_ctrl_layout.addWidget(QLabel("2D field plot"))
        poly_var_row = QHBoxLayout()
        poly_var_row.addWidget(QLabel("variable"))
        self.poly_var_combo = QComboBox()
        poly_var_row.addWidget(self.poly_var_combo, 1)
        poly_ctrl_layout.addLayout(poly_var_row)

        self.poly_grid_only_check = QCheckBox("grid only (no colormap)")
        self.poly_grid_only_check.setChecked(False)
        poly_ctrl_layout.addWidget(self.poly_grid_only_check)
        poly_ctrl_layout.addStretch(1)

        self._mon_ctrl_tab = QWidget()
        mon_ctrl_layout = QVBoxLayout(self._mon_ctrl_tab)
        mon_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        mon_ctrl_layout.setSpacing(6)
        mon_ctrl_layout.addWidget(QLabel("Monitor: Te/Ne at OMP + target"))
        mon_ctrl_layout.addStretch(1)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Plot tabs (Profiles vs Time history)
        self.plot_tabs = QTabWidget()

        # ---- 1D plot tabs ----
        prof_plot_tab = QWidget()
        prof_plot_layout = QVBoxLayout(prof_plot_tab)
        prof_plot_layout.setContentsMargins(0, 0, 0, 0)
        prof_plot_layout.setSpacing(6)

        self.figure = Figure(figsize=(10.5, 7.5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        prof_plot_layout.addWidget(self.toolbar)
        prof_plot_layout.addWidget(self.canvas, 1)

        slider_row = QHBoxLayout()
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setSingleStep(1)
        self.time_slider.setPageStep(1)
        self.time_slider.setValue(0)
        self.time_readout = QLabel("time index = 0")
        self._time_index_label_1d = QLabel("time index")
        slider_row.addWidget(self._time_index_label_1d)
        slider_row.addWidget(self.time_slider, 1)
        slider_row.addWidget(self.time_readout)
        prof_plot_layout.addLayout(slider_row)

        hist_plot_tab = QWidget()
        hist_plot_layout = QVBoxLayout(hist_plot_tab)
        hist_plot_layout.setContentsMargins(0, 0, 0, 0)
        hist_plot_layout.setSpacing(6)

        self.hist_figure = Figure(figsize=(10.5, 7.5))
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_toolbar = NavigationToolbar(self.hist_canvas, self)
        hist_plot_layout.addWidget(self.hist_toolbar)
        hist_plot_layout.addWidget(self.hist_canvas, 1)

        self.hist_time_readout = QLabel("Time history")
        hist_plot_layout.addWidget(self.hist_time_readout)

        self._tab1d_profiles = prof_plot_tab
        self._tab1d_history = hist_plot_tab

        # ---- 2D plot tabs (created once; attached depending on mode) ----
        self._tab2d_poloidal = QWidget()
        pol_plot_layout = QVBoxLayout(self._tab2d_poloidal)
        pol_plot_layout.setContentsMargins(0, 0, 0, 0)
        pol_plot_layout.setSpacing(6)
        self.pol_figure = Figure(figsize=(10.5, 7.5))
        self.pol_canvas = FigureCanvas(self.pol_figure)
        self.pol_toolbar = NavigationToolbar(self.pol_canvas, self)
        pol_plot_layout.addWidget(self.pol_toolbar)
        pol_plot_layout.addWidget(self.pol_canvas, 1)

        self._tab2d_radial = QWidget()
        rad_plot_layout = QVBoxLayout(self._tab2d_radial)
        rad_plot_layout.setContentsMargins(0, 0, 0, 0)
        rad_plot_layout.setSpacing(6)
        self.rad_figure = Figure(figsize=(10.5, 7.5))
        self.rad_canvas = FigureCanvas(self.rad_figure)
        self.rad_toolbar = NavigationToolbar(self.rad_canvas, self)
        rad_plot_layout.addWidget(self.rad_toolbar)
        rad_plot_layout.addWidget(self.rad_canvas, 1)

        self._tab2d_polygon = QWidget()
        poly_plot_layout = QVBoxLayout(self._tab2d_polygon)
        poly_plot_layout.setContentsMargins(0, 0, 0, 0)
        poly_plot_layout.setSpacing(6)
        self.poly_figure = Figure(figsize=(10.5, 7.5))
        self.poly_canvas = FigureCanvas(self.poly_figure)
        self.poly_toolbar = NavigationToolbar(self.poly_canvas, self)
        poly_plot_layout.addWidget(self.poly_toolbar)
        poly_plot_layout.addWidget(self.poly_canvas, 1)

        self._tab2d_monitor = QWidget()
        mon_plot_layout = QVBoxLayout(self._tab2d_monitor)
        mon_plot_layout.setContentsMargins(0, 0, 0, 0)
        mon_plot_layout.setSpacing(6)
        self.mon_figure = Figure(figsize=(10.5, 7.5))
        self.mon_canvas = FigureCanvas(self.mon_figure)
        self.mon_toolbar = NavigationToolbar(self.mon_canvas, self)
        mon_plot_layout.addWidget(self.mon_toolbar)
        mon_plot_layout.addWidget(self.mon_canvas, 1)

        # Attach initial (1D) tabs
        self._configure_tabs(is_2d=False)
        right_layout.addWidget(self.plot_tabs, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout(root)
        main_layout.addWidget(splitter)

        # Wire signals
        self.load_btn.clicked.connect(lambda: self.load_dataset(replace=True))
        self.add_btn.clicked.connect(lambda: self.load_dataset(replace=False))
        self.deselect_btn.clicked.connect(self.deselect_all_vars)

        self.search_edit.textChanged.connect(self._on_search_change)
        self.vars_list.itemChanged.connect(self._on_var_item_changed)
        self.vars_list.itemDoubleClicked.connect(self._on_var_item_double_clicked)
        self.vars_list.customContextMenuRequested.connect(self._on_var_list_context_menu)
        self.time_slider.valueChanged.connect(lambda _v: self.redraw())
        self.time_slider_2d.valueChanged.connect(lambda _v: self.redraw())
        self.time_slider_2d.valueChanged.connect(lambda _v: self._update_time_readout())
        self.pol_region_combo.currentIndexChanged.connect(lambda _i: self.redraw())
        self.pol_sepadd_spin.valueChanged.connect(lambda _v: self.redraw())
        self.rad_region_combo.currentIndexChanged.connect(lambda _i: self.redraw())
        self.poly_var_combo.currentIndexChanged.connect(lambda _i: self.redraw())
        self.poly_grid_only_check.toggled.connect(lambda _v: self.redraw())
        self.guard_replace_check.toggled.connect(lambda _v: self.redraw())
        self.guard_replace_check.toggled.connect(lambda _v: self.request_time_history_redraw())
        self.hist_upstream_spin.valueChanged.connect(lambda _v: self.request_time_history_redraw())
        self.hist_target_spin.valueChanged.connect(lambda _v: self.request_time_history_redraw())
        self.hist_time_slices_spin.valueChanged.connect(lambda _v: self.request_time_history_redraw())

        # Redraw correct tab when switching
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)

    def _configure_tabs(self, *, is_2d: bool) -> None:
        """
        Swap between 1D and 2D GUI layouts (tabs + controls).

        1D:
          - Profiles + Time history
        2D:
          - Poloidal 1D + Radial 1D + 2D field + Monitor
        """
        self._mode_is_2d = bool(is_2d)

        # Plot tabs
        try:
            self.plot_tabs.blockSignals(True)
            self.plot_tabs.clear()
        finally:
            try:
                self.plot_tabs.blockSignals(False)
            except Exception:
                pass

        # Controls tabs
        try:
            self.controls_tabs.blockSignals(True)
            self.controls_tabs.clear()
        finally:
            try:
                self.controls_tabs.blockSignals(False)
            except Exception:
                pass

        if self._mode_is_2d:
            self.setWindowTitle(f"Hermes-3 GUI (2D) - Qt ({_QT_API})")
            try:
                self.add_btn.setEnabled(False)
            except Exception:
                pass
            self.plot_tabs.addTab(self._tab2d_poloidal, "Poloidal 1D")
            self.plot_tabs.addTab(self._tab2d_radial, "Radial 1D")
            self.plot_tabs.addTab(self._tab2d_polygon, "2D field")
            self.plot_tabs.addTab(self._tab2d_monitor, "Monitor")

            self.controls_tabs.addTab(self._pol_ctrl_tab, "Poloidal 1D")
            self.controls_tabs.addTab(self._rad_ctrl_tab, "Radial 1D")
            self.controls_tabs.addTab(self._poly_ctrl_tab, "2D field")
            self.controls_tabs.addTab(self._mon_ctrl_tab, "Monitor")

            # Show global 2D time control; hide 1D slider embedded in profiles tab.
            try:
                self.time2d_widget.setVisible(True)
            except Exception:
                pass
            for w in (getattr(self, "_time_index_label_1d", None), getattr(self, "time_slider", None), getattr(self, "time_readout", None)):
                try:
                    if w is not None:
                        w.setVisible(False)
                except Exception:
                    pass
        else:
            self.setWindowTitle(f"Hermes-3 GUI (1D) - Qt ({_QT_API})")
            try:
                self.add_btn.setEnabled(True)
            except Exception:
                pass
            self.plot_tabs.addTab(self._tab1d_profiles, "Profiles")
            self.plot_tabs.addTab(self._tab1d_history, "Time history")

            self.controls_tabs.addTab(self._ctrl1d_profiles, "Profiles")
            self.controls_tabs.addTab(self._ctrl1d_history, "Time history")

            try:
                self.time2d_widget.setVisible(False)
            except Exception:
                pass
            for w in (getattr(self, "_time_index_label_1d", None), getattr(self, "time_slider", None), getattr(self, "time_readout", None)):
                try:
                    if w is not None:
                        w.setVisible(True)
                except Exception:
                    pass

        # Keep controls tab aligned with plot tab
        try:
            self.controls_tabs.setCurrentIndex(int(self.plot_tabs.currentIndex()))
        except Exception:
            pass

    def _active_time_slider(self) -> "QSlider":
        return self.time_slider_2d if self._mode_is_2d else self.time_slider

    def _active_time_readout(self) -> "QLabel":
        return self.time_readout_2d if self._mode_is_2d else self.time_readout

    # ---------- Status / datasets ----------
    def set_status(self, msg: str, *, is_error: bool = False) -> None:
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #b00020;" if is_error else "color: #333;")

    def _on_plot_tab_changed(self, index: int) -> None:
 
        try:
            # Sync the controls tab with the plot tab
            self.controls_tabs.setCurrentIndex(int(index))
        except Exception:
            pass
        if self._mode_is_2d:
            self._redraw_2d_current_tab()
        else:
            # Redraw both (cheap enough and keeps state consistent)
            self.redraw()
            self.request_time_history_redraw()

    def request_time_history_redraw(self) -> None:

        if self._mode_is_2d:
            # 2D mode uses the Monitor tab instead of the 1D time-history view.
            self._redraw_2d_monitor()
            return

        try:
            self._hist_redraw_timer.start(120)
        except Exception:
            # Fallback: redraw immediately
            self._do_redraw_time_history()

    def _update_datasets_list(self) -> None:
        if not self.cases:
            self.datasets_label.setText("Loaded datasets: (none)")
            return
        labels = [f"{c.label}{' (2D)' if getattr(c, 'is_2d', False) else ' (1D)'}" for c in self.cases.values()]
        if len(labels) <= 2:
            items = "\n".join(f"- {lbl}" for lbl in labels)
            self.datasets_label.setText(f"Loaded datasets ({len(labels)}):\n{items}")
        else:
            shown = "\n".join(f"- {lbl}" for lbl in labels[:2])
            self.datasets_label.setText(
                f"Loaded datasets ({len(labels)}):\n{shown}\n... and {len(labels) - 2} more"
            )

    # ---------- Variables list ----------
    def _on_search_change(self, text: str) -> None:
        self._var_filter = text or ""
        self._render_var_list()

    def _filtered_vars(self) -> List[str]:
        vars_all = list(self.state.get("vars") or [])
        q = (self._var_filter or "").strip().lower()
        if not q:
            return vars_all
        return [v for v in vars_all if q in v.lower()]

    def _item_text_for_var(self, name: str) -> str:
        ymode = self._yscale_by_var.get(name, "linear")
        ylim = self._ylim_mode_by_var.get(name, "auto")
        return f"{name}   [y:{ymode}, ylim:{ylim}]"

    def _render_var_list(self) -> None:
        self.vars_list.blockSignals(True)
        try:
            self.vars_list.clear()
            vars_all = list(self.state.get("vars") or [])
            if not vars_all:
                return
            for name in self._filtered_vars():
                item = QListWidgetItem(self._item_text_for_var(name))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(_qt_checked() if name in self._selected_set else _qt_unchecked())
                # Store the raw varname so label edits don't break lookups
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.vars_list.addItem(item)
        finally:
            self.vars_list.blockSignals(False)

    def _find_item_by_var(self, varname: str) -> Optional["QListWidgetItem"]:
        for i in range(self.vars_list.count()):
            it = self.vars_list.item(i)
            if it is None:
                continue
            if it.data(Qt.ItemDataRole.UserRole) == varname:
                return it
        return None

    def _on_var_item_changed(self, item: "QListWidgetItem") -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return
        checked = item.checkState() == _qt_checked()
        if checked and name not in self._selected_set:
            self._selected_set.add(name)
            self.selected_vars.append(name)
            self._yscale_by_var.setdefault(name, "linear")
            self._ylim_mode_by_var.setdefault(name, "auto")
        elif (not checked) and name in self._selected_set:
            self._selected_set.remove(name)
            self.selected_vars = [v for v in self.selected_vars if v != name]

        # Update display text (so mode info stays visible)
        item.setText(self._item_text_for_var(name))
        self.redraw()
        self.request_time_history_redraw()

    def _on_var_item_double_clicked(self, item: "QListWidgetItem") -> None:
        """
        Double-clicking anywhere on the row (including the text) toggles the checkbox.
        """
        try:
            cur = item.checkState()
            nxt = _qt_unchecked() if cur == _qt_checked() else _qt_checked()
            item.setCheckState(nxt)  # triggers _on_var_item_changed
        except Exception:
            pass

    def deselect_all_vars(self) -> None:
        self._selected_set = set()
        self.selected_vars = []
        self.vars_list.blockSignals(True)
        try:
            for i in range(self.vars_list.count()):
                it = self.vars_list.item(i)
                if it is not None:
                    it.setCheckState(_qt_unchecked())
        finally:
            self.vars_list.blockSignals(False)
        self.redraw()
        self.request_time_history_redraw()

    def _cycle_yscale(self, current: str) -> str:
        order = ["linear", "log", "symlog"]
        try:
            i = order.index(current)
        except ValueError:
            return "linear"
        return order[(i + 1) % len(order)]

    def _yscale_label(self, mode: str) -> str:
        if mode == "log":
            return "y:log"
        if mode == "symlog":
            return "y:symlog"
        return "y:lin"

    def _cycle_ylim_mode(self, current: str) -> str:
        order = ["auto", "final", "global"]
        try:
            i = order.index(current)
        except ValueError:
            return "auto"
        return order[(i + 1) % len(order)]

    def _ylim_mode_label(self, mode: str) -> str:
        if mode == "final":
            return "ylim:final"
        if mode == "global":
            return "ylim:max"
        return "ylim:auto"

    def _on_var_list_context_menu(self, pos) -> None:
        item = self.vars_list.itemAt(pos)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return

        menu = QMenu(self)

        act_cycle_y = QAction("Cycle y-scale (linear → log → symlog)", self)
        act_cycle_ylim = QAction("Cycle y-limits (auto → final → global)", self)
        menu.addAction(act_cycle_y)
        menu.addAction(act_cycle_ylim)
        menu.addSeparator()

        # Explicit set menus
        m_y = menu.addMenu("Set y-scale")
        for mode in ("linear", "log", "symlog"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._yscale_by_var.get(name, "linear") == mode)
            a.triggered.connect(lambda _=False, m=mode: self._set_var_yscale(name, m))
            m_y.addAction(a)

        m_ylim = menu.addMenu("Set y-limits")
        for mode in ("auto", "final", "global"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._ylim_mode_by_var.get(name, "auto") == mode)
            a.triggered.connect(lambda _=False, m=mode: self._set_var_ylim_mode(name, m))
            m_ylim.addAction(a)

        def _do_cycle_y():
            cur = self._yscale_by_var.get(name, "linear")
            self._yscale_by_var[name] = self._cycle_yscale(cur)
            self._refresh_var_item(name)
            self.redraw()

        def _do_cycle_ylim():
            cur = self._ylim_mode_by_var.get(name, "auto")
            self._ylim_mode_by_var[name] = self._cycle_ylim_mode(cur)
            self._refresh_var_item(name)
            self.redraw()

        act_cycle_y.triggered.connect(_do_cycle_y)
        act_cycle_ylim.triggered.connect(_do_cycle_ylim)

        menu.exec(self.vars_list.mapToGlobal(pos))

    def _refresh_var_item(self, varname: str) -> None:
        it = self._find_item_by_var(varname)
        if it is None:
            return
        self.vars_list.blockSignals(True)
        try:
            it.setText(self._item_text_for_var(varname))
        finally:
            self.vars_list.blockSignals(False)

    def _set_var_yscale(self, varname: str, mode: str) -> None:
        self._yscale_by_var[varname] = mode
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    def _set_var_ylim_mode(self, varname: str, mode: str) -> None:
        self._ylim_mode_by_var[varname] = mode
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    # ---------- Overlay buttons on plots (Option B) ----------
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        """
        Reposition overlay buttons when the canvas resizes.
        """
        try:
            if obj is self.canvas:
                et = event.type()
                # PyQt6/PySide6 both expose QEvent.Type.Resize; keep a fallback just in case.
                if et == QEvent.Type.Resize or et == getattr(QEvent, "Resize", None):
                    self._position_overlay_buttons()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _clear_overlay_buttons(self) -> None:
        for ylim_btn, yscale_btn in list(self._overlay_buttons.values()):
            try:
                ylim_btn.hide()
                yscale_btn.hide()
                ylim_btn.deleteLater()
                yscale_btn.deleteLater()
            except Exception:
                pass
        self._overlay_buttons = {}
        self._overlay_axes_by_var = {}

    def _sync_overlay_buttons(self, vars_to_plot: List[str], axes: List["object"]) -> None:

        # Remove buttons for vars no longer plotted
        keep = set(vars_to_plot)
        for v in list(self._overlay_buttons.keys()):
            if v not in keep:
                try:
                    ylim_btn, yscale_btn = self._overlay_buttons.pop(v)
                    ylim_btn.hide()
                    yscale_btn.hide()
                    ylim_btn.deleteLater()
                    yscale_btn.deleteLater()
                except Exception:
                    pass
                self._overlay_axes_by_var.pop(v, None)

        # Update axes mapping (zip in selection order)
        self._overlay_axes_by_var = {v: ax for v, ax in zip(vars_to_plot, axes)}

        # Create buttons for any new vars
        for v in vars_to_plot:
            if v in self._overlay_buttons:
                self._refresh_overlay_button_labels(v)
                continue

            # Create as children of the canvas so they overlay the plot area.
            ylim_btn = QPushButton(self.canvas)
            yscale_btn = QPushButton(self.canvas)

            ylim_btn.setText(self._ylim_mode_label(self._ylim_mode_by_var.get(v, "auto")))
            yscale_btn.setText(self._yscale_label(self._yscale_by_var.get(v, "linear")))

            ylim_btn.setFixedHeight(self._overlay_btn_h)
            yscale_btn.setFixedHeight(self._overlay_btn_h)
            ylim_btn.setFixedWidth(self._overlay_btn_w_ylim)
            yscale_btn.setFixedWidth(self._overlay_btn_w_yscale)

            try:
                style = (
                    "QPushButton {"
                    " background: rgba(250, 250, 250, 210);"
                    " border: 1px solid rgba(0,0,0,80);"
                    " border-radius: 4px;"
                    " padding: 1px 4px;"
                    " font-size: 10px;"
                    "}"
                    "QPushButton:pressed { background: rgba(230, 230, 230, 230); }"
                )
                ylim_btn.setStyleSheet(style)
                yscale_btn.setStyleSheet(style)
            except Exception:
                pass

            # Click actions
            ylim_btn.clicked.connect(partial(self._on_overlay_ylim_clicked, v))
            yscale_btn.clicked.connect(partial(self._on_overlay_yscale_clicked, v))

            ylim_btn.show()
            yscale_btn.show()
            ylim_btn.raise_()
            yscale_btn.raise_()

            self._overlay_buttons[v] = (ylim_btn, yscale_btn)

        # Position now (and again on draw_event/resize).
        self._position_overlay_buttons()

    def _refresh_overlay_button_labels(self, varname: str) -> None:
        pair = self._overlay_buttons.get(varname)
        if not pair:
            return
        ylim_btn, yscale_btn = pair
        ylim_btn.setText(self._ylim_mode_label(self._ylim_mode_by_var.get(varname, "auto")))
        yscale_btn.setText(self._yscale_label(self._yscale_by_var.get(varname, "linear")))

    def _on_overlay_yscale_clicked(self, varname: str) -> None:
        cur = self._yscale_by_var.get(varname, "linear")
        self._yscale_by_var[varname] = self._cycle_yscale(cur)
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    def _on_overlay_ylim_clicked(self, varname: str) -> None:
        cur = self._ylim_mode_by_var.get(varname, "auto")
        self._ylim_mode_by_var[varname] = self._cycle_ylim_mode(cur)
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    def _position_overlay_buttons(self) -> None:
        """
        Position overlay buttons in canvas pixel coordinates.

        Matplotlib Axes positions are in figure fraction coordinates with origin at bottom-left.
        Qt widget positions are in pixels with origin at top-left.
        """
        if not self._overlay_buttons or not self._overlay_axes_by_var:
            return

        try:
            w, h = self.canvas.get_width_height()
        except Exception:
            return
        if not w or not h:
            return

        pad = int(self._overlay_pad)
        bh = int(self._overlay_btn_h)
        bw_y = int(self._overlay_btn_w_yscale)
        bw_l = int(self._overlay_btn_w_ylim)

        for v, (ylim_btn, yscale_btn) in list(self._overlay_buttons.items()):
            ax = self._overlay_axes_by_var.get(v)
            if ax is None:
                try:
                    ylim_btn.hide()
                    yscale_btn.hide()
                except Exception:
                    pass
                continue

            try:
                pos = ax.get_position()  # figure fraction coords
                x_right = int(pos.x1 * w)
                y_top = int((1.0 - pos.y1) * h)
            except Exception:
                continue

            # Place yscale at top-right inside axes; ylim just to its left.
            y = max(0, y_top + pad)
            x_yscale = max(0, x_right - bw_y - pad)
            x_ylim = max(0, x_yscale - bw_l - pad)

            try:
                yscale_btn.setGeometry(x_yscale, y, bw_y, bh)
                ylim_btn.setGeometry(x_ylim, y, bw_l, bh)
                ylim_btn.show()
                yscale_btn.show()
                ylim_btn.raise_()
                yscale_btn.raise_()
            except Exception:
                pass

    # ---------- Data loading ----------
    def _load_case(self, case_path: str) -> _LoadedCase:
        case_path = str(Path(case_path).expanduser().resolve())
        label = _format_case_label(case_path)
        case_dir = Path(case_path)

        # Decide 1D vs 2D cheaply (metadata-only) to avoid double-loading.
        # Add a fallback path so loading a 2D case while in "1D mode" (or vice versa)
        # still works and automatically switches the GUI after load.
        is_2d_probe = _probe_is_2d_case(case_dir)

        def _load_2d():
            # For now assume the grid file lives alongside the case output.
            # Most cases specify this in BOUT.inp under [mesh] file = ...
            grid_name = _parse_mesh_grid_filename_from_bout_inp(case_dir / "BOUT.inp")
            if not grid_name:
                raise FileNotFoundError(
                    "Detected a 2D case but could not find the grid file in BOUT.inp under:\n"
                    "  [mesh]\n"
                    "  file = \"...\"\n"
                    f"Case directory: {case_dir}"
                )
            grid_path = (case_dir / grid_name).resolve()
            if not grid_path.exists():
                raise FileNotFoundError(
                    "Detected a 2D case but the grid file listed in BOUT.inp was not found:\n"
                    f"  grid: {grid_path}\n"
                    f"  case: {case_dir}"
                )
            cs2 = self.Load.case_2D(case_path, gridfilepath=str(grid_path), verbose=False)
            try:
                _ensure_sdtools_2d_metadata(cs2.ds)
            except Exception:
                pass
            return cs2

        def _load_1d():
            # Newer xHermes versions may not provide ds.hermes.guard_replace_1d(),
            # which sdtools tries to call when guard_replace=True. Disable it here.
            return self.Load.case_1D(case_path, verbose=False, guard_replace=False)

        cs = None
        is_2d = False
        if is_2d_probe:
            try:
                cs = _load_2d()
                is_2d = True
            except Exception:
                cs = _load_1d()
                is_2d = False
        else:
            try:
                cs = _load_1d()
                is_2d = False
            except Exception as e:
                msg = str(e).lower()
                if ("toroidal" in msg and "grid" in msg) or ("topology" in msg and "grid" in msg) or ("provide grid" in msg):
                    cs = _load_2d()
                    is_2d = True
                else:
                    raise
        tdim = _infer_time_dim(cs.ds)
        n_time = int(cs.ds.sizes[tdim]) if tdim and tdim in cs.ds.dims else 1
        return _LoadedCase(label=label, case_path=case_path, ds=cs.ds, n_time=n_time, is_2d=is_2d)

    def _recompute_all_vars(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        if not self.cases:
            return [], None, None
        first_case = next(iter(self.cases.values()))
        first = first_case.ds
        tdim = _infer_time_dim(first)
        is_2d = bool(getattr(first_case, "is_2d", False))

        if is_2d:
            # 2D Hermes-3 typically uses dims ('x','theta') (+ optional time dim)
            all_vars = set(_list_plottable_vars_2d(first, time_dim=tdim))
            for c in list(self.cases.values())[1:]:
                all_vars |= set(_list_plottable_vars_2d(c.ds, time_dim=tdim))
            return sorted(all_vars), "theta", tdim

        sdim = self.spatial_dim_forced or _infer_spatial_dim(first)
        all_vars = set(_list_plottable_vars(first, spatial_dim=sdim, time_dim=tdim))
        for c in list(self.cases.values())[1:]:
            all_vars |= set(_list_plottable_vars(c.ds, spatial_dim=sdim, time_dim=tdim))
        return sorted(all_vars), sdim, tdim

    def _set_time_range(self, n_t: int) -> None:
        n_t = max(1, int(n_t))
        for slider in (getattr(self, "time_slider", None), getattr(self, "time_slider_2d", None)):
            if slider is None:
                continue
            slider.blockSignals(True)
            try:
                slider.setMinimum(0)
                slider.setMaximum(max(0, n_t - 1))
                # set to final time step by default
                slider.setValue(n_t - 1)
            finally:
                slider.blockSignals(False)

    def _update_after_load(self) -> None:
        vars_, sdim, tdim = self._recompute_all_vars()
        self.state["vars"] = vars_
        self.state["spatial_dim"] = sdim
        self.state["time_dim"] = tdim

        # Switch UI mode to match data dimensionality
        try:
            first_case = next(iter(self.cases.values()))
            is_2d = bool(getattr(first_case, "is_2d", False))
        except Exception:
            is_2d = False
        if bool(is_2d) != bool(self._mode_is_2d):
            self._configure_tabs(is_2d=bool(is_2d))

        # Drop selections that no longer exist
        if vars_:
            keep = [v for v in self.selected_vars if v in vars_]
            self.selected_vars = keep
            self._selected_set = set(keep)
            if not self.selected_vars:
                default_var = "Te" if "Te" in vars_ else vars_[0]
                self.selected_vars = [default_var]
                self._selected_set = {default_var}
                self._yscale_by_var.setdefault(default_var, "linear")
                self._ylim_mode_by_var.setdefault(default_var, "auto")
        else:
            self.selected_vars = []
            self._selected_set = set()

        # Time axis values from the first dataset (for display)
        ds0 = next(iter(self.cases.values())).ds
        t_values = None
        if tdim is not None and tdim in ds0.coords:
            try:
                t_values = np.asarray(ds0[tdim].values)
            except Exception:
                t_values = None
        self.state["t_values"] = t_values

        # Slider range based on maximum time steps across cases
        max_n_t = max((c.n_time for c in self.cases.values()), default=1)
        self._set_time_range(max_n_t)

        # Populate 2D field variable selector
        try:
            self.poly_var_combo.blockSignals(True)
            self.poly_var_combo.clear()
            for v in vars_:
                self.poly_var_combo.addItem(v)
            if "Te" in vars_:
                self.poly_var_combo.setCurrentText("Te")
            elif vars_:
                self.poly_var_combo.setCurrentIndex(0)
        finally:
            try:
                self.poly_var_combo.blockSignals(False)
            except Exception:
                pass

        self._render_var_list()
        self._update_time_readout()

    def load_dataset(self, *, replace: bool) -> None:
        p = (self.path_edit.text() or "").strip()
        if not p:
            self.set_status("Please enter a case directory path.", is_error=True)
            return
        try:
            lc = self._load_case(p)

            # Prevent mixing 1D and 2D cases in one session (UI/plotting differs).
            if self.cases and (not replace):
                existing_is_2d = bool(next(iter(self.cases.values())).is_2d)
                if bool(lc.is_2d) != bool(existing_is_2d):
                    raise ValueError(
                        "Cannot mix 1D and 2D cases in the same session. "
                        "Use 'Load dataset' (replace) to switch modes."
                    )

            # For now keep 2D mode single-case (simplifies polygon + monitor plotting).
            if self.cases and (not replace) and bool(next(iter(self.cases.values())).is_2d):
                raise ValueError("2D mode currently supports a single loaded case. Use 'Load dataset' (replace).")

            if replace:
                self.cases.clear()
            self.cases[lc.label] = lc
            self._update_after_load()
            self._update_datasets_list()
            self.set_status("")
            self.redraw()
            # New dataset -> invalidate cached time history and redraw (debounced)
            try:
                self._hist_cache.clear()
            except Exception:
                pass
            try:
                self._mon_cache.clear()
            except Exception:
                pass
            self.request_time_history_redraw()
        except Exception as e:
            self.set_status(f"Failed to load dataset: {e}", is_error=True)

    # ---------- Plotting ----------
    def _get_time_index(self) -> int:
        try:
            return int(self._active_time_slider().value())
        except Exception:
            return 0

    def _get_time_index_for_case(self, case: _LoadedCase) -> int:
        ti = self._get_time_index()
        return min(ti, case.n_time - 1)

    def _guard_replace_enabled(self) -> bool:
        try:
            return (not self._mode_is_2d) and bool(self.guard_replace_check.isChecked())
        except Exception:
            return False

    def _isel_1d_with_guard_replace(self, da, *, sdim: str, idx: int):
        """
        For 1D datasets, emulate guard replacement when selecting inlet/target indices.
        """
        if not self._guard_replace_enabled():
            return da.isel({sdim: idx})

        n = None
        try:
            n = int(da.sizes.get(sdim))
        except Exception:
            n = None
        if not n or n < 4:
            return da.isel({sdim: idx})

        # Convert negative to positive for comparison
        idx_pos = idx if idx >= 0 else (n + idx)
        if idx_pos == 1:
            return 0.5 * (da.isel({sdim: 1}) + da.isel({sdim: 2}))
        if idx_pos == (n - 2):
            return 0.5 * (da.isel({sdim: -2}) + da.isel({sdim: -3}))
        return da.isel({sdim: idx})

    def _update_time_readout(self) -> None:
        ti = self._get_time_index()
        tdim = self.state.get("time_dim")
        tvals = self.state.get("t_values")
        if tdim and tvals is not None and ti < len(tvals):
            try:
                self._active_time_readout().setText(f"{tdim} = {tvals[ti] * 1e3:.4f} ms")
                return
            except Exception:
                pass
        self._active_time_readout().setText(f"time index = {ti}")

    def _compute_ylim_for_final(self, varname: str, tdim: Optional[str], yscale: str) -> Tuple[Optional[float], Optional[float]]:
        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            if varname not in ds:
                continue
            da = ds[varname]
            try:
                final_ti = c.n_time - 1
                if tdim is not None and tdim in da.dims:
                    da1 = da.isel({tdim: final_ti})
                else:
                    da1 = da
                yv = np.asarray(da1.values)
                yv = yv[np.isfinite(yv)]
                if yscale == "log":
                    yv = yv[yv > 0]
                if yv.size:
                    ys_all.append(yv)
            except Exception:
                continue
        if not ys_all:
            return None, None
        ys = np.concatenate(ys_all)
        if ys.size == 0:
            return None, None
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        return ymin - margin, ymax + margin

    def _compute_ylim_for_global(self, varname: str, yscale: str) -> Tuple[Optional[float], Optional[float]]:
        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            if varname not in ds:
                continue
            da = ds[varname]
            try:
                yv = np.asarray(da.values)
                yv = yv[np.isfinite(yv)]
                if yscale == "log":
                    yv = yv[yv > 0]
                if yv.size:
                    ys_all.append(yv)
            except Exception:
                continue
        if not ys_all:
            return None, None
        ys = np.concatenate(ys_all)
        if ys.size == 0:
            return None, None
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        return ymin - margin, ymax + margin

    # ---------- 2D plotting ----------
    def _primary_case(self) -> Optional[_LoadedCase]:
        if not self.cases:
            return None
        return next(iter(self.cases.values()))

    def _ds_at_time(self, case: _LoadedCase):
        ds = case.ds
        tdim = self.state.get("time_dim")
        if tdim and tdim in getattr(ds, "dims", {}):
            try:
                ti = self._get_time_index_for_case(case)
                ds_t = ds.isel({tdim: ti})

                # IMPORTANT: sdtools stores crucial geometry info on `ds.metadata`
                # (a Python attribute, not xarray attrs). Slicing can drop it.
                # Reattach metadata/options so selectors keep working.
                try:
                    if hasattr(ds, "metadata") and not hasattr(ds_t, "metadata"):
                        ds_t.metadata = ds.metadata  # type: ignore[attr-defined]
                    elif hasattr(ds, "metadata") and hasattr(ds_t, "metadata"):
                        # Ensure derived keys like omp_a/imp_a survive.
                        try:
                            ds_t.metadata.update(ds.metadata)  # type: ignore[attr-defined]
                        except Exception:
                            ds_t.metadata = ds.metadata  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    if hasattr(ds, "options") and not hasattr(ds_t, "options"):
                        ds_t.options = ds.options  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Back-fill sdtools-derived geometry keys if xHermes didn't provide them.
                try:
                    _ensure_sdtools_2d_metadata(ds_t)
                except Exception:
                    pass

                return ds_t
            except Exception:
                # Even if time slicing fails, make sure sdtools-required geometry aliases exist.
                try:
                    _ensure_sdtools_2d_metadata(ds)
                except Exception:
                    pass
                return ds
        return ds

    def _redraw_2d_current_tab(self) -> None:
        idx = int(self.plot_tabs.currentIndex())
        if idx == 0:
            self._redraw_2d_poloidal()
        elif idx == 1:
            self._redraw_2d_radial()
        elif idx == 2:
            self._redraw_2d_polygon()
        else:
            self._redraw_2d_monitor()

    def _redraw_2d_poloidal(self) -> None:
        self._update_time_readout()
        self.pol_figure.clear()
        try:
            self.pol_figure.set_facecolor("white")
        except Exception:
            pass

        if not self.cases:
            ax = self.pol_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            self.pol_canvas.draw_idle()
            return

        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            ax = self.pol_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
            self.pol_canvas.draw_idle()
            return

        region = str(self.pol_region_combo.currentText() or "outer_lower")
        sepadd = int(self.pol_sepadd_spin.value())

        # Layout similar to 1D profiles
        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))
        gs = self.pol_figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35, wspace=0.30)
        axes = [self.pol_figure.add_subplot(gs[i % nrows, i // nrows]) for i in range(n)]

        try:
            from hermes3.selectors import get_1d_poloidal_data  # type: ignore
        except Exception as e:
            ax = self.pol_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"sdtools import error:\n{e}", ha="center", va="center", transform=ax.transAxes)
            self.pol_canvas.draw_idle()
            return

        for ax, name in zip(axes, vars_to_plot):
            ax.set_title(f"{name} ({region}, sepadd={sepadd})", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Units (match 1D GUI behavior)
            units = None
            for c in self.cases.values():
                try:
                    if name in c.ds:
                        units = c.ds[name].attrs.get("units", None)
                        if units:
                            break
                except Exception:
                    continue
            ax.set_ylabel(f"({units})" if units else "")

            for c in self.cases.values():
                ds_t = self._ds_at_time(c)
                try:
                    df = get_1d_poloidal_data(ds_t, params=[name], region=region, sepadd=sepadd, target_first=False)
                    x = np.asarray(df["Spar"].values)
                    y = np.asarray(df[name].values)
                    ax.plot(x, y, label=c.label)
                except Exception as e:
                    self.set_status(f"Poloidal extract failed for {name}: {e}", is_error=True)
                    continue
            if len(self.cases) > 1:
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(r"S$_\parallel$ (m)")

        self.pol_canvas.draw_idle()

    def _redraw_2d_radial(self) -> None:
        self._update_time_readout()
        self.rad_figure.clear()
        try:
            self.rad_figure.set_facecolor("white")
        except Exception:
            pass

        if not self.cases:
            ax = self.rad_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            self.rad_canvas.draw_idle()
            return

        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            ax = self.rad_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
            self.rad_canvas.draw_idle()
            return

        region = str(self.rad_region_combo.currentText() or "omp")

        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))
        gs = self.rad_figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35, wspace=0.30)
        axes = [self.rad_figure.add_subplot(gs[i % nrows, i // nrows]) for i in range(n)]

        try:
            # sdtools currently provides two implementations; the "_old" version avoids
            # relying on an 'xr' alias being present.
            from hermes3.selectors import get_1d_radial_data_old as get_1d_radial_data  # type: ignore
        except Exception as e:
            ax = self.rad_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"sdtools import error:\n{e}", ha="center", va="center", transform=ax.transAxes)
            self.rad_canvas.draw_idle()
            return

        for ax, name in zip(axes, vars_to_plot):
            ax.set_title(f"{name} ({region})", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Units (match 1D GUI behavior)
            units = None
            for c in self.cases.values():
                try:
                    if name in c.ds:
                        units = c.ds[name].attrs.get("units", None)
                        if units:
                            break
                except Exception:
                    continue
            ax.set_ylabel(f"({units})" if units else "")

            for c in self.cases.values():
                ds_t = self._ds_at_time(c)
                try:
                    df = get_1d_radial_data(ds_t, params=[name], region=region, guards=False, sol=True, core=True)
                    x = np.asarray(df["Srad"].values)
                    y = np.asarray(df[name].values) if name in df else None
                    if y is None:
                        continue
                    ax.plot(x, y, label=c.label)
                except Exception as e:
                    self.set_status(f"Radial extract failed for {name}: {e}", is_error=True)
                    continue
            if len(self.cases) > 1:
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(r"$r^\prime - r_{sep}$ (m)")

        self.rad_canvas.draw_idle()

    def _redraw_2d_polygon(self) -> None:
        self._update_time_readout()
        self.poly_figure.clear()
        try:
            self.poly_figure.set_facecolor("white")
        except Exception:
            pass

        case = self._primary_case()
        if case is None:
            ax = self.poly_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            self.poly_canvas.draw_idle()
            return

        var = str(self.poly_var_combo.currentText() or "").strip()
        if not var:
            ax = self.poly_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Select a variable.", ha="center", va="center", transform=ax.transAxes)
            self.poly_canvas.draw_idle()
            return

        ds_t = self._ds_at_time(case)
        ax = self.poly_figure.add_subplot(1, 1, 1)
        try:
            data = ds_t[var]
            # Clean guards for nicer visuals and use xbout polygon plotting
            grid_only = bool(self.poly_grid_only_check.isChecked())
            if grid_only:
                data.hermesm.clean_guards().bout.polygon(
                    ax=ax,
                    grid_only=True,
                    linecolor="k",
                    linewidth=0.2,
                    antialias=True,
                    separatrix=False,
                    targets=False,
                    add_colorbar=False,
                )
                ax.set_title("grid", fontsize=11)
            else:
                data.hermesm.clean_guards().bout.polygon(
                    ax=ax,
                    cmap="Spectral_r",
                    # Semi-transparent gridlines over colormap
                    linecolor=(0, 0, 0, 0.15),
                    linewidth=0,
                    antialias=True,
                    separatrix=True,
                    separatrix_kwargs={"linewidth": 0.2, "color": "k"},
                    targets=False,
                    add_colorbar=True,
                )
                ax.set_title(f"{var}", fontsize=11)
        except Exception as e:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"2D plot failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
        self.poly_canvas.draw_idle()

    def _redraw_2d_monitor(self) -> None:
        self.mon_figure.clear()
        try:
            self.mon_figure.set_facecolor("white")
        except Exception:
            pass

        case = self._primary_case()
        if case is None:
            ax = self.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            self.mon_canvas.draw_idle()
            return

        ds = case.ds
        tdim = self.state.get("time_dim") or "t"
        if tdim not in getattr(ds, "coords", {}):
            ax = self.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"No time coordinate '{tdim}' in dataset.", ha="center", va="center", transform=ax.transAxes)
            self.mon_canvas.draw_idle()
            return

        # Build monitor from 1D poloidal extraction (per user rule):
        # for outer_lower SOL leg, index 0 ~ OMP and index -2 ~ target.
        try:
            from hermes3.selectors import get_1d_poloidal_data  # type: ignore
        except Exception as e:
            ax = self.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"sdtools import error:\n{e}", ha="center", va="center", transform=ax.transAxes)
            self.mon_canvas.draw_idle()
            return

        try:
            t_ms = np.asarray(ds[tdim].values) * 1e3
        except Exception:
            t_ms = np.arange(case.n_time)

        region = "outer_lower"
        sepadd = int(self.pol_sepadd_spin.value()) if hasattr(self, "pol_sepadd_spin") else 0
        ck = (case.label, region, sepadd)
        cached = self._mon_cache.get(ck)
        if cached is None:
            Ne_omp = []
            Te_omp = []
            Ne_targ = []
            Te_targ = []

            def _slice_with_metadata(i: int):
                try:
                    ds_i = ds.isel({tdim: int(i)})
                except Exception:
                    ds_i = ds
                # Reattach metadata/options and backfill geometry aliases
                try:
                    if hasattr(ds, "metadata"):
                        ds_i.metadata = ds.metadata  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    if hasattr(ds, "options"):
                        ds_i.options = ds.options  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    _ensure_sdtools_2d_metadata(ds_i)
                except Exception:
                    pass
                return ds_i

            for i in range(int(case.n_time)):
                dsi = _slice_with_metadata(i)
                try:
                    df = get_1d_poloidal_data(dsi, params=["Ne", "Te"], region=region, sepadd=sepadd, target_first=False)
                    Ne = np.asarray(df["Ne"].values)
                    Te = np.asarray(df["Te"].values)
                    # user rule: OMP is first index; target is 2nd-to-last
                    Ne_omp.append(float(Ne[0]) if Ne.size else np.nan)
                    Te_omp.append(float(Te[0]) if Te.size else np.nan)
                    if Ne.size >= 2:
                        Ne_targ.append(float(Ne[-2]))
                        Te_targ.append(float(Te[-2]))
                    else:
                        Ne_targ.append(np.nan)
                        Te_targ.append(np.nan)
                except Exception:
                    Ne_omp.append(np.nan)
                    Te_omp.append(np.nan)
                    Ne_targ.append(np.nan)
                    Te_targ.append(np.nan)

            Ne_omp = np.asarray(Ne_omp, dtype=float)
            Te_omp = np.asarray(Te_omp, dtype=float)
            Ne_targ = np.asarray(Ne_targ, dtype=float)
            Te_targ = np.asarray(Te_targ, dtype=float)
            self._mon_cache[ck] = (t_ms, Ne_omp, Te_omp, Ne_targ, Te_targ)
            cached = self._mon_cache[ck]

        t_ms, Ne_omp, Te_omp, Ne_targ, Te_targ = cached

        gs = self.mon_figure.add_gridspec(nrows=2, ncols=2, hspace=0.35, wspace=0.30)
        ax00 = self.mon_figure.add_subplot(gs[0, 0])
        ax01 = self.mon_figure.add_subplot(gs[0, 1], sharex=ax00)
        ax10 = self.mon_figure.add_subplot(gs[1, 0], sharex=ax00)
        ax11 = self.mon_figure.add_subplot(gs[1, 1], sharex=ax00)

        def _plot(ax, y, title):
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            if y is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                return
            y = np.asarray(y, dtype=float)
            if not np.any(np.isfinite(y)):
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                return
            ax.plot(t_ms[: len(y)], y, lw=1.5)

        _plot(ax00, Ne_omp, r"$N_e^{omp}$")
        _plot(ax01, Te_omp, r"$T_e^{omp}$")
        _plot(ax10, Ne_targ, r"$N_e^{targ}$")
        _plot(ax11, Te_targ, r"$T_e^{targ}$")

        for ax in (ax10, ax11):
            ax.set_xlabel("Time (ms)")

        self.mon_figure.suptitle(f"{case.label} monitor", fontsize=12)
        try:
            self.mon_figure.tight_layout()
        except Exception:
            pass
        self.mon_canvas.draw_idle()

    def redraw(self) -> None:
        if self._mode_is_2d:
            self._redraw_2d_current_tab()
            return

        self._update_time_readout()

        self.figure.clear()
        # Explicit facecolor to avoid inheriting dark appearances on some platforms.
        try:
            self.figure.set_facecolor("white")
        except Exception:
            pass

        if not self.cases:
            # No plots -> no overlay buttons
            self._clear_overlay_buttons()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No dataset loaded.\nLoad a case directory to view variables.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            self.canvas.draw_idle()
            return

        sdim = self.state.get("spatial_dim")
        tdim = self.state.get("time_dim")
        vars_to_plot = list(self.selected_vars)

        if not vars_to_plot:
            self._clear_overlay_buttons()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No variables selected.\nCheck variables on the left to plot.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            self.canvas.draw_idle()
            return

        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))

        gs = self.figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35, wspace=0.30)

        sharex_ref: List[Optional["object"]] = [None] * ncols
        axes: List["object"] = []

        for idx in range(n):
            col = idx // nrows
            row = idx % nrows
            sharex = sharex_ref[col]
            ax = self.figure.add_subplot(gs[row, col], sharex=sharex)
            if sharex_ref[col] is None:
                sharex_ref[col] = ax
            axes.append(ax)

        # Determine bottom-most axis per column for x-label and tick labels
        bottom_idx_by_col: Dict[int, int] = {}
        for col in range(ncols):
            inds = [i for i in range(n) if (i // nrows) == col]
            if inds:
                bottom_idx_by_col[col] = max(inds)

        for i, ax in enumerate(axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if not is_bottom:
                ax.tick_params(labelbottom=False)

        for ax, name in zip(axes, vars_to_plot):
            mode = self._yscale_by_var.get(name, "linear")
            ylim_mode = self._ylim_mode_by_var.get(name, "auto")

            # Configure y-scale before plotting
            linthresh = None
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    ys_all = []
                    for c in self.cases.values():
                        ds = c.ds
                        if name not in ds:
                            continue
                        da = ds[name]
                        case_ti = self._get_time_index_for_case(c)
                        if tdim is not None and tdim in da.dims:
                            da1 = da.isel({tdim: case_ti})
                        else:
                            da1 = da
                        yv = np.asarray(da1.values)
                        yv = yv[np.isfinite(yv)]
                        if yv.size:
                            ys_all.append(yv)
                    if ys_all:
                        ys = np.concatenate(ys_all)
                        amax = float(np.nanmax(np.abs(ys))) if ys.size else 1.0
                        linthresh = max(1e-12, 1e-3 * amax)
                    else:
                        linthresh = 1e-6
                    ax.set_yscale("symlog", linthresh=linthresh)
                else:
                    ax.set_yscale("linear")
            except Exception as e:
                self.set_status(f"Y-scale error for {name}: {e}", is_error=True)

            # Extract units from first dataset that has var
            units = None
            for c in self.cases.values():
                ds = c.ds
                if name in ds:
                    try:
                        units = ds[name].attrs.get("units", None)
                        if units:
                            break
                    except Exception:
                        pass

            for c in self.cases.values():
                ds = c.ds
                if name not in ds:
                    continue
                da = ds[name]
                try:
                    case_ti = self._get_time_index_for_case(c)
                    if tdim is not None and tdim in da.dims:
                        da1 = da.isel({tdim: case_ti})
                    else:
                        da1 = da

                    if sdim and sdim in ds.coords:
                        x = np.asarray(ds[sdim].values)
                    else:
                        x = np.arange(int(ds.sizes.get(sdim, da1.size))) if sdim else np.arange(da1.size)

                    y = np.asarray(da1.values)
                    if self._guard_replace_enabled() and y.ndim == 1 and sdim in getattr(da1, "dims", ()):
                        x, y = _guard_replace_1d_profile_xy(x, y)
                    if mode == "log":
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(x, y, label=c.label)
                except Exception as e:
                    self.set_status(f"Plot error for {name}: {e}", is_error=True)

            ax.set_title(name, fontsize=10)
            ax.set_ylabel(f"({units})" if units else "")
            ax.grid(True, which="both", alpha=0.3)
            if len(self.cases) > 1:
                ax.legend(loc="upper left", fontsize=9)

            # Apply y-limit mode
            try:
                if ylim_mode == "auto":
                    ax.relim()
                    ax.autoscale_view()
                elif ylim_mode == "final":
                    ymin, ymax = self._compute_ylim_for_final(name, tdim, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
                elif ylim_mode == "global":
                    ymin, ymax = self._compute_ylim_for_global(name, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
            except Exception:
                pass

        # X label on bottom-most axis in each column
        for i, ax in enumerate(axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if is_bottom:
                ax.set_xlabel(r"S$_\parallel$ (m)")

        # Sync overlay buttons to the current subplot grid.
        self._sync_overlay_buttons(vars_to_plot=vars_to_plot, axes=axes)
        self.canvas.draw_idle()

    # ---------- Time history plotting ----------
    def redraw_time_history(self) -> None:
        """
        Backwards-compatible entry point.
        Prefer `request_time_history_redraw()` for better interactivity.
        """
        self.request_time_history_redraw()

    def _do_redraw_time_history(self) -> None:
        """
        Plot time traces of selected variables at an upstream and target index, based on
        `plot_time_history_optimized` in `convergence_functions.py`.

        For each selected variable we draw 2 subplots (rows):
        - upstream value vs time
        - target value vs time
        """
        self.hist_figure.clear()
        try:
            self.hist_figure.set_facecolor("white")
        except Exception:
            pass

        if not self.cases:
            ax = self.hist_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No dataset loaded.\nLoad a case directory to view time history.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            self.hist_canvas.draw_idle()
            return

        tdim = self.state.get("time_dim") or "t"
        # Shared selection with profiles
        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            ax = self.hist_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No variables selected.\nCheck variables on the left to plot time history.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            self.hist_canvas.draw_idle()
            return

        upstream_index = int(self.hist_upstream_spin.value())
        target_index = int(self.hist_target_spin.value())
        time_slices = int(self.hist_time_slices_spin.value())

        n_cols = max(1, len(vars_to_plot))
        n_rows = 2
        gs = self.hist_figure.add_gridspec(nrows=n_rows, ncols=n_cols, hspace=0.35, wspace=0.30)

        last_time_ms = None
        for i, var in enumerate(vars_to_plot):
            ax_u = self.hist_figure.add_subplot(gs[0, i])
            ax_t = self.hist_figure.add_subplot(gs[1, i], sharex=ax_u)

            units = None
            # Choose a scale similar to convergence_functions (log if huge)
            log_threshold = 1e6
            max_abs = 0.0

            for c in self.cases.values():
                ds = c.ds
                if var not in ds:
                    continue
                da = ds[var]
                try:
                    units = units or da.attrs.get("units", None)
                except Exception:
                    pass

                # Pick time dim
                if tdim not in da.dims:
                    # Not a time-varying variable -> skip
                    continue

                # Spatial dimension: try "y" then "pos" then inferred spatial dim
                sdim = None
                for cand in ("y", "pos", self.state.get("spatial_dim")):
                    if cand and cand in da.dims:
                        sdim = cand
                        break
                if sdim is None:
                    continue

                # Clamp indices (per-case, per-var)
                n_s = int(ds.sizes.get(sdim, 0))
                if n_s <= 0:
                    continue
                upi = upstream_index if upstream_index >= 0 else max(0, n_s + upstream_index)
                tgi = target_index if target_index >= 0 else max(0, n_s + target_index)
                upi = int(np.clip(upi, 0, n_s - 1))
                tgi = int(np.clip(tgi, 0, n_s - 1))

                # Cache key: (case_label, var, tdim, sdim, upi, tgi)
                ck = (c.label, var, tdim, sdim, upi, tgi)
                cached = self._hist_cache.get(ck)
                if cached is None:
                    try:
                        t_full = np.asarray(ds[tdim].values) * 1e3
                        y_up_full = np.asarray(self._isel_1d_with_guard_replace(da, sdim=sdim, idx=upi).values).squeeze()
                        y_tg_full = np.asarray(self._isel_1d_with_guard_replace(da, sdim=sdim, idx=tgi).values).squeeze()
                        self._hist_cache[ck] = (t_full, y_up_full, y_tg_full)
                        cached = self._hist_cache[ck]
                    except Exception:
                        continue

                t_full, y_up_full, y_tg_full = cached
                if t_full is None:
                    continue

                n_t = int(len(t_full))
                if n_t <= 0:
                    continue
                n_sel = min(time_slices, n_t)
                sl = slice(-n_sel, None)
                tvals = t_full[sl]
                y_up = y_up_full[sl]
                y_tg = y_tg_full[sl]

                # Downsample for responsiveness (keep last N points evenly)
                try:
                    if self._hist_max_points and len(tvals) > int(self._hist_max_points):
                        stride = int(np.ceil(len(tvals) / float(self._hist_max_points)))
                        tvals = tvals[::stride]
                        y_up = y_up[::stride]
                        y_tg = y_tg[::stride]
                except Exception:
                    pass

                # Track scale decision
                try:
                    max_abs = max(max_abs, float(np.nanmax(np.abs(y_up))), float(np.nanmax(np.abs(y_tg))))
                except Exception:
                    pass

                ax_u.plot(tvals, y_up, "-", linewidth=1.5, label=c.label)
                ax_t.plot(tvals, y_tg, "--", linewidth=1.5, label=c.label)

                if tvals.size:
                    last_time_ms = float(tvals[-1])

            scale = "log" if (max_abs > log_threshold and max_abs > 0) else "linear"
            ax_u.set_yscale(scale)
            ax_t.set_yscale(scale)

            ax_u.set_title(f"Upstream {var}", fontsize=10)
            ax_t.set_title(f"Target {var}", fontsize=10)
            ax_t.set_xlabel("Time (ms)")
            # Variable name is already in the subplot title; keep y-label to units only.
            ylabel = f"({units})" if units else ""
            ax_u.set_ylabel(ylabel)
            ax_t.set_ylabel(ylabel)
            ax_u.grid(True, alpha=0.3)
            ax_t.grid(True, alpha=0.3)

            if len(self.cases) > 1:
                ax_u.legend(loc="best", fontsize=8)

            # Hide x tick labels on top row
            ax_u.tick_params(labelbottom=False)

        # if last_time_ms is not None:
        #     self.hist_time_readout.setText(f"Time history (last: {last_time_ms:.4f} ms)")
        #     self.hist_figure.suptitle(f"Time History (Last: {last_time_ms:.3f} ms)", fontsize=12)
        # else:
        #     self.hist_time_readout.setText("Time history")

        try:
            self.hist_figure.tight_layout()
        except Exception:
            pass
        self.hist_canvas.draw_idle()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes-3 1D GUI (PyQt + embedded Matplotlib).")
    parser.add_argument(
        "casepath",
        nargs="?",
        default=None,
        help="Path to Hermes-3 1D case directory (contains BOUT.dmp.*.nc and BOUT.inp).",
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
        help="GUI theme override. 'system' follows OS, 'light' forces light mode.",
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    if args.theme == "light":
        _apply_qt_light_theme(app)
        _apply_mpl_light_theme()
    elif args.theme == "dark":
        # Keep Qt system palette by default; for Matplotlib, a dark theme is available.
        # If you want Qt dark, we'd set a dark QPalette here.
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
    win = Hermes3QtMainWindow(initial_case_path=args.casepath, spatial_dim=args.spatial_dim)
    win.resize(1400, 850)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

