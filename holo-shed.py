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
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402


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
    # Include derived/geometry coordinates (e.g. R, Z) as plottable options too.
    # Avoid listing the coordinate axes themselves.
    ignore = {spatial_dim}
    if time_dim:
        ignore.add(time_dim)
    for name in getattr(ds, "variables", {}).keys():
        if name in ignore:
            continue
        try:
            da = ds[name]
            if _is_plottable_1d_var(da, spatial_dim=spatial_dim, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(set(out))


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
    # Include derived/geometry coordinates (e.g. R, Z) as plottable options too.
    ignore = {"x", "theta", "y"}
    if time_dim:
        ignore.add(time_dim)
    for name in getattr(ds, "variables", {}).keys():
        if name in ignore:
            continue
        try:
            da = ds[name]
            if _is_plottable_2d_var(da, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(set(out))


def _selector_params_only(vars_to_plot: List[str]) -> List[str]:
    """
    sdtools selectors always provide geometry columns like Spar/Spol/R/Z.
    Avoid requesting those as "params" to keep selectors robust.
    """
    geom = {"Spar", "Spol", "Srad", "R", "Z"}
    return [v for v in vars_to_plot if v not in geom]


def _xpoint_idx_bpxy_valley(bp: np.ndarray) -> Optional[int]:
    """
    Heuristic X-point index from a 1D Bpxy trace along the field line.

    We walk from the start and keep updating the minimum |Bpxy| until the signal
    has started rising again for a few consecutive points. This avoids picking the
    target-side low-field region after the X-point.
    """
    a = np.asarray(bp, dtype=float)
    m = np.isfinite(a)
    if not np.any(m):
        return None
    a = np.abs(a[m])
    if a.size < 3:
        return int(np.argmin(a))

    best_i = 0
    best = float(a[0])
    rise = 0
    # how many consecutive rises indicate we're past the minimum
    rise_needed = 3
    eps = 0.0

    prev = float(a[0])
    for i in range(1, int(a.size)):
        cur = float(a[i])
        if cur < best:
            best = cur
            best_i = i
            rise = 0
        # Rising after we have a minimum
        if i > best_i:
            if cur > prev + eps:
                rise += 1
            else:
                rise = 0
            if rise >= rise_needed and (i - best_i) >= 2:
                break
        prev = cur
    return int(best_i)


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


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if max_points is None or max_points <= 0:
        return x, y
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        return x, y
    n = int(x.size)
    if n <= int(max_points):
        return x, y
    stride = int(np.ceil(n / float(max_points)))
    return x[::stride], y[::stride]


def _parse_optional_float(text: str) -> Optional[float]:
    """
    Parse a float from a text box. Empty/'auto' -> None.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() == "auto":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _get_option_float(ds, keys: List[str]) -> Optional[float]:
    """
    Best-effort extraction of a float from ds.options for a list of candidate keys.
    Supports both nested options (ds.options["mesh"]["length_xpt"]) and flat keys.
    """
    try:
        opt = getattr(ds, "options", None)
        if opt is None:
            return None
    except Exception:
        return None

    for k in keys:
        # Nested form: "mesh:length_xpt"
        if ":" in k:
            sec, name = k.split(":", 1)
            try:
                v = opt[sec][name]
                return float(v)
            except Exception:
                pass
        # Flat form: "length_xpt"
        try:
            v = opt[k]
            return float(v)
        except Exception:
            pass
    return None


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

    # Some workflows produce a single combined file under this name
    dmp_single = case_dir / "BOUT.dmp.nc"
    if dmp_single.exists():
        return dmp_single

    # Prefer the first dump file if present (common naming)
    d0 = case_dir / "BOUT.dmp.0.nc"
    if d0.exists():
        return d0

    # Fall back to any dump file
    dmps = sorted(case_dir.glob("BOUT.dmp.*.nc"))
    if dmps:
        return dmps[0]

    raise FileNotFoundError(
        f"Could not find BOUT output in {case_dir} (expected BOUT.squash.nc, BOUT.dmp.nc, or BOUT.dmp.*.nc)."
    )


def _should_use_squash_for_load(case_dir: Path) -> bool:
    """
    Decide whether to ask sdtools to use (or create) BOUT.squash.nc.

    User preference: always use squash for multi-file outputs, because even
    ~20 dump files can be a noticeable slowdown.

    Notes:
    - If BOUT.squash.nc already exists, always use it.
    - If we have per-timestep files (BOUT.dmp.*.nc), always use/create squash.
    - If the case already has a single combined file (e.g. BOUT.dmp.nc) and no
      per-timestep files, we skip squash (already fast, and squash tooling may
      assume BOUT.dmp.*.nc exists).
    """
    try:
        if (case_dir / "BOUT.squash.nc").exists():
            return True
    except Exception:
        pass

    # Any per-timestep dump files -> always squash
    try:
        for _ in case_dir.glob("BOUT.dmp.*.nc"):
            return True
    except Exception:
        pass

    return False


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
    from PySide6.QtCore import QEvent, Qt, QTimer  # type: ignore
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


class _RegionOverlayWindow(QMainWindow):
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
        self._ylim_mode_by_var: Dict[str, str] = {}  # var -> {"auto","final","max"} (legacy: "global" == "max")
        self._var_filter: str = ""

        self._build_ui()

        # Overlay controls (Qt buttons positioned on top of each subplot).
        # var -> (ylim_button, yscale_button)
        self._overlay_buttons: Dict[str, Tuple["QPushButton", "QPushButton"]] = {}
        # var -> matplotlib Axes (for positioning)
        self._overlay_axes_by_var: Dict[str, "object"] = {}
        # 2D poloidal overlays
        self._overlay_buttons_pol: Dict[str, Tuple["QPushButton", "QPushButton"]] = {}
        self._overlay_axes_by_var_pol: Dict[str, "object"] = {}
        # 2D radial overlays
        self._overlay_buttons_rad: Dict[str, Tuple["QPushButton", "QPushButton"]] = {}
        self._overlay_axes_by_var_rad: Dict[str, "object"] = {}
        # Geometry constants (pixels, in canvas coordinates)
        self._overlay_btn_h = 22
        self._overlay_btn_w_yscale = 56
        self._overlay_btn_w_ylim = 72
        self._overlay_pad = 6

        # Keep overlay buttons positioned correctly on draw + resize.
        self.canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons())
        self.canvas.installEventFilter(self)
        # 2D canvases (created in _build_ui) also use overlay buttons.
        # These may not exist in early init, so guard with getattr.
        try:
            self.pol_canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons_pol())
            self.pol_canvas.installEventFilter(self)
        except Exception:
            pass
        try:
            self.rad_canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons_rad())
            self.rad_canvas.installEventFilter(self)
        except Exception:
            pass

        # Time-history performance helpers:
        # - debounce redraws (avoid redrawing many times while user drags/scrolls)
        # - cache extracted time series per case/var/index (avoid repeated xarray slicing)
        self._hist_redraw_timer = QTimer(self)
        self._hist_redraw_timer.setSingleShot(True)
        self._hist_redraw_timer.timeout.connect(self._do_redraw_time_history)
        self._hist_cache: Dict[tuple, tuple] = {}
        self._hist_max_points = 2000  # downsample long traces for responsiveness
        self._mon_cache: Dict[tuple, tuple] = {}
        self._pol_cache: Dict[tuple, object] = {}
        self._rad_cache: Dict[tuple, object] = {}
        # Cached y-limits for 2D extracted 1D plots (poloidal/radial)
        self._pol_ylim_cache: Dict[tuple, Tuple[Optional[float], Optional[float]]] = {}
        self._rad_ylim_cache: Dict[tuple, Tuple[Optional[float], Optional[float]]] = {}
        self._profile_max_points = 2500
        self._fast_slider_step = 10  # Shift+Arrow moves by this many indices
        # 2D polygon colorbar limits are only applied when user confirms
        self._poly_vmin_active: Optional[float] = None
        self._poly_vmax_active: Optional[float] = None
        # Preserve 1D toolbar zoom/pan across time scrubbing
        self._last_draw_state_1d_profiles: Optional[tuple] = None
        # 2D polygon artist reuse (fast time slider updates)
        self._poly_plot_state: Optional[tuple] = None
        self._poly_ax = None
        self._poly_polys = None
        self._poly_cbar = None
        # Optional popout windows showing cut location on 2D colormap
        self._region2d_pol = None
        self._region2d_rad = None

        # 2D time-slider redraw throttle (keeps dragging responsive)
        self._time2d_redraw_timer = QTimer(self)
        self._time2d_redraw_timer.setSingleShot(True)
        self._time2d_redraw_timer.timeout.connect(self._do_redraw)
        self._time2d_dragging = False

        # Global keyboard shortcuts (arrow keys)
        self._time_shortcuts: List["QShortcut"] = []

        # Skip re-rendering 2D tabs when nothing changed
        self._last_draw_state_2d: Dict[str, object] = {"pol": None, "rad": None, "poly": None, "mon": None}

        # General redraw debounce (profiles + 2D tabs)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._do_redraw)

        # Install shortcuts once widgets exist
        self._install_time_shortcuts()
        # Also install a global key event filter so plain Left/Right always work
        # (Qt can treat bare arrow keys as navigation rather than shortcuts).
        try:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
        except Exception:
            pass

        if initial_case_path:
            self.path_edit.setText(str(initial_case_path))
            self.load_dataset(replace=True)
        else:
            self.set_status("Enter a case directory path and click 'Load dataset'.")
            self.redraw()
            self.request_time_history_redraw()

    # ---------- Keyboard shortcuts ----------
    def _install_time_shortcuts(self) -> None:
        """
        Global shortcuts for time scrubbing (work without focusing the slider).
        """
        self._time_shortcuts = []

        def add(seq: str, delta: int) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(lambda d=delta: self._nudge_time_slider(d))
            self._time_shortcuts.append(sc)

        add("Left", -1)
        add("Right", 1)
        add("Shift+Left", -int(self._fast_slider_step))
        add("Shift+Right", int(self._fast_slider_step))

    def _nudge_time_slider(self, delta: int) -> None:
        slider = self._active_time_slider()
        try:
            v = int(slider.value()) + int(delta)
            v = max(int(slider.minimum()), min(int(slider.maximum()), v))
            slider.setValue(v)
            self._set_time_readout_for_index(v)
        except Exception:
            return

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
        # Allow continuous updates while dragging; redraw is throttled separately.
        try:
            self.time_slider_2d.setTracking(True)
        except Exception:
            pass
        time2d_row.addWidget(self.time_slider_2d, 1)

        # Controls to the right of the slider (replacing the old readout label)
        time2d_row.addWidget(QLabel("idx"))
        self.time_spin_2d = QSpinBox()
        self.time_spin_2d.setRange(0, 0)
        self.time_spin_2d.setValue(0)
        try:
            # Don't emit while user types, only when value is committed/stepped
            self.time_spin_2d.setKeyboardTracking(False)
        except Exception:
            pass
        time2d_row.addWidget(self.time_spin_2d)

        time2d_row.addWidget(QLabel("t [ms]"))
        self.time_ms_spin_2d = QDoubleSpinBox()
        self.time_ms_spin_2d.setDecimals(4)
        self.time_ms_spin_2d.setSingleStep(0.1)
        self.time_ms_spin_2d.setRange(0.0, 1.0e12)
        self.time_ms_spin_2d.setValue(0.0)
        try:
            self.time_ms_spin_2d.setKeyboardTracking(False)
        except Exception:
            pass
        time2d_row.addWidget(self.time_ms_spin_2d)

        # Keep the old label for backwards compatibility (must be parented to avoid becoming a top-level window)
        self.time_readout_2d = QLabel("time index = 0", self.time2d_widget)
        self.time_readout_2d.setVisible(False)
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

        self.pol_use_spol_check = QCheckBox("Plot vs Spol (instead of Spar)")
        self.pol_use_spol_check.setChecked(False)
        pol_ctrl_layout.addWidget(self.pol_use_spol_check)

        pol_xpt_row = QHBoxLayout()
        pol_xpt_row.addWidget(QLabel("X-point method"))
        self.pol_xpoint_combo = QComboBox()
        self.pol_xpoint_combo.addItems(["Bxy extrema", "Bpxy valley", "min R"])
        try:
            self.pol_xpoint_combo.setCurrentIndex(0)  # default
        except Exception:
            pass
        pol_xpt_row.addWidget(self.pol_xpoint_combo, 1)
        pol_ctrl_layout.addLayout(pol_xpt_row)

        self.pol_show_region2d_check = QCheckBox("Show region in 2D")
        self.pol_show_region2d_check.setChecked(False)
        pol_ctrl_layout.addWidget(self.pol_show_region2d_check)
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

        self.rad_show_region2d_check = QCheckBox("Show region in 2D")
        self.rad_show_region2d_check.setChecked(False)
        rad_ctrl_layout.addWidget(self.rad_show_region2d_check)
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

        self.poly_log_check = QCheckBox("log scale")
        self.poly_log_check.setChecked(False)
        poly_ctrl_layout.addWidget(self.poly_log_check)

        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("cbar min"))
        self.poly_vmin_edit = QLineEdit()
        self.poly_vmin_edit.setPlaceholderText("auto")
        clim_row.addWidget(self.poly_vmin_edit, 1)
        clim_row.addWidget(QLabel("max"))
        self.poly_vmax_edit = QLineEdit()
        self.poly_vmax_edit.setPlaceholderText("auto")
        clim_row.addWidget(self.poly_vmax_edit, 1)
        poly_ctrl_layout.addLayout(clim_row)

        apply_row = QHBoxLayout()
        self.poly_apply_clim_btn = QPushButton("Apply cbar limits")
        apply_row.addWidget(self.poly_apply_clim_btn)
        apply_row.addStretch(1)
        poly_ctrl_layout.addLayout(apply_row)
        poly_ctrl_layout.addStretch(1)

        self._mon_ctrl_tab = QWidget()
        mon_ctrl_layout = QVBoxLayout(self._mon_ctrl_tab)
        mon_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        mon_ctrl_layout.setSpacing(6)
        mon_ctrl_layout.addWidget(QLabel("Time history at OMP + target (from selected variables)"))
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
        slider_row.addWidget(self.time_slider, 1)

        # Controls to the right of the slider (replacing the old readout label)
        slider_row.addWidget(QLabel("idx"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 0)
        self.time_spin.setValue(0)
        try:
            self.time_spin.setKeyboardTracking(False)
        except Exception:
            pass
        slider_row.addWidget(self.time_spin)

        slider_row.addWidget(QLabel("t [ms]"))
        self.time_ms_spin = QDoubleSpinBox()
        self.time_ms_spin.setDecimals(4)
        self.time_ms_spin.setSingleStep(0.1)
        self.time_ms_spin.setRange(0.0, 1.0e12)
        self.time_ms_spin.setValue(0.0)
        try:
            self.time_ms_spin.setKeyboardTracking(False)
        except Exception:
            pass
        slider_row.addWidget(self.time_ms_spin)

        # Keep old widgets for backwards compatibility (parent + hidden to avoid top-level popup windows)
        self.time_readout = QLabel("time index = 0", prof_plot_tab)
        self.time_readout.setVisible(False)
        self._time_index_label_1d = QLabel("time index", prof_plot_tab)
        self._time_index_label_1d.setVisible(False)
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

        self._tab2d_timehist = QWidget()
        mon_plot_layout = QVBoxLayout(self._tab2d_timehist)
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
        self.time_slider.valueChanged.connect(self._on_time_slider_1d_changed)
        self.time_slider_2d.valueChanged.connect(self._on_time_slider_2d_changed)
        self.time_slider_2d.sliderMoved.connect(lambda v: self._set_time_readout_for_index(int(v)))
        self.time_slider_2d.sliderPressed.connect(lambda: self._set_time2d_dragging(True))
        self.time_slider_2d.sliderReleased.connect(lambda: self._set_time2d_dragging(False, redraw=True))
        self.time_spin.valueChanged.connect(self._on_time_spin_1d_changed)
        self.time_spin_2d.valueChanged.connect(self._on_time_spin_2d_changed)
        self.time_ms_spin.valueChanged.connect(self._on_time_ms_spin_1d_changed)
        self.time_ms_spin_2d.valueChanged.connect(self._on_time_ms_spin_2d_changed)
        self.pol_region_combo.currentIndexChanged.connect(lambda _i: self.request_redraw())
        self.pol_sepadd_spin.valueChanged.connect(lambda _v: self.request_redraw())
        self.pol_use_spol_check.toggled.connect(lambda _v: self.request_redraw())
        self.pol_xpoint_combo.currentIndexChanged.connect(lambda _i: self.request_redraw())
        self.pol_show_region2d_check.toggled.connect(self._on_pol_show_region2d_toggled)
        self.rad_region_combo.currentIndexChanged.connect(lambda _i: self.request_redraw())
        self.rad_show_region2d_check.toggled.connect(self._on_rad_show_region2d_toggled)
        self.poly_var_combo.currentIndexChanged.connect(lambda _i: self.request_redraw())
        self.poly_grid_only_check.toggled.connect(lambda _v: self.request_redraw())
        self.poly_log_check.toggled.connect(lambda _v: self.request_redraw())
        self.poly_apply_clim_btn.clicked.connect(self._apply_poly_clim)
        self.guard_replace_check.toggled.connect(lambda _v: self.request_redraw())
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
          - Poloidal 1D + Radial 1D + 2D field + Time history
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
            self.plot_tabs.addTab(self._tab2d_timehist, "Time history")

            self.controls_tabs.addTab(self._pol_ctrl_tab, "Poloidal 1D")
            self.controls_tabs.addTab(self._rad_ctrl_tab, "Radial 1D")
            self.controls_tabs.addTab(self._poly_ctrl_tab, "2D field")
            self.controls_tabs.addTab(self._mon_ctrl_tab, "Time history")

            # Show global 2D time control; hide 1D slider embedded in profiles tab.
            try:
                self.time2d_widget.setVisible(True)
            except Exception:
                pass
            for w in (
                getattr(self, "time_spin", None),
                getattr(self, "time_ms_spin", None),
                getattr(self, "time_slider", None),
            ):
                try:
                    if w is not None:
                        w.setVisible(False)
                except Exception:
                    pass
        else:
            self.setWindowTitle(f"Hermes-3 GUI (1D) - Qt ({_QT_API})")
            # Leaving 2D mode -> close any region-overlay popouts
            try:
                self._close_region2d_window("pol")
                self._close_region2d_window("rad")
            except Exception:
                pass
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
            for w in (
                getattr(self, "time_spin", None),
                getattr(self, "time_ms_spin", None),
                getattr(self, "time_slider", None),
            ):
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

    def _active_time_spin(self) -> "QSpinBox":
        return self.time_spin_2d if self._mode_is_2d else self.time_spin

    def _active_time_ms_spin(self) -> "QDoubleSpinBox":
        return self.time_ms_spin_2d if self._mode_is_2d else self.time_ms_spin

    def _t_values_ms(self) -> Optional[np.ndarray]:
        tvals = self.state.get("t_values")
        if tvals is None:
            return None
        try:
            arr = np.asarray(tvals, dtype=float) * 1e3  # stored in seconds, display in ms
            arr = arr[np.isfinite(arr)]
            return arr if arr.size else None
        except Exception:
            return None

    def _nearest_time_index_for_ms(self, t_ms: float) -> Optional[int]:
        tvals = self.state.get("t_values")
        if tvals is None:
            return None
        try:
            arr = np.asarray(tvals, dtype=float) * 1e3
            if arr.size == 0:
                return None
            # Use finite values for distance; map back to original indices
            finite_mask = np.isfinite(arr)
            if not np.any(finite_mask):
                return None
            idxs = np.nonzero(finite_mask)[0]
            arr_f = arr[finite_mask]
            j = int(np.argmin(np.abs(arr_f - float(t_ms))))
            return int(idxs[j])
        except Exception:
            return None

    def _set_time_ms_spin_for_index(self, ti: int) -> None:
        tvals = self.state.get("t_values")
        if tvals is None:
            try:
                self._active_time_ms_spin().setEnabled(False)
            except Exception:
                pass
            return
        try:
            tvals = np.asarray(tvals, dtype=float)
            if ti < 0 or ti >= tvals.size or not np.isfinite(tvals[ti]):
                return
            ms = float(tvals[ti]) * 1e3
            spin = self._active_time_ms_spin()
            spin.setEnabled(True)
            spin.blockSignals(True)
            spin.setValue(ms)
        finally:
            try:
                self._active_time_ms_spin().blockSignals(False)
            except Exception:
                pass

    def _set_time_spin_for_index(self, ti: int) -> None:
        try:
            spin = self._active_time_spin()
            spin.blockSignals(True)
            spin.setValue(int(ti))
        finally:
            try:
                spin.blockSignals(False)
            except Exception:
                pass

    def _on_time_spin_1d_changed(self, v: int) -> None:
        # Jump to index via numeric input (1D)
        try:
            self.time_slider.setValue(int(v))
        except Exception:
            pass
        self.request_redraw()

    def _on_time_spin_2d_changed(self, v: int) -> None:
        # Jump to index via numeric input (2D)
        try:
            self.time_slider_2d.setValue(int(v))
        except Exception:
            pass
        # 2D redraw can be heavy; this is an intentional action so redraw immediately
        self.request_redraw()

    def _on_time_ms_spin_1d_changed(self, v: float) -> None:
        ti = self._nearest_time_index_for_ms(float(v))
        if ti is None:
            return
        try:
            self.time_slider.setValue(int(ti))
        except Exception:
            pass
        self.request_redraw()

    def _on_time_ms_spin_2d_changed(self, v: float) -> None:
        ti = self._nearest_time_index_for_ms(float(v))
        if ti is None:
            return
        try:
            self.time_slider_2d.setValue(int(ti))
        except Exception:
            pass
        self.request_redraw()

    def _on_time_slider_1d_changed(self, v: int) -> None:
        # Keep numeric input synced with slider
        try:
            self.time_spin.blockSignals(True)
            self.time_spin.setValue(int(v))
        finally:
            try:
                self.time_spin.blockSignals(False)
            except Exception:
                pass
        try:
            self._set_time_ms_spin_for_index(int(v))
        except Exception:
            pass
        self.request_redraw()

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
            self.request_redraw()
        else:
            # Redraw both (cheap enough and keeps state consistent)
            self.request_redraw()
            self.request_time_history_redraw()

    def request_time_history_redraw(self) -> None:

        if self._mode_is_2d:
            # 2D mode uses the Time history tab instead of the 1D time-history view.
            self._redraw_2d_monitor()
            return

        try:
            self._hist_redraw_timer.start(120)
        except Exception:
            # Fallback: redraw immediately
            self._do_redraw_time_history()

    def request_redraw(self) -> None:
        """
        Immediate redraw for the currently active plot tab.

        (We previously debounced this for performance, but it makes the time slider
        feel laggy when scrubbing.)
        """
        try:
            self.redraw()
        except Exception:
            self._do_redraw()

    def _do_redraw(self) -> None:
        try:
            self.redraw()
        except Exception:
            # avoid crashing the UI loop on unexpected plot errors
            pass

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
        # Keep variable list clean; controls live on plots / context menu.
        return str(name)

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
        self.request_redraw()
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
        self.request_redraw()
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
        # Legacy support: older sessions may store "global"
        if current == "global":
            current = "max"
        order = ["auto", "final", "max"]
        try:
            i = order.index(current)
        except ValueError:
            return "auto"
        return order[(i + 1) % len(order)]

    def _ylim_mode_label(self, mode: str) -> str:
        if mode == "final":
            return "ylim:final"
        if mode in ("global", "max"):
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
        act_cycle_ylim = QAction("Cycle y-limits (auto → final → max)", self)
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
        for mode in ("auto", "final", "max"):
            a = QAction(mode, self)
            a.setCheckable(True)
            curm = self._ylim_mode_by_var.get(name, "auto")
            if curm == "global":
                curm = "max"
            a.setChecked(curm == mode)
            a.triggered.connect(lambda _=False, m=mode: self._set_var_ylim_mode(name, m))
            m_ylim.addAction(a)

        def _do_cycle_y():
            cur = self._yscale_by_var.get(name, "linear")
            self._yscale_by_var[name] = self._cycle_yscale(cur)
            self._refresh_var_item(name)
            self.request_redraw()

        def _do_cycle_ylim():
            cur = self._ylim_mode_by_var.get(name, "auto")
            self._ylim_mode_by_var[name] = self._cycle_ylim_mode(cur)
            self._refresh_var_item(name)
            self.request_redraw()

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
        self.request_redraw()

    def _set_var_ylim_mode(self, varname: str, mode: str) -> None:
        # Legacy: allow "global" as alias for "max"
        if mode == "global":
            mode = "max"
        self._ylim_mode_by_var[varname] = mode
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.request_redraw()

    # ---------- Overlay buttons on plots (Option B) ----------
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        """
        Reposition overlay buttons when the canvas resizes.
        """
        try:
            # Global key handling (plain arrow keys) for time scrubbing
            et = event.type()
            if et == QEvent.Type.KeyPress or et == getattr(QEvent, "KeyPress", None):
                try:
                    key = int(event.key())
                    if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                        mods = event.modifiers()
                        is_shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                        delta = int(self._fast_slider_step) if is_shift else 1
                        if key == Qt.Key.Key_Left:
                            delta = -delta
                        self._nudge_time_slider(delta)
                        return True
                except Exception:
                    pass

            if obj is self.canvas:
                # PyQt6/PySide6 both expose QEvent.Type.Resize; keep a fallback just in case.
                if et == QEvent.Type.Resize or et == getattr(QEvent, "Resize", None):
                    self._position_overlay_buttons()
            if obj is getattr(self, "pol_canvas", None):
                if et == QEvent.Type.Resize or et == getattr(QEvent, "Resize", None):
                    self._position_overlay_buttons_pol()
            if obj is getattr(self, "rad_canvas", None):
                if et == QEvent.Type.Resize or et == getattr(QEvent, "Resize", None):
                    self._position_overlay_buttons_rad()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _on_time_slider_2d_changed(self, v: int) -> None:
        """
        Live scrubbing for 2D time slider, but throttle redraw while dragging.
        """
        try:
            self._set_time_readout_for_index(int(v))
        except Exception:
            pass
        if getattr(self, "_time2d_dragging", False):
            # Restart timer; draw at most ~1 every 60ms while dragging
            try:
                self._time2d_redraw_timer.start(60)
            except Exception:
                self.request_redraw()
        else:
            self.request_redraw()

    def _set_time2d_dragging(self, dragging: bool, *, redraw: bool = False) -> None:
        try:
            self._time2d_dragging = bool(dragging)
        except Exception:
            return
        if redraw:
            # Ensure final position redraws immediately on release
            try:
                self._time2d_redraw_timer.stop()
            except Exception:
                pass
            self.request_redraw()

    def _apply_poly_clim(self) -> None:
        """
        Apply 2D field colorbar limits from the text boxes.
        """
        self._poly_vmin_active = _parse_optional_float(self.poly_vmin_edit.text())
        self._poly_vmax_active = _parse_optional_float(self.poly_vmax_edit.text())
        self.request_redraw()

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

    def _clear_overlay_buttons_pol(self) -> None:
        for ylim_btn, yscale_btn in list(self._overlay_buttons_pol.values()):
            try:
                ylim_btn.hide()
                yscale_btn.hide()
                ylim_btn.deleteLater()
                yscale_btn.deleteLater()
            except Exception:
                pass
        self._overlay_buttons_pol = {}
        self._overlay_axes_by_var_pol = {}

    def _clear_overlay_buttons_rad(self) -> None:
        for ylim_btn, yscale_btn in list(self._overlay_buttons_rad.values()):
            try:
                ylim_btn.hide()
                yscale_btn.hide()
                ylim_btn.deleteLater()
                yscale_btn.deleteLater()
            except Exception:
                pass
        self._overlay_buttons_rad = {}
        self._overlay_axes_by_var_rad = {}

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
        for d in (self._overlay_buttons, self._overlay_buttons_pol, self._overlay_buttons_rad):
            pair = d.get(varname)
            if not pair:
                continue
            ylim_btn, yscale_btn = pair
            ylim_btn.setText(self._ylim_mode_label(self._ylim_mode_by_var.get(varname, "auto")))
            yscale_btn.setText(self._yscale_label(self._yscale_by_var.get(varname, "linear")))

    def _on_overlay_yscale_clicked(self, varname: str) -> None:
        cur = self._yscale_by_var.get(varname, "linear")
        self._yscale_by_var[varname] = self._cycle_yscale(cur)
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.request_redraw()

    def _on_overlay_ylim_clicked(self, varname: str) -> None:
        cur = self._ylim_mode_by_var.get(varname, "auto")
        self._ylim_mode_by_var[varname] = self._cycle_ylim_mode(cur)
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.request_redraw()

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

    def _position_overlay_buttons_pol(self) -> None:
        return self._position_overlay_buttons_for_canvas(
            canvas=getattr(self, "pol_canvas", None),
            overlay_buttons=self._overlay_buttons_pol,
            overlay_axes_by_var=self._overlay_axes_by_var_pol,
        )

    def _position_overlay_buttons_rad(self) -> None:
        return self._position_overlay_buttons_for_canvas(
            canvas=getattr(self, "rad_canvas", None),
            overlay_buttons=self._overlay_buttons_rad,
            overlay_axes_by_var=self._overlay_axes_by_var_rad,
        )

    def _position_overlay_buttons_for_canvas(self, *, canvas, overlay_buttons, overlay_axes_by_var) -> None:
        if canvas is None:
            return
        if not overlay_buttons or not overlay_axes_by_var:
            return
        try:
            w, h = canvas.get_width_height()
        except Exception:
            return
        if not w or not h:
            return

        pad = int(self._overlay_pad)
        bh = int(self._overlay_btn_h)
        bw_y = int(self._overlay_btn_w_yscale)
        bw_l = int(self._overlay_btn_w_ylim)

        for v, (ylim_btn, yscale_btn) in list(overlay_buttons.items()):
            ax = overlay_axes_by_var.get(v)
            if ax is None:
                try:
                    ylim_btn.hide()
                    yscale_btn.hide()
                except Exception:
                    pass
                continue
            try:
                pos = ax.get_position()
                x_right = int(pos.x1 * w)
                y_top = int((1.0 - pos.y1) * h)
            except Exception:
                continue

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

    def _sync_overlay_buttons_pol(self, vars_to_plot: List[str], axes: List["object"]) -> None:
        self._sync_overlay_buttons_for_canvas(
            canvas=getattr(self, "pol_canvas", None),
            vars_to_plot=vars_to_plot,
            axes=axes,
            overlay_buttons=self._overlay_buttons_pol,
            overlay_axes_by_var=self._overlay_axes_by_var_pol,
        )
        self._position_overlay_buttons_pol()

    def _sync_overlay_buttons_rad(self, vars_to_plot: List[str], axes: List["object"]) -> None:
        self._sync_overlay_buttons_for_canvas(
            canvas=getattr(self, "rad_canvas", None),
            vars_to_plot=vars_to_plot,
            axes=axes,
            overlay_buttons=self._overlay_buttons_rad,
            overlay_axes_by_var=self._overlay_axes_by_var_rad,
        )
        self._position_overlay_buttons_rad()

    def _sync_overlay_buttons_for_canvas(
        self,
        *,
        canvas,
        vars_to_plot: List[str],
        axes: List["object"],
        overlay_buttons: Dict[str, Tuple["QPushButton", "QPushButton"]],
        overlay_axes_by_var: Dict[str, "object"],
    ) -> None:
        if canvas is None:
            return
        # Remove buttons for vars no longer plotted
        keep = set(vars_to_plot)
        for v in list(overlay_buttons.keys()):
            if v not in keep:
                try:
                    ylim_btn, yscale_btn = overlay_buttons.pop(v)
                    ylim_btn.hide()
                    yscale_btn.hide()
                    ylim_btn.deleteLater()
                    yscale_btn.deleteLater()
                except Exception:
                    pass
                overlay_axes_by_var.pop(v, None)

        overlay_axes_by_var.clear()
        overlay_axes_by_var.update({v: ax for v, ax in zip(vars_to_plot, axes)})

        for v in vars_to_plot:
            if v in overlay_buttons:
                self._refresh_overlay_button_labels(v)
                continue

            ylim_btn = QPushButton(canvas)
            yscale_btn = QPushButton(canvas)
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

            ylim_btn.clicked.connect(partial(self._on_overlay_ylim_clicked, v))
            yscale_btn.clicked.connect(partial(self._on_overlay_yscale_clicked, v))

            ylim_btn.show()
            yscale_btn.show()
            ylim_btn.raise_()
            yscale_btn.raise_()
            overlay_buttons[v] = (ylim_btn, yscale_btn)

    # ---------- Data loading ----------
    def _load_case(self, case_path: str) -> _LoadedCase:
        case_path = str(Path(case_path).expanduser().resolve())
        label = _format_case_label(case_path)
        case_dir = Path(case_path)

        # Decide 1D vs 2D cheaply (metadata-only) to avoid double-loading.
        # Add a fallback path so loading a 2D case while in "1D mode" (or vice versa)
        # still works and automatically switches the GUI after load.
        is_2d_probe = _probe_is_2d_case(case_dir)
        use_squash = _should_use_squash_for_load(case_dir)

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
            cs2 = self.Load.case_2D(
                case_path,
                gridfilepath=str(grid_path),
                verbose=False,
                use_squash=use_squash,
                force_squash=False,
            )
            try:
                _ensure_sdtools_2d_metadata(cs2.ds)
            except Exception:
                pass
            return cs2

        def _load_1d():
            # Newer xHermes versions may not provide ds.hermes.guard_replace_1d(),
            # which sdtools tries to call when guard_replace=True. Disable it here.
            return self.Load.case_1D(
                case_path,
                verbose=False,
                guard_replace=False,
                use_squash=use_squash,
                force_squash=False,
            )

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
                if ("toroidal" in msg and "grid" in msg) or ("topology" in msg and "grid" in msg) or ("provide grid in data directory" in msg):
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
        for slider, spin in (
            (getattr(self, "time_slider", None), getattr(self, "time_spin", None)),
            (getattr(self, "time_slider_2d", None), getattr(self, "time_spin_2d", None)),
        ):
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
            if spin is not None:
                try:
                    spin.blockSignals(True)
                    spin.setRange(0, max(0, n_t - 1))
                    spin.setValue(n_t - 1)
                finally:
                    try:
                        spin.blockSignals(False)
                    except Exception:
                        pass

        # Update ms spin ranges (if time coordinate available)
        tms = self._t_values_ms()
        for ms_spin in (getattr(self, "time_ms_spin", None), getattr(self, "time_ms_spin_2d", None)):
            if ms_spin is None:
                continue
            if tms is None:
                try:
                    ms_spin.setEnabled(False)
                except Exception:
                    pass
                continue
            try:
                ms_spin.setEnabled(True)
                ms_spin.blockSignals(True)
                ms_spin.setRange(float(np.nanmin(tms)), float(np.nanmax(tms)))
                ms_spin.setValue(float(tms[-1]))
            finally:
                try:
                    ms_spin.blockSignals(False)
                except Exception:
                    pass

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
            self.request_redraw()
            # New dataset -> invalidate cached time history and redraw (debounced)
            try:
                self._hist_cache.clear()
            except Exception:
                pass
            try:
                self._mon_cache.clear()
            except Exception:
                pass
            try:
                self._pol_cache.clear()
                self._rad_cache.clear()
            except Exception:
                pass
            try:
                self._pol_ylim_cache.clear()
                self._rad_ylim_cache.clear()
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
        self._set_time_readout_for_index(self._get_time_index())

    def _set_time_readout_for_index(self, ti: int) -> None:
        """
        Update the active time readout label for an arbitrary index (used for 2D slider dragging).
        """
        try:
            self._set_time_spin_for_index(int(ti))
        except Exception:
            pass
        try:
            self._set_time_ms_spin_for_index(int(ti))
        except Exception:
            pass
        # Prefer the active slider's range for a stable "idx/Max" display
        try:
            tmax = int(self._active_time_slider().maximum())
        except Exception:
            tmax = -1
        idx_txt = f"time index = {ti}" if tmax < 0 else f"time index = {ti}/{tmax}"

        tdim = self.state.get("time_dim")
        tvals = self.state.get("t_values")
        if tdim and tvals is not None and ti < len(tvals):
            try:
                self._active_time_readout().setText(f"{idx_txt}    ({tdim} = {tvals[ti] * 1e3:.4f} ms)")
                return
            except Exception:
                pass
        self._active_time_readout().setText(idx_txt)

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

    def _maybe_capture_view_2d(self, *, kind: str, state_key_no_ti: tuple) -> Optional[Dict[str, tuple]]:
        """
        Preserve toolbar zoom/pan across time-slider redraws.

        We only preserve view state when the redraw is "same plot config, different time index",
        i.e. when the last drawn key (with ti removed) matches the incoming key.
        """
        try:
            last = self._last_draw_state_2d.get(kind)
        except Exception:
            last = None
        if not last:
            return None
        try:
            # Both pol and rad state tuples are of the form:
            #   (kind, label, ti, ...rest...)
            # We strip ti generically so adding new fields doesn't break zoom-preserve.
            last_no_ti = (last[0], last[1], *last[3:])
            if kind == "pol":
                axes_map = self._overlay_axes_by_var_pol
                fig_axes = set(getattr(self.pol_figure, "axes", []))
            else:
                axes_map = self._overlay_axes_by_var_rad
                fig_axes = set(getattr(self.rad_figure, "axes", []))
        except Exception:
            return None
        if last_no_ti != state_key_no_ti:
            return None

        out: Dict[str, tuple] = {}
        for v, ax in list(axes_map.items()):
            try:
                if ax not in fig_axes:
                    continue
                out[v] = (ax.get_xlim(), ax.get_ylim())
            except Exception:
                continue
        return out or None

    def _restore_view_2d(self, *, views: Optional[Dict[str, tuple]], vars_to_plot: List[str], axes: List["object"]) -> None:
        if not views:
            return
        for v, ax in zip(vars_to_plot, axes):
            lims = views.get(v)
            if not lims:
                continue
            xlim, ylim = lims
            try:
                ax.set_xlim(xlim)
            except Exception:
                pass
            try:
                ax.set_ylim(ylim)
            except Exception:
                pass

    def _reset_toolbar_home(self, toolbar) -> None:
        """
        When we rebuild axes (figure.clear()), the Matplotlib navigation toolbar's
        stored view stack can point to old axes, making the Home button unreliable.

        Reset the stack and set "home" to the current default view.
        """
        if toolbar is None:
            return
        try:
            st = getattr(toolbar, "_nav_stack", None)
            if st is not None:
                try:
                    st.clear()
                except Exception:
                    # Fallback for older matplotlib Stack implementation
                    try:
                        st._elements.clear()  # type: ignore[attr-defined]
                        st._pos = 0  # type: ignore[attr-defined]
                    except Exception:
                        pass
            # Push current view as the new home
            try:
                toolbar.push_current()
            except Exception:
                pass
            try:
                toolbar.set_history_buttons()
            except Exception:
                pass
        except Exception:
            pass

    def _close_region2d_window(self, kind: str) -> None:
        st = self._region2d_pol if kind == "pol" else self._region2d_rad
        if not st:
            return
        try:
            win = st.get("win")
            if win is not None:
                win.close()
        except Exception:
            pass
        if kind == "pol":
            self._region2d_pol = None
        else:
            self._region2d_rad = None

    def _ensure_region2d_window(self, kind: str):
        if kind == "pol" and self._region2d_pol:
            return self._region2d_pol
        if kind == "rad" and self._region2d_rad:
            return self._region2d_rad

        def _on_close():
            # If user closes the window manually, untoggle the checkbox
            try:
                if kind == "pol" and hasattr(self, "pol_show_region2d_check"):
                    self.pol_show_region2d_check.blockSignals(True)
                    self.pol_show_region2d_check.setChecked(False)
                    self.pol_show_region2d_check.blockSignals(False)
                if kind == "rad" and hasattr(self, "rad_show_region2d_check"):
                    self.rad_show_region2d_check.blockSignals(True)
                    self.rad_show_region2d_check.setChecked(False)
                    self.rad_show_region2d_check.blockSignals(False)
            except Exception:
                pass
            if kind == "pol":
                self._region2d_pol = None
            else:
                self._region2d_rad = None

        title = "Show cut in 2D: Poloidal" if kind == "pol" else "Show cut in 2D: Radial"
        win = _RegionOverlayWindow(title, on_close=_on_close)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        fig = Figure(figsize=(7.5, 6.0))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, win)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        win.setCentralWidget(central)

        st = {"win": win, "fig": fig, "canvas": canvas, "toolbar": toolbar, "ax": None, "polys": None, "line": None, "arrow": None, "state": None}
        if kind == "pol":
            self._region2d_pol = st
        else:
            self._region2d_rad = st
        try:
            win.resize(900, 700)
        except Exception:
            pass
        try:
            win.show()
        except Exception:
            pass
        return st

    def _get_region2d_state(self):
        case = self._primary_case()
        if case is None:
            return None, None, None
        ds_t = self._ds_at_time(case)
        var = str(getattr(self, "poly_var_combo", None).currentText() or "").strip() if hasattr(self, "poly_var_combo") else ""
        if not var:
            var = "Te" if ("Te" in ds_t) else (next(iter(ds_t.data_vars.keys())) if getattr(ds_t, "data_vars", {}) else "")
        if not var or var not in ds_t:
            return case, ds_t, None
        logscale = bool(getattr(self, "poly_log_check", None).isChecked()) if hasattr(self, "poly_log_check") else False
        vmin = self._poly_vmin_active
        vmax = self._poly_vmax_active
        return case, ds_t, (var, logscale, vmin, vmax)

    def _update_region2d_overlay(self, kind: str) -> None:
        """
        Update the optional popout window showing the cut location over a 2D colormap.
        """
        st = self._region2d_pol if kind == "pol" else self._region2d_rad
        if not st:
            return
        case, ds_t, bg = self._get_region2d_state()
        fig = st.get("fig")
        canvas = st.get("canvas")
        if case is None or ds_t is None or bg is None or fig is None or canvas is None:
            return

        var, logscale, vmin, vmax = bg
        state_key = (case.label, var, bool(logscale), vmin, vmax)

        rebuild = (st.get("state") != state_key) or (st.get("ax") is None) or (st.get("polys") is None) or (st.get("line") is None)

        if rebuild:
            fig.clear()
            try:
                fig.set_facecolor("white")
            except Exception:
                pass
            ax = fig.add_subplot(1, 1, 1)
            st["ax"] = ax
            st["polys"] = None
            st["line"] = None
            st["arrow"] = None
            st["state"] = state_key
            try:
                data = ds_t[var]
                data.hermesm.clean_guards().bout.polygon(
                    ax=ax,
                    cmap="Spectral_r",
                    linecolor=(0, 0, 0, 0.10),
                    linewidth=0.0,
                    antialias=False,
                    logscale=bool(logscale),
                    vmin=vmin,
                    vmax=vmax,
                    separatrix=True,
                    separatrix_kwargs={"linewidth": 0.6, "color": "k"},
                    targets=False,
                    add_colorbar=False,
                )
                st["polys"] = ax.collections[-1] if ax.collections else None
            except Exception as e:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"2D background failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
            try:
                (ln,) = ax.plot([], [], color="red", linewidth=2.0, linestyle="--", alpha=0.7)
                ln.set_zorder(10)
                st["line"] = ln
            except Exception:
                st["line"] = None
            try:
                self._reset_toolbar_home(st.get("toolbar"))
            except Exception:
                pass
        else:
            # Fast update for time changes: update polygon colors only
            try:
                polys = st.get("polys")
                if polys is not None:
                    data = ds_t[var].hermesm.clean_guards()
                    polys.set_array(np.asarray(data.data).flatten())
            except Exception:
                pass

        ax = st.get("ax")
        ln = st.get("line")
        if ax is None or ln is None:
            canvas.draw_idle()
            return

        # Determine cut R,Z
        try:
            if kind == "pol":
                region = str(self.pol_region_combo.currentText() or "outer_lower")
                sepadd = int(self.pol_sepadd_spin.value())
                ti = self._get_time_index_for_case(case)
                ck = (case.label, ti, region, sepadd)
                df = self._pol_cache.get(ck)
                if df is None:
                    from hermes3.selectors import get_1d_poloidal_data  # type: ignore

                    df = get_1d_poloidal_data(ds_t, params=[], region=region, sepadd=sepadd, target_first=False)
                    self._pol_cache[ck] = df
            else:
                region = str(self.rad_region_combo.currentText() or "omp")
                ti = self._get_time_index_for_case(case)
                ck = (case.label, ti, region)
                df = self._rad_cache.get(ck)
                if df is None:
                    try:
                        import xarray as xr  # type: ignore
                        import hermes3.selectors as _sel  # type: ignore

                        setattr(_sel, "xr", xr)
                        get_1d_radial_data = _sel.get_1d_radial_data  # type: ignore[attr-defined]
                    except Exception:
                        from hermes3.selectors import get_1d_radial_data_old as get_1d_radial_data  # type: ignore
                    # Need R/Z in the output to draw the overlay. The selector does not
                    # include these unless explicitly requested.
                    df = get_1d_radial_data(ds_t, params=["R", "Z"], region=region, guards=False, sol=True, core=True)
                    self._rad_cache[ck] = df
            if df is None or ("R" not in df) or ("Z" not in df):
                raise KeyError("No R/Z columns in extracted cut data.")
            R0 = np.asarray(df["R"].values, dtype=float)
            Z0 = np.asarray(df["Z"].values, dtype=float)
            # Coordinate used to define "increasing direction" along the cut
            if kind == "pol":
                try:
                    use_spol = bool(self.pol_use_spol_check.isChecked())
                except Exception:
                    use_spol = False
                ccol = "Spol" if use_spol else "Spar"
            else:
                ccol = "Srad"
            c0 = np.asarray(df[ccol].values, dtype=float) if ccol in df else None

            m = np.isfinite(R0) & np.isfinite(Z0)
            if c0 is not None:
                m = m & np.isfinite(c0)
            R = R0[m]
            Z = Z0[m]
            cc = c0[m] if c0 is not None else None
            ln.set_data(R, Z)

            # Update direction arrow (at end of increasing coordinate)
            try:
                arr = st.get("arrow")
                if arr is not None:
                    arr.remove()
            except Exception:
                pass
            st["arrow"] = None
            if cc is not None and R.size >= 2:
                try:
                    order = np.argsort(cc)
                    end_i = int(order[-1])
                    prev_i = int(order[-2])
                    # ensure the arrow has non-zero length
                    k = 2
                    while k <= order.size and (R[end_i] == R[prev_i] and Z[end_i] == Z[prev_i]):
                        prev_i = int(order[-k])
                        k += 1
                    arrow = ax.annotate(
                        "",
                        xy=(float(R[end_i]), float(Z[end_i])),
                        xytext=(float(R[prev_i]), float(Z[prev_i])),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=getattr(ln, "get_color", lambda: "k")(),
                            lw=2.0,
                            alpha=0.7,
                        ),
                    )
                    try:
                        arrow.set_zorder(11)
                    except Exception:
                        pass
                    st["arrow"] = arrow
                except Exception:
                    pass
        except Exception as e:
            # No cut available -> clear line
            try:
                ln.set_data([], [])
            except Exception:
                pass
            try:
                arr = st.get("arrow")
                if arr is not None:
                    arr.remove()
            except Exception:
                pass
            st["arrow"] = None
            try:
                ax.text(0.02, 0.02, f"Cut overlay unavailable: {e}", transform=ax.transAxes, fontsize=9)
            except Exception:
                pass

        canvas.draw_idle()

    def _on_pol_show_region2d_toggled(self, checked: bool) -> None:
        if not checked:
            self._close_region2d_window("pol")
            return
        # Only valid in 2D mode
        if not getattr(self, "_mode_is_2d", False):
            try:
                self.pol_show_region2d_check.blockSignals(True)
                self.pol_show_region2d_check.setChecked(False)
                self.pol_show_region2d_check.blockSignals(False)
            except Exception:
                pass
            return
        self._ensure_region2d_window("pol")
        self._update_region2d_overlay("pol")

    def _on_rad_show_region2d_toggled(self, checked: bool) -> None:
        if not checked:
            self._close_region2d_window("rad")
            return
        if not getattr(self, "_mode_is_2d", False):
            try:
                self.rad_show_region2d_check.blockSignals(True)
                self.rad_show_region2d_check.setChecked(False)
                self.rad_show_region2d_check.blockSignals(False)
            except Exception:
                pass
            return
        self._ensure_region2d_window("rad")
        self._update_region2d_overlay("rad")

    def _maybe_capture_view_1d(self, *, state_key_no_ti: tuple) -> Optional[Dict[str, tuple]]:
        """
        Preserve toolbar zoom/pan across 1D time-slider redraws.

        Only capture when the last drawn key (with ti removed) matches the incoming key.
        """
        last = getattr(self, "_last_draw_state_1d_profiles", None)
        if not last:
            return None
        try:
            # ("prof", case_labels, ti, sdim, vars_to_plot, modes, guard_replace)
            last_no_ti = ("prof", last[1], last[3], last[4], last[5], last[6])
        except Exception:
            return None
        if last_no_ti != state_key_no_ti:
            return None

        out: Dict[str, tuple] = {}
        try:
            fig_axes = set(getattr(self.figure, "axes", []))
        except Exception:
            fig_axes = set()
        for v, ax in list(getattr(self, "_overlay_axes_by_var", {}).items()):
            try:
                if ax not in fig_axes:
                    continue
                out[v] = (ax.get_xlim(), ax.get_ylim())
            except Exception:
                continue
        return out or None

    def _ds_at_time_index(self, case: _LoadedCase, ti: int):
        """
        Like _ds_at_time(), but for an explicit time index (used for ylim computations).
        Ensures sdtools metadata/options survive slicing.
        """
        ds = case.ds
        tdim = self.state.get("time_dim")
        if tdim and tdim in getattr(ds, "dims", {}):
            try:
                ds_t = ds.isel({tdim: int(ti)})
            except Exception:
                ds_t = ds
        else:
            ds_t = ds
        # Reattach metadata/options so selectors keep working
        try:
            if hasattr(ds, "metadata") and not hasattr(ds_t, "metadata"):
                ds_t.metadata = ds.metadata  # type: ignore[attr-defined]
            elif hasattr(ds, "metadata") and hasattr(ds_t, "metadata"):
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
        try:
            _ensure_sdtools_2d_metadata(ds_t)
        except Exception:
            pass
        return ds_t

    def _with_margin(self, ymin: float, ymax: float) -> Tuple[float, float]:
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return ymin, ymax
        if ymax > ymin:
            m = 0.05 * (ymax - ymin)
            return ymin - m, ymax + m
        # Degenerate
        m = 0.1 * (abs(ymax) if ymax != 0 else 1.0)
        return ymin - m, ymax + m

    def _compute_ylim_poloidal_extracted(
        self,
        *,
        case: _LoadedCase,
        region: str,
        sepadd: int,
        varname: str,
        yscale: str,
        mode: str,
        get_1d_poloidal_data,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute y-limits for 2D poloidal extracted 1D profiles.
        Modes match the 1D GUI: auto/final/max (legacy: global==max).
        """
        key = ("pol", case.label, region, int(sepadd), varname, yscale, mode, int(case.n_time))
        hit = self._pol_ylim_cache.get(key)
        if hit is not None:
            return hit

        def _accum_from_df(df, cur_min, cur_max):
            try:
                y = np.asarray(df[varname].values, dtype=float)
            except Exception:
                return cur_min, cur_max
            if yscale == "log":
                y = y[y > 0]
            y = y[np.isfinite(y)]
            if y.size == 0:
                return cur_min, cur_max
            ym, yM = float(np.nanmin(y)), float(np.nanmax(y))
            if cur_min is None or ym < cur_min:
                cur_min = ym
            if cur_max is None or yM > cur_max:
                cur_max = yM
            return cur_min, cur_max

        ymin: Optional[float] = None
        ymax: Optional[float] = None

        if mode == "global":
            mode = "max"
        if mode == "final":
            tis = [max(0, int(case.n_time) - 1)]
        elif mode == "max":
            # User-requested: compute across all time slices
            tis = list(range(int(case.n_time)))
        else:
            return None, None

        for ti in tis:
            ck = (case.label, int(ti), region, int(sepadd))
            df = self._pol_cache.get(ck)
            if df is None:
                try:
                    ds_t = self._ds_at_time_index(case, int(ti))
                    df = get_1d_poloidal_data(ds_t, params=[varname], region=region, sepadd=int(sepadd), target_first=False)
                except Exception:
                    df = None
                self._pol_cache[ck] = df
            else:
                try:
                    missing = [varname] if varname not in df.columns else []
                except Exception:
                    missing = [varname]
                if missing:
                    try:
                        ds_t = self._ds_at_time_index(case, int(ti))
                        df_new = get_1d_poloidal_data(ds_t, params=missing, region=region, sepadd=int(sepadd), target_first=False)
                        if df_new is not None and varname in df_new:
                            df[varname] = df_new[varname].values
                        self._pol_cache[ck] = df
                    except Exception:
                        pass
            if df is None:
                continue
            ymin, ymax = _accum_from_df(df, ymin, ymax)

        if ymin is None or ymax is None:
            out = (None, None)
        else:
            a, b = self._with_margin(float(ymin), float(ymax))
            out = (a, b)
        self._pol_ylim_cache[key] = out
        return out

    def _compute_ylim_radial_extracted(
        self,
        *,
        case: _LoadedCase,
        region: str,
        varname: str,
        yscale: str,
        mode: str,
        get_1d_radial_data,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute y-limits for 2D radial extracted 1D profiles.
        Modes match the 1D GUI: auto/final/max (legacy: global==max).
        """
        key = ("rad", case.label, region, varname, yscale, mode, int(case.n_time))
        hit = self._rad_ylim_cache.get(key)
        if hit is not None:
            return hit

        def _accum_from_df(df, cur_min, cur_max):
            try:
                y = np.asarray(df[varname].values, dtype=float)
            except Exception:
                return cur_min, cur_max
            if yscale == "log":
                y = y[y > 0]
            y = y[np.isfinite(y)]
            if y.size == 0:
                return cur_min, cur_max
            ym, yM = float(np.nanmin(y)), float(np.nanmax(y))
            if cur_min is None or ym < cur_min:
                cur_min = ym
            if cur_max is None or yM > cur_max:
                cur_max = yM
            return cur_min, cur_max

        ymin: Optional[float] = None
        ymax: Optional[float] = None

        if mode == "global":
            mode = "max"
        if mode == "final":
            tis = [max(0, int(case.n_time) - 1)]
        elif mode == "max":
            tis = list(range(int(case.n_time)))
        else:
            return None, None

        for ti in tis:
            ck = (case.label, int(ti), region)
            df = self._rad_cache.get(ck)
            if df is None:
                try:
                    ds_t = self._ds_at_time_index(case, int(ti))
                    df = get_1d_radial_data(ds_t, params=[varname], region=region, guards=False, sol=True, core=True)
                except Exception:
                    df = None
                self._rad_cache[ck] = df
            else:
                try:
                    missing = [varname] if varname not in df.columns else []
                except Exception:
                    missing = [varname]
                if missing:
                    try:
                        ds_t = self._ds_at_time_index(case, int(ti))
                        df_new = get_1d_radial_data(ds_t, params=missing, region=region, guards=False, sol=True, core=True)
                        if df_new is not None and varname in df_new:
                            df[varname] = df_new[varname].values
                        self._rad_cache[ck] = df
                    except Exception:
                        pass
            if df is None:
                continue
            ymin, ymax = _accum_from_df(df, ymin, ymax)

        if ymin is None or ymax is None:
            out = (None, None)
        else:
            a, b = self._with_margin(float(ymin), float(ymax))
            out = (a, b)
        self._rad_ylim_cache[key] = out
        return out

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

        # If nothing relevant changed since last draw, don't rebuild the figure.
        _view_restore = None
        try:
            case = self._primary_case()
            ti = self._get_time_index_for_case(case) if case else -1
            region = str(self.pol_region_combo.currentText() or "outer_lower")
            sepadd = int(self.pol_sepadd_spin.value())
            use_spol = bool(getattr(self, "pol_use_spol_check", None).isChecked()) if hasattr(self, "pol_use_spol_check") else False
            xpt_mode = str(getattr(self, "pol_xpoint_combo", None).currentText() or "Bpxy valley") if hasattr(self, "pol_xpoint_combo") else "Bpxy valley"
            vars_to_plot = tuple(self.selected_vars)
            modes = tuple((v, self._yscale_by_var.get(v, "linear"), self._ylim_mode_by_var.get(v, "auto")) for v in vars_to_plot)
            state_key = ("pol", getattr(case, "label", None), ti, region, sepadd, use_spol, xpt_mode, vars_to_plot, modes)
            if self._last_draw_state_2d.get("pol") == state_key and self.pol_figure.axes:
                try:
                    self._position_overlay_buttons_pol()
                except Exception:
                    pass
                self.pol_canvas.draw_idle()
                return
            # Preserve zoom/pan across time changes (same config, different ti)
            state_no_ti = ("pol", getattr(case, "label", None), region, sepadd, use_spol, xpt_mode, vars_to_plot, modes)
            _view_restore = self._maybe_capture_view_2d(kind="pol", state_key_no_ti=state_no_ti)
            self._last_draw_state_2d["pol"] = state_key
        except Exception:
            pass

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
        use_spol = bool(getattr(self, "pol_use_spol_check", None).isChecked()) if hasattr(self, "pol_use_spol_check") else False
        xpt_mode = str(getattr(self, "pol_xpoint_combo", None).currentText() or "Bpxy valley") if hasattr(self, "pol_xpoint_combo") else "Bpxy valley"
        xcol = "Spol" if use_spol else "Spar"
        xlab = (r"S$_{pol}$ (m)" if use_spol else r"S$_\parallel$ (m)")

        # Layout similar to 1D profiles
        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))
        # Extra vertical padding to avoid title/xlabel overlap between stacked subplots
        gs = self.pol_figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.60, wspace=0.30)
        axes = [self.pol_figure.add_subplot(gs[i % nrows, i // nrows]) for i in range(n)]

        try:
            from hermes3.selectors import get_1d_poloidal_data  # type: ignore
        except Exception as e:
            ax = self.pol_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"sdtools import error:\n{e}", ha="center", va="center", transform=ax.transAxes)
            self.pol_canvas.draw_idle()
            return

        # Reference: mark the X-point location along the field line.
        xline_rmin = None
        try:
            c0 = self._primary_case()
            if c0 is not None:
                ds0_t = self._ds_at_time(c0)
                ti0 = self._get_time_index_for_case(c0)
                ck0 = (c0.label, ti0, region, sepadd)
                df0 = self._pol_cache.get(ck0)
                if df0 is None:
                    try:
                        # params=[] still returns geometry columns like Spar/R
                        df0 = get_1d_poloidal_data(ds0_t, params=[], region=region, sepadd=sepadd, target_first=False)
                    except Exception:
                        df0 = None
                    self._pol_cache[ck0] = df0
                # Ensure Bpxy/Bxy exist in the cached df (incremental add)
                if df0 is not None:
                    try:
                        need_bpxy = "Bpxy" not in df0.columns
                    except Exception:
                        need_bpxy = True
                    if need_bpxy:
                        try:
                            df_new = get_1d_poloidal_data(ds0_t, params=["Bpxy"], region=region, sepadd=sepadd, target_first=False)
                            if df_new is not None and "Bpxy" in df_new:
                                try:
                                    df0["Bpxy"] = df_new["Bpxy"].values
                                except Exception:
                                    pass
                            self._pol_cache[ck0] = df0
                        except Exception:
                            pass
                    try:
                        need_bxy = "Bxy" not in df0.columns
                    except Exception:
                        need_bxy = True
                    if need_bxy:
                        try:
                            df_new = get_1d_poloidal_data(ds0_t, params=["Bxy"], region=region, sepadd=sepadd, target_first=False)
                            if df_new is not None and "Bxy" in df_new:
                                try:
                                    df0["Bxy"] = df_new["Bxy"].values
                                except Exception:
                                    pass
                            self._pol_cache[ck0] = df0
                        except Exception:
                            pass
                # Need geometry coordinate for the chosen x-axis.
                if df0 is not None and "R" in df0 and ("Spar" in df0 or "Spol" in df0):
                    r = np.asarray(df0["R"].values)
                    s = np.asarray(df0.get(xcol, df0.get("Spar")).values)  # type: ignore[union-attr]
                    if r.size and s.size:
                        rr = np.asarray(r, dtype=float)
                        ss = np.asarray(s, dtype=float)
                        m = np.isfinite(rr) & np.isfinite(ss)
                        if np.any(m):
                            s_use = ss[m]
                            if xpt_mode.strip().lower().startswith("min r"):
                                try:
                                    rr_use = rr[m]
                                    j0 = int(np.nanargmin(rr_use))
                                    if 0 <= j0 < s_use.size:
                                        xline_rmin = float(s_use[j0])
                                except Exception:
                                    pass
                            elif xpt_mode.strip().lower().startswith("bxy"):
                                # Outer: max(Bxy), Inner: min(Bxy)
                                try:
                                    bxy = np.asarray(df0["Bxy"].values, dtype=float)
                                    bxy_use = bxy[m]
                                    mb = np.isfinite(bxy_use)
                                    if np.any(mb):
                                        b2 = np.asarray(bxy_use[mb], dtype=float)
                                        s2 = np.asarray(s_use[mb], dtype=float)
                                        # Avoid target-side extrema: search only the middle 75%
                                        n2 = int(b2.size)
                                        lo = int(np.floor(0.125 * n2))
                                        hi = int(np.ceil(0.875 * n2))
                                        if hi <= lo:
                                            lo, hi = 0, n2
                                        b_mid = b2[lo:hi]
                                        s_mid = s2[lo:hi]
                                        if b_mid.size:
                                            if str(region).startswith("outer"):
                                                jmid = int(np.nanargmax(b_mid))
                                            else:
                                                jmid = int(np.nanargmin(b_mid))
                                            jmid = int(np.clip(jmid, 0, b_mid.size - 1))
                                            xline_rmin = float(s_mid[jmid])
                                except Exception:
                                    pass
                            else:
                                # Default: Bpxy "valley" minimum (until it starts rising again)
                                try:
                                    bp = np.asarray(df0["Bpxy"].values, dtype=float)
                                    bp_use = bp[m]
                                    mbp = np.isfinite(bp_use)
                                    if np.any(mbp):
                                        bp2 = np.asarray(bp_use[mbp], dtype=float)
                                        s2 = np.asarray(s_use[mbp], dtype=float)
                                        j = _xpoint_idx_bpxy_valley(bp2)
                                        if j is not None and 0 <= int(j) < s2.size:
                                            xline_rmin = float(s2[int(j)])
                                except Exception:
                                    # Fallback: closest-to-zero R
                                    try:
                                        rr_use = rr[m]
                                        j0 = int(np.nanargmin(np.abs(rr_use)))
                                        if 0 <= j0 < s_use.size:
                                            xline_rmin = float(s_use[j0])
                                    except Exception:
                                        pass
        except Exception:
            xline_rmin = None

        # Batch extract once per case/time/region/sepadd for speed
        # Configure y-scales per variable before plotting
        for ax, name in zip(axes, vars_to_plot):
            mode = self._yscale_by_var.get(name, "linear")
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    ax.set_yscale("symlog", linthresh=1e-6)
                else:
                    ax.set_yscale("linear")
            except Exception:
                pass

        for c in self.cases.values():
            ds_t = self._ds_at_time(c)
            ti = self._get_time_index_for_case(c)
            # Incremental cache: keep one df per (case,time,region,sepadd) and extend
            ck = (c.label, ti, region, sepadd)
            df = self._pol_cache.get(ck)
            params = _selector_params_only(list(vars_to_plot))
            if df is None:
                try:
                    df = get_1d_poloidal_data(ds_t, params=list(params), region=region, sepadd=sepadd, target_first=False)
                except Exception as e:
                    self.set_status(f"Poloidal extract failed: {e}", is_error=True)
                    df = None
                self._pol_cache[ck] = df
            else:
                try:
                    missing = [v for v in params if v not in df.columns]
                except Exception:
                    missing = list(params)
                if missing:
                    try:
                        df_new = get_1d_poloidal_data(ds_t, params=list(missing), region=region, sepadd=sepadd, target_first=False)
                        # Merge missing columns by index (same length/order expected)
                        for v in list(missing):
                            if v in df_new:
                                try:
                                    df[v] = df_new[v].values
                                except Exception:
                                    pass
                        self._pol_cache[ck] = df
                    except Exception as e:
                        self.set_status(f"Poloidal extract failed: {e}", is_error=True)
            if df is None:
                continue

            try:
                x = np.asarray(df.get(xcol, df.get("Spar")).values)  # type: ignore[union-attr]
            except Exception:
                x = np.asarray(df["Spar"].values)
            for ax, name in zip(axes, vars_to_plot):
                ax.set_title(f"{name} ({region}, sepadd={sepadd})", fontsize=10)
                ax.grid(True, alpha=0.3)
                try:
                    y = np.asarray(df[name].values)
                except Exception:
                    continue
                # If log scale, mask non-positive values
                if self._yscale_by_var.get(name, "linear") == "log":
                    y = np.where(y > 0, y, np.nan)
                ax.plot(x, y, label=c.label)

        for ax, name in zip(axes, vars_to_plot):
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
            ax.set_ylabel(f"{units}" if units else "")
            if len(self.cases) > 1:
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(xlab)
            if xline_rmin is not None and np.isfinite(xline_rmin):
                try:
                    ax.axvline(xline_rmin, color="k", linewidth=1.0, linestyle="--")
                    ax.text(
                        xline_rmin,
                        0.98,
                        "X-point",
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                        va="top",
                        ha="left",
                        fontsize=8,
                        color="k",
                    )
                except Exception:
                    pass

            # Apply ylim mode (match 1D behavior: auto/final/global)
            try:
                ylim_mode = self._ylim_mode_by_var.get(name, "auto")
                if ylim_mode == "global":
                    ylim_mode = "max"
                if ylim_mode == "auto":
                    ax.relim()
                    ax.autoscale_view()
                else:
                    c0 = self._primary_case()
                    if c0 is not None:
                        ymin, ymax = self._compute_ylim_poloidal_extracted(
                            case=c0,
                            region=region,
                            sepadd=sepadd,
                            varname=name,
                            yscale=self._yscale_by_var.get(name, "linear"),
                            mode=ylim_mode,
                            get_1d_poloidal_data=get_1d_poloidal_data,
                        )
                        if ymin is not None and ymax is not None:
                            ax.set_ylim(ymin, ymax)
            except Exception:
                pass

        # Overlay buttons for yscale/ylim (same as 1D)
        self._sync_overlay_buttons_pol(vars_to_plot=vars_to_plot, axes=axes)
        # Ensure toolbar Home resets to the default view (not the preserved zoom)
        try:
            self._reset_toolbar_home(getattr(self, "pol_toolbar", None))
        except Exception:
            pass
        # Restore user zoom/pan (if applicable)
        try:
            self._restore_view_2d(views=_view_restore, vars_to_plot=vars_to_plot, axes=axes)
        except Exception:
            pass
        try:
            self.pol_figure.tight_layout()
        except Exception:
            pass
        self.pol_canvas.draw_idle()

    def _redraw_2d_radial(self) -> None:
        self._update_time_readout()

        # If nothing relevant changed since last draw, don't rebuild the figure.
        _view_restore = None
        try:
            case = self._primary_case()
            ti = self._get_time_index_for_case(case) if case else -1
            region = str(self.rad_region_combo.currentText() or "omp")
            vars_to_plot = tuple(self.selected_vars)
            modes = tuple((v, self._yscale_by_var.get(v, "linear"), self._ylim_mode_by_var.get(v, "auto")) for v in vars_to_plot)
            state_key = ("rad", getattr(case, "label", None), ti, region, vars_to_plot, modes)
            if self._last_draw_state_2d.get("rad") == state_key and self.rad_figure.axes:
                try:
                    self._position_overlay_buttons_rad()
                except Exception:
                    pass
                self.rad_canvas.draw_idle()
                return
            # Preserve zoom/pan across time changes (same config, different ti)
            state_no_ti = ("rad", getattr(case, "label", None), region, vars_to_plot, modes)
            _view_restore = self._maybe_capture_view_2d(kind="rad", state_key_no_ti=state_no_ti)
            self._last_draw_state_2d["rad"] = state_key
        except Exception:
            pass

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
        # Extra vertical padding to avoid title/xlabel overlap between stacked subplots
        gs = self.rad_figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.60, wspace=0.30)
        axes = [self.rad_figure.add_subplot(gs[i % nrows, i // nrows]) for i in range(n)]

        try:
            # Prefer the vectorized sdtools implementation (much faster for omp/imp),
            # but it expects an `xr` alias in the module. Provide it here.
            import xarray as xr  # type: ignore
            import hermes3.selectors as _sel  # type: ignore

            setattr(_sel, "xr", xr)
            get_1d_radial_data = _sel.get_1d_radial_data  # type: ignore[attr-defined]
        except Exception as e:
            # Fallback: older loop-based implementation (slower for omp/imp but robust).
            try:
                from hermes3.selectors import get_1d_radial_data_old as get_1d_radial_data  # type: ignore
            except Exception as e2:
                ax = self.rad_figure.add_subplot(1, 1, 1)
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"sdtools import error:\n{e}\n{e2}", ha="center", va="center", transform=ax.transAxes)
                self.rad_canvas.draw_idle()
                return

        # Batch extract once per case/time/region for speed
        # Configure y-scales per variable before plotting
        for ax, name in zip(axes, vars_to_plot):
            mode = self._yscale_by_var.get(name, "linear")
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    ax.set_yscale("symlog", linthresh=1e-6)
                else:
                    ax.set_yscale("linear")
            except Exception:
                pass

        for c in self.cases.values():
            ds_t = self._ds_at_time(c)
            ti = self._get_time_index_for_case(c)
            # Incremental cache: keep one df per (case,time,region) and extend
            ck = (c.label, ti, region)
            df = self._rad_cache.get(ck)
            params = _selector_params_only(list(vars_to_plot))
            if df is None:
                try:
                    df = get_1d_radial_data(ds_t, params=list(params), region=region, guards=False, sol=True, core=True)
                except Exception as e:
                    self.set_status(f"Radial extract failed: {e}", is_error=True)
                    df = None
                self._rad_cache[ck] = df
            else:
                try:
                    missing = [v for v in params if v not in df.columns]
                except Exception:
                    missing = list(params)
                if missing:
                    try:
                        df_new = get_1d_radial_data(ds_t, params=list(missing), region=region, guards=False, sol=True, core=True)
                        for v in list(missing):
                            if v in df_new:
                                try:
                                    df[v] = df_new[v].values
                                except Exception:
                                    pass
                        self._rad_cache[ck] = df
                    except Exception as e:
                        self.set_status(f"Radial extract failed: {e}", is_error=True)
            if df is None:
                continue

            # Plot radial coordinate in mm
            x = np.asarray(df["Srad"].values) * 1e3
            for ax, name in zip(axes, vars_to_plot):
                ax.set_title(f"{name} ({region})", fontsize=10)
                ax.grid(True, alpha=0.3)
                try:
                    y = np.asarray(df[name].values)
                except Exception:
                    continue
                if self._yscale_by_var.get(name, "linear") == "log":
                    y = np.where(y > 0, y, np.nan)
                ax.plot(x, y, label=c.label)

        for ax, name in zip(axes, vars_to_plot):
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
            ax.set_ylabel(f"{units}" if units else "")
            if len(self.cases) > 1:
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(r"$r^\prime - r_{sep}$ (mm)")
            # Separatrix marker
            try:
                ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
                ax.text(
                    0.0,
                    0.98,
                    "separatrix",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="k",
                )
            except Exception:
                pass

            try:
                ylim_mode = self._ylim_mode_by_var.get(name, "auto")
                if ylim_mode == "global":
                    ylim_mode = "max"
                if ylim_mode == "auto":
                    ax.relim()
                    ax.autoscale_view()
                else:
                    c0 = self._primary_case()
                    if c0 is not None:
                        ymin, ymax = self._compute_ylim_radial_extracted(
                            case=c0,
                            region=region,
                            varname=name,
                            yscale=self._yscale_by_var.get(name, "linear"),
                            mode=ylim_mode,
                            get_1d_radial_data=get_1d_radial_data,
                        )
                        if ymin is not None and ymax is not None:
                            ax.set_ylim(ymin, ymax)
            except Exception:
                pass

        self._sync_overlay_buttons_rad(vars_to_plot=vars_to_plot, axes=axes)
        # Ensure toolbar Home resets to the default view (not the preserved zoom)
        try:
            self._reset_toolbar_home(getattr(self, "rad_toolbar", None))
        except Exception:
            pass
        # Restore user zoom/pan (if applicable)
        try:
            self._restore_view_2d(views=_view_restore, vars_to_plot=vars_to_plot, axes=axes)
        except Exception:
            pass
        try:
            self.rad_figure.tight_layout()
        except Exception:
            pass
        self.rad_canvas.draw_idle()

    def _redraw_2d_polygon(self) -> None:
        self._update_time_readout()

        case = self._primary_case()
        if case is None:
            self.poly_figure.clear()
            ax = self.poly_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            self.poly_canvas.draw_idle()
            return

        var = str(self.poly_var_combo.currentText() or "").strip()
        if not var:
            self.poly_figure.clear()
            ax = self.poly_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Select a variable.", ha="center", va="center", transform=ax.transAxes)
            self.poly_canvas.draw_idle()
            return

        ds_t = self._ds_at_time(case)
        grid_only = bool(self.poly_grid_only_check.isChecked())
        logscale = bool(self.poly_log_check.isChecked())
        vmin = self._poly_vmin_active
        vmax = self._poly_vmax_active

        # Reuse the PatchCollection when only time index changes (fast).
        state = (case.label, var, grid_only, logscale, vmin, vmax)
        if self._poly_plot_state == state and self._poly_ax is not None and self._poly_polys is not None:
            try:
                data = ds_t[var].hermesm.clean_guards()
                self._poly_polys.set_array(np.asarray(data.data).flatten())
                if self._poly_cbar is not None:
                    try:
                        self._poly_cbar.update_normal(self._poly_polys)
                    except Exception:
                        pass
                self.poly_canvas.draw_idle()
                return
            except Exception:
                # fall through to rebuild
                pass

        # Otherwise rebuild the plot (variable/settings changed)
        self.poly_figure.clear()
        try:
            self.poly_figure.set_facecolor("white")
        except Exception:
            pass
        ax = self.poly_figure.add_subplot(1, 1, 1)
        try:
            data = ds_t[var]
            # Clean guards for nicer visuals and use xbout polygon plotting
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
                ax.set_title("Computational grid", fontsize=11)
            else:
                data.hermesm.clean_guards().bout.polygon(
                    ax=ax,
                    cmap="Spectral_r",
                   
                    linecolor=(0, 0, 0, 0.15),
                    linewidth=0,
                    antialias=True,
                    logscale=logscale,
                    vmin=vmin,
                    vmax=vmax,
                    separatrix=True,
                    separatrix_kwargs={"linewidth": 0.2, "color": "k"},
                    targets=False,
                    add_colorbar=False,
                )
                ax.set_title(f"{var}", fontsize=11)
                # Create and keep our own colorbar so we can update it efficiently
                try:
                    polys = ax.collections[-1] if ax.collections else None
                    if polys is not None:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        label = ""
                        try:
                            label = var
                            if "units" in data.attrs:
                                label = f"{label} [{data.attrs['units']}]"
                        except Exception:
                            pass
                        self._poly_cbar = self.poly_figure.colorbar(polys, cax=cax, label=label)
                        try:
                            cax.grid(which="both", visible=False)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"2D plot failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
        # Save handles for fast time updates
        try:
            self._poly_plot_state = state
            self._poly_ax = ax
            self._poly_polys = ax.collections[-1] if ax.collections else None
        except Exception:
            self._poly_plot_state = None
            self._poly_ax = None
            self._poly_polys = None
            self._poly_cbar = None
        self.poly_canvas.draw_idle()

    def _redraw_2d_monitor(self) -> None:
        # If nothing relevant changed since last draw, don't rebuild the figure.
        try:
            case = self._primary_case()
            sepadd = int(self.pol_sepadd_spin.value()) if hasattr(self, "pol_sepadd_spin") else 0
            region = str(self.pol_region_combo.currentText() or "outer_lower") if hasattr(self, "pol_region_combo") else "outer_lower"
            vars_to_plot = tuple(self.selected_vars)
            state_key = ("mon", getattr(case, "label", None), sepadd, region, vars_to_plot)
            if self._last_draw_state_2d.get("mon") == state_key and self.mon_figure.axes:
                self.mon_canvas.draw_idle()
                return
            self._last_draw_state_2d["mon"] = state_key
        except Exception:
            pass

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

        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            ax = self.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
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

        region = str(self.pol_region_combo.currentText() or "outer_lower") if hasattr(self, "pol_region_combo") else "outer_lower"
        sepadd = int(self.pol_sepadd_spin.value()) if hasattr(self, "pol_sepadd_spin") else 0
        ck = (case.label, region, sepadd, tuple(vars_to_plot))
        cached = self._mon_cache.get(ck)
        if cached is None:
            omp = np.full((int(case.n_time), len(vars_to_plot)), np.nan, dtype=float)
            targ = np.full((int(case.n_time), len(vars_to_plot)), np.nan, dtype=float)

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
                    params = _selector_params_only(list(vars_to_plot))
                    df = get_1d_poloidal_data(dsi, params=list(params), region=region, sepadd=sepadd, target_first=False)
                    for j, name in enumerate(vars_to_plot):
                        try:
                            y = np.asarray(df[name].values, dtype=float)
                        except Exception:
                            continue
                        if y.size:
                            omp[i, j] = float(y[0])
                            if y.size >= 2:
                                targ[i, j] = float(y[-2])
                except Exception:
                    continue

            self._mon_cache[ck] = (t_ms, omp, targ, tuple(vars_to_plot))
            cached = self._mon_cache[ck]

        t_ms, omp, targ, vorder = cached
        vorder = list(vorder)

        n = len(vorder)
        # Arrange as 2 rows x N columns:
        #   top row   -> OMP
        #   bottom row-> target
        gs = self.mon_figure.add_gridspec(nrows=2, ncols=max(1, n), hspace=0.45, wspace=0.35)
        ax_omp = []
        ax_targ = []
        for i in range(max(1, n)):
            if i == 0:
                a0 = self.mon_figure.add_subplot(gs[0, i])
                a1 = self.mon_figure.add_subplot(gs[1, i], sharex=a0)
            else:
                a0 = self.mon_figure.add_subplot(gs[0, i], sharex=ax_omp[0])
                a1 = self.mon_figure.add_subplot(gs[1, i], sharex=ax_omp[0])
            ax_omp.append(a0)
            ax_targ.append(a1)

        for i, name in enumerate(vorder):
            a0 = ax_omp[i]
            a1 = ax_targ[i]
            a0.set_title(f"{name} (OMP)", fontsize=10)
            a1.set_title(f"{name} (target)", fontsize=10)
            for ax in (a0, a1):
                ax.grid(True, alpha=0.3)

            y0 = np.asarray(omp[:, i], dtype=float)
            y1 = np.asarray(targ[:, i], dtype=float)
            if self._yscale_by_var.get(name, "linear") == "log":
                y0 = np.where(y0 > 0, y0, np.nan)
                y1 = np.where(y1 > 0, y1, np.nan)
            a0.plot(t_ms[: len(y0)], y0, lw=1.2)
            a1.plot(t_ms[: len(y1)], y1, lw=1.2)

            # Apply y-scale mode (re-using the same per-variable setting as other tabs)
            mode = self._yscale_by_var.get(name, "linear")
            for ax in (a0, a1):
                try:
                    if mode == "log":
                        ax.set_yscale("log")
                    elif mode == "symlog":
                        ax.set_yscale("symlog", linthresh=1e-6)
                    else:
                        ax.set_yscale("linear")
                except Exception:
                    pass

            # Units (if known)
            units = None
            try:
                if name in ds:
                    units = ds[name].attrs.get("units", None)
            except Exception:
                units = None
            yl = f"{units}" if units else ""
            a0.set_ylabel(yl)
            a1.set_ylabel(yl)

        # X labels only on bottom row
        for ax in ax_targ:
            ax.set_xlabel("Time (ms)")

        # No suptitle (save vertical space)
        try:
            self.mon_figure.tight_layout()
        except Exception:
            pass
        self.mon_canvas.draw_idle()

    def redraw(self) -> None:
        if self._mode_is_2d:
            self._redraw_2d_current_tab()
            # Also keep optional region-overlay popouts in sync
            try:
                if self._region2d_pol:
                    self._update_region2d_overlay("pol")
            except Exception:
                pass
            try:
                if self._region2d_rad:
                    self._update_region2d_overlay("rad")
            except Exception:
                pass
            return

        self._update_time_readout()

        # Preserve toolbar zoom/pan across time changes (same config, different ti)
        _view_restore_1d = None
        try:
            sdim0 = self.state.get("spatial_dim")
            vars0 = tuple(self.selected_vars)
            modes0 = tuple((v, self._yscale_by_var.get(v, "linear"), self._ylim_mode_by_var.get(v, "auto")) for v in vars0)
            guard0 = bool(self._guard_replace_enabled())
            case_labels0 = tuple(self.cases.keys())
            ti0 = int(self._get_time_index())
            state_key = ("prof", case_labels0, ti0, sdim0, vars0, modes0, guard0)
            state_no_ti = ("prof", case_labels0, sdim0, vars0, modes0, guard0)
            _view_restore_1d = self._maybe_capture_view_1d(state_key_no_ti=state_no_ti)
            self._last_draw_state_1d_profiles = state_key
        except Exception:
            _view_restore_1d = None

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
                    # Downsample long traces for responsiveness
                    try:
                        x, y = _downsample_xy(x, y, int(self._profile_max_points))
                    except Exception:
                        pass
                    if mode == "log":
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(x, y, label=c.label)
                except Exception as e:
                    self.set_status(f"Plot error for {name}: {e}", is_error=True)

            # X-point marker from options (1D only)
            try:
                ds0 = next(iter(self.cases.values())).ds
                xpt = _get_option_float(ds0, ["length_xpt", "mesh:length_xpt"])
                if xpt is not None and np.isfinite(xpt):
                    ax.axvline(xpt, color="k", linewidth=1.0, alpha=0.7, linestyle="--")
                    ax.text(
                        xpt,
                        0.98,
                        "X-point",
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                        va="top",
                        ha="left",
                        fontsize=8,
                        color="k",
                        alpha=0.7,
                    )
            except Exception:
                pass

            ax.set_title(name, fontsize=10)
            ax.set_ylabel(f"{units}" if units else "")
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
                elif ylim_mode == "global" or ylim_mode == "max":
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
        # Ensure toolbar Home resets to the default view (not the preserved zoom)
        try:
            self._reset_toolbar_home(getattr(self, "toolbar", None))
        except Exception:
            pass
        # Restore user zoom/pan if applicable
        try:
            self._restore_view_2d(views=_view_restore_1d, vars_to_plot=vars_to_plot, axes=axes)
        except Exception:
            pass
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

