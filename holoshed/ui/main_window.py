"""Main Qt window for holo-shed."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("QtAgg", force=True)
from matplotlib import rcParams  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from holoshed.dataset_utils import (
    downsample_xy,
    ensure_sdtools_2d_metadata,
    format_case_label,
    get_option_float,
    guard_replace_1d_profile_xy,
    infer_spatial_dim,
    infer_time_dim,
    list_plottable_vars,
    list_plottable_vars_2d,
    parse_optional_float,
    selector_params_only,
    xpoint_idx_bpxy_valley,
)
from holoshed.models import LoadedCase
from holoshed.paths import ensure_sdtools_on_path
from holoshed.plotting.coordinator import PlotCoordinator
from holoshed.ui.qt import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QEvent,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPoint,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    qt_checked,
    qt_unchecked,
)
from holoshed.ui.mixins.overlay_buttons import OverlayButtonsMixin
from holoshed.ui.mixins.region2d import Region2dMixin
from holoshed.ui.mixins.subplot_menus import SubplotMenusMixin
from holoshed.ui.mixins.variable_list import VariableListMixin
from holoshed.ui.mixins.layout import LayoutMixin
from holoshed.ui.mixins.case_sliders import CaseSlidersMixin
from holoshed.ui.mixins.time_controls import TimeControlsMixin
from holoshed.ui.mixins.data_loading import DataLoadingMixin

# Backward-compat aliases used in plotting modules
_qt_checked = qt_checked
_qt_unchecked = qt_unchecked

class MainWindow(
    QMainWindow,
    LayoutMixin,
    CaseSlidersMixin,
    TimeControlsMixin,
    DataLoadingMixin,
    OverlayButtonsMixin,
    Region2dMixin,
    VariableListMixin,
    SubplotMenusMixin,
):

    def __init__(self, *, initial_case_path: Optional[str], spatial_dim: Optional[str]):
        super().__init__()

        ensure_sdtools_on_path()
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

        self.cases: Dict[str, LoadedCase] = {}
        self._case_time_indices: Dict[str, int] = {}  # per-case time indices (1D)
        self._case_sliders: Dict[str, "QSlider"] = {}  # per-case slider widgets (1D)
        self._case_spinboxes: Dict[str, "QSpinBox"] = {}  # per-case index spinbox widgets (1D)
        self._case_ms_spinboxes: Dict[str, "QDoubleSpinBox"] = {}  # per-case time (ms) spinbox widgets (1D)
        self._case_slider_widgets: Dict[str, "QWidget"] = {}  # per-case slider row widgets (1D)
        # 2D per-case time slider tracking
        self._case_time_indices_2d: Dict[str, int] = {}  # per-case time indices (2D)
        self._case_sliders_2d: Dict[str, "QSlider"] = {}  # per-case slider widgets (2D)
        self._case_spinboxes_2d: Dict[str, "QSpinBox"] = {}  # per-case index spinbox widgets (2D)
        self._case_ms_spinboxes_2d: Dict[str, "QDoubleSpinBox"] = {}  # per-case time (ms) spinbox widgets (2D)
        self._case_slider_widgets_2d: Dict[str, "QWidget"] = {}  # per-case slider row widgets (2D)
        self._normalize_time_enabled: bool = False  # time normalization state
        self.spatial_dim_forced = spatial_dim
        self.state = dict(spatial_dim=None, time_dim=None, vars=[], t_values=None)
        self._mode_is_2d = False

        self.selected_vars: List[str] = []  # preserve selection order
        self._selected_set: set[str] = set()
        self._yscale_by_var: Dict[str, str] = {}  # var -> {"linear","log","symlog"}
        self._ylim_mode_by_var: Dict[str, str] = {}  # var -> {"auto","final","max"} (legacy: "global" == "max")
        self._xscale_by_var: Dict[str, str] = {}  # var -> {"linear","log"}
        self._overlay_vars: Dict[str, List[str]] = {}  # primary_var -> [overlaid_vars]
        self._var_filter: str = ""
        # Time history specific state
        self._hist_yscale_by_var: Dict[str, str] = {}  # var -> {"linear","log","symlog"} for time history
        self._hist_overlay_vars: Dict[str, List[str]] = {}  # primary_var -> [overlaid_vars] for time history
        self._hist_axes_by_var: Dict[str, tuple] = {}  # var -> (ax_upstream, ax_target)

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
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.installEventFilter(self)
        # 2D canvases (created in _build_ui) also use overlay buttons.
        # These may not exist in early init, so guard with getattr.
        try:
            self.pol_canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons_pol())
            self.pol_canvas.mpl_connect("button_press_event", self._on_pol_canvas_button_press)
            self.pol_canvas.installEventFilter(self)
        except Exception:
            pass
        try:
            self.rad_canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons_rad())
            self.rad_canvas.mpl_connect("button_press_event", self._on_rad_canvas_button_press)
            self.rad_canvas.installEventFilter(self)
        except Exception:
            pass
        # Time history canvas context menu
        try:
            self.hist_canvas.mpl_connect("button_press_event", self._on_hist_canvas_button_press)
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
        # Cached y-limits for 1D profile plots
        self._1d_ylim_cache: Dict[tuple, Tuple[Optional[float], Optional[float]]] = {}
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
        # Multi-case 2D polygon tracking (for side-by-side comparison)
        self._poly_axes_multi: Optional[List] = None
        self._poly_polys_multi: Optional[List] = None
        self._poly_cbars_multi: Optional[List] = None
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
            self._update_case_sliders()
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
        self._update_case_sliders()







































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



    def _apply_poly_clim(self) -> None:
        """
        Apply 2D field colorbar limits from the text boxes.
        """
        self._poly_vmin_active = parse_optional_float(self.poly_vmin_edit.text())
        self._poly_vmax_active = parse_optional_float(self.poly_vmax_edit.text())
        self.request_redraw()





















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



    def _compute_ylim_for_final(self, varname: str, tdim: Optional[str], yscale: str) -> Tuple[Optional[float], Optional[float]]:
        # Include overlay variables in the y-limit calculation
        overlay_vars = tuple(self._overlay_vars.get(varname, []))
        all_vars = [varname] + list(overlay_vars)
        # Cache key includes: mode, varname, overlays, yscale, and case n_times
        case_info = tuple((c.label, c.n_time) for c in self.cases.values())
        cache_key = ("final", varname, overlay_vars, yscale, case_info)
        hit = self._1d_ylim_cache.get(cache_key)
        if hit is not None:
            return hit

        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            for vname in all_vars:
                if vname not in ds:
                    continue
                da = ds[vname]
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
            self._1d_ylim_cache[cache_key] = (None, None)
            return None, None
        ys = np.concatenate(ys_all)
        if ys.size == 0:
            self._1d_ylim_cache[cache_key] = (None, None)
            return None, None
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        result = (ymin - margin, ymax + margin)
        self._1d_ylim_cache[cache_key] = result
        return result

    def _compute_ylim_for_global(self, varname: str, yscale: str) -> Tuple[Optional[float], Optional[float]]:
        # Include overlay variables in the y-limit calculation
        overlay_vars = tuple(self._overlay_vars.get(varname, []))
        all_vars = [varname] + list(overlay_vars)
        # Cache key includes: mode, varname, overlays, yscale, and case n_times
        case_info = tuple((c.label, c.n_time) for c in self.cases.values())
        cache_key = ("global", varname, overlay_vars, yscale, case_info)
        hit = self._1d_ylim_cache.get(cache_key)
        if hit is not None:
            return hit

        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            for vname in all_vars:
                if vname not in ds:
                    continue
                da = ds[vname]
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
            self._1d_ylim_cache[cache_key] = (None, None)
            return None, None
        ys = np.concatenate(ys_all)
        if ys.size == 0:
            self._1d_ylim_cache[cache_key] = (None, None)
            return None, None
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        result = (ymin - margin, ymax + margin)
        self._1d_ylim_cache[cache_key] = result
        return result

    # ---------- 2D plotting ----------
    def _primary_case(self) -> Optional[LoadedCase]:
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







    def _maybe_capture_view_1d(self, *, state_key_no_ti: tuple) -> Optional[Dict[str, tuple]]:
        """
        Preserve toolbar zoom/pan across 1D time-slider redraws.

        Only capture when the last drawn key (with ti removed) matches the incoming key.
        """
        last = getattr(self, "_last_draw_state_1d_profiles", None)
        if not last:
            return None
        try:
            # State key is ("prof", case_labels, ti, sdim, vars, modes, guard, overlays, ...)
            # Strip ti (index 2) to compare config without time index
            last_no_ti = ("prof", last[1], *last[3:])
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

    def _ds_at_time_index(self, case: LoadedCase, ti: int):
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
            ensure_sdtools_2d_metadata(ds_t)
        except Exception:
            pass
        return ds_t




    def _ds_at_time(self, case: LoadedCase):
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
                    ensure_sdtools_2d_metadata(ds_t)
                except Exception:
                    pass

                return ds_t
            except Exception:
                # Even if time slicing fails, make sure sdtools-required geometry aliases exist.
                try:
                    ensure_sdtools_2d_metadata(ds)
                except Exception:
                    pass
                return ds
        return ds

    def _redraw_2d_current_tab(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_2d_current_tab(self)

    def _redraw_2d_poloidal(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_2d_poloidal(self)

    def _redraw_2d_radial(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_2d_radial(self)

    def _redraw_2d_polygon(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_2d_polygon(self)

    def _redraw_2d_monitor(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_2d_monitor(self)

    def redraw(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_profiles(self)

    def redraw_time_history(self) -> None:
        """
        Backwards-compatible entry point.
        Prefer `request_time_history_redraw()` for better interactivity.
        """
        self.request_time_history_redraw()

    def _do_redraw_time_history(self) -> None:
        from holoshed.plotting.coordinator import PlotCoordinator
        PlotCoordinator.redraw_time_history(self)

    def _with_margin(self, ymin: float, ymax: float) -> Tuple[float, float]:
        from holoshed.plotting import ylim as ylim_mod
        return ylim_mod.with_margin(ymin, ymax)

    def _compute_ylim_poloidal_extracted(self, **kwargs):
        from holoshed.plotting import ylim as ylim_mod
        return ylim_mod.compute_ylim_poloidal_extracted(self, **kwargs)

    def _compute_ylim_radial_extracted(self, **kwargs):
        from holoshed.plotting import ylim as ylim_mod
        return ylim_mod.compute_ylim_radial_extracted(self, **kwargs)

