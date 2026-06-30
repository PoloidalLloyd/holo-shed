"""Extracted from MainWindow: LayoutMixin."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.ui.qt import (
    _QT_API,
    QAbstractItemView,
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget, Qt, qt_checked, qt_unchecked,
)

class LayoutMixin:

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

        # Plot style option: datasets by colour or linestyle
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Distinguish datasets by:"))
        self.dataset_style_combo = QComboBox()
        self.dataset_style_combo.addItems(["colour", "linestyle"])
        self.dataset_style_combo.setCurrentIndex(1)  # Default: linestyle
        self.dataset_style_combo.currentIndexChanged.connect(lambda _: self.request_redraw())
        style_row.addWidget(self.dataset_style_combo)
        style_row.addStretch(1)
        left_layout.addLayout(style_row)

        # Option to normalize time to start from zero
        self.normalize_time_check = QCheckBox("Normalize time (start from 0)")
        self.normalize_time_check.setChecked(False)
        self.normalize_time_check.stateChanged.connect(self._on_normalize_time_changed)
        left_layout.addWidget(self.normalize_time_check)

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
        # Dataset name label for main 2D slider (shown when multiple datasets loaded)
        self._main_slider_label_2d = QLabel("")
        self._main_slider_label_2d.setMinimumWidth(60)
        self._main_slider_label_2d.setMaximumWidth(120)
        self._main_slider_label_2d.setVisible(False)
        time2d_row.addWidget(self._main_slider_label_2d)

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

        self._main_arrow_keys_check_2d = QCheckBox("keys")
        self._main_arrow_keys_check_2d.setToolTip(
            "When checked, left/right arrow keys adjust this case's time index"
        )
        self._main_arrow_keys_check_2d.setChecked(True)
        time2d_row.addWidget(self._main_arrow_keys_check_2d)

        # Keep the old label for backwards compatibility (must be parented to avoid becoming a top-level window)
        self.time_readout_2d = QLabel("time index = 0", self.time2d_widget)
        self.time_readout_2d.setVisible(False)
        time2d_layout.addLayout(time2d_row)

        # Container for per-case 2D time sliders (below main slider, populated dynamically)
        self._case_sliders_container_2d = QWidget()
        self._case_sliders_layout_2d = QVBoxLayout(self._case_sliders_container_2d)
        self._case_sliders_layout_2d.setContentsMargins(0, 0, 0, 0)
        self._case_sliders_layout_2d.setSpacing(2)
        time2d_layout.addWidget(self._case_sliders_container_2d)

        self.time2d_widget.setVisible(False)
        left_layout.addWidget(self.time2d_widget)

        # Per-view controls (selection is shared)
        self.controls_tabs = QTabWidget()

        # Profiles controls (informational only; buttons live on plots)
        prof_ctrl_tab = QWidget()
        prof_ctrl_layout = QVBoxLayout(prof_ctrl_tab)
        prof_ctrl_layout.setContentsMargins(6, 6, 6, 6)
        prof_ctrl_layout.setSpacing(6)
        prof_ctrl_layout.addWidget(QLabel("Profiles controls:\n- Use the overlay buttons on each plot for y-scale and y-limits.\n- Shift+right-click a subplot for more options."))
        self.guard_replace_check = QCheckBox("Replace guard cells (1D only)")
        self.guard_replace_check.setChecked(True)
        prof_ctrl_layout.addWidget(self.guard_replace_check)
        self.constrain_overlay_units_check = QCheckBox("Constrain subplot overlays by units")
        self.constrain_overlay_units_check.setChecked(True)
        prof_ctrl_layout.addWidget(self.constrain_overlay_units_check)
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
        pol_ctrl_layout.addWidget(QLabel("Poloidal 1D (SOL ring)\nShift+right-click subplot for context options."))
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
        rad_ctrl_layout.addWidget(QLabel("Radial 1D\nShift+right-click subplot for context options."))
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

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("colormap"))
        self.poly_cmap_combo = QComboBox()
        self.poly_cmap_combo.addItems([
            "Spectral_r", "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdBu_r", "seismic", "hot", "jet", "turbo",
        ])
        self.poly_cmap_combo.setCurrentText("Spectral_r")
        cmap_row.addWidget(self.poly_cmap_combo, 1)
        poly_ctrl_layout.addLayout(cmap_row)

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
        mon_ctrl_layout.addWidget(QLabel("Time history at midplane + target (from selected variables)"))

        th_region_row = QHBoxLayout()
        th_region_row.addWidget(QLabel("region"))
        self.timehist_region_combo = QComboBox()
        self.timehist_region_combo.addItems(["outer_lower", "outer_upper", "inner_lower", "inner_upper"])
        try:
            self.timehist_region_combo.setCurrentText("outer_lower")
        except Exception:
            pass
        th_region_row.addWidget(self.timehist_region_combo, 1)
        mon_ctrl_layout.addLayout(th_region_row)
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
        # Dataset name label (shown when multiple datasets loaded)
        self._main_slider_label = QLabel("")
        self._main_slider_label.setMinimumWidth(60)
        self._main_slider_label.setMaximumWidth(120)
        self._main_slider_label.setVisible(False)
        slider_row.addWidget(self._main_slider_label)

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

        self._main_arrow_keys_check = QCheckBox("keys")
        self._main_arrow_keys_check.setToolTip(
            "When checked, left/right arrow keys adjust this case's time index"
        )
        self._main_arrow_keys_check.setChecked(True)
        slider_row.addWidget(self._main_arrow_keys_check)

        # Keep old widgets for backwards compatibility (parent + hidden to avoid top-level popup windows)
        self.time_readout = QLabel("time index = 0", prof_plot_tab)
        self.time_readout.setVisible(False)
        self._time_index_label_1d = QLabel("time index", prof_plot_tab)
        self._time_index_label_1d.setVisible(False)
        prof_plot_layout.addLayout(slider_row)

        # Container for per-case time sliders (below main slider, populated dynamically)
        self._case_sliders_container = QWidget()
        self._case_sliders_layout = QVBoxLayout(self._case_sliders_container)
        self._case_sliders_layout.setContentsMargins(0, 0, 0, 0)
        self._case_sliders_layout.setSpacing(2)
        prof_plot_layout.addWidget(self._case_sliders_container)

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
        self.timehist_region_combo.currentIndexChanged.connect(lambda _i: self.request_time_history_redraw())
        self.poly_var_combo.currentIndexChanged.connect(lambda _i: self.request_redraw())
        self.poly_grid_only_check.toggled.connect(lambda _v: self.request_redraw())
        self.poly_log_check.toggled.connect(lambda _v: self.request_redraw())
        self.poly_cmap_combo.currentIndexChanged.connect(lambda _i: self.request_redraw())
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
            # Keep add button enabled - 2D mode now supports up to 3 datasets for comparison
            try:
                self.add_btn.setEnabled(True)
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

