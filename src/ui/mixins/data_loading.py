"""Extracted from MainWindow: DataLoadingMixin."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import List, Optional, Tuple

import numpy as np

from src.dataset_utils import (
    cases_have_mixed_backends,
    infer_spatial_dim,
    infer_time_dim,
    list_plottable_vars,
    list_plottable_vars_2d,
    merge_case_variable_sets,
    time_reference_case,
)
from src.models import LoadedCase

class DataLoadingMixin:

    def _load_case(self, case_path: str) -> LoadedCase:
        from src.backends.factory import load_case

        return load_case(case_path, load_cls=self.Load)

    def _recompute_all_vars(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        if not self.cases:
            return [], None, None

        case_list = list(self.cases.values())
        first_case = case_list[0]
        mixed = cases_have_mixed_backends(self.cases)
        is_2d = bool(getattr(first_case, "is_2d", False))
        var_sets: List[set[str]] = []
        tdim: Optional[str] = None

        for c in case_list:
            if c.backend is not None:
                tdim_c, _tvals = c.backend.time_coordinate(c)
                var_sets.append(set(c.backend.list_variables(c)))
            else:
                ds = c.ds
                tdim_c = infer_time_dim(ds)
                if is_2d:
                    var_sets.append(set(list_plottable_vars_2d(ds, time_dim=tdim_c)))
                else:
                    sdim_c = self.spatial_dim_forced or infer_spatial_dim(ds)
                    var_sets.append(set(list_plottable_vars(ds, spatial_dim=sdim_c, time_dim=tdim_c)))

        ref = time_reference_case(self.cases)
        if ref is not None and ref.backend is not None:
            tdim, _tvals = ref.backend.time_coordinate(ref)
        elif ref is not None:
            tdim = infer_time_dim(ref.ds)
        else:
            tdim = None

        all_vars = merge_case_variable_sets(var_sets, mixed_backends=mixed)
        sdim = "theta" if is_2d else (self.spatial_dim_forced or infer_spatial_dim(first_case.ds))
        return all_vars, sdim, tdim

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
        first_case = next(iter(self.cases.values()), None)
        try:
            is_2d = bool(getattr(first_case, "is_2d", False)) if first_case is not None else False
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

        # Time axis for readouts: use the longest transient case, not load order.
        ref_case = time_reference_case(self.cases) or first_case
        t_values = None
        if ref_case is not None and ref_case.backend is not None:
            _tdim, tvals = ref_case.backend.time_coordinate(ref_case)
            try:
                t_values = np.asarray(tvals, dtype=float)
            except Exception:
                t_values = None
        elif ref_case is not None:
            ds0 = ref_case.ds
            if tdim is not None and hasattr(ds0, "coords") and tdim in ds0.coords:
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

    def _invalidate_plot_caches(self) -> None:
        for attr in (
            "_hist_cache",
            "_mon_cache",
            "_pol_cache",
            "_rad_cache",
            "_pol_ylim_cache",
            "_rad_ylim_cache",
            "_1d_ylim_cache",
        ):
            try:
                getattr(self, attr).clear()
            except Exception:
                pass
        try:
            self._last_draw_state_2d = {"pol": None, "rad": None, "poly": None, "mon": None}
        except Exception:
            pass

    def remove_case(self, label: str) -> None:
        """Remove one loaded case from the session."""
        if label not in self.cases:
            return
        self.cases.pop(label)
        self._case_time_indices.pop(label, None)
        self._case_time_indices_2d.pop(label, None)
        self._case_arrow_keys_enabled.pop(label, None)

        if not self.cases:
            self.state = dict(spatial_dim=None, time_dim=None, vars=[], t_values=None)
            self.selected_vars = []
            self._selected_set = set()
            self._mode_is_2d = False
            self._configure_tabs(is_2d=False)
            self._render_var_list()
            self._update_time_readout()
            self.set_status("Enter a case directory path and click 'Load case'.")
        else:
            self._update_after_load()
            self._sync_main_arrow_keys_checkboxes()
            self.set_status("")

        self._update_datasets_list()
        self._invalidate_plot_caches()
        self.request_full_redraw()

    def _get_unique_case_label(self, label: str) -> str:
        """Generate a unique label by adding numeric suffix if label already exists."""
        if label not in self.cases:
            return label
        # Find the next available numeric suffix
        i = 2
        while f"{label} ({i})" in self.cases:
            i += 1
        return f"{label} ({i})"

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
                        "Use 'New session' to switch modes."
                    )

            # Limit 2D mode to 3 cases for comparison (keeps plots readable)
            if self.cases and (not replace) and bool(next(iter(self.cases.values())).is_2d):
                if len(self.cases) >= 3:
                    raise ValueError("2D mode supports up to 3 datasets for comparison. Use 'New session' to start fresh.")

            if replace:
                self.cases.clear()
                self._case_time_indices.clear()
                self._case_arrow_keys_enabled.clear()

            # Ensure unique label by adding numeric suffix if needed
            unique_label = self._get_unique_case_label(lc.label)
            if unique_label != lc.label:
                lc = dataclass_replace(lc, label=unique_label)
            self.cases[lc.label] = lc
            self._ensure_case_arrow_keys_enabled(lc.label)
            self._update_after_load()
            self._sync_main_arrow_keys_checkboxes()
            self._update_datasets_list()
            if cases_have_mixed_backends(self.cases) and not self.state.get("vars"):
                self.set_status(
                    "Mixed Hermes + SOLPS session: no common plottable variables found.",
                    is_error=True,
                )
            else:
                self.set_status("")
            self._invalidate_plot_caches()
            self.request_full_redraw()
        except Exception as e:
            self.set_status(f"Failed to load dataset: {e}", is_error=True)

    # ---------- Plotting ----------

