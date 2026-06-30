"""Extracted from MainWindow: TimeControlsMixin."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.dataset_utils import infer_time_dim
from src.models import LoadedCase
from src.ui.qt import QCheckBox, QKeySequence, QShortcut, Qt

class TimeControlsMixin:

    def _is_case_arrow_keys_enabled(self, label: str) -> bool:
        return bool(self._case_arrow_keys_enabled.get(label, True))

    def _set_case_arrow_keys_enabled(self, label: str, enabled: bool) -> None:
        self._case_arrow_keys_enabled[label] = bool(enabled)

    def _ensure_case_arrow_keys_enabled(self, label: str) -> None:
        if label not in self._case_arrow_keys_enabled:
            self._case_arrow_keys_enabled[label] = True

    def _sync_main_arrow_keys_checkboxes(self) -> None:
        labels = list(self.cases.keys())
        enabled = self._is_case_arrow_keys_enabled(labels[0]) if labels else True
        for chk in (getattr(self, "_main_arrow_keys_check", None), getattr(self, "_main_arrow_keys_check_2d", None)):
            if chk is None:
                continue
            chk.blockSignals(True)
            try:
                chk.setChecked(enabled)
            finally:
                chk.blockSignals(False)

    def _wire_arrow_keys_checkboxes(self) -> None:
        for chk in (getattr(self, "_main_arrow_keys_check", None), getattr(self, "_main_arrow_keys_check_2d", None)):
            if chk is None:
                continue
            chk.toggled.connect(self._on_main_arrow_keys_toggled)

    def _on_main_arrow_keys_toggled(self, checked: bool) -> None:
        labels = list(self.cases.keys())
        if not labels:
            return
        self._set_case_arrow_keys_enabled(labels[0], checked)
        self._sync_main_arrow_keys_checkboxes()

    def _make_case_arrow_keys_checkbox(self, label: str) -> "QCheckBox":
        chk = QCheckBox("keys")
        chk.setToolTip("When checked, left/right arrow keys adjust this case's time index")
        chk.setChecked(self._is_case_arrow_keys_enabled(label))

        def on_toggled(checked: bool, case_label: str = label) -> None:
            self._set_case_arrow_keys_enabled(case_label, checked)
            labels = list(self.cases.keys())
            if labels and case_label == labels[0]:
                self._sync_main_arrow_keys_checkboxes()

        chk.toggled.connect(on_toggled)
        return chk

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
        labels = list(self.cases.keys())
        first_label = labels[0] if labels else None
        moved = False

        # Move the main slider (first loaded case)
        if first_label and self._is_case_arrow_keys_enabled(first_label):
            slider = self._active_time_slider()
            try:
                v = int(slider.value()) + int(delta)
                v = max(int(slider.minimum()), min(int(slider.maximum()), v))
                slider.setValue(v)
                self._set_time_readout_for_index(v)
                moved = True
            except Exception:
                pass

        # Also move per-case sliders (for synchronized arrow key control)
        # Handle 1D per-case sliders
        for label, case_slider in self._case_sliders.items():
            if not self._is_case_arrow_keys_enabled(label):
                continue
            try:
                cv = int(case_slider.value()) + int(delta)
                cv = max(int(case_slider.minimum()), min(int(case_slider.maximum()), cv))
                case_slider.blockSignals(True)
                case_slider.setValue(cv)
                case_slider.blockSignals(False)
                self._case_time_indices[label] = cv
                # Also update the index spinbox
                spinbox = self._case_spinboxes.get(label)
                if spinbox is not None:
                    spinbox.blockSignals(True)
                    spinbox.setValue(cv)
                    spinbox.blockSignals(False)
                # Also update the ms spinbox
                ms_spinbox = self._case_ms_spinboxes.get(label)
                if ms_spinbox is not None and label in self.cases:
                    t_ms = self._get_time_ms_for_case(self.cases[label], cv)
                    ms_spinbox.blockSignals(True)
                    ms_spinbox.setValue(t_ms)
                    ms_spinbox.blockSignals(False)
                moved = True
            except Exception:
                pass

        # Handle 2D per-case sliders
        for label, case_slider in self._case_sliders_2d.items():
            if not self._is_case_arrow_keys_enabled(label):
                continue
            try:
                cv = int(case_slider.value()) + int(delta)
                cv = max(int(case_slider.minimum()), min(int(case_slider.maximum()), cv))
                case_slider.blockSignals(True)
                case_slider.setValue(cv)
                case_slider.blockSignals(False)
                self._case_time_indices_2d[label] = cv
                # Also update the index spinbox
                spinbox = self._case_spinboxes_2d.get(label)
                if spinbox is not None:
                    spinbox.blockSignals(True)
                    spinbox.setValue(cv)
                    spinbox.blockSignals(False)
                # Also update the ms spinbox
                ms_spinbox = self._case_ms_spinboxes_2d.get(label)
                if ms_spinbox is not None and label in self.cases:
                    t_ms = self._get_time_ms_for_case(self.cases[label], cv)
                    ms_spinbox.blockSignals(True)
                    ms_spinbox.setValue(t_ms)
                    ms_spinbox.blockSignals(False)
                moved = True
            except Exception:
                pass

        # Trigger single redraw after all sliders updated
        if moved:
            self.request_redraw()

    # ---------- UI ----------

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
            # Apply normalization if enabled (search in normalized space)
            if self._is_time_normalized() and arr.size > 0:
                arr = arr - arr[0]
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
        # Use first case's time values with normalization support
        if self.cases:
            try:
                first_case = next(iter(self.cases.values()))
                clamped_ti = min(ti, first_case.n_time - 1)
                ms = self._get_time_ms_for_case(first_case, clamped_ti)
                spin = self._active_time_ms_spin()
                spin.setEnabled(True)
                spin.blockSignals(True)
                spin.setValue(ms)
                spin.blockSignals(False)
                return
            except Exception:
                pass

        # Fallback to state t_values if no cases loaded
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
            # Apply normalization if enabled
            if self._is_time_normalized() and tvals.size > 0:
                ms = ms - float(tvals[0]) * 1e3
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
        self.time_spin.blockSignals(True)
        self.time_spin.setValue(int(v))
        self.time_spin.blockSignals(False)

        # Update ms spinbox using first dataset (with normalization support)
        if self.cases:
            first_case = next(iter(self.cases.values()))
            clamped_v = min(int(v), first_case.n_time - 1)
            t_ms = self._get_time_ms_for_case(first_case, clamped_v)
            self.time_ms_spin.blockSignals(True)
            self.time_ms_spin.setValue(t_ms)
            self.time_ms_spin.blockSignals(False)
        else:
            self._set_time_ms_spin_for_index(int(v))
        self.request_redraw()

    # ---------- Status / datasets ----------

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

    def _time_values_for_case(self, case: "LoadedCase"):
        """Return (time_dim_name, time_values_seconds) for a loaded case."""
        backend = case.backend
        if backend is not None:
            tdim, tvals = backend.time_coordinate(case)
            try:
                return tdim, np.asarray(tvals, dtype=float)
            except Exception:
                return tdim, None

        ds = case.ds
        tdim = self.state.get("time_dim") or infer_time_dim(ds)
        if tdim is None:
            return None, None
        try:
            if hasattr(ds, "coords") and tdim in ds.coords:
                return tdim, np.asarray(ds[tdim].values, dtype=float)
        except Exception:
            pass
        try:
            if tdim in ds:
                return tdim, np.asarray(ds[tdim].values, dtype=float)
        except Exception:
            pass
        return tdim, None

    def _get_time_ms_for_case(self, case: "LoadedCase", ti: int) -> float:
        """Get the time in ms for a given time index in a case's dataset."""
        _tdim, t_values = self._time_values_for_case(case)
        if t_values is not None and ti < len(t_values):
            t_ms = float(t_values[ti]) * 1e3  # Convert to ms
            # Apply normalization if enabled (t - t[0])
            if self._is_time_normalized():
                t0_ms = float(t_values[0]) * 1e3
                t_ms = t_ms - t0_ms
            return t_ms
        return 0.0

    def _find_time_index_for_ms(self, case: "LoadedCase", t_ms: float) -> int:
        """Find the nearest time index for a given time in ms."""
        try:
            _tdim, t_values = self._time_values_for_case(case)
            if t_values is not None:
                t_ms_arr = np.asarray(t_values, dtype=float) * 1e3  # Convert to ms
                # Apply normalization if enabled (search in normalized space)
                if self._is_time_normalized():
                    t_ms_arr = t_ms_arr - t_ms_arr[0]
                # Find nearest index
                idx = int(np.argmin(np.abs(t_ms_arr - t_ms)))
                return max(0, min(idx, case.n_time - 1))
        except Exception:
            pass
        return 0

    def _is_time_normalized(self) -> bool:
        """Check if time normalization is enabled."""
        # Read checkbox state directly to avoid synchronization issues
        try:
            return bool(self.normalize_time_check.isChecked())
        except Exception:
            return getattr(self, '_normalize_time_enabled', False)

    def _on_normalize_time_changed(self, state) -> None:
        """Handle normalize time checkbox state change."""
        # Store the state - check if non-zero (checked) or use isChecked() directly
        # Qt.Checked = 2, Qt.Unchecked = 0, but PySide6 might pass enum
        try:
            self._normalize_time_enabled = bool(self.normalize_time_check.isChecked())
        except Exception:
            self._normalize_time_enabled = bool(state)
        # Update all ms spinboxes to reflect the new normalization setting
        self._update_all_time_displays()

    def _update_all_time_displays(self) -> None:
        """Update all time (ms) spinboxes to reflect current normalization setting."""
        # Update main slider ms spinbox (uses first dataset)
        ti = self._get_time_index()
        if self.cases:
            first_case = next(iter(self.cases.values()))
            clamped_ti = min(ti, first_case.n_time - 1)
            t_ms = self._get_time_ms_for_case(first_case, clamped_ti)
            # Update the 1D ms spinbox directly
            if hasattr(self, 'time_ms_spin') and self.time_ms_spin is not None:
                self.time_ms_spin.blockSignals(True)
                self.time_ms_spin.setValue(t_ms)
                self.time_ms_spin.blockSignals(False)
            # Update 2D spinbox if in 2D mode
            if hasattr(self, 'time_ms_spin_2d') and self.time_ms_spin_2d is not None:
                self.time_ms_spin_2d.blockSignals(True)
                self.time_ms_spin_2d.setValue(t_ms)
                self.time_ms_spin_2d.blockSignals(False)
        else:
            self._set_time_ms_spin_for_index(ti)

        # Update 1D per-case ms spinboxes (each uses its own t[0] for normalization)
        for label, ms_spin in self._case_ms_spinboxes.items():
            if label in self.cases:
                case_ti = self._case_time_indices.get(label, 0)
                t_ms = self._get_time_ms_for_case(self.cases[label], case_ti)
                ms_spin.blockSignals(True)
                ms_spin.setValue(t_ms)
                ms_spin.blockSignals(False)

        # Update 2D per-case ms spinboxes
        for label, ms_spin in self._case_ms_spinboxes_2d.items():
            if label in self.cases:
                case_ti = self._case_time_indices_2d.get(label, 0)
                t_ms = self._get_time_ms_for_case(self.cases[label], case_ti)
                ms_spin.blockSignals(True)
                ms_spin.setValue(t_ms)
                ms_spin.blockSignals(False)

    # ---------- Variables list ----------

    def _get_time_index(self) -> int:
        try:
            return int(self._active_time_slider().value())
        except Exception:
            return 0

    def _get_time_index_for_case(self, case: LoadedCase) -> int:
        # Use per-case time index if available (for multiple datasets)
        # Check appropriate dict based on current mode (1D vs 2D)
        if self._mode_is_2d:
            if case.label in self._case_time_indices_2d and len(self.cases) > 1:
                ti = self._case_time_indices_2d[case.label]
            else:
                ti = self._get_time_index()
        else:
            if case.label in self._case_time_indices and len(self.cases) > 1:
                ti = self._case_time_indices[case.label]
            else:
                ti = self._get_time_index()
        return min(ti, case.n_time - 1)

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

