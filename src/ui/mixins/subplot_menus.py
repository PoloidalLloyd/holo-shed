"""Extracted from MainWindow: SubplotMenusMixin."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.ui.qt import QAction, QMenu, QPoint, Qt, qt_checked, qt_unchecked

_qt_checked = qt_checked
_qt_unchecked = qt_unchecked

class SubplotMenusMixin:

    def _datasets_by_colour(self) -> bool:
        """Return True if datasets should be distinguished by colour (vs linestyle)."""
        try:
            return self.dataset_style_combo.currentText() == "colour"
        except Exception:
            return True  # Default to colour

    # ---------- Right-click context menu on subplots ----------

    def _var_from_axes(self, ax, axes_map: Optional[Dict[str, object]] = None) -> Optional[str]:
        """Given an axes object, find which variable it displays."""
        if axes_map is None:
            axes_map = self._overlay_axes_by_var
        for var, ax_obj in axes_map.items():
            if ax_obj is ax:
                return var
        return None

    def _build_subplot_context_menu(self, varname: str, ax, canvas_type: str = "main") -> "QMenu":
        """Build context menu for a subplot showing the given variable."""
        menu = QMenu(self)

        # --- Y-limits submenu ---
        m_ylim = menu.addMenu("Y-limits")
        for mode in ("auto", "final", "max"):
            a = QAction(mode, self)
            a.setCheckable(True)
            cur_mode = self._ylim_mode_by_var.get(varname, "auto")
            if cur_mode == "global":
                cur_mode = "max"
            a.setChecked(cur_mode == mode)
            a.triggered.connect(partial(self._set_var_ylim_mode, varname, mode))
            m_ylim.addAction(a)

        # --- Y-scale submenu ---
        m_yscale = menu.addMenu("Y-scale")
        for mode in ("linear", "log", "symlog"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._yscale_by_var.get(varname, "linear") == mode)
            a.triggered.connect(partial(self._set_var_yscale, varname, mode))
            m_yscale.addAction(a)

        # --- X-scale submenu ---
        m_xscale = menu.addMenu("X-scale")
        for mode in ("linear", "log"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._xscale_by_var.get(varname, "linear") == mode)
            a.triggered.connect(partial(self._set_var_xscale, varname, mode))
            m_xscale.addAction(a)

        menu.addSeparator()

        # --- Add variable to this subplot (overlay) ---
        m_add = menu.addMenu("Add variable to subplot")
        # Make the submenu scrollable for long variable lists
        m_add.setStyleSheet("QMenu { menu-scrollable: 1; }")
        # Get all available vars minus already selected/overlaid
        all_vars = list(self.state.get("vars") or [])
        current_overlays = self._overlay_vars.get(varname, [])
        excluded = self._selected_set | set(current_overlays) | {varname}
        primary_units = self._get_var_units(varname)
        # Optionally filter to only show variables with matching units
        constrain_by_units = self.constrain_overlay_units_check.isChecked()
        if constrain_by_units:
            available_vars = [
                v for v in all_vars
                if v not in excluded and self._get_var_units(v) == primary_units
            ]
        else:
            available_vars = [v for v in all_vars if v not in excluded]
        if available_vars:
            for v in available_vars:
                units = self._get_var_units(v)
                label = f"{v} [{units}]" if units else v
                a = QAction(label, self)
                a.triggered.connect(partial(self._add_var_to_subplot, v, varname))
                m_add.addAction(a)
        else:
            if constrain_by_units and primary_units:
                info = QAction(f"(no variables with units [{primary_units}])", self)
            else:
                info = QAction("(no additional variables available)", self)
            info.setEnabled(False)
            m_add.addAction(info)

        # --- Show current overlays with option to remove ---
        current_overlays = self._overlay_vars.get(varname, [])
        if current_overlays:
            m_remove_overlay = menu.addMenu("Remove overlay")
            for ov in current_overlays:
                units = self._get_var_units(ov)
                label = f"{ov} [{units}]" if units else ov
                a = QAction(label, self)
                a.triggered.connect(partial(self._remove_overlay_from_subplot, ov, varname))
                m_remove_overlay.addAction(a)

        menu.addSeparator()

        # --- Remove this subplot ---
        act_remove = QAction(f"Remove subplot ({varname})", self)
        act_remove.triggered.connect(partial(self._remove_subplot_var, varname))
        menu.addAction(act_remove)

        return menu

    def _add_var_to_subplot(self, new_var: str, target_var: str) -> None:
        """Add a variable to be overlaid on the target variable's subplot."""
        if target_var not in self._overlay_vars:
            self._overlay_vars[target_var] = []
        if new_var not in self._overlay_vars[target_var]:
            self._overlay_vars[target_var].append(new_var)
        # Reset y-limits to auto so the subplot rescales to show all variables
        self._ylim_mode_by_var[target_var] = "auto"
        self._refresh_var_item(target_var)
        self._refresh_overlay_button_labels(target_var)
        self.request_redraw()

    def _remove_overlay_from_subplot(self, overlay_var: str, target_var: str) -> None:
        """Remove an overlaid variable from a subplot."""
        if target_var in self._overlay_vars:
            if overlay_var in self._overlay_vars[target_var]:
                self._overlay_vars[target_var].remove(overlay_var)
            if not self._overlay_vars[target_var]:
                del self._overlay_vars[target_var]
        # Reset y-limits to auto so the subplot rescales
        self._ylim_mode_by_var[target_var] = "auto"
        self._refresh_var_item(target_var)
        self._refresh_overlay_button_labels(target_var)
        self.request_redraw()

    def _remove_subplot_var(self, varname: str) -> None:
        """Uncheck a variable from the selection, removing its subplot."""
        # Also clean up any overlays associated with this variable
        if varname in self._overlay_vars:
            del self._overlay_vars[varname]
        # Also clean up time history overlays
        if varname in self._hist_overlay_vars:
            del self._hist_overlay_vars[varname]

        # Directly update selection state (don't rely solely on signal)
        if varname in self._selected_set:
            self._selected_set.remove(varname)
            self.selected_vars = [v for v in self.selected_vars if v != varname]

        # Uncheck from vars_list (this may also trigger _on_var_item_changed)
        item = self._find_item_by_var(varname)
        if item is not None:
            # Block signals to avoid double-processing
            self.vars_list.blockSignals(True)
            try:
                item.setCheckState(_qt_unchecked())
            finally:
                self.vars_list.blockSignals(False)

        # Trigger redraw
        self.request_redraw()
        self.request_time_history_redraw()

    def _is_shift_held(self) -> bool:
        """Check if Shift key is currently held using Qt."""
        try:
            from PyQt6.QtWidgets import QApplication
            mods = QApplication.keyboardModifiers()
            return bool(mods & Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            pass
        try:
            from PySide6.QtWidgets import QApplication
            mods = QApplication.keyboardModifiers()
            return bool(mods & Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            pass
        return False

    def _on_canvas_button_press(self, event) -> None:
        """Handle Shift+right-click on the main 1D profiles canvas for context menu."""
        if event.button != 3:  # Right-click only
            return
        # Require Shift modifier to avoid interfering with matplotlib zoom
        if not self._is_shift_held():
            return
        if event.inaxes is None:
            return
        varname = self._var_from_axes(event.inaxes, self._overlay_axes_by_var)
        if varname is None:
            return
        menu = self._build_subplot_context_menu(varname, event.inaxes, canvas_type="main")
        # Convert matplotlib coords to Qt global coords
        global_pos = self.canvas.mapToGlobal(
            QPoint(int(event.x), int(self.canvas.height() - event.y))
        )
        menu.exec(global_pos)

    def _on_pol_canvas_button_press(self, event) -> None:
        """Handle Shift+right-click on the 2D poloidal canvas for context menu."""
        if event.button != 3:
            return
        if not self._is_shift_held():
            return
        if event.inaxes is None:
            return
        varname = self._var_from_axes(event.inaxes, self._overlay_axes_by_var_pol)
        if varname is None:
            return
        menu = self._build_subplot_context_menu(varname, event.inaxes, canvas_type="pol")
        global_pos = self.pol_canvas.mapToGlobal(
            QPoint(int(event.x), int(self.pol_canvas.height() - event.y))
        )
        menu.exec(global_pos)

    def _on_rad_canvas_button_press(self, event) -> None:
        """Handle Shift+right-click on the 2D radial canvas for context menu."""
        if event.button != 3:
            return
        if not self._is_shift_held():
            return
        if event.inaxes is None:
            return
        varname = self._var_from_axes(event.inaxes, self._overlay_axes_by_var_rad)
        if varname is None:
            return
        menu = self._build_subplot_context_menu(varname, event.inaxes, canvas_type="rad")
        global_pos = self.rad_canvas.mapToGlobal(
            QPoint(int(event.x), int(self.rad_canvas.height() - event.y))
        )
        menu.exec(global_pos)

    # ---------- Time History context menu ----------

    def _on_hist_canvas_button_press(self, event) -> None:
        """Handle Shift+right-click on the time history canvas for context menu."""
        if event.button != 3:  # Right-click only
            return
        # Require Shift modifier to avoid interfering with matplotlib zoom
        if not self._is_shift_held():
            return
        if event.inaxes is None:
            return
        varname = self._var_from_hist_axes(event.inaxes)
        if varname is None:
            return
        menu = self._build_hist_subplot_context_menu(varname, event.inaxes)
        # Convert matplotlib coords to Qt global coords
        global_pos = self.hist_canvas.mapToGlobal(
            QPoint(int(event.x), int(self.hist_canvas.height() - event.y))
        )
        menu.exec(global_pos)

    def _var_from_hist_axes(self, ax) -> Optional[str]:
        """Find which variable a time history axes belongs to."""
        for var, (ax_u, ax_t) in self._hist_axes_by_var.items():
            if ax is ax_u or ax is ax_t:
                return var
        return None

    def _build_hist_subplot_context_menu(self, varname: str, ax) -> "QMenu":
        """Build context menu for a time history subplot."""
        menu = QMenu(self)

        # --- Y-scale submenu ---
        m_yscale = menu.addMenu("Y-scale")
        for mode in ("linear", "log", "symlog"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._hist_yscale_by_var.get(varname, "auto") == mode)
            a.triggered.connect(partial(self._set_hist_var_yscale, varname, mode))
            m_yscale.addAction(a)
        # Add "auto" option that uses the original threshold-based logic
        a_auto = QAction("auto (threshold)", self)
        a_auto.setCheckable(True)
        a_auto.setChecked(self._hist_yscale_by_var.get(varname, "auto") == "auto")
        a_auto.triggered.connect(partial(self._set_hist_var_yscale, varname, "auto"))
        m_yscale.addAction(a_auto)

        menu.addSeparator()

        # --- Add variable to this subplot (overlay) ---
        m_add = menu.addMenu("Add variable to subplot")
        # Make the submenu scrollable for long variable lists
        m_add.setStyleSheet("QMenu { menu-scrollable: 1; }")
        # Get all available vars minus already selected/overlaid
        all_vars = list(self.state.get("vars") or [])
        current_overlays = self._hist_overlay_vars.get(varname, [])
        excluded = self._selected_set | set(current_overlays) | {varname}
        primary_units = self._get_var_units(varname)
        # Optionally filter to only show variables with matching units
        constrain_by_units = self.constrain_overlay_units_check.isChecked()
        if constrain_by_units:
            available_vars = [
                v for v in all_vars
                if v not in excluded and self._get_var_units(v) == primary_units
            ]
        else:
            available_vars = [v for v in all_vars if v not in excluded]
        if available_vars:
            for v in available_vars:
                units = self._get_var_units(v)
                label = f"{v} [{units}]" if units else v
                a = QAction(label, self)
                a.triggered.connect(partial(self._add_var_to_hist_subplot, v, varname))
                m_add.addAction(a)
        else:
            if constrain_by_units and primary_units:
                info = QAction(f"(no variables with units [{primary_units}])", self)
            else:
                info = QAction("(no additional variables available)", self)
            info.setEnabled(False)
            m_add.addAction(info)

        # --- Show current overlays with option to remove ---
        current_overlays = self._hist_overlay_vars.get(varname, [])
        if current_overlays:
            m_remove_overlay = menu.addMenu("Remove overlay")
            for ov in current_overlays:
                units = self._get_var_units(ov)
                label = f"{ov} [{units}]" if units else ov
                a = QAction(label, self)
                a.triggered.connect(partial(self._remove_overlay_from_hist_subplot, ov, varname))
                m_remove_overlay.addAction(a)

        menu.addSeparator()

        # --- Remove this subplot ---
        act_remove = QAction(f"Remove subplot ({varname})", self)
        act_remove.triggered.connect(partial(self._remove_subplot_var, varname))
        menu.addAction(act_remove)

        return menu

    def _set_hist_var_yscale(self, varname: str, mode: str) -> None:
        """Set the y-axis scale for a time history variable."""
        self._hist_yscale_by_var[varname] = mode
        self.request_time_history_redraw()

    def _add_var_to_hist_subplot(self, new_var: str, target_var: str) -> None:
        """Add a variable to be overlaid on the target variable's time history subplot."""
        if target_var not in self._hist_overlay_vars:
            self._hist_overlay_vars[target_var] = []
        if new_var not in self._hist_overlay_vars[target_var]:
            self._hist_overlay_vars[target_var].append(new_var)
        self.request_time_history_redraw()

    def _remove_overlay_from_hist_subplot(self, overlay_var: str, target_var: str) -> None:
        """Remove an overlaid variable from a time history subplot."""
        if target_var in self._hist_overlay_vars:
            if overlay_var in self._hist_overlay_vars[target_var]:
                self._hist_overlay_vars[target_var].remove(overlay_var)
            if not self._hist_overlay_vars[target_var]:
                del self._hist_overlay_vars[target_var]
        self.request_time_history_redraw()

    # ---------- Overlay buttons on plots (Option B) ----------

