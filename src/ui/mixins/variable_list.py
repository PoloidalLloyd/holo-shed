"""Extracted from MainWindow: VariableListMixin."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.ui.qt import QAction, QListWidgetItem, QMenu, Qt, qt_checked, qt_unchecked

_qt_checked = qt_checked
_qt_unchecked = qt_unchecked

class VariableListMixin:

    def _on_search_change(self, text: str) -> None:
        self._var_filter = text or ""
        self._render_var_list()

    def _filtered_vars(self) -> List[str]:
        vars_all = list(self.state.get("vars") or [])
        q = (self._var_filter or "").strip().lower()
        if not q:
            return vars_all
        return [v for v in vars_all if q in v.lower()]

    def _get_var_units(self, varname: str) -> Optional[str]:
        """Get units string for a variable from the loaded dataset(s) or derived variables."""
        # First check derived variable registry (these have explicitly defined units)
        try:
            from derived_variables import get_derived_var_units
            units = get_derived_var_units(varname)
            if units:
                return units
        except Exception:
            pass
        # Fall back to dataset attrs
        cases = getattr(self, "cases", None)
        if cases:
            for c in cases.values():
                ds = getattr(c, "ds", None)
                if ds is None:
                    continue
                if varname in ds:
                    try:
                        units = ds[varname].attrs.get("units", None)
                        if units:
                            return units
                    except Exception:
                        pass
        return None

    def _item_text_for_var(self, name: str) -> str:
        # Show variable name with units if available
        units = self._get_var_units(name)
        if units:
            return f"{name} [{units}]"
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

    def _set_var_xscale(self, varname: str, mode: str) -> None:
        """Set x-scale for a variable's subplot."""
        self._xscale_by_var[varname] = mode
        self.request_redraw()

