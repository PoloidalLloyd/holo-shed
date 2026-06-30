"""Extracted from MainWindow: OverlayButtonsMixin."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from functools import partial

from holoshed.ui.qt import QPushButton

class OverlayButtonsMixin:

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

