"""Extracted from MainWindow: Region2dMixin."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from src.ui.qt import QVBoxLayout, QWidget
from src.ui.region_overlay import RegionOverlayWindow

class Region2dMixin:

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
        win = RegionOverlayWindow(title, on_close=_on_close)
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
        cmap = str(getattr(self, "poly_cmap_combo", None).currentText() or "Spectral_r") if hasattr(self, "poly_cmap_combo") else "Spectral_r"
        vmin = self._poly_vmin_active
        vmax = self._poly_vmax_active
        return case, ds_t, (var, logscale, cmap, vmin, vmax)

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

        var, logscale, cmap, vmin, vmax = bg
        state_key = (case.label, var, bool(logscale), cmap, vmin, vmax)

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
                    cmap=cmap,
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
                    from src.plotting.common import get_poloidal_profile

                    df = get_poloidal_profile(
                        case,
                        time_index=int(ti),
                        region=region,
                        sepadd=int(sepadd),
                        params=[],
                    )
                    self._pol_cache[ck] = df
            else:
                region = str(self.rad_region_combo.currentText() or "omp")
                ti = self._get_time_index_for_case(case)
                ck = (case.label, ti, region)
                df = self._rad_cache.get(ck)
                if df is None:
                    from src.plotting.common import get_radial_profile

                    df = get_radial_profile(
                        case,
                        time_index=int(ti),
                        region=region,
                        params=["R", "Z"],
                    )
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
            c0 = np.asarray(df[ccol].values, dtype=float) if (df is not None and ccol in df) else None

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

