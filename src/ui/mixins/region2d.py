"""Extracted from MainWindow: Region2dMixin."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from src.models import LoadedCase
from src.plotting.common import get_poloidal_profile, get_radial_profile, radial_distance_column
from src.plotting.polygon_2d import plot_region2d_background
from src.ui.qt import QVBoxLayout, QWidget
from src.ui.region_overlay import RegionOverlayWindow


class Region2dMixin:

    def _cases_2d_for_region(self) -> List[LoadedCase]:
        return [c for c in self.cases.values() if getattr(c, "is_2d", False)][:3]

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

    def _fit_region2d_layout(self, st, n_cases: int) -> None:
        """Size figure and window for side-by-side panels without over-stretching."""
        n = max(1, min(int(n_cases), 3))
        per_panel_w_in = 3.6
        fig_h_in = 5.2
        fig = st.get("fig")
        canvas = st.get("canvas")
        win = st.get("win")
        if fig is None:
            return
        fig.set_size_inches(per_panel_w_in * n, fig_h_in, forward=True)
        try:
            dpi = float(fig.get_dpi())
        except Exception:
            dpi = 100.0
        chrome_w, chrome_h = 40, 130
        width = int(per_panel_w_in * n * dpi) + chrome_w
        height = int(fig_h_in * dpi) + chrome_h
        width = min(max(width, 620), 1380)
        height = min(max(height, 620), 820)
        if win is not None:
            try:
                win.resize(width, height)
            except Exception:
                pass
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _ensure_region2d_window(self, kind: str):
        if kind == "pol" and self._region2d_pol:
            return self._region2d_pol
        if kind == "rad" and self._region2d_rad:
            return self._region2d_rad

        def _on_close():
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
        fig = Figure(figsize=(3.6, 5.2))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, win)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        win.setCentralWidget(central)

        st = {
            "win": win,
            "fig": fig,
            "canvas": canvas,
            "toolbar": toolbar,
            "panels": [],
            "state": None,
        }
        if kind == "pol":
            self._region2d_pol = st
        else:
            self._region2d_rad = st
        try:
            self._fit_region2d_layout(st, 1)
        except Exception:
            pass
        try:
            win.show()
        except Exception:
            pass
        return st

    def _get_region2d_bg(self) -> Optional[Tuple[str, bool, str, object, object]]:
        var = str(getattr(self, "poly_var_combo", None).currentText() or "").strip() if hasattr(self, "poly_var_combo") else ""
        if not var:
            return None
        logscale = bool(getattr(self, "poly_log_check", None).isChecked()) if hasattr(self, "poly_log_check") else False
        cmap = str(getattr(self, "poly_cmap_combo", None).currentText() or "Spectral_r") if hasattr(self, "poly_cmap_combo") else "Spectral_r"
        vmin = self._poly_vmin_active
        vmax = self._poly_vmax_active
        return var, logscale, cmap, vmin, vmax

    def _case_supports_var(self, case: LoadedCase, var: str) -> bool:
        if getattr(case, "backend_kind", "hermes") == "solps" and case.backend is not None:
            try:
                case.backend._resolve_param(case.ds, var)
                return True
            except Exception:
                try:
                    return var in case.backend.list_variables(case)
                except Exception:
                    return False
        try:
            ds_t = self._ds_at_time(case)
            return var in ds_t
        except Exception:
            return False

    def _region2d_cut_df(self, case: LoadedCase, kind: str):
        if kind == "pol":
            region = str(self.pol_region_combo.currentText() or "outer_lower")
            sepadd = int(self.pol_sepadd_spin.value())
            ti = self._get_time_index_for_case(case)
            ck = (case.label, ti, region, sepadd)
            df = self._pol_cache.get(ck)
            if df is None:
                df = get_poloidal_profile(
                    case,
                    time_index=int(ti),
                    region=region,
                    sepadd=int(sepadd),
                    params=[],
                )
                self._pol_cache[ck] = df
            ccol = "Spol" if (
                bool(getattr(self, "pol_use_spol_check", None).isChecked())
                if hasattr(self, "pol_use_spol_check")
                else False
            ) else "Spar"
        else:
            region = str(self.rad_region_combo.currentText() or "omp")
            ti = self._get_time_index_for_case(case)
            ck_geom = (case.label, ti, region, "__cut_rz__")
            df = self._rad_cache.get(ck_geom)
            if df is None:
                df = get_radial_profile(
                    case,
                    time_index=int(ti),
                    region=region,
                    params=["R", "Z"],
                )
                self._rad_cache[ck_geom] = df
            ccol = radial_distance_column(df) or "Srad"
        return df, ccol

    def _region2d_cut_rz(self, case: LoadedCase, kind: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        df, ccol = self._region2d_cut_df(case, kind)
        if df is None or "R" not in df.columns:
            raise KeyError("No R column in extracted cut data.")
        R0 = np.asarray(df["R"].values, dtype=float)
        if "Z" in df.columns:
            Z0 = np.asarray(df["Z"].values, dtype=float)
        elif kind == "rad":
            Z0 = np.zeros_like(R0)
        else:
            raise KeyError("No Z column in extracted cut data.")
        c0 = np.asarray(df[ccol].values, dtype=float) if ccol in df.columns else None
        m = np.isfinite(R0) & np.isfinite(Z0)
        if c0 is not None:
            m = m & np.isfinite(c0)
        return R0[m], Z0[m], c0[m] if c0 is not None else None

    def _draw_region2d_cut(self, ax, ln, arrow, R: np.ndarray, Z: np.ndarray, cc: Optional[np.ndarray]):
        ln.set_data(R, Z)
        try:
            if arrow is not None:
                arrow.remove()
        except Exception:
            pass
        new_arrow = None
        if cc is not None and R.size >= 2:
            try:
                order = np.argsort(cc)
                end_i = int(order[-1])
                prev_i = int(order[-2])
                k = 2
                while k <= order.size and (R[end_i] == R[prev_i] and Z[end_i] == Z[prev_i]):
                    prev_i = int(order[-k])
                    k += 1
                new_arrow = ax.annotate(
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
                    new_arrow.set_zorder(11)
                except Exception:
                    pass
            except Exception:
                pass
        return new_arrow

    def _update_region2d_overlay(self, kind: str) -> None:
        """Update the popout window showing cut location(s) over 2D colormap(s)."""
        st = self._region2d_pol if kind == "pol" else self._region2d_rad
        if not st:
            return

        cases = self._cases_2d_for_region()
        bg = self._get_region2d_bg()
        fig = st.get("fig")
        canvas = st.get("canvas")
        if not cases or bg is None or fig is None or canvas is None:
            return

        var, logscale, cmap, vmin, vmax = bg
        if kind == "pol":
            region = str(self.pol_region_combo.currentText() or "outer_lower")
            sepadd = int(self.pol_sepadd_spin.value())
            cut_key = (region, sepadd)
        else:
            region = str(self.rad_region_combo.currentText() or "omp")
            cut_key = (region,)

        case_info = tuple((c.label, self._get_time_index_for_case(c)) for c in cases)
        state_key = (kind, case_info, var, bool(logscale), cmap, vmin, vmax, cut_key)
        panels = st.get("panels") or []
        rebuild = st.get("state") != state_key or len(panels) != len(cases)

        if rebuild:
            fig.clear()
            try:
                fig.set_facecolor("white")
            except Exception:
                pass
            panels = []
            n_cases = len(cases)
            for i, case in enumerate(cases):
                ax = fig.add_subplot(1, n_cases, i + 1)
                short = case.label if len(case.label) <= 22 else case.label[:19] + "..."
                panel = {"ax": ax, "polys": None, "line": None, "arrow": None, "case": case.label}

                if not self._case_supports_var(case, var):
                    ax.set_axis_off()
                    ax.text(
                        0.5,
                        0.5,
                        f"{short}\nVariable '{var}' not available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    panels.append(panel)
                    continue

                try:
                    panel["polys"] = plot_region2d_background(
                        self,
                        case,
                        ax,
                        var=var,
                        logscale=logscale,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_title(f"{short}\n{var}", fontsize=10)
                except Exception as e:
                    ax.set_axis_off()
                    ax.text(
                        0.5,
                        0.5,
                        f"{short}\n2D background failed:\n{e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )

                try:
                    (ln,) = ax.plot([], [], color="red", linewidth=2.0, linestyle="--", alpha=0.85, label="cut")
                    ln.set_zorder(10)
                    panel["line"] = ln
                except Exception:
                    pass
                panels.append(panel)

            st["panels"] = panels
            st["state"] = state_key
            try:
                self._reset_toolbar_home(st.get("toolbar"))
            except Exception:
                pass
            try:
                self._fit_region2d_layout(st, len(cases))
            except Exception:
                pass
        else:
            # Fast path: Hermes time-only updates for polygon colours
            for panel, case in zip(panels, cases):
                if getattr(case, "backend_kind", "hermes") == "solps":
                    continue
                polys = panel.get("polys")
                if polys is None or not self._case_supports_var(case, var):
                    continue
                try:
                    ds_t = self._ds_at_time(case)
                    data = ds_t[var].hermesm.clean_guards()
                    polys.set_array(np.asarray(data.data).flatten())
                except Exception:
                    pass

        # Update cut overlays on every panel
        for panel, case in zip(panels, cases):
            ax = panel.get("ax")
            ln = panel.get("line")
            if ax is None or ln is None:
                continue
            try:
                R, Z, cc = self._region2d_cut_rz(case, kind)
                panel["arrow"] = self._draw_region2d_cut(ax, ln, panel.get("arrow"), R, Z, cc)
            except Exception as e:
                try:
                    ln.set_data([], [])
                except Exception:
                    pass
                try:
                    arr = panel.get("arrow")
                    if arr is not None:
                        arr.remove()
                except Exception:
                    pass
                panel["arrow"] = None
                try:
                    ax.text(
                        0.02,
                        0.02,
                        f"Cut overlay unavailable: {e}",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="darkred",
                    )
                except Exception:
                    pass

        try:
            fig.tight_layout()
        except Exception:
            pass
        canvas.draw_idle()

    def _on_pol_show_region2d_toggled(self, checked: bool) -> None:
        if not checked:
            self._close_region2d_window("pol")
            return
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
