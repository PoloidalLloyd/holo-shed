"""Plotting implementation extracted from holo-shed."""

from __future__ import annotations

import numpy as np

from src.dataset_utils import (
    downsample_xy,
    ensure_sdtools_2d_metadata,
    get_option_float,
    guard_replace_1d_profile_xy,
    infer_time_dim,
    parse_optional_float,
    selector_params_only,
    xpoint_idx_bpxy_valley,
)
from src.models import LoadedCase

def redraw_monitor(win):
        # If nothing relevant changed since last draw, don't rebuild the figure.
        try:
            case = win._primary_case()
            sepadd = int(win.pol_sepadd_spin.value()) if hasattr(win, "pol_sepadd_spin") else 0
            region = (
                str(win.timehist_region_combo.currentText() or "outer_lower")
                if hasattr(win, "timehist_region_combo")
                else "outer_lower"
            )
            vars_to_plot = tuple(win.selected_vars)
            state_key = ("mon", getattr(case, "label", None), sepadd, region, vars_to_plot)
            if win._last_draw_state_2d.get("mon") == state_key and win.mon_figure.axes:
                win.mon_canvas.draw_idle()
                return
            win._last_draw_state_2d["mon"] = state_key
        except Exception:
            pass

        win.mon_figure.clear()
        try:
            win.mon_figure.set_facecolor("white")
        except Exception:
            pass

        case = win._primary_case()
        if case is None:
            ax = win.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            win.mon_canvas.draw_idle()
            return

        vars_to_plot = list(win.selected_vars)
        if not vars_to_plot:
            ax = win.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
            win.mon_canvas.draw_idle()
            return

        ds = case.ds
        tdim = win.state.get("time_dim") or "t"
        if tdim not in getattr(ds, "coords", {}):
            ax = win.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"No time coordinate '{tdim}' in dataset.", ha="center", va="center", transform=ax.transAxes)
            win.mon_canvas.draw_idle()
            return

        # Build monitor from 1D poloidal extraction (per user rule):
        # for outer_lower SOL leg, index 0 ~ OMP and index -2 ~ target.
        from src.plotting.common import get_poloidal_profile

        try:
            t_ms = np.asarray(ds[tdim].values) * 1e3
        except Exception:
            t_ms = np.arange(case.n_time)

        region = (
            str(win.timehist_region_combo.currentText() or "outer_lower")
            if hasattr(win, "timehist_region_combo")
            else "outer_lower"
        )
        sepadd = int(win.pol_sepadd_spin.value()) if hasattr(win, "pol_sepadd_spin") else 0
        ck = (case.label, region, sepadd, tuple(vars_to_plot))
        cached = win._mon_cache.get(ck)
        if cached is None:
            omp = np.full((int(case.n_time), len(vars_to_plot)), np.nan, dtype=float)
            targ = np.full((int(case.n_time), len(vars_to_plot)), np.nan, dtype=float)

            for i in range(int(case.n_time)):
                try:
                    params = selector_params_only(list(vars_to_plot))
                    df = get_poloidal_profile(
                        case,
                        time_index=int(i),
                        region=region,
                        sepadd=sepadd,
                        params=list(params),
                    )
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

            win._mon_cache[ck] = (t_ms, omp, targ, tuple(vars_to_plot))
            cached = win._mon_cache[ck]

        t_ms, omp, targ, vorder = cached
        vorder = list(vorder)

        n = len(vorder)
        # Arrange as 2 rows x N columns:
        #   top row   -> OMP
        #   bottom row-> target
        gs = win.mon_figure.add_gridspec(nrows=2, ncols=max(1, n), hspace=0.45, wspace=0.35)
        ax_omp = []
        ax_targ = []
        for i in range(max(1, n)):
            if i == 0:
                a0 = win.mon_figure.add_subplot(gs[0, i])
                a1 = win.mon_figure.add_subplot(gs[1, i], sharex=a0)
            else:
                a0 = win.mon_figure.add_subplot(gs[0, i], sharex=ax_omp[0])
                a1 = win.mon_figure.add_subplot(gs[1, i], sharex=ax_omp[0])
            ax_omp.append(a0)
            ax_targ.append(a1)

        mid_label = "omp" if str(region).startswith("outer") else "imp"
        tgt_label = f"{region}_target"

        for i, name in enumerate(vorder):
            a0 = ax_omp[i]
            a1 = ax_targ[i]
            a0.set_title(f"{name} ({mid_label})", fontsize=10)
            a1.set_title(f"{name} ({tgt_label})", fontsize=10)
            for ax in (a0, a1):
                ax.grid(True, alpha=0.3)

            y0 = np.asarray(omp[:, i], dtype=float)
            y1 = np.asarray(targ[:, i], dtype=float)
            if win._yscale_by_var.get(name, "linear") == "log":
                y0 = np.where(y0 > 0, y0, np.nan)
                y1 = np.where(y1 > 0, y1, np.nan)
            a0.plot(t_ms[: len(y0)], y0, lw=1.2)
            a1.plot(t_ms[: len(y1)], y1, lw=1.2)

            # Apply y-scale mode (re-using the same per-variable setting as other tabs)
            mode = win._yscale_by_var.get(name, "linear")
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
            win.mon_figure.tight_layout()
        except Exception:
            pass
        win.mon_canvas.draw_idle()

