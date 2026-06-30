"""Plotting implementation extracted from holo-shed."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.dataset_utils import selector_params_only
from src.models import LoadedCase
from src.plotting.common import get_poloidal_profile, resolve_profile_column


def _case_time_ms(win, case: LoadedCase) -> np.ndarray:
    if getattr(case, "backend_kind", "hermes") == "solps" and case.backend is not None:
        _, tvals = case.backend.time_coordinate(case)
        try:
            return np.asarray(tvals, dtype=float) * 1e3
        except Exception:
            return np.arange(int(case.n_time), dtype=float)

    tdim = win.state.get("time_dim") or "t"
    ds = case.ds
    try:
        if tdim in getattr(ds, "coords", {}):
            return np.asarray(ds[tdim].values, dtype=float) * 1e3
    except Exception:
        pass
    return np.arange(int(case.n_time), dtype=float)


def extract_monitor_traces(
    case: LoadedCase,
    win,
    *,
    region: str,
    sepadd: int,
    vars_to_plot: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return (t_ms, omp[var], target[var]) traces for one case."""
    n_t = int(case.n_time)
    t_ms = _case_time_ms(win, case)
    omp: Dict[str, np.ndarray] = {v: np.full(n_t, np.nan, dtype=float) for v in vars_to_plot}
    targ: Dict[str, np.ndarray] = {v: np.full(n_t, np.nan, dtype=float) for v in vars_to_plot}
    params = selector_params_only(list(vars_to_plot))

    for i in range(n_t):
        try:
            df = get_poloidal_profile(
                case,
                time_index=int(i),
                region=region,
                sepadd=int(sepadd),
                params=list(params),
            )
        except Exception:
            continue
        for v in vars_to_plot:
            col = resolve_profile_column(case, v, df)
            if col is None:
                continue
            try:
                y = np.asarray(df[col].values, dtype=float)
            except Exception:
                continue
            if y.size:
                omp[v][i] = float(y[0])
                if y.size >= 2:
                    targ[v][i] = float(y[-2])

    return t_ms, omp, targ


def redraw_monitor(win):
    region = (
        str(win.timehist_region_combo.currentText() or "outer_lower")
        if hasattr(win, "timehist_region_combo")
        else "outer_lower"
    )
    sepadd = int(win.pol_sepadd_spin.value()) if hasattr(win, "pol_sepadd_spin") else 0
    vars_to_plot = list(win.selected_vars)

    # If nothing relevant changed since last draw, don't rebuild the figure.
    try:
        case_info = tuple((c.label, int(c.n_time)) for c in win.cases.values())
        state_key = ("mon", case_info, sepadd, region, tuple(vars_to_plot))
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

    if not win.cases:
        ax = win.mon_figure.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
        win.mon_canvas.draw_idle()
        return

    if not vars_to_plot:
        ax = win.mon_figure.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
        win.mon_canvas.draw_idle()
        return

    # Hermes cases need a time coordinate; SOLPS uses backend.time_coordinate instead.
    hermes_cases = [
        c for c in win.cases.values() if getattr(c, "backend_kind", "hermes") != "solps"
    ]
    if hermes_cases:
        tdim = win.state.get("time_dim") or "t"
        ds0 = hermes_cases[0].ds
        if tdim not in getattr(ds0, "coords", {}):
            ax = win.mon_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"No time coordinate '{tdim}' in dataset.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            win.mon_canvas.draw_idle()
            return

    case_traces: List[Tuple[LoadedCase, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []
    for case in win.cases.values():
        ck = (case.label, region, sepadd, tuple(vars_to_plot))
        cached = win._mon_cache.get(ck)
        if cached is None:
            try:
                cached = extract_monitor_traces(
                    case, win, region=region, sepadd=sepadd, vars_to_plot=vars_to_plot
                )
            except Exception:
                continue
            win._mon_cache[ck] = cached
        case_traces.append((case, *cached))

    if not case_traces:
        ax = win.mon_figure.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Could not extract time history for the selected variables/region.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        win.mon_canvas.draw_idle()
        return

    steady_state = all(int(c.n_time) <= 1 for c in win.cases.values())
    n = len(vars_to_plot)
    gs = win.mon_figure.add_gridspec(nrows=2, ncols=max(1, n), hspace=0.45, wspace=0.35)
    ax_omp: List = []
    ax_targ: List = []
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
    datasets_by_colour = win._datasets_by_colour()
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    var_colors: Dict[str, str] = {}
    if not datasets_by_colour:
        import matplotlib.pyplot as plt

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for vname in vars_to_plot:
            if vname not in var_colors:
                var_colors[vname] = color_cycle[len(var_colors) % len(color_cycle)]

    for i, name in enumerate(vars_to_plot):
        a0 = ax_omp[i]
        a1 = ax_targ[i]
        a0.set_title(f"{name} ({mid_label})", fontsize=10)
        a1.set_title(f"{name} ({tgt_label})", fontsize=10)
        for ax in (a0, a1):
            ax.grid(True, alpha=0.3)

        plotted_any = False
        for case_idx, (case, t_ms, omp, targ) in enumerate(case_traces):
            y0 = np.asarray(omp.get(name, []), dtype=float)
            y1 = np.asarray(targ.get(name, []), dtype=float)
            if win._yscale_by_var.get(name, "linear") == "log":
                y0 = np.where(y0 > 0, y0, np.nan)
                y1 = np.where(y1 > 0, y1, np.nan)

            label = case.label if len(case_traces) > 1 else None
            plot_kw: dict = {"label": label}
            if steady_state or len(t_ms) <= 1:
                plot_kw.update(marker="o", markersize=7, linestyle="None")
            else:
                plot_kw.update(lw=1.2, linestyle=linestyles[case_idx % len(linestyles)])

            if datasets_by_colour:
                a0.plot(t_ms[: len(y0)], y0, **plot_kw)
                a1.plot(t_ms[: len(y1)], y1, **plot_kw)
            else:
                plot_kw["color"] = var_colors.get(name)
                a0.plot(t_ms[: len(y0)], y0, **plot_kw)
                a1.plot(t_ms[: len(y1)], y1, **plot_kw)
            if np.any(np.isfinite(y0)) or np.any(np.isfinite(y1)):
                plotted_any = True

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

        if steady_state:
            for ax in (a0, a1):
                try:
                    ax.set_xlim(-0.5, 0.5)
                except Exception:
                    pass

        units = None
        for case, _, _, _ in case_traces:
            ds = case.ds
            try:
                if hasattr(ds, "data_vars") and name in ds:
                    units = ds[name].attrs.get("units", None)
                    if units:
                        break
            except Exception:
                pass
        yl = f"{units}" if units else ""
        a0.set_ylabel(yl)
        a1.set_ylabel(yl)

        if len(case_traces) > 1 and plotted_any:
            a0.legend(loc="best", fontsize=8)

    for ax in ax_targ:
        ax.set_xlabel("Time (ms)" if not steady_state else "Time (ms) — steady-state")

    if steady_state:
        try:
            win.mon_figure.suptitle(
                "Steady-state case(s): markers show midplane / target values at t = 0",
                fontsize=10,
                y=0.995,
            )
        except Exception:
            pass

    try:
        win.mon_figure.tight_layout()
    except Exception:
        pass
    win.mon_canvas.draw_idle()
