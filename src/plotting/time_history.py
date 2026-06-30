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

def redraw_time_history_impl(win):
        """
        Plot time traces of selected variables at an upstream and target index, based on
        `plot_time_history_optimized` in `convergence_functions.py`.

        For each selected variable we draw 2 subplots (rows):
        - upstream value vs time
        - target value vs time

        Supports overlay variables and user-selected yscale via right-click menu.
        """
        win.hist_figure.clear()
        win._hist_axes_by_var.clear()  # Reset axes tracking for right-click detection
        try:
            win.hist_figure.set_facecolor("white")
        except Exception:
            pass

        if not win.cases:
            ax = win.hist_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No dataset loaded.\nLoad a case directory to view time history.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            win.hist_canvas.draw_idle()
            return

        tdim = win.state.get("time_dim") or "t"
        # Shared selection with profiles
        vars_to_plot = list(win.selected_vars)
        if not vars_to_plot:
            ax = win.hist_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No variables selected.\nCheck variables on the left to plot time history.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            win.hist_canvas.draw_idle()
            return

        upstream_index = int(win.hist_upstream_spin.value())
        target_index = int(win.hist_target_spin.value())
        time_slices = int(win.hist_time_slices_spin.value())

        n_cols = max(1, len(vars_to_plot))
        n_rows = 2
        gs = win.hist_figure.add_gridspec(nrows=n_rows, ncols=n_cols, hspace=0.35, wspace=0.30)

        # Color cycle for overlays
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        last_time_ms = None
        for i, var in enumerate(vars_to_plot):
            ax_u = win.hist_figure.add_subplot(gs[0, i])
            ax_t = win.hist_figure.add_subplot(gs[1, i], sharex=ax_u)

            # Store axes for right-click detection
            win._hist_axes_by_var[var] = (ax_u, ax_t)

            # Get overlay variables for this primary variable
            overlay_vars = win._hist_overlay_vars.get(var, [])
            all_vars_for_subplot = [var] + overlay_vars

            units = None
            # Choose a scale similar to convergence_functions (log if huge)
            log_threshold = 1e6
            max_abs = 0.0

            # Plot each variable (primary + overlays)
            for var_idx, plot_var in enumerate(all_vars_for_subplot):
                color = default_colors[var_idx % len(default_colors)]
                is_overlay = (var_idx > 0)

                for c in win.cases.values():
                    ds = c.ds
                    if plot_var not in ds:
                        continue
                    da = ds[plot_var]
                    try:
                        if not is_overlay:
                            units = units or da.attrs.get("units", None)
                    except Exception:
                        pass

                    # Pick time dim
                    if tdim not in da.dims:
                        # Not a time-varying variable -> skip
                        continue

                    # Spatial dimension: try "y" then "pos" then inferred spatial dim
                    sdim = None
                    for cand in ("y", "pos", win.state.get("spatial_dim")):
                        if cand and cand in da.dims:
                            sdim = cand
                            break
                    if sdim is None:
                        continue

                    # Clamp indices (per-case, per-var)
                    n_s = int(ds.sizes.get(sdim, 0))
                    if n_s <= 0:
                        continue
                    upi = upstream_index if upstream_index >= 0 else max(0, n_s + upstream_index)
                    tgi = target_index if target_index >= 0 else max(0, n_s + target_index)
                    upi = int(np.clip(upi, 0, n_s - 1))
                    tgi = int(np.clip(tgi, 0, n_s - 1))

                    # Cache key: (case_label, var, tdim, sdim, upi, tgi)
                    ck = (c.label, plot_var, tdim, sdim, upi, tgi)
                    cached = win._hist_cache.get(ck)
                    if cached is None:
                        try:
                            t_full = np.asarray(ds[tdim].values) * 1e3
                            y_up_full = np.asarray(win._isel_1d_with_guard_replace(da, sdim=sdim, idx=upi).values).squeeze()
                            y_tg_full = np.asarray(win._isel_1d_with_guard_replace(da, sdim=sdim, idx=tgi).values).squeeze()
                            win._hist_cache[ck] = (t_full, y_up_full, y_tg_full)
                            cached = win._hist_cache[ck]
                        except Exception:
                            continue

                    t_full, y_up_full, y_tg_full = cached
                    if t_full is None:
                        continue

                    n_t = int(len(t_full))
                    if n_t <= 0:
                        continue
                    n_sel = min(time_slices, n_t)
                    sl = slice(-n_sel, None)
                    tvals = t_full[sl]
                    y_up = y_up_full[sl]
                    y_tg = y_tg_full[sl]

                    # Downsample for responsiveness (keep last N points evenly)
                    try:
                        if win._hist_max_points and len(tvals) > int(win._hist_max_points):
                            stride = int(np.ceil(len(tvals) / float(win._hist_max_points)))
                            tvals = tvals[::stride]
                            y_up = y_up[::stride]
                            y_tg = y_tg[::stride]
                    except Exception:
                        pass

                    # Track scale decision (for auto mode)
                    try:
                        max_abs = max(max_abs, float(np.nanmax(np.abs(y_up))), float(np.nanmax(np.abs(y_tg))))
                    except Exception:
                        pass

                    # Build label: include variable name for overlays, case label for multiple cases
                    if is_overlay:
                        if len(win.cases) > 1:
                            label = f"{plot_var} ({c.label})"
                        else:
                            label = plot_var
                    else:
                        label = c.label if len(win.cases) > 1 else None

                    ax_u.plot(tvals, y_up, "-", linewidth=1.5, label=label, color=color)
                    ax_t.plot(tvals, y_tg, "--", linewidth=1.5, label=label, color=color)

                    if tvals.size:
                        last_time_ms = float(tvals[-1])

            # Apply yscale - check user setting first, then fall back to auto
            user_scale = win._hist_yscale_by_var.get(var, "auto")
            if user_scale == "auto":
                scale = "log" if (max_abs > log_threshold and max_abs > 0) else "linear"
            else:
                scale = user_scale
            ax_u.set_yscale(scale)
            ax_t.set_yscale(scale)

            # Build title with overlay info
            if overlay_vars:
                title_suffix = f" + {len(overlay_vars)} overlay(s)"
            else:
                title_suffix = ""
            ax_u.set_title(f"Upstream {var}{title_suffix}", fontsize=10)
            ax_t.set_title(f"Target {var}{title_suffix}", fontsize=10)
            ax_t.set_xlabel("Time (ms)")
            # Variable name is already in the subplot title; keep y-label to units only.
            ylabel = f"({units})" if units else ""
            ax_u.set_ylabel(ylabel)
            ax_t.set_ylabel(ylabel)
            ax_u.grid(True, alpha=0.3)
            ax_t.grid(True, alpha=0.3)

            # Show legend if we have multiple cases or overlays
            if len(win.cases) > 1 or overlay_vars:
                ax_u.legend(loc="best", fontsize=8)

            # Hide x tick labels on top row
            ax_u.tick_params(labelbottom=False)

        # if last_time_ms is not None:
        #     win.hist_time_readout.setText(f"Time history (last: {last_time_ms:.4f} ms)")
        #     win.hist_figure.suptitle(f"Time History (Last: {last_time_ms:.3f} ms)", fontsize=12)
        # else:
        #     win.hist_time_readout.setText("Time history")

        try:
            win.hist_figure.tight_layout()
        except Exception:
            pass
        win.hist_canvas.draw_idle()
