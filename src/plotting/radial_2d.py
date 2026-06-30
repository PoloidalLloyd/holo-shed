"""Plotting implementation extracted from holo-shed."""

from __future__ import annotations

from typing import Dict, List

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
from src.plotting.common import RADIAL_XLABEL, get_radial_profile, radial_distance_mm, resolve_profile_column

def redraw_radial(win):
        win._update_time_readout()

        # If nothing relevant changed since last draw, don't rebuild the figure.
        _view_restore = None
        try:
            case_info = win._all_case_time_state()
            region = str(win.rad_region_combo.currentText() or "omp")
            vars_to_plot = tuple(win.selected_vars)
            modes = tuple((v, win._yscale_by_var.get(v, "linear"), win._ylim_mode_by_var.get(v, "auto")) for v in vars_to_plot)
            # Include overlay variables in state key so view isn't restored when overlays change
            overlays = tuple((v, tuple(win._overlay_vars.get(v, []))) for v in vars_to_plot)
            state_key = ("rad", case_info, region, vars_to_plot, modes, overlays)
            if win._last_draw_state_2d.get("rad") == state_key and win.rad_figure.axes:
                try:
                    win._position_overlay_buttons_rad()
                except Exception:
                    pass
                win.rad_canvas.draw_idle()
                return
            # Preserve zoom/pan across time changes (same config, different ti)
            state_no_ti = ("rad", tuple(label for label, _ in case_info), region, vars_to_plot, modes, overlays)
            _view_restore = win._maybe_capture_view_2d(kind="rad", state_key_no_ti=state_no_ti)
            win._last_draw_state_2d["rad"] = state_key
        except Exception:
            pass

        win.rad_figure.clear()
        try:
            win.rad_figure.set_facecolor("white")
        except Exception:
            pass

        if not win.cases:
            # No plots -> no overlay buttons
            win._clear_overlay_buttons_rad()
            ax = win.rad_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            win.rad_canvas.draw_idle()
            return

        vars_to_plot = list(win.selected_vars)
        if not vars_to_plot:
            # No plots -> no overlay buttons
            win._clear_overlay_buttons_rad()
            ax = win.rad_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
            win.rad_canvas.draw_idle()
            return

        region = str(win.rad_region_combo.currentText() or "omp")

        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))
        # Extra vertical padding to avoid title/xlabel overlap between stacked subplots
        gs = win.rad_figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.60, wspace=0.30)
        axes = [win.rad_figure.add_subplot(gs[i % nrows, i // nrows]) for i in range(n)]

        # Batch extract once per case/time/region for speed
        # Configure y-scales and x-scales per variable before plotting
        for ax, name in zip(axes, vars_to_plot):
            mode = win._yscale_by_var.get(name, "linear")
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    ax.set_yscale("symlog", linthresh=1e-6)
                else:
                    ax.set_yscale("linear")
            except Exception:
                pass
            # X-scale
            xscale_mode = win._xscale_by_var.get(name, "linear")
            try:
                if xscale_mode == "log":
                    ax.set_xscale("log")
                else:
                    ax.set_xscale("linear")
            except Exception:
                pass

        # Collect all overlay variables to include in extraction
        all_overlay_vars: List[str] = []
        for name in vars_to_plot:
            all_overlay_vars.extend(win._overlay_vars.get(name, []))

        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        datasets_by_colour = win._datasets_by_colour()

        # Build variable-to-color map for linestyle mode (same variable = same color across datasets)
        var_colors: Dict[str, str] = {}
        if not datasets_by_colour:
            import matplotlib.pyplot as plt
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            all_vars = list(vars_to_plot) + all_overlay_vars
            for vname in all_vars:
                if vname not in var_colors:
                    var_colors[vname] = color_cycle[len(var_colors) % len(color_cycle)]

        for case_idx, c in enumerate(win.cases.values()):
            ds_t = win._ds_at_time(c)
            ti = win._get_time_index_for_case(c)
            # Incremental cache: keep one df per (case,time,region) and extend
            ck = (c.label, ti, region)
            df = win._rad_cache.get(ck)
            # Include overlay vars in params for extraction
            params = selector_params_only(list(vars_to_plot) + list(all_overlay_vars))
            if df is None:
                try:
                    df = get_radial_profile(
                        c,
                        time_index=int(ti),
                        region=region,
                        params=list(params),
                    )
                except Exception as e:
                    win.set_status(f"Radial extract failed: {e}", is_error=True)
                    df = None
                win._rad_cache[ck] = df
            else:
                try:
                    missing = [
                        v for v in params
                        if resolve_profile_column(c, v, df) is None
                    ]
                except Exception:
                    missing = list(params)
                if missing:
                    try:
                        df_new = get_radial_profile(
                            c,
                            time_index=int(ti),
                            region=region,
                            params=list(missing),
                        )
                        for v in list(missing):
                            col = resolve_profile_column(c, v, df_new) if df_new is not None else None
                            if col is not None and df_new is not None:
                                try:
                                    df[v] = df_new[col].values
                                except Exception:
                                    pass
                        win._rad_cache[ck] = df
                    except Exception as e:
                        win.set_status(f"Radial extract failed: {e}", is_error=True)
            if df is None:
                continue

            x = radial_distance_mm(df)
            if x is None:
                win.set_status("Radial extract missing Srad/dist coordinate.", is_error=True)
                continue

            # Hermes-only: radial-only derived variables from xarray dataset
            if getattr(c, "backend_kind", "hermes") != "solps":
                all_needed_vars = list(vars_to_plot) + list(all_overlay_vars)
                for vname in all_needed_vars:
                    if resolve_profile_column(c, vname, df) is not None:
                        continue
                    if vname not in ds_t:
                        continue
                    try:
                        da = ds_t[vname]
                        dims = tuple(da.dims)
                        if 'x' in dims and 'theta' not in dims and 'y' not in dims:
                            if len(dims) == 1 and dims[0] == 'x':
                                vals = np.asarray(da.values)
                            else:
                                continue
                            if len(vals) == len(df):
                                df[vname] = vals
                    except Exception:
                        pass

            for ax, name in zip(axes, vars_to_plot):
                overlay_vars = win._overlay_vars.get(name, [])
                if overlay_vars:
                    title = f"{name} + {', '.join(overlay_vars)} ({region})"
                else:
                    title = f"{name} ({region})"
                ax.set_title(title, fontsize=10)
                ax.grid(True, alpha=0.3)

                col = resolve_profile_column(c, name, df)
                y = None
                if col is not None:
                    try:
                        y = np.asarray(df[col].values)
                    except Exception:
                        pass
                if y is None and getattr(c, "backend_kind", "hermes") != "solps" and name in ds_t:
                    # Try to extract radial-only derived variable directly
                    try:
                        da = ds_t[name]
                        dims = tuple(da.dims)
                        if 'x' in dims and 'theta' not in dims:
                            y = np.asarray(da.values)
                            # Check length matches x-axis
                            if len(y) != len(x):
                                y = None
                    except Exception:
                        pass
                if y is None:
                    continue
                if win._yscale_by_var.get(name, "linear") == "log":
                    y = np.where(y > 0, y, np.nan)
                # Label with variable name if overlays exist
                if overlay_vars:
                    plot_label = f"{name} ({c.label})" if len(win.cases) > 1 else name
                else:
                    plot_label = c.label

                if datasets_by_colour:
                    # Datasets by colour: primary var solid, let matplotlib pick color
                    line, = ax.plot(x, y, label=plot_label, linestyle=linestyles[0])
                    case_color = line.get_color()
                    case_ls = linestyles[0]
                else:
                    # Datasets by linestyle: each dataset gets different linestyle, same color per variable
                    case_ls = linestyles[case_idx % len(linestyles)]
                    var_color = var_colors.get(name)
                    line, = ax.plot(x, y, label=plot_label, linestyle=case_ls, color=var_color)
                    case_color = line.get_color()

                # Plot overlay variables
                for ov_idx, ov_name in enumerate(overlay_vars):
                    ov_y = None
                    ov_col = resolve_profile_column(c, ov_name, df)
                    if ov_col is not None:
                        try:
                            ov_y = np.asarray(df[ov_col].values)
                        except Exception:
                            pass
                    if ov_y is None and getattr(c, "backend_kind", "hermes") != "solps" and ov_name in ds_t:
                        # Try to extract radial-only derived variable directly
                        try:
                            da = ds_t[ov_name]
                            dims = tuple(da.dims)
                            if 'x' in dims and 'theta' not in dims:
                                ov_y = np.asarray(da.values)
                                # Check length matches x-axis
                                if len(ov_y) != len(x):
                                    ov_y = None
                        except Exception:
                            pass
                    if ov_y is None:
                        continue
                    try:
                        if win._yscale_by_var.get(name, "linear") == "log":
                            ov_y = np.where(ov_y > 0, ov_y, np.nan)
                        ov_label = f"{ov_name} ({c.label})" if len(win.cases) > 1 else ov_name
                        if datasets_by_colour:
                            # Overlays use different linestyle, same color per case
                            ls = linestyles[(ov_idx + 1) % len(linestyles)]
                            ax.plot(x, ov_y, label=ov_label, linestyle=ls, color=case_color)
                        else:
                            # Overlays use same color per variable, different linestyle per case
                            ov_color = var_colors.get(ov_name)
                            ax.plot(x, ov_y, label=ov_label, linestyle=case_ls, color=ov_color)
                    except Exception:
                        pass

        for ax, name in zip(axes, vars_to_plot):
            # Units (match 1D GUI behavior)
            units = None
            for c in win.cases.values():
                try:
                    if hasattr(c.ds, "data_vars") and name in c.ds:
                        units = c.ds[name].attrs.get("units", None)
                        if units:
                            break
                except Exception:
                    continue
            ax.set_ylabel(f"{units}" if units else "")
            ax.set_xlabel(RADIAL_XLABEL)
            try:
                ax.axvline(
                    0.0,
                    color="k",
                    linestyle="--",
                    linewidth=1.0,
                    label="separatrix",
                    zorder=1,
                )
            except Exception:
                pass

            overlay_vars = win._overlay_vars.get(name, [])
            try:
                ax.legend(loc="best", fontsize=8)
            except Exception:
                pass

            try:
                ylim_mode = win._ylim_mode_by_var.get(name, "auto")
                if ylim_mode == "global":
                    ylim_mode = "max"
                if ylim_mode == "auto":
                    # Force full autoscale to include all plotted data (including overlays)
                    ax.relim()
                    ax.autoscale(enable=True, axis='y', tight=False)
                    ax.autoscale_view(scalex=False, scaley=True)
                else:
                    c0 = win._primary_case()
                    if c0 is not None:
                        ymin, ymax = win._compute_ylim_radial_extracted(
                            case=c0,
                            region=region,
                            varname=name,
                            yscale=win._yscale_by_var.get(name, "linear"),
                            mode=ylim_mode,
                        )
                        if ymin is not None and ymax is not None:
                            ax.set_ylim(ymin, ymax)
            except Exception:
                pass

        win._sync_overlay_buttons_rad(vars_to_plot=vars_to_plot, axes=axes)
        # Ensure toolbar Home resets to the default view (not the preserved zoom)
        try:
            win._reset_toolbar_home(getattr(win, "rad_toolbar", None))
        except Exception:
            pass
        # Restore user zoom/pan (if applicable)
        try:
            win._restore_view_2d(views=_view_restore, vars_to_plot=vars_to_plot, axes=axes)
        except Exception:
            pass
        try:
            win.rad_figure.tight_layout()
        except Exception:
            pass
        win.rad_canvas.draw_idle()

