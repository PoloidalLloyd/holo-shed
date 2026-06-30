"""Plotting implementation extracted from holo-shed."""

from __future__ import annotations

import numpy as np

from holoshed.dataset_utils import (
    downsample_xy,
    ensure_sdtools_2d_metadata,
    get_option_float,
    guard_replace_1d_profile_xy,
    infer_time_dim,
    parse_optional_float,
    selector_params_only,
    xpoint_idx_bpxy_valley,
)
from holoshed.models import LoadedCase

def redraw_profiles(win):
        if win._mode_is_2d:
            win._redraw_2d_current_tab()
            # Also keep optional region-overlay popouts in sync
            try:
                if win._region2d_pol:
                    win._update_region2d_overlay("pol")
            except Exception:
                pass
            try:
                if win._region2d_rad:
                    win._update_region2d_overlay("rad")
            except Exception:
                pass
            return

        win._update_time_readout()

        # Preserve toolbar zoom/pan across time changes (same config, different ti)
        _view_restore_1d = None
        try:
            sdim0 = win.state.get("spatial_dim")
            vars0 = tuple(win.selected_vars)
            modes0 = tuple((v, win._yscale_by_var.get(v, "linear"), win._ylim_mode_by_var.get(v, "auto")) for v in vars0)
            guard0 = bool(win._guard_replace_enabled())
            case_labels0 = tuple(win.cases.keys())
            ti0 = int(win._get_time_index())
            # Include overlay variables in state key so view isn't restored when overlays change
            overlays0 = tuple((v, tuple(win._overlay_vars.get(v, []))) for v in vars0)
            state_key = ("prof", case_labels0, ti0, sdim0, vars0, modes0, guard0, overlays0)
            state_no_ti = ("prof", case_labels0, sdim0, vars0, modes0, guard0, overlays0)
            _view_restore_1d = win._maybe_capture_view_1d(state_key_no_ti=state_no_ti)
            win._last_draw_state_1d_profiles = state_key
        except Exception:
            _view_restore_1d = None

        win.figure.clear()
        # Explicit facecolor to avoid inheriting dark appearances on some platforms.
        try:
            win.figure.set_facecolor("white")
        except Exception:
            pass

        if not win.cases:
            # No plots -> no overlay buttons
            win._clear_overlay_buttons()
            ax = win.figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No dataset loaded.\nLoad a case directory to view variables.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            win.canvas.draw_idle()
            return

        sdim = win.state.get("spatial_dim")
        tdim = win.state.get("time_dim")
        vars_to_plot = list(win.selected_vars)

        if not vars_to_plot:
            win._clear_overlay_buttons()
            ax = win.figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No variables selected.\nCheck variables on the left to plot.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            win.canvas.draw_idle()
            return

        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))

        gs = win.figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35, wspace=0.30)

        sharex_ref: List[Optional["object"]] = [None] * ncols
        axes: List["object"] = []

        for idx in range(n):
            col = idx // nrows
            row = idx % nrows
            sharex = sharex_ref[col]
            ax = win.figure.add_subplot(gs[row, col], sharex=sharex)
            if sharex_ref[col] is None:
                sharex_ref[col] = ax
            axes.append(ax)

        # Determine bottom-most axis per column for x-label and tick labels
        bottom_idx_by_col: Dict[int, int] = {}
        for col in range(ncols):
            inds = [i for i in range(n) if (i // nrows) == col]
            if inds:
                bottom_idx_by_col[col] = max(inds)

        for i, ax in enumerate(axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if not is_bottom:
                ax.tick_params(labelbottom=False)

        for ax, name in zip(axes, vars_to_plot):
            mode = win._yscale_by_var.get(name, "linear")
            ylim_mode = win._ylim_mode_by_var.get(name, "auto")

            # Configure y-scale before plotting
            linthresh = None
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    ys_all = []
                    for c in win.cases.values():
                        ds = c.ds
                        if name not in ds:
                            continue
                        da = ds[name]
                        case_ti = win._get_time_index_for_case(c)
                        if tdim is not None and tdim in da.dims:
                            da1 = da.isel({tdim: case_ti})
                        else:
                            da1 = da
                        yv = np.asarray(da1.values)
                        yv = yv[np.isfinite(yv)]
                        if yv.size:
                            ys_all.append(yv)
                    if ys_all:
                        ys = np.concatenate(ys_all)
                        amax = float(np.nanmax(np.abs(ys))) if ys.size else 1.0
                        linthresh = max(1e-12, 1e-3 * amax)
                    else:
                        linthresh = 1e-6
                    ax.set_yscale("symlog", linthresh=linthresh)
                else:
                    ax.set_yscale("linear")
            except Exception as e:
                win.set_status(f"Y-scale error for {name}: {e}", is_error=True)

            # Configure x-scale
            xscale_mode = win._xscale_by_var.get(name, "linear")
            try:
                if xscale_mode == "log":
                    ax.set_xscale("log")
                else:
                    ax.set_xscale("linear")
            except Exception as e:
                win.set_status(f"X-scale error for {name}: {e}", is_error=True)

            # Extract units from first dataset that has var
            units = None
            for c in win.cases.values():
                ds = c.ds
                if name in ds:
                    try:
                        units = ds[name].attrs.get("units", None)
                        if units:
                            break
                    except Exception:
                        pass

            # Track colors/styles per case for consistent differentiation
            case_colors: Dict[str, str] = {}
            overlay_vars = win._overlay_vars.get(name, [])
            linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
            datasets_by_colour = win._datasets_by_colour()

            # Build variable-to-color map for linestyle mode (same variable = same color across datasets)
            var_colors: Dict[str, str] = {}
            if not datasets_by_colour:
                import matplotlib.pyplot as plt
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                all_vars = [name] + list(overlay_vars)
                for vname in all_vars:
                    if vname not in var_colors:
                        var_colors[vname] = color_cycle[len(var_colors) % len(color_cycle)]

            for case_idx, c in enumerate(win.cases.values()):
                ds = c.ds
                if name not in ds:
                    continue
                da = ds[name]
                try:
                    case_ti = win._get_time_index_for_case(c)
                    if tdim is not None and tdim in da.dims:
                        da1 = da.isel({tdim: case_ti})
                    else:
                        da1 = da

                    if sdim and sdim in ds.coords:
                        x = np.asarray(ds[sdim].values)
                    else:
                        x = np.arange(int(ds.sizes.get(sdim, da1.size))) if sdim else np.arange(da1.size)

                    y = np.asarray(da1.values)
                    if win._guard_replace_enabled() and y.ndim == 1 and sdim in getattr(da1, "dims", ()):
                        x, y = guard_replace_1d_profile_xy(x, y)
                    # Downsample long traces for responsiveness
                    try:
                        x, y = downsample_xy(x, y, int(win._profile_max_points))
                    except Exception:
                        pass
                    if mode == "log":
                        y = np.where(y > 0, y, np.nan)
                    # Use variable name as label if overlays exist, else case label
                    if overlay_vars:
                        plot_label = f"{name} ({c.label})" if len(win.cases) > 1 else name
                    else:
                        plot_label = c.label

                    if datasets_by_colour:
                        # Datasets by colour: primary var solid, let matplotlib pick color
                        line, = ax.plot(x, y, label=plot_label, linestyle=linestyles[0])
                        case_colors[c.label] = line.get_color()
                    else:
                        # Datasets by linestyle: each dataset gets different linestyle, same color per variable
                        ls = linestyles[case_idx % len(linestyles)]
                        var_color = var_colors.get(name)
                        line, = ax.plot(x, y, label=plot_label, linestyle=ls, color=var_color)
                        case_colors[c.label] = (line.get_color(), ls)
                except Exception as e:
                    win.set_status(f"Plot error for {name}: {e}", is_error=True)

            # Plot overlay variables on the same axes
            for ov_idx, ov_name in enumerate(overlay_vars):
                for case_idx, c in enumerate(win.cases.values()):
                    ds = c.ds
                    if ov_name not in ds:
                        continue
                    try:
                        da = ds[ov_name]
                        case_ti = win._get_time_index_for_case(c)
                        if tdim is not None and tdim in da.dims:
                            da1 = da.isel({tdim: case_ti})
                        else:
                            da1 = da
                        if sdim and sdim in ds.coords:
                            x = np.asarray(ds[sdim].values)
                        else:
                            x = np.arange(int(ds.sizes.get(sdim, da1.size))) if sdim else np.arange(da1.size)
                        y = np.asarray(da1.values)
                        if win._guard_replace_enabled() and y.ndim == 1 and sdim in getattr(da1, "dims", ()):
                            x, y = guard_replace_1d_profile_xy(x, y)
                        try:
                            x, y = downsample_xy(x, y, int(win._profile_max_points))
                        except Exception:
                            pass
                        if mode == "log":
                            y = np.where(y > 0, y, np.nan)
                        ov_label = f"{ov_name} ({c.label})" if len(win.cases) > 1 else ov_name

                        if datasets_by_colour:
                            # Datasets by colour: overlays use different linestyle, same color per case
                            ls = linestyles[(ov_idx + 1) % len(linestyles)]
                            color = case_colors.get(c.label)
                            ax.plot(x, y, label=ov_label, linestyle=ls, color=color)
                        else:
                            # Datasets by linestyle: overlays use same color per variable, different linestyle per case
                            stored = case_colors.get(c.label)
                            ls = stored[1] if stored else linestyles[case_idx % len(linestyles)]
                            ov_color = var_colors.get(ov_name)
                            ax.plot(x, y, label=ov_label, linestyle=ls, color=ov_color)
                    except Exception as e:
                        win.set_status(f"Overlay plot error for {ov_name}: {e}", is_error=True)

            # X-point marker from options (1D only)
            try:
                ds0 = next(iter(win.cases.values())).ds
                xpt = get_option_float(ds0, ["length_xpt", "mesh:length_xpt"])
                if xpt is not None and np.isfinite(xpt):
                    ax.axvline(xpt, color="k", linewidth=1.0, alpha=0.7, linestyle="--")
                    ax.text(
                        xpt,
                        0.98,
                        "X-point",
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                        va="top",
                        ha="left",
                        fontsize=8,
                        color="k",
                        alpha=0.7,
                    )
            except Exception:
                pass

            # Update title to show overlaid variables
            overlay_vars = win._overlay_vars.get(name, [])
            if overlay_vars:
                title = f"{name} + {', '.join(overlay_vars)}"
            else:
                title = name
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(f"{units}" if units else "")
            ax.grid(True, which="both", alpha=0.3)
            # Show legend if multiple cases or overlays
            if len(win.cases) > 1 or overlay_vars:
                ax.legend(loc="upper left", fontsize=8)

            # Apply y-limit mode
            try:
                if ylim_mode == "auto":
                    # Force full autoscale to include all plotted data (including overlays)
                    ax.relim()
                    ax.autoscale(enable=True, axis='y', tight=False)
                    ax.autoscale_view(scalex=False, scaley=True)
                elif ylim_mode == "final":
                    ymin, ymax = win._compute_ylim_for_final(name, tdim, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
                elif ylim_mode == "global" or ylim_mode == "max":
                    ymin, ymax = win._compute_ylim_for_global(name, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
            except Exception:
                pass

        # X label on bottom-most axis in each column
        for i, ax in enumerate(axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if is_bottom:
                ax.set_xlabel(r"S$_\parallel$ (m)")

        # Sync overlay buttons to the current subplot grid.
        win._sync_overlay_buttons(vars_to_plot=vars_to_plot, axes=axes)
        # Ensure toolbar Home resets to the default view (not the preserved zoom)
        try:
            win._reset_toolbar_home(getattr(win, "toolbar", None))
        except Exception:
            pass
        # Restore user zoom/pan if applicable
        try:
            win._restore_view_2d(views=_view_restore_1d, vars_to_plot=vars_to_plot, axes=axes)
        except Exception:
            pass
        win.canvas.draw_idle()

    # ---------- Time history plotting ----------
