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
from holoshed.plotting.common import get_poloidal_profile
from holoshed.models import LoadedCase

def redraw_poloidal(win):
        win._update_time_readout()

        # If nothing relevant changed since last draw, don't rebuild the figure.
        _view_restore = None
        try:
            case = win._primary_case()
            ti = win._get_time_index_for_case(case) if case else -1
            region = str(win.pol_region_combo.currentText() or "outer_lower")
            sepadd = int(win.pol_sepadd_spin.value())
            use_spol = bool(getattr(win, "pol_use_spol_check", None).isChecked()) if hasattr(win, "pol_use_spol_check") else False
            xpt_mode = str(getattr(win, "pol_xpoint_combo", None).currentText() or "Bpxy valley") if hasattr(win, "pol_xpoint_combo") else "Bpxy valley"
            vars_to_plot = tuple(win.selected_vars)
            modes = tuple((v, win._yscale_by_var.get(v, "linear"), win._ylim_mode_by_var.get(v, "auto")) for v in vars_to_plot)
            # Include overlay variables in state key so view isn't restored when overlays change
            overlays = tuple((v, tuple(win._overlay_vars.get(v, []))) for v in vars_to_plot)
            state_key = ("pol", getattr(case, "label", None), ti, region, sepadd, use_spol, xpt_mode, vars_to_plot, modes, overlays)
            if win._last_draw_state_2d.get("pol") == state_key and win.pol_figure.axes:
                try:
                    win._position_overlay_buttons_pol()
                except Exception:
                    pass
                win.pol_canvas.draw_idle()
                return
            # Preserve zoom/pan across time changes (same config, different ti)
            state_no_ti = ("pol", getattr(case, "label", None), region, sepadd, use_spol, xpt_mode, vars_to_plot, modes, overlays)
            _view_restore = win._maybe_capture_view_2d(kind="pol", state_key_no_ti=state_no_ti)
            win._last_draw_state_2d["pol"] = state_key
        except Exception:
            pass

        win.pol_figure.clear()
        try:
            win.pol_figure.set_facecolor("white")
        except Exception:
            pass

        if not win.cases:
            # No plots -> no overlay buttons
            win._clear_overlay_buttons_pol()
            ax = win.pol_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            win.pol_canvas.draw_idle()
            return

        vars_to_plot = list(win.selected_vars)
        if not vars_to_plot:
            # No plots -> no overlay buttons
            win._clear_overlay_buttons_pol()
            ax = win.pol_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No variables selected.", ha="center", va="center", transform=ax.transAxes)
            win.pol_canvas.draw_idle()
            return

        region = str(win.pol_region_combo.currentText() or "outer_lower")
        sepadd = int(win.pol_sepadd_spin.value())
        use_spol = bool(getattr(win, "pol_use_spol_check", None).isChecked()) if hasattr(win, "pol_use_spol_check") else False
        xpt_mode = str(getattr(win, "pol_xpoint_combo", None).currentText() or "Bpxy valley") if hasattr(win, "pol_xpoint_combo") else "Bpxy valley"
        xcol = "Spol" if use_spol else "Spar"
        xlab = (r"S$_{pol}$ (m)" if use_spol else r"S$_\parallel$ (m)")

        # Layout similar to 1D profiles
        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))
        # Extra vertical padding to avoid title/xlabel overlap between stacked subplots
        gs = win.pol_figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.60, wspace=0.30)
        axes = [win.pol_figure.add_subplot(gs[i % nrows, i // nrows]) for i in range(n)]

        # Reference: mark the X-point location along the field line.
        xline_rmin = None
        try:
            c0 = win._primary_case()
            if c0 is not None:
                ds0_t = win._ds_at_time(c0)
                ti0 = win._get_time_index_for_case(c0)
                ck0 = (c0.label, ti0, region, sepadd)
                df0 = win._pol_cache.get(ck0)
                if df0 is None:
                    try:
                        # params=[] still returns geometry columns like Spar/R
                        df0 = get_poloidal_profile(c0, time_index=ti0, region=region, sepadd=sepadd, params=[])
                    except Exception:
                        df0 = None
                    win._pol_cache[ck0] = df0
                # Ensure Bpxy/Bxy exist in the cached df (incremental add)
                if df0 is not None:
                    try:
                        need_bpxy = "Bpxy" not in df0.columns
                    except Exception:
                        need_bpxy = True
                    if need_bpxy:
                        try:
                            df_new = get_poloidal_profile(c0, time_index=ti0, region=region, sepadd=sepadd, params=["Bpxy"])
                            if df_new is not None and "Bpxy" in df_new:
                                try:
                                    df0["Bpxy"] = df_new["Bpxy"].values
                                except Exception:
                                    pass
                            win._pol_cache[ck0] = df0
                        except Exception:
                            pass
                    try:
                        need_bxy = "Bxy" not in df0.columns
                    except Exception:
                        need_bxy = True
                    if need_bxy:
                        try:
                            df_new = get_poloidal_profile(c0, time_index=ti0, region=region, sepadd=sepadd, params=["Bxy"])
                            if df_new is not None and "Bxy" in df_new:
                                try:
                                    df0["Bxy"] = df_new["Bxy"].values
                                except Exception:
                                    pass
                            win._pol_cache[ck0] = df0
                        except Exception:
                            pass
                # Need geometry coordinate for the chosen x-axis.
                if df0 is not None and "R" in df0 and ("Spar" in df0 or "Spol" in df0):
                    r = np.asarray(df0["R"].values)
                    s = np.asarray(df0.get(xcol, df0.get("Spar")).values)  # type: ignore[union-attr]
                    if r.size and s.size:
                        rr = np.asarray(r, dtype=float)
                        ss = np.asarray(s, dtype=float)
                        m = np.isfinite(rr) & np.isfinite(ss)
                        if np.any(m):
                            s_use = ss[m]
                            if xpt_mode.strip().lower().startswith("min r"):
                                try:
                                    rr_use = rr[m]
                                    j0 = int(np.nanargmin(rr_use))
                                    if 0 <= j0 < s_use.size:
                                        xline_rmin = float(s_use[j0])
                                except Exception:
                                    pass
                            elif xpt_mode.strip().lower().startswith("bxy"):
                                # Outer: max(Bxy), Inner: min(Bxy)
                                try:
                                    bxy = np.asarray(df0["Bxy"].values, dtype=float)
                                    bxy_use = bxy[m]
                                    mb = np.isfinite(bxy_use)
                                    if np.any(mb):
                                        b2 = np.asarray(bxy_use[mb], dtype=float)
                                        s2 = np.asarray(s_use[mb], dtype=float)
                                        # Avoid target-side extrema: search only the middle 75%
                                        n2 = int(b2.size)
                                        lo = int(np.floor(0.125 * n2))
                                        hi = int(np.ceil(0.875 * n2))
                                        if hi <= lo:
                                            lo, hi = 0, n2
                                        b_mid = b2[lo:hi]
                                        s_mid = s2[lo:hi]
                                        if b_mid.size:
                                            if str(region).startswith("outer"):
                                                jmid = int(np.nanargmax(b_mid))
                                            else:
                                                jmid = int(np.nanargmin(b_mid))
                                            jmid = int(np.clip(jmid, 0, b_mid.size - 1))
                                            xline_rmin = float(s_mid[jmid])
                                except Exception:
                                    pass
                            else:
                                # Default: Bpxy "valley" minimum (until it starts rising again)
                                try:
                                    bp = np.asarray(df0["Bpxy"].values, dtype=float)
                                    bp_use = bp[m]
                                    mbp = np.isfinite(bp_use)
                                    if np.any(mbp):
                                        bp2 = np.asarray(bp_use[mbp], dtype=float)
                                        s2 = np.asarray(s_use[mbp], dtype=float)
                                        j = xpoint_idx_bpxy_valley(bp2)
                                        if j is not None and 0 <= int(j) < s2.size:
                                            xline_rmin = float(s2[int(j)])
                                except Exception:
                                    # Fallback: closest-to-zero R
                                    try:
                                        rr_use = rr[m]
                                        j0 = int(np.nanargmin(np.abs(rr_use)))
                                        if 0 <= j0 < s_use.size:
                                            xline_rmin = float(s_use[j0])
                                    except Exception:
                                        pass
        except Exception:
            xline_rmin = None

        # Batch extract once per case/time/region/sepadd for speed
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
            # Incremental cache: keep one df per (case,time,region,sepadd) and extend
            ck = (c.label, ti, region, sepadd)
            df = win._pol_cache.get(ck)
            # Include overlay vars in params for extraction
            params = selector_params_only(list(vars_to_plot) + list(all_overlay_vars))
            if df is None:
                try:
                    df = get_poloidal_profile(c, time_index=ti, region=region, sepadd=sepadd, params=list(params))
                except Exception as e:
                    win.set_status(f"Poloidal extract failed: {e}", is_error=True)
                    df = None
                win._pol_cache[ck] = df
            else:
                try:
                    missing = [v for v in params if v not in df.columns]
                except Exception:
                    missing = list(params)
                if missing:
                    try:
                        df_new = get_poloidal_profile(c, time_index=ti, region=region, sepadd=sepadd, params=list(missing))
                        # Merge missing columns by index (same length/order expected)
                        for v in list(missing):
                            if v in df_new:
                                try:
                                    df[v] = df_new[v].values
                                except Exception:
                                    pass
                        win._pol_cache[ck] = df
                    except Exception as e:
                        win.set_status(f"Poloidal extract failed: {e}", is_error=True)
            if df is None:
                continue

            try:
                x = np.asarray(df.get(xcol, df.get("Spar")).values)  # type: ignore[union-attr]
            except Exception:
                x = np.asarray(df["Spar"].values)

            for ax, name in zip(axes, vars_to_plot):
                overlay_vars = win._overlay_vars.get(name, [])
                if overlay_vars:
                    title = f"{name} + {', '.join(overlay_vars)} ({region}, sepadd={sepadd})"
                else:
                    title = f"{name} ({region}, sepadd={sepadd})"
                ax.set_title(title, fontsize=10)
                ax.grid(True, alpha=0.3)
                try:
                    y = np.asarray(df[name].values)
                except Exception:
                    continue
                # If log scale, mask non-positive values
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
                    try:
                        ov_y = np.asarray(df[ov_name].values)
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
                    if name in c.ds:
                        units = c.ds[name].attrs.get("units", None)
                        if units:
                            break
                except Exception:
                    continue
            ax.set_ylabel(f"{units}" if units else "")
            # Show legend if multiple cases or overlays
            overlay_vars = win._overlay_vars.get(name, [])
            if len(win.cases) > 1 or overlay_vars:
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(xlab)
            if xline_rmin is not None and np.isfinite(xline_rmin):
                try:
                    ax.axvline(xline_rmin, color="k", linewidth=1.0, linestyle="--")
                    ax.text(
                        xline_rmin,
                        0.98,
                        "X-point",
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                        va="top",
                        ha="left",
                        fontsize=8,
                        color="k",
                    )
                except Exception:
                    pass

            # Apply ylim mode (match 1D behavior: auto/final/global)
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
                        ymin, ymax = win._compute_ylim_poloidal_extracted(
                            case=c0,
                            region=region,
                            sepadd=sepadd,
                            varname=name,
                            yscale=win._yscale_by_var.get(name, "linear"),
                            mode=ylim_mode,
                        )
                        if ymin is not None and ymax is not None:
                            ax.set_ylim(ymin, ymax)
            except Exception:
                pass

        # Overlay buttons for yscale/ylim (same as 1D)
        win._sync_overlay_buttons_pol(vars_to_plot=vars_to_plot, axes=axes)
        # Ensure toolbar Home resets to the default view (not the preserved zoom)
        try:
            win._reset_toolbar_home(getattr(win, "pol_toolbar", None))
        except Exception:
            pass
        # Restore user zoom/pan (if applicable)
        try:
            win._restore_view_2d(views=_view_restore, vars_to_plot=vars_to_plot, axes=axes)
        except Exception:
            pass
        try:
            win.pol_figure.tight_layout()
        except Exception:
            pass
        win.pol_canvas.draw_idle()

