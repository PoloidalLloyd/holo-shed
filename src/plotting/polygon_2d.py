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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.models import LoadedCase


def _plot_polygon_hermes(win, case, ax, *, var, grid_only, logscale, cmap, vmin, vmax, n_cases, i):
    ds_t = win._ds_at_time(case)
    data = ds_t[var]
    short_label = case.label if len(case.label) <= 25 else case.label[:22] + "..."

    if grid_only:
        data.hermesm.clean_guards().bout.polygon(
            ax=ax,
            grid_only=True,
            linecolor="k",
            linewidth=0.2,
            antialias=True,
            separatrix=False,
            targets=False,
            add_colorbar=False,
        )
        ax.set_title(f"{short_label}\n(grid)", fontsize=10)
        return None, None

    data.hermesm.clean_guards().bout.polygon(
        ax=ax,
        cmap=cmap,
        linecolor=(0, 0, 0, 0.15),
        linewidth=0,
        antialias=True,
        logscale=logscale,
        vmin=vmin,
        vmax=vmax,
        separatrix=True,
        separatrix_kwargs={"linewidth": 0.2, "color": "k"},
        targets=False,
        add_colorbar=False,
    )
    ax.set_title(f"{short_label}\n{var}", fontsize=10)
    polys = ax.collections[-1] if ax.collections else None
    cbar = None
    if polys is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        label = var
        try:
            if "units" in data.attrs:
                label = f"{label} [{data.attrs['units']}]"
        except Exception:
            pass
        cbar = win.poly_figure.colorbar(polys, cax=cax, label=label if i == n_cases - 1 else "")
        try:
            cax.grid(which="both", visible=False)
        except Exception:
            pass
    return polys, cbar


def _plot_polygon_solps(win, case, ax, *, var, grid_only, logscale, cmap, vmin, vmax, n_cases, i):
    backend = case.backend
    if backend is None:
        raise RuntimeError(f"Case {case.label} has no backend attached")
    ti = win._get_time_index_for_case(case)
    backend.plot_2d_field(
        case,
        param=var,
        ax=ax,
        time_index=ti,
        grid_only=grid_only,
        logscale=logscale,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )
    short_label = case.label if len(case.label) <= 25 else case.label[:22] + "..."
    title = f"{short_label}\n(grid)" if grid_only else f"{short_label}\n{var}"
    ax.set_title(title, fontsize=10)
    polys = ax.collections[-1] if ax.collections else None
    cbar = None
    if polys is not None and not grid_only:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = win.poly_figure.colorbar(polys, cax=cax, label=var if i == n_cases - 1 else "")
        try:
            cax.grid(which="both", visible=False)
        except Exception:
            pass
    return polys, cbar


def plot_region2d_background(
    win,
    case: LoadedCase,
    ax,
    *,
    var: str,
    logscale: bool,
    cmap: str,
    vmin,
    vmax,
):
    """Draw a 2D field background for the region-overlay popout (no colorbar)."""
    if getattr(case, "backend_kind", "hermes") == "solps":
        backend = case.backend
        if backend is None:
            raise RuntimeError(f"Case {case.label} has no backend attached")
        ti = win._get_time_index_for_case(case)
        backend.plot_2d_field(
            case,
            param=var,
            ax=ax,
            time_index=ti,
            grid_only=False,
            logscale=logscale,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            separatrix=True,
        )
        return ax.collections[-1] if ax.collections else None

    ds_t = win._ds_at_time(case)
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
    return ax.collections[-1] if ax.collections else None


def redraw_polygon(win):
        win._update_time_readout()

        # Get all 2D cases (limit to 3 for comparison)
        cases_2d = [c for c in win.cases.values() if getattr(c, 'is_2d', False)][:3]
        if not cases_2d:
            win.poly_figure.clear()
            ax = win.poly_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No 2D dataset loaded.", ha="center", va="center", transform=ax.transAxes)
            win.poly_canvas.draw_idle()
            return

        var = str(win.poly_var_combo.currentText() or "").strip()
        if not var:
            win.poly_figure.clear()
            ax = win.poly_figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Select a variable.", ha="center", va="center", transform=ax.transAxes)
            win.poly_canvas.draw_idle()
            return

        grid_only = bool(win.poly_grid_only_check.isChecked())
        logscale = bool(win.poly_log_check.isChecked())
        cmap = str(win.poly_cmap_combo.currentText() or "Spectral_r")
        vmin = win._poly_vmin_active
        vmax = win._poly_vmax_active

        n_cases = len(cases_2d)
        # Build state key for settings (excludes time indices - those change frequently)
        case_labels = tuple(c.label for c in cases_2d)
        settings_state = (case_labels, var, grid_only, logscale, cmap, vmin, vmax)

        # Reuse the PatchCollections when only time index changes (fast update; Hermes only).
        all_hermes = all(getattr(c, "backend_kind", "hermes") != "solps" for c in cases_2d)
        if (all_hermes and
            win._poly_plot_state == settings_state and
            win._poly_axes_multi is not None and
            win._poly_polys_multi is not None and
            len(win._poly_axes_multi) == n_cases):
            try:
                for i, case in enumerate(cases_2d):
                    ds_t = win._ds_at_time(case)
                    data = ds_t[var].hermesm.clean_guards()
                    if win._poly_polys_multi[i] is not None:
                        win._poly_polys_multi[i].set_array(np.asarray(data.data).flatten())
                        if win._poly_cbars_multi and i < len(win._poly_cbars_multi) and win._poly_cbars_multi[i] is not None:
                            try:
                                win._poly_cbars_multi[i].update_normal(win._poly_polys_multi[i])
                            except Exception:
                                pass
                win.poly_canvas.draw_idle()
                return
            except Exception:
                # fall through to rebuild
                pass

        # Otherwise rebuild the plot (variable/settings changed or cases changed)
        win.poly_figure.clear()
        try:
            win.poly_figure.set_facecolor("white")
        except Exception:
            pass

        # Initialize tracking lists for multi-case support
        win._poly_axes_multi = []
        win._poly_polys_multi = []
        win._poly_cbars_multi = []

        # Create side-by-side subplots (1 row, N columns)
        for i, case in enumerate(cases_2d):
            ax = win.poly_figure.add_subplot(1, n_cases, i + 1)
            win._poly_axes_multi.append(ax)

            try:
                if getattr(case, "backend_kind", "hermes") == "solps":
                    polys, cbar = _plot_polygon_solps(
                        win,
                        case,
                        ax,
                        var=var,
                        grid_only=grid_only,
                        logscale=logscale,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        n_cases=n_cases,
                        i=i,
                    )
                else:
                    polys, cbar = _plot_polygon_hermes(
                        win,
                        case,
                        ax,
                        var=var,
                        grid_only=grid_only,
                        logscale=logscale,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        n_cases=n_cases,
                        i=i,
                    )
                win._poly_polys_multi.append(polys)
                win._poly_cbars_multi.append(cbar)
            except Exception as e:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"2D plot failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
                win._poly_polys_multi.append(None)
                win._poly_cbars_multi.append(None)

        # Adjust layout for side-by-side plots
        try:
            win.poly_figure.tight_layout()
        except Exception:
            pass

        # Save state for fast time updates
        win._poly_plot_state = settings_state
        # Keep legacy single-case attributes for backwards compatibility
        win._poly_ax = win._poly_axes_multi[0] if win._poly_axes_multi else None
        win._poly_polys = win._poly_polys_multi[0] if win._poly_polys_multi else None
        win._poly_cbar = win._poly_cbars_multi[0] if win._poly_cbars_multi else None

        win.poly_canvas.draw_idle()

