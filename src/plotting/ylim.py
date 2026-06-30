"""Y-limit computation for 2D extracted 1D profiles."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.models import LoadedCase
from src.plotting.common import get_poloidal_profile, get_radial_profile


def with_margin(ymin: float, ymax: float) -> Tuple[float, float]:
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return ymin, ymax
    if ymax > ymin:
        m = 0.05 * (ymax - ymin)
        return ymin - m, ymax + m
    m = 0.1 * (abs(ymax) if ymax != 0 else 1.0)
    return ymin - m, ymax + m


def compute_ylim_poloidal_extracted(
    win,
    *,
    case: LoadedCase,
    region: str,
    sepadd: int,
    varname: str,
    yscale: str,
    mode: str,
) -> Tuple[Optional[float], Optional[float]]:
    overlay_vars = tuple(win._overlay_vars.get(varname, []))
    all_vars = [varname] + list(overlay_vars)
    key = ("pol", case.label, region, int(sepadd), varname, overlay_vars, yscale, mode, int(case.n_time))
    hit = win._pol_ylim_cache.get(key)
    if hit is not None:
        return hit

    def _accum_from_df(df, vname, cur_min, cur_max):
        try:
            y = np.asarray(df[vname].values, dtype=float)
        except Exception:
            return cur_min, cur_max
        if yscale == "log":
            y = y[y > 0]
        y = y[np.isfinite(y)]
        if y.size == 0:
            return cur_min, cur_max
        ym, yM = float(np.nanmin(y)), float(np.nanmax(y))
        if cur_min is None or ym < cur_min:
            cur_min = ym
        if cur_max is None or yM > cur_max:
            cur_max = yM
        return cur_min, cur_max

    ymin: Optional[float] = None
    ymax: Optional[float] = None

    if mode == "global":
        mode = "max"
    if mode == "final":
        tis = [max(0, int(case.n_time) - 1)]
    elif mode == "max":
        tis = list(range(int(case.n_time)))
    else:
        return None, None

    for ti in tis:
        ck = (case.label, int(ti), region, int(sepadd))
        df = win._pol_cache.get(ck)
        if df is None:
            try:
                df = get_poloidal_profile(
                    case,
                    time_index=int(ti),
                    region=region,
                    sepadd=int(sepadd),
                    params=all_vars,
                )
            except Exception:
                df = None
            win._pol_cache[ck] = df
        else:
            try:
                missing = [v for v in all_vars if v not in df.columns]
            except Exception:
                missing = all_vars
            if missing:
                try:
                    df_new = get_poloidal_profile(
                        case,
                        time_index=int(ti),
                        region=region,
                        sepadd=int(sepadd),
                        params=missing,
                    )
                    if df_new is not None:
                        for v in missing:
                            if v in df_new:
                                df[v] = df_new[v].values
                    win._pol_cache[ck] = df
                except Exception:
                    pass
        if df is None:
            continue
        for vname in all_vars:
            ymin, ymax = _accum_from_df(df, vname, ymin, ymax)

    if ymin is None or ymax is None:
        out = (None, None)
    else:
        a, b = with_margin(float(ymin), float(ymax))
        out = (a, b)
    win._pol_ylim_cache[key] = out
    return out


def compute_ylim_radial_extracted(
    win,
    *,
    case: LoadedCase,
    region: str,
    varname: str,
    yscale: str,
    mode: str,
) -> Tuple[Optional[float], Optional[float]]:
    overlay_vars = tuple(win._overlay_vars.get(varname, []))
    all_vars = [varname] + list(overlay_vars)
    key = ("rad", case.label, region, varname, overlay_vars, yscale, mode, int(case.n_time))
    hit = win._rad_ylim_cache.get(key)
    if hit is not None:
        return hit

    def _accum_from_df(df, vname, cur_min, cur_max):
        try:
            y = np.asarray(df[vname].values, dtype=float)
        except Exception:
            return cur_min, cur_max
        if yscale == "log":
            y = y[y > 0]
        y = y[np.isfinite(y)]
        if y.size == 0:
            return cur_min, cur_max
        ym, yM = float(np.nanmin(y)), float(np.nanmax(y))
        if cur_min is None or ym < cur_min:
            cur_min = ym
        if cur_max is None or yM > cur_max:
            cur_max = yM
        return cur_min, cur_max

    ymin: Optional[float] = None
    ymax: Optional[float] = None

    if mode == "global":
        mode = "max"
    if mode == "final":
        tis = [max(0, int(case.n_time) - 1)]
    elif mode == "max":
        tis = list(range(int(case.n_time)))
    else:
        return None, None

    for ti in tis:
        ck = (case.label, int(ti), region)
        df = win._rad_cache.get(ck)
        if df is None:
            try:
                df = get_radial_profile(
                    case,
                    time_index=int(ti),
                    region=region,
                    params=all_vars,
                )
            except Exception:
                df = None
            win._rad_cache[ck] = df
        else:
            try:
                missing = [v for v in all_vars if v not in df.columns]
            except Exception:
                missing = all_vars
            if missing:
                try:
                    df_new = get_radial_profile(
                        case,
                        time_index=int(ti),
                        region=region,
                        params=missing,
                    )
                    if df_new is not None:
                        for v in missing:
                            if v in df_new:
                                df[v] = df_new[v].values
                    win._rad_cache[ck] = df
                except Exception:
                    pass
        if df is None:
            continue
        for vname in all_vars:
            ymin, ymax = _accum_from_df(df, vname, ymin, ymax)

    if ymin is None or ymax is None:
        out = (None, None)
    else:
        a, b = with_margin(float(ymin), float(ymax))
        out = (a, b)
    win._rad_ylim_cache[key] = out
    return out
