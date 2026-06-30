"""Hermes xarray dataset helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np

def infer_time_dim(ds) -> Optional[str]:
    for cand in ("t", "time"):
        if cand in ds.dims:
            return cand
    # fall back: any 1D dimension with monotonic coordinate
    for d in ds.dims:
        if d in ds.coords and ds[d].ndim == 1:
            return d
    return None


def infer_spatial_dim(ds) -> str:
    # Hermes-3 1D typically uses "pos"
    for cand in ("pos", "y", "x", "s"):
        if cand in ds.dims:
            return cand
    # fall back: choose a non-time dimension
    tdim = infer_time_dim(ds)
    for d in ds.dims:
        if d != tdim:
            return d
    # last resort
    return list(ds.dims)[0]


def is_plottable_1d_var(da, spatial_dim: str, time_dim: Optional[str]) -> bool:
    dims = tuple(da.dims)
    if spatial_dim not in dims:
        return False
    if len(dims) == 1 and dims[0] == spatial_dim:
        return True
    if time_dim is None:
        return False
    if len(dims) == 2 and set(dims) == {time_dim, spatial_dim}:
        return True
    return False


def list_plottable_vars(ds, spatial_dim: str, time_dim: Optional[str]) -> List[str]:
    out: List[str] = []
    # Include derived/geometry coordinates (e.g. R, Z) as plottable options too.
    # Avoid listing the coordinate axes themselves.
    ignore = {spatial_dim}
    if time_dim:
        ignore.add(time_dim)
    for name in getattr(ds, "variables", {}).keys():
        if name in ignore:
            continue
        try:
            da = ds[name]
            if is_plottable_1d_var(da, spatial_dim=spatial_dim, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(set(out))


def is_plottable_2d_var(da, time_dim: Optional[str]) -> bool:
    """
    Heuristic for Hermes-3 2D fields.

    sdtools uses dims ('x', 'theta') (optionally with 't') for 2D tokamak data.
    """
    dims = tuple(getattr(da, "dims", ()))
    if "x" not in dims or "theta" not in dims:
        return False
    if time_dim is None:
        # allow static fields
        return set(dims) >= {"x", "theta"}
    # common cases: (x,theta) or (t,x,theta)
    if set(dims) >= {"x", "theta"} and (time_dim in dims or time_dim not in dims):
        return True
    return False


def is_radial_only_2d_var(da, time_dim: Optional[str]) -> bool:
    """
    Check for derived variables that only have radial ('x') dimension.

    These are reduced quantities like detachment_location that have been
    computed along the poloidal direction, leaving only radial variation.
    They can be plotted in the radial profiles view.
    """
    dims = tuple(getattr(da, "dims", ()))
    if "x" not in dims:
        return False
    # Don't include full 2D vars (those go through is_plottable_2d_var)
    if "theta" in dims or "y" in dims:
        return False
    # Accept (x,) only
    if len(dims) == 1 and dims[0] == "x":
        return True
    # Accept (x, time_dim) with any common time dimension name
    if len(dims) == 2:
        other_dim = [d for d in dims if d != "x"][0]
        # Check if the other dimension is a time dimension
        if time_dim and other_dim == time_dim:
            return True
        # Also accept common time dimension names
        if other_dim in ("t", "time"):
            return True
    return False


def list_plottable_vars_2d(ds, time_dim: Optional[str]) -> List[str]:
    out: List[str] = []
    # Include derived/geometry coordinates (e.g. R, Z) as plottable options too.
    ignore = {"x", "theta", "y"}
    if time_dim:
        ignore.add(time_dim)
    for name in getattr(ds, "variables", {}).keys():
        if name in ignore:
            continue
        try:
            da = ds[name]
            # Include both full 2D vars and radial-only derived vars
            if is_plottable_2d_var(da, time_dim=time_dim) or is_radial_only_2d_var(da, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(set(out))


def selector_params_only(vars_to_plot: List[str]) -> List[str]:
    """
    sdtools selectors always provide geometry columns like Spar/Spol/R/Z.
    Avoid requesting those as "params" to keep selectors robust.
    """
    geom = {"Spar", "Spol", "Srad", "R", "Z"}
    return [v for v in vars_to_plot if v not in geom]


def xpoint_idx_bpxy_valley(bp: np.ndarray) -> Optional[int]:
    """
    Heuristic X-point index from a 1D Bpxy trace along the field line.

    We walk from the start and keep updating the minimum |Bpxy| until the signal
    has started rising again for a few consecutive points. This avoids picking the
    target-side low-field region after the X-point.
    """
    a = np.asarray(bp, dtype=float)
    m = np.isfinite(a)
    if not np.any(m):
        return None
    a = np.abs(a[m])
    if a.size < 3:
        return int(np.argmin(a))

    best_i = 0
    best = float(a[0])
    rise = 0
    # how many consecutive rises indicate we're past the minimum
    rise_needed = 3
    eps = 0.0

    prev = float(a[0])
    for i in range(1, int(a.size)):
        cur = float(a[i])
        if cur < best:
            best = cur
            best_i = i
            rise = 0
        # Rising after we have a minimum
        if i > best_i:
            if cur > prev + eps:
                rise += 1
            else:
                rise = 0
            if rise >= rise_needed and (i - best_i) >= 2:
                break
        prev = cur
    return int(best_i)




def guard_replace_1d_profile_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Guard-replace for *profiles* (x,y arrays).

    Behavior (as requested):
    - Strip the *outer* guard cells (drop first and last points)
    - Replace the remaining inner guard cell values by averaging with the adjacent
      last/first real cell, so the endpoints represent the last real face values.

    Assumed indexing (common in Hermes-3 1D outputs):
    - index 0 and -1 are the unused "outer" guards
    - index 1 is the inlet-side inner guard, index 2 is first real cell
    - index -2 is the target-side inner guard, index -3 is last real cell
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 4:
        return x, y

    # Strip outer guard cells
    xx = x[1:-1].copy()
    yy = y[1:-1].copy()

    try:
        # Inlet face: average (inner guard, first real)
        yy[0] = 0.5 * (y[1] + y[2])
        xx[0] = 0.5 * (x[1] + x[2])

        # Target face: average (inner guard, last real)
        yy[-1] = 0.5 * (y[-2] + y[-3])
        xx[-1] = 0.5 * (x[-2] + x[-3])
    except Exception:
        return x, y

    return xx, yy


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if max_points is None or max_points <= 0:
        return x, y
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        return x, y
    n = int(x.size)
    if n <= int(max_points):
        return x, y
    stride = int(np.ceil(n / float(max_points)))
    return x[::stride], y[::stride]


def parse_optional_float(text: str) -> Optional[float]:
    """
    Parse a float from a text box. Empty/'auto' -> None.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() == "auto":
        return None
    try:
        return float(s)
    except Exception:
        return None


def get_option_float(ds, keys: List[str]) -> Optional[float]:
    """
    Best-effort extraction of a float from ds.options for a list of candidate keys.
    Supports both nested options (ds.options["mesh"]["length_xpt"]) and flat keys.
    """
    try:
        opt = getattr(ds, "options", None)
        if opt is None:
            return None
    except Exception:
        return None

    for k in keys:
        # Nested form: "mesh:length_xpt"
        if ":" in k:
            sec, name = k.split(":", 1)
            try:
                v = opt[sec][name]
                return float(v)
            except Exception:
                pass
        # Flat form: "length_xpt"
        try:
            v = opt[k]
            return float(v)
        except Exception:
            pass
    return None


def format_case_label(case_path: str) -> str:
    p = Path(case_path).expanduser().resolve()
    return p.name or str(p)


def pick_bout_output_for_probe(case_dir: Path) -> Path:
    """Pick a representative BOUT output file for cheap metadata probing."""
    squash = case_dir / "BOUT.squash.nc"
    if squash.exists():
        return squash
    dmp_single = case_dir / "BOUT.dmp.nc"
    if dmp_single.exists():
        return dmp_single
    d0 = case_dir / "BOUT.dmp.0.nc"
    if d0.exists():
        return d0
    dmps = sorted(case_dir.glob("BOUT.dmp.*.nc"))
    if dmps:
        return dmps[0]
    raise FileNotFoundError(
        f"Could not find BOUT output in {case_dir} "
        "(expected BOUT.squash.nc, BOUT.dmp.nc, or BOUT.dmp.*.nc)."
    )


def should_use_squash_for_load(case_dir: Path) -> bool:
    try:
        if (case_dir / "BOUT.squash.nc").exists():
            return True
    except Exception:
        pass
    try:
        for _ in case_dir.glob("BOUT.dmp.*.nc"):
            return True
    except Exception:
        pass
    return False


def probe_is_2d_case(case_dir: Path) -> bool:
    probe_path = pick_bout_output_for_probe(case_dir)
    try:
        from netCDF4 import Dataset  # type: ignore

        with Dataset(str(probe_path), mode="r") as ds:
            dims = ds.dimensions
            if "x" not in dims:
                return False
            nx = int(len(dims["x"]))
            has_pol = ("theta" in dims) or ("y" in dims)
            if not has_pol:
                return False
            return nx > 1
    except Exception:
        try:
            import xarray as xr  # type: ignore

            with xr.open_dataset(str(probe_path), decode_cf=False, mask_and_scale=False) as ds:
                if "x" not in ds.dims:
                    return False
                nx = int(ds.sizes.get("x", 0))
                has_pol = ("theta" in ds.dims) or ("y" in ds.dims)
                if not has_pol:
                    return False
                return nx > 1
        except Exception as e:
            raise RuntimeError(
                f"Failed to probe case dimensionality from {probe_path}.\n"
                f"Tried netCDF4 and xarray. Original error:\n{e}"
            ) from e


def parse_mesh_grid_filename_from_bout_inp(inp_path: Path) -> Optional[str]:
    try:
        txt = inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    section = None
    for raw in txt:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^\[(.+?)\]\s*$", line)
        if m:
            section = m.group(1).strip().lower()
            continue
        if section != "mesh" or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip().lower() != "file":
            continue
        val = val.strip().strip('"').strip("'").strip()
        return val or None
    return None


def ensure_sdtools_2d_metadata(ds) -> None:
    """Back-fill ds.metadata keys expected by sdtools 2D selectors."""
    try:
        m = ds.metadata
        if not isinstance(m, dict):
            return
    except Exception:
        return
    try:
        dims = getattr(ds, "dims", {})
        if "x" not in dims or "theta" not in dims:
            return
    except Exception:
        pass
    need_midplanes = not all(k in m for k in ("omp_a", "omp_b", "imp_a", "imp_b"))
    try:
        topology = str(m.get("topology", "")).lower()
        MYG = int(m.get("MYG", 0))
        MXG = int(m.get("MXG", 0))
        j1_1 = int(m["jyseps1_1"])
        j1_2 = int(m["jyseps1_2"])
        j2_1 = int(m["jyseps2_1"])
        j2_2 = int(m["jyseps2_2"])
        ixseps1 = int(m.get("ixseps1", 0))
        ixseps2 = int(m.get("ixseps2", ixseps1))
    except Exception:
        topology = str(m.get("topology", "")).lower()
        MYG = int(m.get("MYG", 0) or 0)
        MXG = int(m.get("MXG", 0) or 0)
        j1_1 = j1_2 = j2_1 = j2_2 = 0
        ixseps1 = int(m.get("ixseps1", 0) or 0)
        ixseps2 = int(m.get("ixseps2", ixseps1) or ixseps1)
    if "single-null" in topology:
        targets = ["inner_lower", "outer_lower"]
    elif "double-null" in topology:
        targets = ["inner_lower", "outer_lower", "inner_upper", "outer_upper"]
    else:
        targets = list(m.get("targets") or [])
    num_targets = len(targets) if targets else 0
    m.setdefault("ixseps1g", ixseps1 - MXG)
    m.setdefault("ixseps2g", ixseps2 - MXG)
    if need_midplanes and num_targets:
        j1_1g = j1_1 + MYG
        j2_1g = j2_1 + MYG
        j1_2g = j1_2 + MYG * (num_targets - 1)
        j2_2g = j2_2 + MYG * (num_targets - 1)
        m.setdefault("j1_1g", j1_1g)
        m.setdefault("j2_1g", j2_1g)
        m.setdefault("j1_2g", j1_2g)
        m.setdefault("j2_2g", j2_2g)
        try:
            omp_a = int((j2_2g - j1_2g) / 2) + j1_2g
            omp_b = omp_a + 1
            imp_a = int((j2_1g - j1_1g) / 2) + j1_1g + 1
            imp_b = int((j2_1g - j1_1g) / 2) + j1_1g
            m.setdefault("omp_a", omp_a)
            m.setdefault("omp_b", omp_b)
            m.setdefault("imp_a", imp_a)
            m.setdefault("imp_b", imp_b)
        except Exception:
            pass
    if targets:
        m.setdefault("targets", targets)
    try:
        if "dpol" not in ds and "dl" in ds:
            ds["dpol"] = ds["dl"]
    except Exception:
        pass
