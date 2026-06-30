"""SOLPS case backend (balance.nc via sdtools code_comparison.solps_pp)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataset_utils import format_case_label, selector_params_only, params_with_requested_geometry
from src.models import LoadedCase
from src.paths import ensure_sdtools_on_path

# Variables that are useful in holo-shed but not 2D grid fields — hide from pickers.
_SKIP_VAR_PREFIXES = ("left", "right", "species", "jsep", "bb", "crx", "cry", "gs", "conn")
_SKIP_VAR_NAMES = {
    "leftix",
    "rightix",
    "topix",
    "bottomix",
    "resignore",
    "resplim",
    "hx",
    "hy",
    "hz",
    "vol",
    "R",
    "Z",
    "Btot",
    "Bpol",
    "Btor",
    "Bx",
    "By",
    "Bz",
    "Babs",
    "bb",
}


def _import_solps_case():
    ensure_sdtools_on_path()
    try:
        from code_comparison.solps_pp import SOLPScase  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Could not import SOLPScase from sdtools.\n"
            "Ensure the sdtools submodule is initialised:\n"
            "    git submodule update --init --recursive\n"
            f"Original error: {e}"
        ) from e
    return SOLPScase


def _slc(case: LoadedCase):
    if case.ds is None:
        raise RuntimeError(f"Case {case.label} has no SOLPS data loaded")
    return case.ds


def _is_plottable_2d(name: str, arr: Any) -> bool:
    if name in _SKIP_VAR_NAMES:
        return False
    if any(name.startswith(p) for p in _SKIP_VAR_PREFIXES):
        return False
    try:
        shape = getattr(arr, "shape", None)
        if shape is None or len(shape) != 2:
            return False
        if int(shape[0]) < 2 or int(shape[1]) < 2:
            return False
    except Exception:
        return False
    return True


def _resolve_param(slc: Any, param: str) -> str:
    """Map holo-shed variable name to a balance-file key."""
    if param in slc.bal:
        return param
    lower = param.lower()
    if lower in slc.bal:
        return lower
    raise KeyError(f"Parameter {param!r} not found in SOLPS balance data")


class SolpsBackend:
    kind = "solps"

    def __init__(self) -> None:
        self._SOLPScase = _import_solps_case()

    def load(self, path: Path, *, load_cls: Any = None) -> LoadedCase:
        case_dir = path.expanduser().resolve()
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case path is not a directory: {case_dir}")
        balance = case_dir / "balance.nc"
        if not balance.exists():
            raise FileNotFoundError(f"SOLPS case missing balance.nc: {balance}")

        case_path = str(case_dir)
        label = format_case_label(case_path)
        slc = self._SOLPScase(case_path)
        n_time = self._detect_n_time(slc)

        return LoadedCase(
            label=label,
            case_path=case_path,
            ds=slc,
            n_time=n_time,
            is_2d=True,
            backend_kind="solps",
            backend=self,
        )

    def _detect_n_time(self, slc: Any) -> int:
        """Steady-state balance files have a single time slice; extend when transient support lands."""
        return 1

    def list_variables(self, case: LoadedCase) -> List[str]:
        slc = _slc(case)
        preferred = [
            "Te",
            "Td+",
            "Ne",
            "Pe",
            "Pd+",
            "Na",
            "Nm",
            "Nn",
            "Ta",
            "Tn",
            "Tm",
            "Pa",
            "Pm",
            "Pn",
            "Vd+",
            "NVd+",
            "M",
            "fhx_total",
            "fhx_density",
            "fhtx",
            "fhtx_density",
            "te",
            "ti",
            "ne",
        ]
        out: List[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name in seen:
                return
            arr = slc.bal.get(name)
            if arr is not None and _is_plottable_2d(name, arr):
                out.append(name)
                seen.add(name)

        for name in preferred:
            if name in slc.bal:
                add(name)
        for name in sorted(slc.bal.keys()):
            add(name)
        return out

    def time_coordinate(self, case: LoadedCase) -> Tuple[str, np.ndarray]:
        slc = _slc(case)
        for key in ("times", "time", "t"):
            if key in slc.bal:
                try:
                    vals = np.asarray(slc.bal[key], dtype=float).ravel()
                    if vals.size:
                        return key, vals
                except Exception:
                    pass
        n = max(1, int(case.n_time))
        return "t", np.arange(n, dtype=float)

    def ds_at_time_index(self, case: LoadedCase, time_index: int) -> Any:
        # Steady-state: return the live SOLPScase object (time index ignored for now).
        return _slc(case)

    def get_poloidal_profile(
        self,
        case: LoadedCase,
        *,
        region: str,
        sepadd: int,
        time_index: int,
        params: list,
    ) -> pd.DataFrame:
        slc = _slc(case)
        p = selector_params_only(list(params))
        resolved = [_resolve_param(slc, x) for x in p] if p else []
        return slc.get_1d_poloidal_data(
            resolved,
            region=str(region),
            sepadd=int(sepadd),
            target_first=False,
            guards=False,
        )

    def get_radial_profile(
        self,
        case: LoadedCase,
        *,
        region: str,
        time_index: int,
        params: list,
    ) -> pd.DataFrame:
        slc = _slc(case)
        requested = list(params)
        p = params_with_requested_geometry(requested)
        resolved = [_resolve_param(slc, x) for x in p] if p else []
        keep_geometry = any(name in ("R", "Z") for name in requested) or any(
            name in ("R", "Z") for name in resolved
        )
        df = slc.get_1d_radial_data(
            resolved,
            region=str(region),
            guards=False,
            keep_geometry=keep_geometry,
        )
        if keep_geometry and "R" in df.columns and "Z" not in df.columns and str(region) in ("omp", "imp"):
            df = df.copy()
            df["Z"] = 0.0
        return df

    def plot_2d_field(
        self,
        case: LoadedCase,
        *,
        param: str,
        ax: Any,
        time_index: int,
        **plot_kw: Any,
    ) -> None:
        slc = _slc(case)
        grid_only = bool(plot_kw.get("grid_only", False))
        if grid_only:
            # plot_2d only uses param for array shape before replacing with zeros.
            shape_key = "te" if "te" in slc.bal else "Te" if "Te" in slc.bal else str(param)
            if shape_key not in slc.bal:
                shape_key = _resolve_param(slc, str(param))
        else:
            shape_key = _resolve_param(slc, str(param))

        kwargs = dict(
            ax=ax,
            grid_only=grid_only,
            logscale=False if grid_only else bool(plot_kw.get("logscale", False)),
            cmap=str(plot_kw.get("cmap", "Spectral_r")),
            vmin=plot_kw.get("vmin"),
            vmax=plot_kw.get("vmax"),
            cbar=bool(plot_kw.get("cbar", False)),
            separatrix=False if grid_only else bool(plot_kw.get("separatrix", True)),
            antialias=bool(plot_kw.get("antialias", True)),
        )
        if grid_only:
            kwargs.update(linecolor="k", linewidth=0.2)
        else:
            kwargs.update(
                linewidth=float(plot_kw.get("linewidth", 0)),
                linecolor=str(plot_kw.get("linecolor", "k")),
            )
        slc.plot_2d(shape_key, **kwargs)

    def supports_2d(self, case: LoadedCase) -> bool:
        return True
