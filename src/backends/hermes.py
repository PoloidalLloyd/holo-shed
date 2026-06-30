"""Hermes-3 / BOUT case backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.dataset_utils import (
    ensure_sdtools_2d_metadata,
    format_case_label,
    infer_spatial_dim,
    infer_time_dim,
    list_plottable_vars,
    list_plottable_vars_2d,
    parse_mesh_grid_filename_from_bout_inp,
    probe_is_2d_case,
    selector_params_only,
    should_use_squash_for_load,
)
from src.models import LoadedCase
from src.paths import ensure_sdtools_on_path


class HermesBackend:
    kind = "hermes"

    def __init__(self) -> None:
        ensure_sdtools_on_path()
        from hermes3.load import Load  # type: ignore

        self._Load = Load
        try:
            import hermes3.accessors  # type: ignore  # noqa: F401
        except Exception:
            pass

    def load(self, path: Path, *, load_cls: Any = None) -> LoadedCase:
        Load = load_cls or self._Load
        case_path = str(path.expanduser().resolve())
        label = format_case_label(case_path)
        case_dir = Path(case_path)
        is_2d_probe = probe_is_2d_case(case_dir)
        use_squash = should_use_squash_for_load(case_dir)

        def _load_2d():
            grid_name = parse_mesh_grid_filename_from_bout_inp(case_dir / "BOUT.inp")
            if not grid_name:
                raise FileNotFoundError(
                    "Detected a 2D case but could not find the grid file in BOUT.inp under:\n"
                    "  [mesh]\n  file = \"...\"\n"
                    f"Case directory: {case_dir}"
                )
            grid_path = (case_dir / grid_name).resolve()
            if not grid_path.exists():
                raise FileNotFoundError(
                    f"Detected a 2D case but grid file not found: {grid_path}"
                )
            cs2 = Load.case_2D(
                case_path,
                gridfilepath=str(grid_path),
                verbose=False,
                use_squash=use_squash,
                force_squash=False,
            )
            try:
                ensure_sdtools_2d_metadata(cs2.ds)
            except Exception:
                pass
            return cs2

        def _load_1d():
            return Load.case_1D(
                case_path,
                verbose=False,
                guard_replace=False,
                use_squash=use_squash,
                force_squash=False,
            )

        cs = None
        is_2d = False
        if is_2d_probe:
            try:
                cs = _load_2d()
                is_2d = True
            except Exception:
                cs = _load_1d()
                is_2d = False
        else:
            try:
                cs = _load_1d()
                is_2d = False
            except Exception as e:
                msg = str(e).lower()
                if (
                    ("toroidal" in msg and "grid" in msg)
                    or ("topology" in msg and "grid" in msg)
                    or ("provide grid in data directory" in msg)
                ):
                    cs = _load_2d()
                    is_2d = True
                else:
                    raise

        try:
            import derived_variables as dv

            print(f"Computing derived variables for {label}...")
            available = dv.get_available_variables()
            if available:
                print(f"  Found {len(available)} registered derived variable(s)")
                cs.ds = dv.compute_derived_variables(cs.ds)
        except ImportError:
            print("derived_variables.py not found - skipping derived quantities")
        except Exception as e:
            print(f"Warning: Error computing derived variables: {e}")

        tdim = infer_time_dim(cs.ds)
        n_time = int(cs.ds.sizes[tdim]) if tdim and tdim in cs.ds.dims else 1
        return LoadedCase(
            label=label,
            case_path=case_path,
            ds=cs.ds,
            n_time=n_time,
            is_2d=is_2d,
            backend_kind="hermes",
            backend=self,
        )

    def list_variables(self, case: LoadedCase) -> List[str]:
        ds = case.ds
        tdim = infer_time_dim(ds)
        if case.is_2d:
            return list_plottable_vars_2d(ds, time_dim=tdim)
        sdim = infer_spatial_dim(ds)
        return list_plottable_vars(ds, spatial_dim=sdim, time_dim=tdim)

    def time_coordinate(self, case: LoadedCase):
        ds = case.ds
        tdim = infer_time_dim(ds)
        if tdim and tdim in ds.coords:
            return tdim, ds[tdim].values
        import numpy as np

        return tdim or "t", np.arange(case.n_time)

    def ds_at_time_index(self, case: LoadedCase, time_index: int):
        ds = case.ds
        tdim = infer_time_dim(ds)
        if tdim and tdim in ds.dims:
            return ds.isel({tdim: int(time_index)})
        return ds

    def get_poloidal_profile(
        self, case: LoadedCase, *, region: str, sepadd: int, time_index: int, params: list
    ) -> pd.DataFrame:
        from hermes3.selectors import get_1d_poloidal_data  # type: ignore

        ds_t = self.ds_at_time_index(case, time_index)
        try:
            if hasattr(case.ds, "metadata"):
                ds_t.metadata = case.ds.metadata  # type: ignore[attr-defined]
            if hasattr(case.ds, "options"):
                ds_t.options = case.ds.options  # type: ignore[attr-defined]
            ensure_sdtools_2d_metadata(ds_t)
        except Exception:
            pass
        p = selector_params_only(list(params))
        return get_1d_poloidal_data(
            ds_t, params=p, region=region, sepadd=int(sepadd), target_first=False
        )

    def get_radial_profile(
        self, case: LoadedCase, *, region: str, time_index: int, params: list
    ) -> pd.DataFrame:
        import hermes3.selectors as sel  # type: ignore

        try:
            from hermes3.selectors import get_1d_radial_data  # type: ignore
        except ImportError:
            from hermes3.selectors import get_1d_radial_data_old as get_1d_radial_data  # type: ignore

        ds_t = self.ds_at_time_index(case, time_index)
        try:
            if hasattr(case.ds, "metadata"):
                ds_t.metadata = case.ds.metadata  # type: ignore[attr-defined]
            if hasattr(case.ds, "options"):
                ds_t.options = case.ds.options  # type: ignore[attr-defined]
            ensure_sdtools_2d_metadata(ds_t)
        except Exception:
            pass
        p = selector_params_only(list(params))
        return get_1d_radial_data(ds_t, params=p, region=region)

    def plot_2d_field(
        self, case: LoadedCase, *, param: str, ax: Any, time_index: int, **plot_kw: Any
    ) -> None:
        raise NotImplementedError("Use polygon_2d module for Hermes 2D rendering")

    def supports_2d(self, case: LoadedCase) -> bool:
        return bool(case.is_2d)
