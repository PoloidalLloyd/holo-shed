"""Shared plotting helpers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.dataset_utils import selector_params_only
from src.models import LoadedCase

RADIAL_XLABEL = r"$r - r_{\mathrm{sep}}$ (mm)"


def radial_distance_mm(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Radial distance from separatrix in millimetres (Hermes Srad or SOLPS dist)."""
    for key in ("Srad", "dist"):
        if key in df.columns:
            try:
                return np.asarray(df[key].values, dtype=float) * 1e3
            except Exception:
                return None
    return None


def radial_distance_column(df: pd.DataFrame) -> Optional[str]:
    for key in ("Srad", "dist"):
        if key in df.columns:
            return key
    return None


def get_poloidal_profile(
    case: LoadedCase,
    *,
    time_index: int,
    region: str,
    sepadd: int,
    params: List[str],
) -> pd.DataFrame:
    backend = case.backend
    if backend is None:
        raise RuntimeError(f"Case {case.label} has no backend attached")
    return backend.get_poloidal_profile(
        case,
        region=region,
        sepadd=int(sepadd),
        time_index=int(time_index),
        params=selector_params_only(list(params)),
    )


def get_radial_profile(
    case: LoadedCase,
    *,
    time_index: int,
    region: str,
    params: List[str],
) -> pd.DataFrame:
    backend = case.backend
    if backend is None:
        raise RuntimeError(f"Case {case.label} has no backend attached")
    return backend.get_radial_profile(
        case,
        region=region,
        time_index=int(time_index),
        params=list(params),
    )


def resolve_profile_column(case: LoadedCase, var: str, df: pd.DataFrame) -> str | None:
    """Map a holo-shed variable name to a column in a poloidal/radial extract DataFrame."""
    try:
        cols = list(df.columns)
    except Exception:
        return None
    if var in cols:
        return var
    backend = case.backend
    if backend is not None and hasattr(backend, "_resolve_param"):
        try:
            resolved = backend._resolve_param(case.ds, var)
            if resolved in cols:
                return resolved
        except Exception:
            pass
    lower = {str(c).lower(): c for c in cols}
    return lower.get(str(var).lower())
