"""Shared plotting helpers."""

from __future__ import annotations

from typing import List

import pandas as pd

from src.dataset_utils import selector_params_only
from src.models import LoadedCase


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
        params=selector_params_only(list(params)),
    )
