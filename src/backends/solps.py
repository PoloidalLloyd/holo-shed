"""SOLPS case backend (stub until sdtools submodule includes SOLPScase)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd

from src.models import LoadedCase

_SOLPS_MSG = (
    "SOLPS support requires an updated sdtools submodule with transient SOLPScase. "
    "See code_comparison/solps_pp.py in sdtools."
)


class SolpsBackend:
    kind = "solps"

    def load(self, path: Path, *, load_cls: Any = None) -> LoadedCase:
        raise NotImplementedError(_SOLPS_MSG)

    def list_variables(self, case: LoadedCase) -> List[str]:
        raise NotImplementedError(_SOLPS_MSG)

    def time_coordinate(self, case: LoadedCase):
        raise NotImplementedError(_SOLPS_MSG)

    def get_poloidal_profile(self, case: LoadedCase, *, region, sepadd, time_index, params) -> pd.DataFrame:
        raise NotImplementedError(_SOLPS_MSG)

    def get_radial_profile(self, case: LoadedCase, *, region, time_index, params) -> pd.DataFrame:
        raise NotImplementedError(_SOLPS_MSG)

    def plot_2d_field(self, case: LoadedCase, *, param, ax, time_index, **plot_kw) -> None:
        raise NotImplementedError(_SOLPS_MSG)

    def supports_2d(self, case: LoadedCase) -> bool:
        return True

    def ds_at_time_index(self, case: LoadedCase, time_index: int):
        raise NotImplementedError(_SOLPS_MSG)
