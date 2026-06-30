"""Case backend protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7
    from typing_extensions import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from holoshed.models import LoadedCase


@runtime_checkable
class CaseBackend(Protocol):
    kind: str

    def load(self, path: Path, *, load_cls: Any = None) -> LoadedCase: ...

    def list_variables(self, case: LoadedCase) -> list: ...

    def time_coordinate(self, case: LoadedCase) -> tuple: ...

    def get_poloidal_profile(
        self, case: LoadedCase, *, region: str, sepadd: int, time_index: int, params: list
    ) -> pd.DataFrame: ...

    def get_radial_profile(
        self, case: LoadedCase, *, region: str, time_index: int, params: list
    ) -> pd.DataFrame: ...

    def plot_2d_field(
        self, case: LoadedCase, *, param: str, ax: Any, time_index: int, **plot_kw: Any
    ) -> None: ...

    def supports_2d(self, case: LoadedCase) -> bool: ...

    def ds_at_time_index(self, case: LoadedCase, time_index: int) -> Any: ...
