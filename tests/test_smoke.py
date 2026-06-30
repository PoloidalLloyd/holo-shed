"""Smoke tests for holo-shed package (no Qt display required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _qt_available() -> bool:
    try:
        from src.ui import qt  # noqa: F401

        return True
    except Exception:
        return False


qt_required = pytest.mark.skipif(not _qt_available(), reason="Qt bindings or system libraries unavailable")


def test_import_src_package():
    import src  # noqa: F401
    from src import dataset_utils, models, paths
    from src.backends import base, factory, hermes, solps
    from src.plotting import common, coordinator, ylim


@qt_required
def test_import_app_requires_qt():
    from src import app  # noqa: F401


def test_infer_time_dim():
    from src.dataset_utils import infer_time_dim

    ds = xr.Dataset(
        {"v": (("t", "x"), np.zeros((3, 5)))},
        coords={"t": [0.0, 1.0, 2.0], "x": np.arange(5)},
    )
    assert infer_time_dim(ds) == "t"


def test_infer_spatial_dim():
    from src.dataset_utils import infer_spatial_dim

    ds = xr.Dataset(
        {"v": (("t", "pos"), np.zeros((2, 4)))},
        coords={"t": [0.0, 1.0], "pos": np.arange(4)},
    )
    assert infer_spatial_dim(ds) == "pos"


def test_detect_backend_hermes(tmp_path: Path):
    from src.backends.factory import detect_backend

    (tmp_path / "BOUT.dmp.0.nc").write_text("")
    assert detect_backend(tmp_path) == "hermes"


def test_detect_backend_solps(tmp_path: Path):
    from src.backends.factory import detect_backend

    (tmp_path / "balance.nc").write_text("")
    assert detect_backend(tmp_path) == "solps"


def test_detect_backend_missing_dir(tmp_path: Path):
    from src.backends.factory import detect_backend

    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        detect_backend(missing)


def test_solps_backend_missing_balance(tmp_path: Path):
    from src.backends.solps import SolpsBackend

    backend = SolpsBackend()
    assert backend.kind == "solps"
    with pytest.raises(FileNotFoundError, match="balance.nc"):
        backend.load(tmp_path)


def test_probe_is_2d_case_no_bout(tmp_path: Path):
    from src.dataset_utils import probe_is_2d_case

    with pytest.raises(FileNotFoundError):
        probe_is_2d_case(tmp_path)


def test_plot_coordinator_routes():
    from src.plotting.coordinator import PlotCoordinator

    assert hasattr(PlotCoordinator, "redraw_profiles")
    assert hasattr(PlotCoordinator, "redraw_2d_poloidal")


def test_merge_case_variable_sets_union():
    from src.dataset_utils import merge_case_variable_sets

    a = {"Te", "Ne", "Bpxy"}
    b = {"Te", "Ne", "Vd+"}
    assert merge_case_variable_sets([a, b], mixed_backends=False) == ["Bpxy", "Ne", "Te", "Vd+"]


def test_merge_case_variable_sets_intersection():
    from src.dataset_utils import merge_case_variable_sets

    a = {"Te", "Ne", "Bpxy"}
    b = {"Te", "Ne", "Vd+"}
    assert merge_case_variable_sets([a, b], mixed_backends=True) == ["Ne", "Te"]


def test_format_case_display_label():
    from src.dataset_utils import format_case_display_label
    from src.models import LoadedCase

    hermes = LoadedCase(label="run_a", case_path="/a", ds=None, is_2d=True, backend_kind="hermes")
    solps = LoadedCase(label="run_b", case_path="/b", ds=None, is_2d=True, backend_kind="solps")
    assert format_case_display_label(hermes) == "run_a (Hermes 2D)"
    assert format_case_display_label(solps) == "run_b (SOLPS 2D)"


def test_resolve_profile_column_case_insensitive():
    import pandas as pd
    from src.models import LoadedCase
    from src.plotting.common import resolve_profile_column

    case = LoadedCase(label="s", case_path="/s", ds=None, is_2d=True, backend_kind="solps")
    df = pd.DataFrame({"Te": [1.0], "Ne": [2.0]})
    assert resolve_profile_column(case, "te", df) == "Te"
    assert resolve_profile_column(case, "Ne", df) == "Ne"
    assert resolve_profile_column(case, "missing", df) is None


def test_params_with_requested_geometry():
    from src.dataset_utils import params_with_requested_geometry, selector_params_only

    assert selector_params_only(["Te", "R", "Z"]) == ["Te"]
    assert params_with_requested_geometry(["Te", "R", "Z"]) == ["Z", "R", "Te"]


def test_radial_distance_mm_solps_dist():
    import pandas as pd
    from src.plotting.common import RADIAL_XLABEL, radial_distance_mm

    df = pd.DataFrame({"dist": [0.0, 0.01, 0.02], "Te": [1.0, 2.0, 3.0]})
    x = radial_distance_mm(df)
    assert x is not None
    assert len(x) == 3
    assert float(x[1]) == 10.0
    assert "sep" in RADIAL_XLABEL and "mm" in RADIAL_XLABEL


def test_ylim_with_margin():
    from src.plotting.ylim import with_margin

    lo, hi = with_margin(0.0, 10.0)
    assert lo < 0.0
    assert hi > 10.0


@qt_required
def test_main_window_import_without_display(monkeypatch):
    """Import MainWindow class without constructing QApplication."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from src.ui.main_window import MainWindow

    assert MainWindow.__name__ == "MainWindow"


def test_entry_point_shim():
    path = Path(__file__).resolve().parents[1] / "holo-shed.py"
    text = path.read_text()
    assert "src.app" in text
    assert "main()" in text
