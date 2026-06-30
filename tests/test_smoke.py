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


def test_solps_backend_stub(tmp_path: Path):
    from src.backends.solps import SolpsBackend

    backend = SolpsBackend()
    assert backend.kind == "solps"
    with pytest.raises(NotImplementedError, match="SOLPS"):
        backend.load(tmp_path)


def test_probe_is_2d_case_no_bout(tmp_path: Path):
    from src.dataset_utils import probe_is_2d_case

    with pytest.raises(FileNotFoundError):
        probe_is_2d_case(tmp_path)


def test_plot_coordinator_routes():
    from src.plotting.coordinator import PlotCoordinator

    assert hasattr(PlotCoordinator, "redraw_profiles")
    assert hasattr(PlotCoordinator, "redraw_2d_poloidal")


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
