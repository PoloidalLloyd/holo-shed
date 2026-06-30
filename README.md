# holo-shed

A Qt GUI for rapid Hermes-3 1D and 2D analysis. SOLPS support is planned via a pluggable backend layer.

## Install

This repo vendors [sdtools](https://github.com/mikekryjak/sdtools) under `external/sdtools` (as a git submodule).

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/PoloidalLloyd/holo-shed.git
cd holo-shed
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 holo-shed.py /path/to/case_dir
```

The entry script is a thin shim; application code is the `src` Python package in this repo.

## Package layout

```
holo-shed/                  # git repo (project name)
  holo-shed.py              # entry point: src.app.main()
  derived_variables.py      # Hermes-only derived xarray variables
  src/                      # Python package (import as `src`)
    app.py
    models.py
    dataset_utils.py
    backends/
    plotting/
    ui/
  tests/
    test_smoke.py
```

## Adding a backend

1. Implement `CaseBackend` in `src/backends/base.py` (see `HermesBackend` for reference).
2. Register detection in `src/backends/factory.py` (`detect_backend` + `get_backend`).
3. Plotting modules call `case.backend.get_poloidal_profile(...)` via `src/plotting/common.py` — no redraw changes needed if the backend returns the same DataFrame columns.

SOLPS: once `external/sdtools` includes transient `SOLPScase`, fill in `SolpsBackend` in `src/backends/solps.py`. Until then, directories with only `balance.nc` fail at load with an actionable message.

## Tests

```bash
python3 -m pytest tests/
```

Smoke tests cover imports, backend detection, and pure helpers without opening a display.

## Notes

- Automatic dimension detection supports both 1D and 2D Hermes cases; 2D analysis requires the grid file in the case directory.
- The 2D monitor tab is basic and may be extended later.
