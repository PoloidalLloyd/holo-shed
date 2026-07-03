# holo-shed

A Qt GUI for rapid Hermes-3 and SOLPS 2D analysis via a pluggable backend layer.

## Install

This repo vendors [sdtools](https://github.com/mikekryjak/sdtools) and [xhermes](https://github.com/boutproject/xhermes) under `external/` (as git submodules).

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
  external/                 # git submodules (sdtools, xhermes)
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

SOLPS cases are detected when a directory contains `balance.nc` (and no BOUT dump files). Load with the same `python3 holo-shed.py /path/to/solps/run` command. Steady-state `balance.nc` and transient cases with `b2time.nc` are supported; use the time slider to step through transient SOLPS slices (`timesa` in seconds). Balance-only quantities (e.g. `Vd+`, `M`, EIRENE `*_bal` terms) remain tied to the final snapshot in `balance.nc`.

**Hermes + SOLPS comparison:** use **Load case** for each directory in any order. Variables are intersected when backends or dimensions are mixed (e.g. `Te`, `Ne`). Poloidal/radial tabs overlay profiles; the 2D field tab shows side-by-side plots (up to 3 cases). **New session** clears and opens a single case.

**Hermes 1D + 2D (or SOLPS):** load a 2D case first (or any 2D case in the session). Hermes 1D cases can then be added and appear on the **Poloidal 1D** tab only (plotted vs `Spar`/`pos`). Radial and 2D field tabs show 2D cases only. Up to 3 datasets total in comparison mode.

## Tests

```bash
python3 -m pytest tests/
```

Smoke tests cover imports, backend detection, and pure helpers without opening a display.

## Notes

- Automatic dimension detection supports both 1D and 2D Hermes cases; 2D analysis requires the grid file in the case directory.
- The 2D monitor tab is basic and may be extended later.
