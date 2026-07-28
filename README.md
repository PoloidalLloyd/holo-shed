# holo-shed

A Qt GUI for rapid Hermes-3 and SOLPS 2D analysis.

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
## Usage with SOLPS-ITER

SOLPS cases are detected when a directory contains `balance.nc` (and no BOUT dump files). Load with the same `python3 holo-shed.py /path/to/solps/run` command. Steady-state `balance.nc` and transient cases with `b2time.nc` are supported. Balance-only quantities (e.g. `Vd+`, `M`, EIRENE `*_bal` terms) remain tied to the final snapshot in `balance.nc`.

## Notes

- Automatic dimension detection supports both 1D and 2D Hermes cases; 2D analysis requires the grid file in the case directory.
