# holo-shed
A very basic gui for rapid Hermes-3 analysis.

## Install

This repo vendors `sdtools` under `external/sdtools` (as a git submodule).

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/PoloidalLloyd/holo-shed.git
cd holo-shed
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Run

```bash
python holo-shed.py /path/to/case_dir
```
