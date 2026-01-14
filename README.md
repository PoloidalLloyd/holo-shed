# holo-shed
A very basic gui for rapid Hermes-3 1D and 2D analysis.

## Install

This repo vendors [sdtools]([url](https://github.com/mikekryjak/sdtools)) under `external/sdtools` (as a git submodule).

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

Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 holo-shed.py /path/to/case_dir
```

## Comments
Attempts have been made to allow for automatic dimension detection to allow for 1D and 2D analysis, however for 2D analysis the grid file must be located in the same directory. Currently 2D version only allows for one data set to be loaded at a time, although this will hopefully be changed in the future.

The monitor functionality for 2D is still very barebones and could be improved. 
