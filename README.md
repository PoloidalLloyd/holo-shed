# holo-shed
A very basic gui for rapid Hermes-3 analysis (initially only in 1D)

## Install

This repo vendors `sdtools` under `external/sdtools` (as a git submodule).

Clone with submodules:

```bash
git clone --recurse-submodules <YOUR_REPO_URL>
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
