"""Python path setup for vendored external libraries."""

from __future__ import annotations

import sys
from pathlib import Path


def _prepend_sys_path(directory: Path) -> None:
    sp = str(directory)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def ensure_xhermes_on_path() -> None:
    """
    Make a best-effort attempt to ensure xhermes is importable.

    Preferred (self-contained) layout:
      - repo_root/external/xhermes  (git submodule)

    Legacy layout:
      - repo_root/xhermes
      - repo_root/analysis/xhermes
    """
    here = Path(__file__).resolve()

    for parent in [here.parent, *here.parents]:
        xhermes_dir = parent / "external" / "xhermes"
        if (xhermes_dir / "xhermes" / "__init__.py").is_file():
            _prepend_sys_path(xhermes_dir)
            return

    for parent in [here.parent, *here.parents]:
        xhermes_dir = parent / "xhermes"
        if (xhermes_dir / "xhermes" / "__init__.py").is_file():
            _prepend_sys_path(xhermes_dir)
            return

    for parent in [here.parent, *here.parents]:
        if parent.name == "analysis":
            xhermes_dir = parent / "xhermes"
            if (xhermes_dir / "xhermes" / "__init__.py").is_file():
                _prepend_sys_path(xhermes_dir)
            return


def ensure_sdtools_on_path() -> None:
    """
    Make a best-effort attempt to ensure sdtools is importable.

    Preferred (self-contained) layout:
      - repo_root/external/sdtools  (git submodule)

    Legacy layout:
      - repo_root/analysis/sdtools
    """
    here = Path(__file__).resolve()

    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "external" / "sdtools"
        if sdtools_dir.exists():
            _prepend_sys_path(sdtools_dir)
            return

    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "analysis" / "sdtools"
        if sdtools_dir.exists():
            _prepend_sys_path(sdtools_dir)
            return

    for parent in [here.parent, *here.parents]:
        if parent.name == "analysis":
            sdtools_dir = parent / "sdtools"
            if sdtools_dir.exists():
                _prepend_sys_path(sdtools_dir)
            return


def ensure_vendored_deps_on_path() -> None:
    """Prepend vendored xhermes and sdtools (xhermes must load before sdtools)."""
    ensure_xhermes_on_path()
    ensure_sdtools_on_path()
