"""Python path setup for vendored sdtools."""

from __future__ import annotations

import sys
from pathlib import Path

def ensure_sdtools_on_path():
    """
    Make a best-effort attempt to ensure sdtools is importable.

    Preferred (self-contained) layout:
      - repo_root/external/sdtools  (git submodule)

    Legacy layout:
      - repo_root/analysis/sdtools
    """
    here = Path(__file__).resolve()

    # Preferred: sdtools is vendored as a submodule under this repo.
    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "external" / "sdtools"
        if sdtools_dir.exists():
            sp = str(sdtools_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return

    # Legacy: some repos keep sdtools under analysis/sdtools.
    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "analysis" / "sdtools"
        if sdtools_dir.exists():
            sp = str(sdtools_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return
    for parent in [here.parent, *here.parents]:
        if parent.name == "analysis":
            sdtools_dir = parent / "sdtools"
            if sdtools_dir.exists():
                sp = str(sdtools_dir)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return
