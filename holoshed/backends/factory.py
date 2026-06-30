"""Backend detection and case loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from holoshed.backends.hermes import HermesBackend
from holoshed.backends.solps import SolpsBackend
from holoshed.models import LoadedCase


def detect_backend(path: Path) -> str:
    p = path.expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Case path is not a directory: {p}")
    has_balance = (p / "balance.nc").exists()
    has_bout = any(p.glob("BOUT.dmp*.nc")) or (p / "BOUT.squash.nc").exists()
    if has_balance and not has_bout:
        return "solps"
    if has_bout:
        return "hermes"
    if has_balance:
        return "solps"
    raise FileNotFoundError(
        f"Could not detect case type in {p} (expected BOUT.dmp*.nc or balance.nc)."
    )


def get_backend(kind: str) -> Any:
    if kind == "hermes":
        return HermesBackend()
    if kind == "solps":
        return SolpsBackend()
    raise ValueError(f"Unknown backend kind: {kind}")


def load_case(path: str, *, load_cls: Any = None) -> LoadedCase:
    p = Path(path)
    kind = detect_backend(p)
    backend = get_backend(kind)
    case = backend.load(p, load_cls=load_cls)
    case.backend = backend
    case.backend_kind = kind
    return case
