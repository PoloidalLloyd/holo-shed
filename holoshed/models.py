"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from holoshed.backends.base import CaseBackend


@dataclass
class LoadedCase:
    label: str
    case_path: str
    ds: Any  # xarray.Dataset for Hermes; optional for SOLPS
    n_time: int = 1
    is_2d: bool = False
    backend_kind: str = "hermes"
    backend: Optional["CaseBackend"] = None


# Alias for backward compatibility during migration
_LoadedCase = LoadedCase
