"""Shared I/O helpers for JSON and pickle."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Load JSON from file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write JSON to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_pickle(path: str | Path) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path: str | Path, obj: Any) -> None:
    """Write object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
