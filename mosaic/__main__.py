"""python -m mosaic：需将含本包与 src/ 的目录加入 PYTHONPATH（通常为仓库根下的 mosaic 目录）。"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
