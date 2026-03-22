#!/usr/bin/env python3
"""从 example/Locomo 全量 conv0 / qa_0 重写 data_mini 下的 mini 文件（与仓库约定一致）。"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
REPO = RUN_DIR.parent.parent.parent
LOCOMO = REPO / "example" / "Locomo"


def main() -> None:
    with open(LOCOMO / "locomo_conv0.json", encoding="utf-8") as f:
        d = json.load(f)
    mini_conv = {
        "speaker_a": d["speaker_a"],
        "speaker_b": d["speaker_b"],
        "session_1_date_time": d["session_1_date_time"],
        "session_1": d["session_1"][:10],
    }
    HERE.mkdir(parents=True, exist_ok=True)
    with open(HERE / "locomo_conv0_mini.json", "w", encoding="utf-8") as f:
        json.dump(mini_conv, f, ensure_ascii=False, indent=2)

    with open(LOCOMO / "qa_0.json", encoding="utf-8") as f:
        qa = json.load(f)
    keep: list = []
    for item in qa:
        ev = item.get("evidence") or []
        if not ev:
            continue
        ok = True
        for e in ev:
            if not str(e).startswith("D1:"):
                ok = False
                break
            try:
                num = int(str(e).split(":")[1])
            except (IndexError, ValueError):
                ok = False
                break
            if num > 10:
                ok = False
                break
        if ok and item.get("category") != 5:
            keep.append(item)
        if len(keep) >= 2:
            break
    with open(HERE / "qa_0_mini.json", "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    print("Wrote", HERE / "locomo_conv0_mini.json", "msgs=", len(mini_conv["session_1"]))
    print("Wrote", HERE / "qa_0_mini.json", "n=", len(keep))


if __name__ == "__main__":
    main()
