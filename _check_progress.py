#!/usr/bin/env python3
import json, os, glob

base = "experiments/locomo/benchmark/runs/update_c1_c3_conv5/results"
files = sorted(glob.glob(os.path.join(base, "*.json*")))
print("Files:", [os.path.basename(f) for f in files])

for f in files:
    try:
        with open(f) as fh:
            data = json.load(fh)
        results = data.get("results", [])
        n = len(results)
        if n == 0:
            print(os.path.basename(f) + ": empty")
            continue
        correct = sum(1 for r in results if r.get("judgment") == "CORRECT")
        wrong = sum(1 for r in results if r.get("judgment") == "WRONG")
        cats = {}
        for r in results:
            c = r.get("category", "?")
            j = r.get("judgment", "?")
            cats.setdefault(c, {"c": 0, "w": 0})
            cats[c]["c" if j == "CORRECT" else "w"] += 1
        pct = correct / n * 100 if n else 0
        print("{}: {}/37 | {} correct, {} wrong | acc={:.1f}%".format(
            os.path.basename(f), n, correct, wrong, pct))
        for c, s in sorted(cats.items()):
            t = s["c"] + s["w"]
            cpct = s["c"] / t * 100 if t else 0
            print("  Cat {}: {}/{} ({:.0f}%)".format(c, s["c"], t, cpct))
    except Exception as e:
        print("{}: ERROR {}".format(os.path.basename(f), e))
