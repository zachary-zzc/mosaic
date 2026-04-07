#!/usr/bin/env python3
"""
Collect experiment results from completed runs into the structured results/ directory.

Run from: experiments/locomo/benchmark/
"""
import json, os, shutil, sys
from pathlib import Path

LOCOMO = Path(__file__).resolve().parent
ARCHIVE = LOCOMO / "archive" / "run_conv0_timed_completed"
RESULTS = LOCOMO / "results"

# ── 1. qa_accuracy: Copy & consolidate QA eval summaries ────────────

qa_dir = RESULTS / "qa_accuracy"
qa_dir.mkdir(parents=True, exist_ok=True)

summaries = {}
for f in sorted((ARCHIVE / "results").glob("qa_*_eval_summary_*.json")):
    shutil.copy2(f, qa_dir / f.name)
    data = json.loads(f.read_text())
    s = data["summary"]
    # derive conv_id and strategy from filename
    # e.g. qa_0_eval_summary_hybrid.json -> conv0, hybrid
    parts = f.stem.replace("qa_", "").replace("_eval_summary_", "|").split("|")
    qa_idx, strategy = parts[0], parts[1]
    conv_id = f"conv{qa_idx}"
    
    cat_map = {"1": "single_hop", "2": "temporal", "3": "multi_hop", "4": "open_domain"}
    cat_acc = {}
    for cat_num, cat_data in s["category_stats"].items():
        cat_name = cat_map.get(cat_num, f"cat{cat_num}")
        cat_acc[cat_name] = {
            "correct": cat_data["correct"],
            "total": cat_data["total"],
            "accuracy": round(cat_data["correct"] / cat_data["total"], 4) if cat_data["total"] > 0 else 0
        }
    
    summaries[f"{conv_id}_{strategy}"] = {
        "conv_id": conv_id,
        "strategy": strategy,
        "overall_accuracy": round(s["overall_accuracy"], 4),
        "total_correct": s["total_correct"],
        "total_questions": s["total_questions"],
        "categories": cat_acc
    }

# Write consolidated summary
with open(qa_dir / "qa_accuracy_all.json", "w") as fp:
    json.dump(summaries, fp, indent=2)

# Compute aggregate (weighted average across conv0+conv7 for each strategy)
for strategy in ["hybrid", "hash_only"]:
    total_q = sum(v["total_questions"] for k, v in summaries.items() if v["strategy"] == strategy)
    total_c = sum(v["total_correct"] for k, v in summaries.items() if v["strategy"] == strategy)
    agg_cats = {}
    for cat in ["single_hop", "temporal", "multi_hop", "open_domain"]:
        c_sum = sum(v["categories"].get(cat, {}).get("correct", 0) for k, v in summaries.items() if v["strategy"] == strategy)
        t_sum = sum(v["categories"].get(cat, {}).get("total", 0) for k, v in summaries.items() if v["strategy"] == strategy)
        agg_cats[cat] = {"correct": c_sum, "total": t_sum, "accuracy": round(c_sum / t_sum, 4) if t_sum > 0 else 0}
    summaries[f"aggregate_{strategy}"] = {
        "strategy": strategy,
        "overall_accuracy": round(total_c / total_q, 4) if total_q > 0 else 0,
        "total_correct": total_c,
        "total_questions": total_q,
        "categories": agg_cats
    }

with open(qa_dir / "qa_accuracy_all.json", "w") as fp:
    json.dump(summaries, fp, indent=2)
print(f"[qa_accuracy] wrote {len(summaries)} entries")

# ── 2. graph_structure: Extract from graph_snapshot JSONs ────────────

gs_dir = RESULTS / "graph_structure"
gs_dir.mkdir(parents=True, exist_ok=True)

graph_stats = {}
for strategy in ["hybrid", "hash_only"]:
    snap_dir = ARCHIVE / "artifacts" / strategy / "graph_snapshots"
    if not snap_dir.exists():
        continue
    for f in sorted(snap_dir.glob("graph_snapshot_conv*_*.json")):
        data = json.loads(f.read_text())
        gi = data.get("graph_info", {})
        # extract conv from filename
        conv_id = f.stem.split("_")[2]  # graph_snapshot_conv0_20260404
        dual_nx = gi.get("dual_nx", {})
        gp = dual_nx.get("G_p", {})
        ga = dual_nx.get("G_a", {})
        
        graph_stats[f"{conv_id}_{strategy}"] = {
            "conv_id": conv_id,
            "strategy": strategy,
            "total_classes": gi.get("total_classes"),
            "E_P_count": gi.get("dual_graph_edge_counts", {}).get("P"),
            "E_A_count": gi.get("dual_graph_edge_counts", {}).get("A"),
            "G_p_nodes": gp.get("nodes"),
            "G_p_edges": gp.get("edges"),
            "G_p_is_dag": gp.get("is_dag"),
            "G_a_nodes": ga.get("nodes"),
            "G_a_edges": ga.get("edges"),
            "snapshot_file": f.name
        }
        # Also copy the snapshot
        shutil.copy2(f, gs_dir / f"{conv_id}_{strategy}_{f.name}")

with open(gs_dir / "graph_structure_all.json", "w") as fp:
    json.dump(graph_stats, fp, indent=2)
print(f"[graph_structure] wrote {len(graph_stats)} entries")

# ── 3. build_metrics: LLM call counts, timing, NCS telemetry ────────

bm_dir = RESULTS / "build_metrics"
bm_dir.mkdir(parents=True, exist_ok=True)

build_data = {}
for strategy in ["hybrid", "hash_only"]:
    metrics_file = ARCHIVE / "artifacts" / strategy / "build_llm_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        # Note: this file was overwritten by conv7 in the shared-dir run
        # so it only has conv7 data. We note this limitation.
        shutil.copy2(metrics_file, bm_dir / f"build_llm_metrics_{strategy}.json")
        build_data[strategy] = data

with open(bm_dir / "build_metrics_all.json", "w") as fp:
    json.dump(build_data, fp, indent=2)
print(f"[build_metrics] wrote {len(build_data)} entries")

# ── 4. timing: Pipeline timing data ─────────────────────────────────

tm_dir = RESULTS / "timing"
tm_dir.mkdir(parents=True, exist_ok=True)

# From task_stdout.log grep results
timing_data = {
    "conv0_hybrid": {
        "graph_build_s": 14152,
        "graph_build_min": 235.9,
        "smoke_query_s": 20,
        "qa_eval_s": 9907,
        "qa_eval_min": 165.1,
        "build_mode": "hybrid",
        "messages": 419,
        "batches": 49,
        "classes_built": 101
    },
    "conv0_hash_only": {
        "graph_build_s": 10,
        "graph_build_min": 0.2,
        "smoke_query_s": 23,
        "qa_eval_s": 10125,
        "qa_eval_min": 168.8,
        "build_mode": "hash_only",
        "messages": 419,
        "batches": 49,
        "classes_built": 40
    },
    "conv7_hybrid": {
        "graph_build_s": 25535,
        "graph_build_min": 425.6,
        "smoke_query_s": 21,
        "qa_eval_s": 16657,
        "qa_eval_min": 277.6,
        "build_mode": "hybrid",
        "messages": 681,
        "batches": 81,
        "classes_built": 168
    },
    "conv7_hash_only": {
        "graph_build_s": 14,
        "graph_build_min": 0.2,
        "smoke_query_s": 20,
        "qa_eval_s": 4479,
        "qa_eval_min": 74.7,
        "qa_eval_note": "resumed run; original was interrupted",
        "build_mode": "hash_only",
        "messages": 681,
        "batches": 81,
        "classes_built": 57
    }
}

with open(tm_dir / "timing_all.json", "w") as fp:
    json.dump(timing_data, fp, indent=2)
print(f"[timing] wrote {len(timing_data)} entries")

# ── 5. raw_eval: Copy full evaluation JSONs (per-question detail) ────

raw_dir = RESULTS / "raw_eval"
raw_dir.mkdir(parents=True, exist_ok=True)

for f in sorted((ARCHIVE / "results").glob("qa_*_eval_full_*.json")):
    shutil.copy2(f, raw_dir / f.name)
print(f"[raw_eval] copied {len(list((ARCHIVE / 'results').glob('qa_*_eval_full_*.json')))} files")

# ── 6. Summary: tab_locomo reference data ────────────────────────────

# For tab_locomo.tex: We need MOSAIC aggregate accuracy per category
# The manuscript uses: Overall, Single-hop, Multi-hop, Temporal
# Our categories: 1=single_hop, 2=temporal, 3=multi_hop, 4=open_domain

# Use hybrid strategy as primary MOSAIC result (that's the full system)
mosaic_hybrid = summaries.get("aggregate_hybrid", {})
mosaic_hash = summaries.get("aggregate_hash_only", {})

tab_locomo_data = {
    "_comment": "Data for tab_locomo.tex. MOSAIC results from this experiment. Baselines from reference_values.py.",
    "mosaic_hybrid": {
        "overall": mosaic_hybrid.get("overall_accuracy"),
        "single_hop": mosaic_hybrid.get("categories", {}).get("single_hop", {}).get("accuracy"),
        "multi_hop": mosaic_hybrid.get("categories", {}).get("multi_hop", {}).get("accuracy"),
        "temporal": mosaic_hybrid.get("categories", {}).get("temporal", {}).get("accuracy"),
        "open_domain": mosaic_hybrid.get("categories", {}).get("open_domain", {}).get("accuracy"),
    },
    "mosaic_hash_only": {
        "overall": mosaic_hash.get("overall_accuracy"),
        "single_hop": mosaic_hash.get("categories", {}).get("single_hop", {}).get("accuracy"),
        "multi_hop": mosaic_hash.get("categories", {}).get("multi_hop", {}).get("accuracy"),
        "temporal": mosaic_hash.get("categories", {}).get("temporal", {}).get("accuracy"),
        "open_domain": mosaic_hash.get("categories", {}).get("open_domain", {}).get("accuracy"),
    }
}

with open(RESULTS / "tab_locomo_data.json", "w") as fp:
    json.dump(tab_locomo_data, fp, indent=2)

# ── 7. Graph structure for tab_graph_structure ───────────────────────

tab_graph_data = {
    "_comment": "Data for tab_graph_structure.tex. Actual measured values from experiment.",
    "sessions": {}
}
for key, gs in graph_stats.items():
    tab_graph_data["sessions"][key] = {
        "|V|_classes": gs["total_classes"],
        "|V|_G_p": gs["G_p_nodes"],
        "|E_P|": gs["E_P_count"],
        "|V|_G_a": gs["G_a_nodes"],
        "|E_A|": gs["E_A_count"],
        "is_dag": gs["G_p_is_dag"]
    }

with open(RESULTS / "tab_graph_structure_data.json", "w") as fp:
    json.dump(tab_graph_data, fp, indent=2)

print("\n=== Results collection complete ===")
print(f"Results directory: {RESULTS}")
for d in sorted(RESULTS.iterdir()):
    if d.is_dir():
        nfiles = len(list(d.iterdir()))
        print(f"  {d.name}/ ({nfiles} files)")
    else:
        print(f"  {d.name}")
