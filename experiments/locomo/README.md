# LoCoMo Experiments

All experiments on the LoCoMo (Long Context and Memory) benchmark dataset.
Data is read from `dataset/locomo/`.

## Sub-Experiments

| Folder | Purpose | Status |
|--------|---------|--------|
| `benchmark/` | Full QA evaluation: hybrid (evolving) vs. hash_only (static) graph | Implemented |
| `ablation/` | Component ablation C0–C6: disable one MOSAIC component at a time | Implemented |
| `scalability/` | Build time and accuracy vs. conversation length (S-50 .. S-600) | Implemented |

## Directory Layout

```
experiments/locomo/
├── README.md
├── _utils.py                       # Shared utilities (path setup, JSON, dataset access)
├── start_experiment.sh             # Run ALL sub-experiments in sequence
├── benchmark/
│   ├── run.py                      # Full pipeline (build + QA + aggregate)
│   ├── start_experiment.sh
│   ├── build_graph.py              # Standalone graph construction
│   ├── qa_eval.py                  # Standalone QA evaluation
│   ├── collect_results.py          # Structure results from completed runs
│   ├── results/
│   └── runs/
├── ablation/
│   ├── run.py                      # C0-C6 ablation with env var overrides
│   ├── start_experiment.sh
│   ├── results/
│   └── runs/
└── scalability/
    ├── run.py                      # Truncated conversations → build + QA
    ├── start_experiment.sh
    ├── results/
    └── runs/
```

## Quick Start

```bash
# Run everything (benchmark → ablation → scalability → export)
cd experiments/locomo && ./start_experiment.sh

# Or run individual sub-experiments:
python experiments/locomo/benchmark/run.py
python experiments/locomo/ablation/run.py
python experiments/locomo/scalability/run.py

# Background execution for any sub-experiment:
cd experiments/locomo/benchmark && nohup ./start_experiment.sh > runs/experiment.log 2>&1 &
```

### Benchmark options
```bash
python experiments/locomo/benchmark/run.py --strategy hybrid        # evolving graph only
python experiments/locomo/benchmark/run.py --strategy hash_only     # static graph only
python experiments/locomo/benchmark/run.py --conv conv0 --max-questions 10
python experiments/locomo/benchmark/run.py --skip-build             # reuse existing graphs
```

### Ablation options
```bash
python experiments/locomo/ablation/run.py --condition C3            # single condition
python experiments/locomo/ablation/run.py --conv conv0              # single conversation
python experiments/locomo/ablation/run.py --skip-build              # query-only ablations
```

### Scalability options
```bash
python experiments/locomo/scalability/run.py --lengths 100 200 400  # custom lengths
python experiments/locomo/scalability/run.py --conv conv0            # single conversation
python experiments/locomo/scalability/run.py --skip-qa              # build timing only
```

## Ablation Conditions

| ID | Condition | What changes | Needs rebuild? |
|----|-----------|-------------|----------------|
| C0 | Full MOSAIC | Reuses benchmark/hybrid | No (reuse) |
| C1 | Static graph | Reuses benchmark/hash_only | No (reuse) |
| C2 | No relationship edges | `MOSAIC_EDGE_SEMANTIC_A=0, MOSAIC_EDGE_PREREQ_LLM=0` | Yes |
| C3 | No graph traversal | `MOSAIC_QUERY_NEIGHBOR_HOPS=0` | No |
| C4 | Prereq-only traversal | `MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=P` | No |
| C5 | Assoc-only traversal | `MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=A` | No |
| C6 | No edges + no traversal | C2 + C3 combined | Yes |

## Manuscript Outputs

| Table | Source |
|-------|--------|
| `tab_locomo.tex` | `benchmark/results/` |
| `tab_ablation_mosaic.tex` | `ablation/results/` |
| `tab_scalability.tex` | `scalability/results/` |
