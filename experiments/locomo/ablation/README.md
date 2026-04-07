# Ablation Experiment

MOSAIC component ablation study (C0–C6) on the LoCoMo dataset.

Each condition disables one component via environment variable overrides,
then re-runs the build + QA pipeline. C0/C1 reuse benchmark results.

## Usage

```bash
python experiments/locomo/ablation/run.py                # all conditions
python experiments/locomo/ablation/run.py --condition C3  # single condition
python experiments/locomo/ablation/run.py --conv conv0    # single conversation
./start_experiment.sh                                     # background execution
```

## Ablation Conditions

| ID | Condition | Env overrides | Needs rebuild? |
|----|-----------|--------------|----------------|
| C0 | Full MOSAIC (control) | Reuses benchmark/hybrid | No |
| C1 | Static graph | Reuses benchmark/hash_only | No |
| C2 | No relationship edges | `MOSAIC_EDGE_SEMANTIC_A=0, MOSAIC_EDGE_PREREQ_LLM=0` | Yes |
| C3 | No graph traversal | `MOSAIC_QUERY_NEIGHBOR_HOPS=0` | No |
| C4 | Prereq-only traversal | `MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=P` | No |
| C5 | Assoc-only traversal | `MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=A` | No |
| C6 | No edges + no traversal | C2 + C3 combined | Yes |

## Prerequisites

Benchmark experiment must have completed first — C0 and C1 reuse its results,
and query-only ablations (C3–C5) reuse benchmark hybrid graphs.

## Results

Results are written to `results/` including `ablation_metrics.json`,
`ablation_table.json`, and `tab_ablation_mosaic.tex`.
