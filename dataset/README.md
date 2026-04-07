# Datasets

This directory contains benchmark datasets used for evaluating MOSAIC. Raw data files are not committed to version control (large files, privacy, or licensing restrictions). Each subdirectory documents the source, format, and usage.

## Available Datasets

### LoCoMo (Long Context and Memory)

- **Purpose**: Long-range conversational memory evaluation (single-hop, multi-hop, temporal queries). Used for the main LoCoMo benchmark experiment.
- **Location**: `locomo/`
- **Source**: [snap-stanford/LoCoMo](https://github.com/snap-stanford/LoCoMo)
- **Contents**: 10 multi-session conversations (~600 turns each), 300+ annotated QA pairs with evidence links.
- **Experiment**: `experiments/locomo/`

## Planned Datasets

The following datasets are described in `experiments/README.md` and may be added as experiments are implemented:

| Dataset | Domain | Status |
|---------|--------|--------|
| HaluMem | Personal life management | Planned |
| MultiWOZ 2.4 | Task-oriented service | Planned |
| MTS-Dialog | Clinical patient management | Planned |
| ABCD | Customer service | Planned |

## Directory Layout

```
dataset/
├── README.md             # This file
└── locomo/               # LoCoMo benchmark data
    ├── experiment_sessions.json   # Session registry for experiments
    ├── locomo_conv0.json          # Conversation 0 (419 messages)
    ├── locomo_conv7.json          # Conversation 7 (681 messages)
    ├── qa_0.json                  # QA pairs for conversation 0
    └── qa_7.json                  # QA pairs for conversation 7
```

## Relationship to Experiments

| Experiment | Data Source |
|------------|------------|
| `experiments/locomo/benchmark/` | `dataset/locomo/` — LoCoMo conversations + QA |
| `experiments/locomo/scalability/` | Synthetic conversations (generated during experiment) |
| `experiments/locomo/ablation/` | Same as LoCoMo; re-runs with components disabled |
