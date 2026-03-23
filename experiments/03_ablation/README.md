# Experiment 3 — MOSAIC 消融

对应 [experiments/README.md](../README.md) 中的 **Experiment 3 — Ablation Study**。

## 条件

C0–C6 定义见总 README「Ablation Conditions」表。

## 产出

- Table 6、Figure 6；C0 数值与 Experiment 1 中 MOSAIC 一致时可复用。

## 运行

```bash
python experiments/03_ablation/run.py
```

当前从 `reference_values.MOSAIC_ABLATION` 导出占位 LaTeX；真实实验需在 MOSAIC 中实现各消融开关后在 LoCoMo 上评测。
