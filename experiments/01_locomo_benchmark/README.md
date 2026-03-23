# Experiment 1 — LoCoMo 基准评测

对应仓库根目录 [experiments/README.md](../README.md) 中的 **Experiment 1 — LoCoMo Benchmark Evaluation**。

## 产出

- Table 1–3、Figure 2–3、Figure 7（定性图可视化）的原始日志与后处理说明见总 README。
- 本目录脚本输出：`results/tab_locomo.tex`、`results/locomo_metrics.json`。

## 脚本

| 脚本 | 作用 |
|------|------|
| `build_graph.py` | 从 `example/Locomo` 的 conv JSON 构图（hash），写入 `results/locomo_cache/` |
| `qa_eval.py` | 对已构图数据跑 QA（与 `mosaic` query 一致） |
| `run.py` | 对 `mosaic/src/locomo results/` 下已就绪的全量 LoCoMo 图跑评测并生成 LaTeX 表 |

## 常用命令（仓库根目录）

```bash
python experiments/01_locomo_benchmark/build_graph.py --all
python experiments/01_locomo_benchmark/qa_eval.py --all
python experiments/01_locomo_benchmark/run.py
# 仅合并已有 qa_*_summary.json：
python experiments/01_locomo_benchmark/run.py --skip-run
```

全量基线 B1–B8 需在统一协议下分别实现各 memory 后端后再汇总；当前 `reference_values.LOCOMO_TABLE` 提供可编辑的参考行。

可选：在干净虚拟环境中对 `example/Locomo` 构图并跑 QA：

`bash experiments/01_locomo_benchmark/run_locomo_env.sh`
