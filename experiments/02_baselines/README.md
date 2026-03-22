# 02 — Baseline Comparison

**对应稿件**：Table 2 (tab:baselines)、Figure 2、Results §3.2。

**目标**：ReAct、Reflexion、Plan-and-Solve、MemGPT、Mem0、Checklist、DualGraph 在四领域的 Acq.、Conc.、Trn；失败模式分类；学习曲线与效率图。

**数据**：与 01_memory_gap 相同 case set，同一批运行七方法。

## DualGraph on LoCoMo（example/Locomo）

使用 **hash 构图 + hash 检索**（不调用 LLM 构图/检索），仅回答与评判使用 Qwen（mosaic 中 `ali_api|qwen3.5-plus`）。

数据约定：长对话 `example/Locomo/locomo_conv0.json`，问答 `example/Locomo/qa_0.json`（可多对：`locomo_conv1.json` + `qa_1.json` 等）。

### 分步执行（推荐：构图耗时可单独跑）

**步骤 1：仅构图**（读取 conv JSON，hash 构图，写出 graph.pkl + tags.json）

```bash
# 仓库根目录下执行

# 方式 A：只构一张图（例如 conv0）
python experiments/02_baselines/step1_build_graph.py \
  --conv example/Locomo/locomo_conv0.json \
  --out experiments/02_baselines/results/locomo_cache/conv_0

# 方式 B：对 example/Locomo 下所有 locomo_conv*.json 依次构图
python experiments/02_baselines/step1_build_graph.py --all
```

输出目录中会生成 `graph.pkl` 和 `tags.json`。构图时间较长，可后台跑。

**步骤 2：仅 query + 打分**（读取已有图与 tags；**作答与对错评判均经 mosaic 与 `llm.py` 相同的 API**。检索默认 **hash**（与步骤 1 一致）；可用 `--method llm` 改为 LLM 检索。）

```bash
# 方式 A：单组 qa + 图
python experiments/02_baselines/step2_qa_eval.py --single \
  --qa example/Locomo/qa_0.json \
  --graph experiments/02_baselines/results/locomo_cache/conv_0/graph.pkl \
  --tags experiments/02_baselines/results/locomo_cache/conv_0/tags.json \
  --out experiments/02_baselines/results

# 同上，但检索用 LLM（与 mosaic query method=llm 一致）
python experiments/02_baselines/step2_qa_eval.py --single \
  --qa example/Locomo/qa_0.json \
  --graph .../graph.pkl \
  --tags .../tags.json \
  --out experiments/02_baselines/results \
  --method llm

# 方式 B：对 locomo_cache 下所有 conv_* 与对应 qa_*.json 逐对跑 QA
python experiments/02_baselines/step2_qa_eval.py --all
```

结果写入 `--out/dualgraph_qa_*_results.json`（逐题完整）与 `dualgraph_qa_*_summary.json`（按 category 与整体统计）。可选 `--max-questions N` 限制题数。

**步骤 3（可选）：仅汇总写表**

步骤 2 完成后，若只需根据已有 summary 生成 Table 2 的 LaTeX：

```bash
python experiments/02_baselines/run.py --run-dualgraph-on-locomo --aggregate-only
```

会读取 `results/dualgraph_qa_*_summary.json`，汇总后写入 `results/tab_baselines.tex` 等。

### 一键执行（构图 + QA 一起跑）

若希望一条命令做完构图和 QA（耗时更长）：

```bash
python experiments/02_baselines/run.py --run-dualgraph-on-locomo
```

或使用自带环境的脚本（在仓库根目录）：

```bash
bash experiments/02_baselines/run_locomo_env.sh
```

### 依赖

mosaic 的 Python 环境（含 langchain、mosaic/config 中 Qwen API 配置）。步骤 1 不调用 Qwen；步骤 2 需要 API 用于作答与评判。
