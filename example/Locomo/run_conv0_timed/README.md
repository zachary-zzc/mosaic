# Locomo conv0 构图运行目录

与 `mosaic/` 源码分离；**构图**通过仓库内的 **`python -m mosaic build`** 执行（与主程序一致）。日志写入本目录 **`log/`**，图与 tags 写入 **`artifacts/`**，QA 评测结果写入 **`results/`**，不污染 `mosaic/results/`。

## 极小数据集 + 分步后台（推荐先跑通）

从 `example/Locomo/locomo_conv0.json` / `qa_0.json` 派生的 **`data_mini/`**（`session_1` 前 10 轮对话 + 2 道 QA），用于在 **mosaic** conda 环境下快速验证整条链路。

| 脚本 | 说明 |
|------|------|
| **`start_background.sh`** | 后台顺序执行：**(1)** `paths_mini.json` 构图 → **(2)** `python -m mosaic query` 单条冒烟 → **(3)** `run_qa_eval.py` 评测（检索+作答+LLM 评判+统计）。控制信息写入 **`log/task_stdout.log`**，mosaic 详情在 **`log_mini/`**（如 `mosaic_server.log`、`qa_eval.log`），产物在 **`artifacts_mini/`**、**`results_mini/`**。 |
| **`run_pipeline_mini.sh`** | 同上三步骤，**前台**运行，便于看报错。 |
| **`start_background_full.sh`** | 仅全量构图（`paths.json`，不跑 QA），等同旧版单步构图。 |

配置见 **`paths_mini.json`**（`AUTO_MINI` 解析到 `data_mini/`、`log_mini/`、`artifacts_mini/`、`results_mini/`）。可选执行 `python data_mini/build_from_source.py` 从全量数据重写 mini 文件。

## 构图完成后，图和 tags 在哪里？

| 文件 | 说明 |
|------|------|
| **`artifacts/graph_network_conv0.pkl`** | 最终类图（`mosaic build --graph-out`），供 `query` / `run_qa_eval.py` 加载 |
| **`artifacts/conv0_tags.json`** | TF-IDF 实例 tags（`--tags-out`），查询检索必需 |
| **`artifacts/graph_snapshots/`** | 构图过程中的快照（`graph_network_*`、`graph_snapshot_*` 等） |

之后在**本目录**跑评测（见下节），或在仓库根用 CLI：

```bash
conda activate mosaic
cd /path/to/LongtermMemory
PYTHONPATH=. python -m mosaic query \
  --graph-pkl example/Locomo/run_conv0_timed/artifacts/graph_network_conv0.pkl \
  --tags-json example/Locomo/run_conv0_timed/artifacts/conv0_tags.json \
  --method llm \
  --question "你的问题"
```

## 配置

1. **`paths.json`**（本目录）  
   - `mosaic_root`、`locomo_conv_json`、`qa_json`、`log_dir`、`results_dir`、`artifacts_dir`、`mosaic_config_path` 填 **`AUTO`** 时按仓库布局自动解析。  
   - `qa_json` 的 `AUTO` → `example/Locomo/qa_0.json`（与 `locomo_conv0` 配套）。  
   - `embedding_model_override`：非空时覆盖 `mosaic/config/config.cfg` 里 `[PATHS] embedding_model`。

2. **`mosaic/config/config.cfg`**  
   - **`[PATHS] embedding_model`**、**API**：与 mosaic 主程序相同，用于构图 LLM、query 作答与 **JUDGE 评判**。

## 前台构图

```bash
conda activate mosaic
cd /path/to/LongtermMemory/example/Locomo/run_conv0_timed
python run.py
```

- **stdout**：任务控制信息（含 `mosaic build` 的 tqdm）。  
- **`--verbose-log`**：`log/run_verbose.log` + 子进程 `mosaic -v`。

## 构图后：QA 评测（与 experiments/01_locomo_benchmark/qa_eval 同源逻辑）

对 **`qa_0.json`** 逐题：**检索 + LLM 作答**（`mosaic/src/query.py`），再用 **`judge_answer_llm`**（与 `llm.py` / `fetch_default_llm_model` 相同 API）判 CORRECT/WRONG，并按 **category** 与**整体**统计。

```bash
conda activate mosaic
cd /path/to/LongtermMemory/example/Locomo/run_conv0_timed
python run_qa_eval.py
# 默认 --method llm（与 LLM 构图一致）；基线对比可用 --method hash
# python run_qa_eval.py --method hash --max-questions 20
```

| 输出 | 内容 |
|------|------|
| **`results/qa_0_eval_full.json`** | 逐题 `generated_answer`、`judgment`、原 `category` 等 |
| **`results/qa_0_eval_summary.json`** | `category_stats`、`overall_accuracy`、`errors` 等汇总 |
| **`log/qa_eval.log`** | mosaic 详细日志（`MOSAIC_SERVER_LOG_BASENAME=qa_eval.log`） |

等价地也可用 **`experiments/01_locomo_benchmark/qa_eval.py`**，将 `--graph` / `--tags` 指向上表中的 `artifacts` 路径，`--method llm`，`--out` 指到本目录 `results` 或任意目录。

## 后台构图

```bash
chmod +x start_background.sh
./start_background.sh
```

## 其他日志文件

| 位置 | 内容 |
|------|------|
| `log/task_stdout.log` | 后台构图时控制信息 + tqdm |
| `log/run_verbose.log` | `--verbose-log` 时的环境与路径 |
| `log/conv0_progress.txt` | 批次与对话消息进度 |
| `log/mosaic_server.log` | 构图阶段 mosaic 详细日志 |
| `log/run.pid` | 后台 PID |

## 环境变量（`run.py` / `run_qa_eval.py` 自动设置）

| 变量 | 构图 `run.py` | 评测 `run_qa_eval.py` |
|------|---------------|------------------------|
| `MOSAIC_LOG_DIR` | `log/` | `log/` |
| `MOSAIC_SERVER_LOG_BASENAME` | `mosaic_server.log` | `qa_eval.log` |
| `MOSAIC_PROGRESS_FILE` | `log/conv0_progress.txt` | （不设置） |
| `GRAPH_SAVE_DIR` | `artifacts/graph_snapshots/` | （不设置） |

子进程构图时：`PYTHONPATH` = 仓库根，`cwd` = 仓库根。
