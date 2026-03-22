# oop_graph
OOP-based graph structure management conversation memory

## Conda 环境与依赖

使用名为 **`mosaic`** 的 conda 环境（与本仓库文档一致）：

```bash
conda activate mosaic
cd mosaic
pip install -r requirements.txt
```

若尚未创建环境：在本目录执行 `conda env create -f environment.yml`，再 `conda activate mosaic` 并 `pip install -r requirements.txt`。**必须在激活 `mosaic` 后再运行 Python**，否则会出现 `No module named 'keybert'` 等错误。

## 子程序与 CLI

| 子程序 | 说明 |
|--------|------|
| **构图** | `src/save.py`：`save` / `save_hash`，按 `conv_message_splitter` 分批（每批最多 10 条消息），主进度为 **tqdm**；详细 TF-IDF/LLM 步骤见日志文件 DEBUG。 |
| **查询** | `src/query.py`：加载 `graph_network*.pkl` + TF-IDF **tags** JSON 后检索并生成答案。 |
| **整套** | 先构图写出 `--graph-out` 与 `--tags-out`，再对同一文件执行查询。 |

命令行（将 `mosaic` 目录加入 `PYTHONPATH`）：

```bash
cd mosaic && PYTHONPATH=. python -m mosaic --help
cd mosaic && PYTHONPATH=. python cli.py --help
```

常用子命令：`build`、`query`、`run`（构图后立刻查）、`chat`（加载图后交互问答）。默认控制台仅 WARNING，构图进度看 **tqdm**；需要终端里打出 TF-IDF/检索细节时用 **`-v` / `--verbose`**（控制台 DEBUG）。

## 脚本入口（兼容）

构图仍可直接调用 `src/save.py`；查询调用 `src/query.py`。
