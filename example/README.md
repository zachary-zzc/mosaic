# Mosaic 示例数据与测试

本目录提供最小示例数据与可执行脚本，用于验证 mosaic 构图与查询流程。

## 文件说明

- **conv_small.json**：最小对话数据（单 session、5 条消息），格式与 `locomo_conv*.json` 一致。
- **qa_small.json**：2 道示例 QA，用于查询测试。
- **build_minimal_graph.py**：仅用 Python 标准库 + `networkx` 构建最小图与 tags，**不依赖 mosaic 或 LLM**。
- **run_example.py**：加载示例图与 QA，执行检索（可选 LLM 作答与评判）。

## 运行方式（在项目根目录 LongtermMemory 下）

### 1. 构建最小图与 tags（无需 API）

```bash
# 仅需 networkx
pip install networkx
python example/build_minimal_graph.py
```

会在 `example/output/` 下生成 `graph_small.pkl` 和 `tags_small.json`。

### 2. 运行示例（需安装 mosaic 依赖）

```bash
# 安装 mosaic 依赖（见 mosaic/requirements.txt）
pip install -r mosaic/requirements.txt

# 仅测试加载与 TF-IDF 检索（不调用 LLM）
PYTHONPATH=mosaic python example/run_example.py --no-llm

# 完整 QA（需在 mosaic/config/config.cfg 中配置 API key）
PYTHONPATH=mosaic python example/run_example.py --max-questions 2
```

配置 API：将 `mosaic/config/config.cfg` 中的 `ali_api_key` 和 `ali_base_url` 设为有效值，或通过环境变量 `MOSAIC_CONFIG_PATH` 指定配置文件路径。

### 3. 在 mosaic 内运行（以 mosaic 为当前目录）

```bash
cd mosaic
python -m src.query   # 或先配置 batch 中的路径后执行
```

## 说明

- 构图完整流程（从 `conv_small.json` 到图）需要 LLM，见 `mosaic/src/save.py`；本示例仅提供预构建的最小图以便快速测试查询。
- 若本地缺少模型或 API，使用 `--no-llm` 可验证图加载与检索链路是否正常。
