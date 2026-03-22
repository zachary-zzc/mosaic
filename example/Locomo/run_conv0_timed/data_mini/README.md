# 极小数据集（冒烟）

- **`locomo_conv0_mini.json`**：从 `example/Locomo/locomo_conv0.json` 截取 `session_1` 前 **10** 条消息（约 1 个构图 batch），字段与全量一致。
- **`qa_0_mini.json`**：从 `example/Locomo/qa_0.json` 选取 **2** 道仅依赖 `D1:1`–`D1:10` 证据的题目，便于与 mini 对话对齐。

重新生成（可选）：

```bash
cd example/Locomo/run_conv0_timed
python data_mini/build_from_source.py
```
