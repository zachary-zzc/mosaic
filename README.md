# LongtermMemory

本仓库包含长期记忆相关的研究代码与论文稿件。

## 目录结构

| 目录 | 说明 |
|------|------|
| **mosaic/** | 主程序：基于 OOP 的图结构对话记忆管理与查询 |
| **Manuscript/** | 论文稿件（LaTeX） |
| **experiments/** | MOSAIC 对话记忆评测三项实验（LoCoMo / 可扩展性 / 消融）；见 [experiments/README.md](experiments/README.md) |
| **dataset/** | 数据集说明与来源（LOCOMO、HaluMem 等；见 [dataset/README.md](dataset/README.md)） |
| **docs/** | 稿件填充说明：placeholder 索引、图表规格、填充顺序（见 `docs/*.md`） |

## Mosaic（程序）

- 图构建与持久化：运行 `save.py`
- 图查询：运行 `query.py`

详见 [mosaic/README.md](mosaic/README.md)。

## Manuscript（稿件）

- 主文件：`Manuscript/manuscript.tex`
- 使用 LaTeX 编译生成 PDF。

## 使用说明

- 程序在服务器环境运行，本地通常无数据集与模型；嵌入模型等大文件未纳入版本控制，需在部署环境中另行配置。
- 本仓库的 `git add`、`git commit`、`git push` 由维护者手动执行。
