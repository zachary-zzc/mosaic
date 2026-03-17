# Manuscript Placeholder 索引

本文档将 `Manuscript/manuscript.tex` 中的占位符与实验/文档产出一一对应，便于按项填充。

---

## 使用说明

- **`\placeholder{...}`**：黄色框，需替换为最终正文或删除框保留内容。
- **[X]、[Y]、[N] 等**：需填入具体数字或范围。
- **`\placeholder{Domain 3}` / D3 / D4**：需先定域名（如 Insurance / IT support），再全局替换。

填充顺序建议：先定 Domain 3/4 名称与数据 → 跑 01–09 实验 → 按表/图/段落逐项填 → 最后补引言文献与附录。

---

## 按稿件位置

### Abstract

| 位置 | 占位内容 | 来源 |
|------|----------|------|
| 领域举例 | `\placeholder{e.g., insurance claims, technical support triage}` | 与 Domain 3/4 最终命名一致 |
| 数值 | 静态图退化 [X]%、DualGraph [Y]%；[N] 例；获得率 [X]%、一致性 [Y]% | 01_memory_gap + 08_pilot（若做） |
| 结尾 | `\placeholder{Refine after experiments.}` | 实验完成后删或改为一句总结 |

### Section 1 Introduction

| 位置 | 占位内容 | 来源 |
|------|----------|------|
| 第二段 | `\placeholder{Paragraph 2: Quantify the problem...}` | 文献调研：Sharma et al.、客服/临床/法律证据、WebArena/AgentBench 终止率 |
| 贡献列表 | `\placeholder{Domain 3}, \placeholder{Domain 4}`；[N] real patient interactions | 定域名；08_pilot 样本量 |

### Section 3 Results

| 小节 | 表格/图 | 占位与来源 |
|------|----------|-------------|
| §3.1 Memory gap | Table 1 | 四域每行 Turns、Acq.、Conc.、Un-om.、p → **01_memory_gap** |
| §3.1 | Figure 1 | 四 panel 描述在 placeholder 内 → 出图后删框 |
| §3.1 | 正文 | [X–Y]%、[A]–[F] 中位轮数、Cohen's d、bootstrap CI → **01_memory_gap** |
| §3.1 | 统计 | `\placeholder{Statistical analysis...}` → 01 效应量与 CI |
| §3.2 Baselines | Table 2 | 7×4 域 Acq./Conc./Trn 每格 → **02_baselines** |
| §3.2 | Figure 2 | 四 panel → 02 出图 |
| §3.2 | 正文 | 70–80%、Domain X L=Y → 02 结果 |
| §3.2 | 统计 | `\placeholder{Statistical comparisons...}` → 02 |
| §3.3 Evolving DAG | Figure 3 | 五 panel → **03_evolving_dag** |
| §3.3 | 正文 | [M]、[K]、[X]%、[Y]%、\|U_t\|、\|V^(t)\|、[Z]% → 03 |
| §3.3 | Table 3 | Pre-specified / Emergent Acq.、Graph growth → 03 |
| §3.3 | 定性 | `\placeholder{Qualitative examples...}` → 03 对话摘录 |
| §3.4 NCS | Table 4 | NCS vs Global 各格 → **04_ncs_mechanism** |
| §3.4 | Figure 4 | 五 panel → 04 |
| §3.4 | 叙述 | `\placeholder{Narrative...}` 与 [X]%、[Y]、[A]×、[Z] → 04 |
| §3.5 Ablation | Table 5 | 七条件 Acq.、ΔAcq.、Conc.、Turns、Coherence → **05_ablation** |
| §3.5 | 正文 | −X%、−Y%、−W%、[Z]% 等 → 05 |
| §3.6 Graph construction | Table 6 | 四域 F1、Auto/Auto+Review/Expert Acq.、Review min → **06_graph_construction** |
| §3.6 | 正文 | [X]%、[Y]%、[Z] min、[K]%、[W]% → 06 |
| §3.6 | Figure 5 | 四 panel → 06 |
| §3.7 Downstream | Table 7 | 各域各方法下游指标；D3/D4 域特定指标 → **07_downstream** |
| §3.7 | 正文 | [X]% (evolving)、因果链 → 07 |
| §3.7 | 因果 | `\placeholder{Causal analysis...}` → 07 |
| §3.7 Pilot | 整段 | `\placeholder{Pilot study design and results...}` → **08_pilot** |

### Section 4 Methods（Evaluation domains 等）

| 位置 | 占位内容 | 来源 |
|------|----------|------|
| Domain 1 | \|V\|、L、密度、社区、200+58 | 01/05 设定与 06 图结构 |
| Domain 2 | \|V\|、L、[N] 例 | 同 01/06 |
| Domain 3/4 | 名称、\|V\|、L、ground truth 来源 | 定名与 dataset/README |
| Table 图结构 | 四域 \|V\|、\|E_P\|、\|E_A\|、L、Δ_max、Communities | 06 或图构建 pipeline |
| Baselines / Simulators / Metrics | 大段 `\placeholder{...}` | 实现说明，与实验配置一致即可 |

### Appendix

| 小节 | 占位内容 | 来源 |
|------|----------|------|
| LoCoMo | Table 各方法各列；`\placeholder{Analysis...}`；`\placeholder{LoCoMo: ...}` 引用 | **09_locomo**；补 bibitem |
| Pilot | `\placeholder{IRB...}` 等 | **08_pilot** |
| Downstream | `\placeholder{Full clinical...}` | **07_downstream** |
| Compute | `\placeholder{Per-turn latency...}` | 04 + 单独计时实验 |
| Evolving | `\placeholder{A. Emergence detection...}` 等 | **03_evolving_dag** 扩展分析 |

---

## 按实验目录反查

| 实验 | 需填的 manuscript 位置 |
|------|------------------------|
| 01_memory_gap | Abstract 部分数值；§3.1 全段、Table 1、Figure 1、统计框 |
| 02_baselines | Table 2、Figure 2、§3.2 叙述与统计框 |
| 03_evolving_dag | Table 3、Figure 3、§3.3 叙述、定性框、Appendix Evolving |
| 04_ncs_mechanism | Table 4、Figure 4、§3.4 叙述与 Narrative 框、Appendix Compute 部分 |
| 05_ablation | Table 5、§3.5 叙述、Appendix 全表 |
| 06_graph_construction | Table 6、Figure 5、§3.6 叙述、Methods 图结构表 |
| 07_downstream | Table 7、§3.7 因果框、Appendix Downstream |
| 08_pilot | §3.7 Pilot 整段、Appendix Pilot |
| 09_locomo | Appendix LoCoMo 表与分析、bibitem |

---

## Domain 3 / Domain 4 全局替换列表

定名后建议全文替换（区分表格表头与正文）：

- `\placeholder{Domain 3}` → 如 `Insurance claims`
- `\placeholder{Domain 4}` → 如 `IT support triage`
- `\placeholder{D3}`、`\placeholder{D4}` → 同上或缩写（与表头一致）

可先用 `grep -n "Domain 3\|Domain 4\|D3\|D4" Manuscript/manuscript.tex` 核对位置后再替换。
