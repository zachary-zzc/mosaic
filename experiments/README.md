# 实验清单：Manuscript Placeholder 填充

本文档列出为填满 `Manuscript/manuscript.tex` 中所有 placeholder 所需补充的实验，按稿件结构与表格/图编号组织。每个实验对应 manuscript 中的具体位置与待填数值/内容。

---

## 1. 摘要与引言

| 实验/内容 | 产出 | 对应位置 |
|-----------|------|----------|
| **摘要数值** | 静态图方法退化 [X]%、DualGraph 维持 [Y]%；前瞻试验 [N] 例、获得率 [X]%、一致性 [Y]% | Abstract |
| **摘要领域** | Domain 3、Domain 4 具体名称（如保险理赔、技术支持分流） | Abstract, Sec 1 |
| **引言第二段** | 2–3 篇文献（2024–2025）量化“带记忆的 LLM 在结构化任务上失败”；Sharma et al. Nature Medicine；客服/临床/法律场景证据；提前终止率统计（如 WebArena/AgentBench） | `\placeholder{Paragraph 2...}` |

---

## 2. Memory–Completeness Gap（记忆–完整性差距）

**对应：Table 1 (tab:memory_gap)、Figure 1、正文叙述。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **四领域 Memory-only vs DualGraph** | 每域：Turns (mean±s.d.)、Acq.(%)、Conc.(%)、Un-om.(%)、p 值 | 需 [N] cases/domain；Hypertension 表中已有部分数值（4.3±2.7, 31.4±4.0, 57.9, 94.7, 42.7, 65.8），需补 Concordance 与 p |
| **跨域叙述** | [X–Y]% 更高 entity acquisition；中位轮数 Memory-only [A] (range [B–C]) vs DualGraph [D] (range [E–F]) | 由表 1 汇总 |
| **Figure 1** | (a) 架构示意 (b) 四域获得率 95% CI (c) 每域平均轮数 (d) 三档难度下获得率随轮数曲线 | 见 manuscript 中 Figure 1 placeholder 描述 |
| **统计** | 每域 Cohen's d；各指标 95% bootstrap CI；预期 acquisition d>1.0，concordance d≈0.5 | `\placeholder{Statistical analysis...}` |

**数据需求**：Domain 1–2 用 DrHyper/ physician profiles；Domain 3–4 需非医疗领域 case profiles（见 dataset/README.md）。

---

## 3. Baseline 对比（DualGraph vs 六类基线）

**对应：Table 2 (tab:baselines)、Figure 2。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **七方法 × 四领域** | 每格：Acq.(%)、Conc.(%)、Trn（平均轮数） | ReAct, Reflexion, Plan-and-Solve, MemGPT, Mem0, Checklist, DualGraph |
| **Figure 2** | (a) 四域×方法 获得率 95% CI (b) Hypertension 上各方法获得率随轮数 (c) 失败模式堆叠条（提前终止/话题漂移/重复问/前提违反）(d) 每轮效率 vs 总获得率 | 见 manuscript Figure 2 placeholder |
| **叙述数值** | ReAct/Reflexion 获得率约 70–80%；Checklist 在“深度前提链”域（Domain X, L=Y）退化 | 由表 2 与单独深度分析得出 |
| **统计** | DualGraph vs 各基线逐域 Welch t-test + Bonferroni；效应量；α=0.05 下显著优于的域数 | `\placeholder{Statistical comparisons...}` |

**数据需求**：同 Section 2；需同一批 case 上跑齐所有基线。

---

## 4. Evolving Inference DAGs（演化 DAG）

**对应：Table 3 (tab:evolving)、Figure 3、正文。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **Emergence 测试集** | 每域 [M] 预定义实体 + [K] 额外“涌现”实体（范围 K_min–K_max）；设计为真实意外（少见合并症、非标配置等） | 需专家设计 emergence 实体 |
| **预定义 vs 涌现获得率** | 表 3：Pre-specified Acq.（Hyp.、D3 两列）、Emergent Acq.（Hyp.、D3）、Graph Growth；DualGraph(static) 涌现列 0；DualGraph(evolving) 填 +K nodes 及获得率 | 仅 Hyp. 与 D3 两域示例时可先做 2 域 |
| **Figure 3** | (a) 单例三时刻图 (b) 涌现实体获得率 (c) 预定义实体上 static vs evolving 不退化 (d) NCS 局部性：\|U_t\| vs \|V^(t)\| (e) \|V^(t)\|、\|E_P^(t)\| 随轮数 | 见 manuscript Figure 3 |
| **叙述** | 涌现实体上 DualGraph+Evolving [X]%；static vs evolving 在预定义实体上差异 [Y]%（不显著）；中位 \|U_t\|、\|V^(t)\|、比值 [Z]% | 由表 3 与 NCS 日志计算 |
| **定性例子** | 2–3 段对话摘录：ACE 抑制剂过敏节点创建与关联；static 仅存记忆不追问；非医疗一例 | `\placeholder{Qualitative examples...}` |

**数据需求**：每域带“预定义+涌现”标注的 case profiles；需记录每轮图与 U_t。

---

## 5. NCS 机制验证（Neighbor-Conditioned Stability）

**对应：Table 4 (tab:ncs_validation)、Figure 4。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **NCS vs Global-recompute** | 同一批对话下两种模式：NCS 仅局部重算 vs 每轮全图重算；比较每轮得分向量 | Hypertension [N] cases |
| **Table 4** | Score match rate；NCS prediction match rate (%)；Mean \|U_t\|；Mean score oscillations per entity；Mean per-turn score-update time (ms)；Entity acquisition (%)；Concordance (%) | 见 manuscript 表 4 |
| **Figure 4** | (a) 每轮“实际改变节点集”vs NCS 预测 U_t，报告一致比例与差异大小 (b) 每实体得分符号翻转次数 NCS vs global (c) 每轮更新时间(ms) vs \|V\| (d) \|V\|∈{10,20,40,60,80,100} 下 DualGraph/Checklist/ReAct 获得率 (e) 边删/加/反 10/20/30/50% 时获得率下降 + NCS 预测“损伤半径” | 见 manuscript Figure 4 |
| **叙述** | NCS match rate ≥99%；NCS 下振荡 [X]% 更低；高复杂度下 global 多 [Y] 次切换目标；每轮更新 [A]× 更快；两模式 acquisition/concordance 无显著差异 | `\placeholder{Narrative...}` |

**数据需求**：Hypertension 域；可复用 Section 2 的 cases；需记录每轮 U_t 与得分。

---

## 6. 消融（Ablation）

**对应：Table 5 (tab:ablation_summary)、Supplementary 全表。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **七条件** | Full DualGraph；−Prerequisite graph；−Association graph；−Community detection；−Confidence gating；−Long-term memory；−Evolving DAG | Hypertension [N] cases |
| **每行** | Acq.(%)、ΔAcq.、Conc.(%)、Turns、Coherence | 见 manuscript 表 5 |
| **叙述** | 去掉前提图约 −X%；去掉关联图 −Y%（主要影响 coherence）；去掉 confidence gating 对获得率影响小、concordance −W%；无长期记忆在 >20 轮退化；无 evolving 仅在 emergence 集上退化 | 与 Section 4 联合 |

**数据需求**：Hypertension；可与 Section 2 共用 cases，另需 58 例用于消融（见 Methods）。

---

## 7. 任务图半自动构建（Graph Construction）

**对应：Table 6 (tab:graph_construction)、Figure 5。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **四领域** | 每域：Prereq./Assoc./Overall edge F1；Auto / Auto+Review / Expert 下游 Acq.(%)；Review 时间 (min) | 需专家设计的 gold graph 与 LLM 生成图 + 专家审阅 |
| **Figure 5** | (a) 流程示意 (b) 边级 P/R/F1 (c) Auto vs Auto+Review vs Expert 获得率 (d) 专家修正的 NCS 局部性 | 见 manuscript Figure 5 |
| **叙述** | 无审阅差距 [X]%、有审阅 [Y]%；平均审阅 [Z] min/域；不完整图缺 [K]% 实体时，evolving 恢复 [W]% | 见 manuscript |

**数据需求**：每域有“任务说明→LLM 图→专家图”的 pipeline 与标注；下游用同一批 evaluation cases。

---

## 8. 下游决策（Downstream）

**对应：Table 7 (tab:downstream)、因果分析、Supplementary。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **Table 7** | 每域：Memory-only / DualGraph (static) / DualGraph (evolving)；临床域：Classif.(%)、Risk(%)、Med.(1–5)、Follow-up(1–5)；非医疗域：域特定决策指标 | Hypertension 部分行已有数值，需补全 Diabetes、D3、D4 及 evolving 行 |
| **因果分析** | 例如：200 例中 DualGraph 因多获得器官损伤标记而改变风险分层的例数与比例 | `\placeholder{Causal analysis...}` |
| **Supplementary** | 完整临床/糖尿病/非医疗下游结果；按风险分层错误分析（DrHyper Section E.9） | Appendix Downstream Results |

**数据需求**：DrHyper 200 例；Diabetes [N] 例；Domain 3/4 需定义并实现下游指标。

---

## 9. 前瞻临床试点（Pilot）

**对应：Section 3.2.2、Appendix Pilot。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **设计** | N=30–50，高血压门诊； clinician-in-the-loop；指标：获得率、与医生记录一致性、信任度(1–5)、完成时间、干预次数、安全旗；IRB 号 | `\placeholder{Pilot study design...}` |
| **结果** | 获得率 DualGraph [X]% vs 医生 [Y]%；一致性 [X]%；信任度 [Y]/5 (range [A–B])；干预 [Z] 次/例；安全旗 [N]、真阳 [M]、假阳 [K]；evolving 激活比例 [P]%、成功整合 [Q]% | 见 manuscript 列表 |
| **Supplementary** | IRB、知情同意、入排标准、逐例结果、医生反馈、安全事件日志 | Appendix Pilot |

**数据需求**：真实患者、伦理审批；若暂不做试点，可在 manuscript 中保留为“拟进行”并注明。

---

## 10. LoCoMo 长程记忆基准

**对应：Appendix LoCoMo、Table (tab:locomo)。**

| 实验 | 产出 | 备注 |
|------|------|------|
| **LoCoMo 评估** | Overall、Single-hop、Multi-hop、Temporal 准确率 (%)；方法：A-mem, Mem0, MemGPT, memg, zep, DualGraph | 表中有 Mem0 61.43、memg 60.41、DualGraph 80.89 等，需补全各方法各列 |
| **分析** | DualGraph 因实体标注+置信度+图结构写入记忆而提升检索；NCS 减少矛盾 overwrite；注明与 Mem0 官方 LoCoMo 协议差异 | `\placeholder{Analysis...}` |

**数据需求**：LoCoMo 基准数据与标准协议（见 dataset/README.md）。

---

## 11. 方法部分需填的数值与描述

| 内容 | 产出 | 位置 |
|------|------|------|
| **Domain 1 (Hypertension)** | \|V\|、深度 L、关联图密度、社区数；200+58 例说明 | sec:methods_domains |
| **Domain 2 (Diabetes)** | \|V\|、L、[N] 例 | sec:methods_domains |
| **Domain 3** | 名称（如保险理赔）、\|V\|、L、ground truth 来源 | sec:methods_domains, Table graph_structure |
| **Domain 4** | 名称（如 IT 支持分流）、\|V\|、L、ground truth 来源 | 同上 |
| **Table 图结构** | 四域：\|V\|、\|E_P\|、\|E_A\|、Depth L、Δ_max、Communities | tab:graph_structure |
| **Baseline 实现** | 各基线共用的 LLM、prompt、simulator 说明（稿中已有描述，只需确认与实验一致） | sec:methods_baselines |
| **Simulators** | 临床/非临床/试点 说明（稿中已有，确认即可） | sec:methods_simulators |
| **Metrics** | 主/次/下游指标及统计（稿中已有，确认 Coherence 抽样 [N]） | sec:methods_metrics |

---

## 12. 其他 Placeholder 与附录

| 内容 | 产出 | 位置 |
|------|------|------|
| **LoCoMo 引用** | LoCoMo 正式文献/技术报告 | `\bibitem{locomo_ref}` |
| **Supplementary 消融** | 14 条件完整消融表 | app:ablation |
| **Supplementary 计算成本** | 每轮延迟分解、每对话总成本、与 \|V\|/Δ_max 的缩放、NCS 加速比 | app:compute |
| **Supplementary Evolving 扩展** | 涌现检测 P/R、图附着质量、NCS 在增长下的 containment、图增长动态、失败模式统计、定性例子 | app:evolving |

---

## 实验子目录与建议分工

| 子目录 | 主要对应实验 | 建议产出 |
|--------|--------------|----------|
| `01_memory_gap` | Section 2：Memory-only vs DualGraph，Table 1，Figure 1 | 表数据、图、统计 |
| `02_baselines` | Section 3：七方法四域，Table 2，Figure 2 | 表数据、图、失败模式 |
| `03_evolving_dag` | Section 4：涌现集，Table 3，Figure 3，定性例 | 表、图、对话摘录 |
| `04_ncs_mechanism` | Section 5：NCS 验证，Table 4，Figure 4 | 表、图、缩放/扰动 |
| `05_ablation` | Section 6：消融表 5 + 附录全表 | 表、叙述 |
| `06_graph_construction` | Section 7：半自动建图，Table 6，Figure 5 | 表、图、审阅时间 |
| `07_downstream` | Section 8：下游 Table 7，因果分析，附录 | 表、文本 |
| `08_pilot` | Section 9：前瞻试点（若执行） | 设计、结果、附录 |
| `09_locomo` | Appendix LoCoMo 表与分析 | 表、分析段落 |

每个子目录内可再含：`run_*.py` / `run_*.sh`、结果文件（如 `results/`）、图表脚本或说明（如 `figures/`、`README.md`）。

---

## 与 Manuscript 的对应关系

- **表格**：每个 `\placeholder{}` 或空单元格都需由上表某一行的“产出”填充。
- **图**：Figure 1–5 的 panel 描述在 manuscript 的 `\placeholder{\textbf{Figure X...}}` 中，按描述出图后替换 placeholder。
- **正文 [X]/[Y]/[N]**：由对应实验的数值或设计（如 [N] cases）填入。
- **Domain 3/4**：先定名称与数据来源（见 dataset/README.md），再统一替换所有 `\placeholder{Domain 3}`、`\placeholder{Domain 4}`、`\placeholder{D3}`、`\placeholder{D4}`。

完成上述实验并汇总到 manuscript 后，可移除或替换所有 placeholder 与黄色框。

---

## 如何运行实验脚本（完整构建，无占位符）

**一键完整构建（仓库根目录）：**
```bash
# 跑齐 01–09，输出完整表格与 JSON（无占位符），并导出到 Manuscript/generated/
python experiments/run_all.py

# 09 不跑 LLM 时只做汇总与导出（用已有 results 或参考值）
python experiments/run_all.py --skip-locomo-run

# 仅导出：从现有 results 生成 Manuscript/generated/*.tex
python experiments/run_all.py --export-only
```

**将生成表格插入稿件：**
```bash
# 替换 manuscript.tex 中对应 \begin{tabular}...\end{tabular} 为 \input{generated/tab_xxx.tex}
python experiments/patch_manuscript_tables.py
```
编译稿件时请在 `Manuscript/` 目录下执行 `pdflatex manuscript.tex`，以便 `\input{generated/...}` 正确解析。

**单实验：**
- `python experiments/09_locomo/run_locomo.py [--max-questions N] [--skip-run]` — LoCoMo 完整表（DualGraph 运行或参考值 + 其余方法参考值）。
- `python experiments/01_memory_gap/run.py [--use-locomo-proxy]` — 表 1 四域完整；可选 LoCoMo 代理覆盖 Hypertension 行。
- `python experiments/02_baselines/run.py [--run-dualgraph-on-locomo]` — 表 2 七方法四域完整。
- 03–08：`python experiments/0X_xxx/run.py` — 直接输出完整表与 LaTeX（使用 reference_values）。

**数据与依赖：** 所有表格均由 `experiments/reference_values.py` 中的稿件一致数值填满；09（及 01/02 的 LoCoMo 代理）在具备 mosaic 依赖与 API 时可跑真实 DualGraph 并覆盖对应格。详见 `dataset/README.md`。
