# Manuscript 填充指南

围绕“填满所有 placeholder、补全实验与说明”的目标，建议按以下顺序与检查清单执行。

---

## 1. 确定 Domain 3 与 Domain 4

- 从 manuscript 示例中二选一或自定：
  - **Domain 3**：如 Insurance claims intake（车险理赔等）。
  - **Domain 4**：如 IT technical support triage（企业 IT 支持分流）。
- 在 `dataset/README.md` 中写明域名、实体数 |V|、深度 L、ground truth 来源。
- 全文替换：`\placeholder{Domain 3}`、`\placeholder{Domain 4}`、`\placeholder{D3}`、`\placeholder{D4}`（见 `docs/placeholder-index.md`）。

---

## 2. 数据与评估协议

- **LoCoMo**：确认基准数据与标准协议来源，补 `\bibitem{locomo_ref}` 和 Appendix 分析段落。
- **DrHyper**：确认高血压 200+58 例可用；Diabetes、D3、D4 的 case 规模与来源在 `dataset/README.md` 写明。
- **Emergence**：为需要演化 DAG 的域设计“预定义 + 涌现实体”标注（至少 Hypertension，可选 D3）。

---

## 3. 实验执行顺序（建议）

1. **01_memory_gap**：四域 Memory-only vs DualGraph → Table 1、Figure 1、摘要与 §3.1 数值。
2. **02_baselines**：同一批 case 跑齐七方法 → Table 2、Figure 2、§3.2。
3. **04_ncs_mechanism**：Hypertension 上 NCS vs global、缩放与扰动 → Table 4、Figure 4、§3.4。
4. **05_ablation**：Hypertension 消融 → Table 5、§3.5。
5. **03_evolving_dag**：涌现实体与演化 DAG → Table 3、Figure 3、§3.3、定性例。
6. **06_graph_construction**：四域 Auto/Review/Expert 图与下游 → Table 6、Figure 5、Methods 图结构表。
7. **07_downstream**：四域下游指标与因果分析 → Table 7、§3.7、Appendix Downstream。
8. **09_locomo**：LoCoMo 各方法各列 → Appendix 表与分析、引用。
9. **08_pilot**：若执行则填 §3.7 Pilot 与 Appendix；否则保留“拟进行”并注明。

---

## 4. 按占位类型填充

- **数值 [X]/[Y]/[N]**：从对应实验表或统计结果中取；同一含义在全稿统一（如“每域 [N] 例”只用一个 N）。
- **`\placeholder{...}` 黄色框**：  
  - 若为“待写段落”（如统计、因果、定性例）：用最终正文替换整段，删除 `\placeholder{}` 与 tcolorbox。  
  - 若为“待填数字/词”：只填内容，保留或去掉框视排版需要。
- **表格空单元格**：每格对应 experiments/README 中该表的“产出”；先填数字，再统一检查小数位与显著性（如 *p*<0.05）。

---

## 5. 引言与附录

- **引言第二段**：补 2–3 篇文献（Sharma et al. Nature Medicine、客服/临床/法律证据、AgentBench/WebArena 终止率）并改写 `\placeholder{Paragraph 2...}`。
- **Appendix**：  
  - LoCoMo：表 + 分析 + 引用。  
  - Pilot：设计、结果、IRB、逐例说明（若做）。  
  - Downstream：完整下游与错误分析。  
  - Compute：延迟分解、成本、缩放、NCS 加速比。  
  - Evolving：涌现检测 P/R、图附着质量、NCS  containment、增长动态、失败模式、定性例。

---

## 6. 检查清单（提交前）

- [ ] Abstract 无未填 [X]/[Y]/[N] 或 placeholder。
- [ ] Domain 3/4 全稿命名一致。
- [ ] Table 1–7 与 Appendix 表无空单元格（或已标“N/A”并说明）。
- [ ] Figure 1–5 已生成并插入，对应 placeholder 已删。
- [ ] 所有 `\placeholder{Statistical...}`、`\placeholder{Causal...}` 等已替换为正文。
- [ ] 参考文献 `locomo_ref` 及文中 [cite] 已补全。
- [ ] 作者、单位、通信邮箱已定稿（若尚未可保留 [ ] 并注明）。

完成以上后，稿中应无黄色框与未定义占位符；若仍有“拟进行”内容（如 pilot），可在文中明确标注“planned”或“to be conducted”。
