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

仓库内 **`experiments/`** 已按 [experiments/README.md](../experiments/README.md) 收敛为三项（对话记忆评测）；表/图编号以该 README 的 **Results Map** 为准。

1. **`01_locomo_benchmark`**：LoCoMo 上 B1–B8 与 MOSAIC；产出 Table 1–3、Figure 2–3、Figure 7 等。
2. **`02_scalability`**：合成对话 S-100…S-2000 上的可扩展性与效率；产出 Table 4–5、Figure 4–5。
3. **`03_ablation`**：MOSAIC 组件消融（C0–C6）；产出 Table 6、Figure 6。

统一跑法：`python experiments/run_all.py`（或分项进入各子目录 README）。

> 若稿件中仍保留旧版「四域临床 / 01–09 编号」占位，需将叙述与表格与上述新实验对齐后，再更新 `docs/placeholder-index.md` 对应行。

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
