# 图表规格说明

Manuscript 中 Figure 1–5 的 placeholder 内已给出各 panel 的设计要求，此处做简要汇总，便于绘图与排版。

---

## Figure 1 — Memory–Completeness Gap

- **(a)** 架构示意：双图（G_P, G_A）、NCS 更新前沿 U_t、置信门控记忆、LLM 控制器；演化 DAG 用虚线标出新节点附着区。
- **(b)** 四域 Entity acquisition rate (%)：Memory-only vs DualGraph，分组柱状图 + 95% CI。
- **(c)** 四域平均对话轮数（DualGraph 更长、更完整）。
- **(d)** 三条获得率随轮数曲线（代表三种难度），标注 community 以体现话题连贯。

---

## Figure 2 — Baseline Comparison

- **(a)** 四域 × 方法 的 entity acquisition，分组柱状图 + 95% bootstrap CI。
- **(b)** Hypertension 上各方法“获得率–轮数”学习曲线；DualGraph 单调上升，ReAct 早平台，Plan-and-Solve 前段上升后停滞。
- **(c)** 失败模式堆叠条：每方法按“提前终止 / 话题漂移 / 重复询问 / 前提违反”占比。
- **(d)** 横轴每轮效率、纵轴总获得率；DualGraph 高获得率且效率可接受；Checklist 在简单实体上效率高但依赖链上停滞。

---

## Figure 3 — Evolving Inference DAGs（核心图）

- **(a)** 单例三时刻：(i) 初始静态图 (ii) 某轮患者提出未预见的合并症，新节点与边、U_{t1} 高亮 (iii) 后续轮已纳入并解析；对比 static 忽略新信息。
- **(b)** 涌现实体上的获得率：DualGraph+Evolving vs static vs 其他基线；static 约 0%，Evolving 目标 [X]%。
- **(c)** 预定义实体上 static vs evolving 获得率，展示 evolving 不退化。
- **(d)** 每次节点附着时 |U_t| vs |V^(t)|，预期 |U_t| ≪ |V^(t)|。
- **(e)** 代表 case 的 |V^(t)|、|E_P^(t)| 随轮数，体现增长稀疏且趋于稳定。

---

## Figure 4 — NCS Mechanism

- **(a)** 每轮“实际得分改变节点集”vs NCS 预测 U_t；报告一致比例与差异大小。
- **(b)** 每实体得分轨迹的符号翻转次数：NCS vs global-recompute；NCS 振荡更低。
- **(c)** 每轮得分更新时间 (ms) vs |V|；NCS 近似平坦，global 近似线性。
- **(d)** |V| ∈ {10,20,40,60,80,100} 下 DualGraph / Checklist / ReAct 获得率；DualGraph 随 |V| 优势增大。
- **(e)** 边删/加/反 10/20/30/50% 时获得率下降 + NCS 预测“损伤半径”与实证损伤对比。

---

## Figure 5 — Graph Construction Validation

- **(a)** 流程：任务说明 → LLM 实体抽取 → LLM 前提识别 → 嵌入关联图 → 专家审阅。
- **(b)** 四域 Prereq. / Assoc. 边 P/R/F1。
- **(c)** Auto vs Auto+Review vs Expert 下游获得率（分组柱）；结论 auto+review ≈ expert。
- **(d)** 专家修正边的 NCS 邻域与下游性能变化是否限于该邻域。

---

## 表格

- 所有 `\placeholder{}` 或空单元格对应 `experiments/README.md` 中各实验产出；表注中的 [N]、[X] 等由同一实验给出。
- Domain 3/4 列名与正文命名保持一致（如 Insurance / IT support）。

图可先做矢量（PDF/SVG）便于 Nature Machine Intelligence 排版；表格保持 `booktabs` 风格，数值统一小数位与显著性标记。
