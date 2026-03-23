# 数据集说明

本仓库实验所用数据集及与稿件的对应关系。数据本身不纳入版本控制（大文件、隐私或许可限制），此处仅记录来源、格式与使用方式。

---

## 已有数据集

### 1. LoCoMo（Long Context and Memory）

- **用途**：长程对话记忆评估（单跳、多跳、时序查询），用于 **Appendix LoCoMo** 及主文对记忆模块的引用。
- **在本项目中的位置**：程序侧见 `mosaic/src/locomo results/`（对话、图、QA 等运行结果）；**原始基准数据与评估协议**需按 LoCoMo 官方发布获取（论文/代码库中的 split、指标定义）。
- **实验**：`experiments/01_locomo_benchmark`（主 LoCoMo 表、分类与检索指标；见 `experiments/README.md`）。
- **备注**：Manuscript 称使用“标准化 LoCoMo 协议”以便与 Mem0 等公平对比；若与 Mem0 官方采用的 LoCoMo 设置不同，需在文中明确说明。

### 2. HaluMem

- **来源**：[MemTensor/HaluMem](https://github.com/MemTensor/HaluMem)（操作级幻觉评估基准，面向 agent 记忆系统）。
- **内容**：Halu-Medium（约 160k tokens/用户）、Halu-Long（约 1M tokens/用户）；多轮对话、memory points、QA、记忆提取/更新/问答任务。
- **用途**：若在 manuscript 中增加**记忆质量/幻觉**相关实验或讨论，可用 HaluMem 做记忆提取准确性、更新一致性、幻觉率等评估；当前稿中未强制要求，属可选扩展。
- **获取**：Hugging Face [IAAR-Shanghai/HaluMem](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)；评估代码见 HaluMem 仓库 `eval/`。
- **实验**：暂无专属实验子目录；若采用可新增 `experiments/10_halumem` 或在现有记忆相关实验中引用。

---

## 实验所需但可能尚未就绪的数据

以下按**实验/稿件需求**列出，便于后续补齐或与外部合作。

### Domain 1：Hypertension（高血压临床问诊）

- **稿件**：Methods § Evaluation domains、Table 图结构、Table 1–2、消融、下游 Table 7、因果分析。
- **描述**：来自 DrHyper；专家标注的患者档案，用于实体图构建与评估。Manuscript 提及约 200 例主评估 + 58 例消融。
- **需具备**：每例结构化 profile（人口学、病史、检验、用药、生活方式等）；与 DualGraph 实体列表对齐的 ground truth；若做下游则需诊断/风险分层/用药合理性等标签。
- **状态**：若 DrHyper 已包含上述 200+58 例，则直接使用；否则需在 `dataset/` 下注明实际来源与规模（如 `dataset/hypertension/` 说明）。

### Domain 2：Diabetes（糖尿病临床问诊）

- **稿件**：与 Domain 1 并列的四域之一；Table 1–2、图结构、下游等。
- **描述**：基于 ADA 标准的 2 型糖尿病初评实体（HbA1c、空腹血糖、BMI、用药、并发症筛查等）；需医生撰写或标注的患者档案 [N] 例。
- **需具备**：与 Domain 2 实体图一致的 profile；下游指标若做则需相应标签。
- **状态**：需确定 [N] 及数据来源（自建合成、合作医院、或公开脱敏数据），并在本 README 或 `dataset/diabetes/` 中说明。

### Domain 3（非医疗，示例：保险理赔 intake）

- **稿件**：Abstract/Intro 中“Domain 3”、Table 1–2、Methods 评估域、图结构表、下游表等。
- **描述**：如车险理赔：事故时间地点、当事人、损伤、警察报告、证人、保单、既往理赔、责任判定等实体。
- **需具备**：结构化 case profiles（合成或合作方脱敏）；与 Domain 3 实体图一致的 ground truth；若做下游则需理赔/责任等相关决策标签。
- **状态**：需确定域名（如“Insurance claims”）、实体数 |V|、深度 L 及数据来源；暂无则标注“待补充”。

### Domain 4（非医疗，示例：IT 技术支持分流）

- **稿件**：与 Domain 3 对称；同上。
- **描述**：如企业 IT 支持：症状、受影响系统、报错、近期变更、环境、网络、复现步骤、业务影响、临时方案等。
- **需具备**：结构化 case profiles（合成工单或合作方）；与 Domain 4 实体图一致的 ground truth；下游可为解决率、升级准确性、解决时间等。
- **状态**：需确定域名（如“IT support triage”）、|V|、L 及数据来源；暂无则标注“待补充”。

### Emergence 测试集（演化 DAG）

- **稿件**：Table 3、Figure 3、Results §3.3。
- **描述**：每域除预定义实体外，增加 [K] 个“涌现”实体（未出现在初始图中但属于合理意外，如少见合并症、非标配置）。
- **需具备**：专家设计的涌现实体列表及在 case 中的标注；或带“预定义 + 涌现”双标注的 case profiles。
- **状态**：可与 Domain 1–4 的 case 同源，在现有 profile 上增加涌现实体标注；若某域暂无，对应 Table 3 列可先留空或只做部分域。

### 前瞻临床试点（Pilot）

- **稿件**：Results §3.2.2、Appendix Pilot。
- **描述**：真实患者、 clinician-in-the-loop、高血压门诊；N=30–50，IRB、知情同意、安全与干预记录。
- **需具备**：伦理批件、纳入/排除标准、逐例去标识化结果、医生反馈与安全事件日志。
- **状态**：若尚未开展，可在 manuscript 中写“拟进行”并在此注明“待 IRB 与临床合作”。

---

## 建议的目录布局（仅说明，不强制提交数据）

```
dataset/
├── README.md                 # 本文件
├── locomo/                   # LoCoMo 基准数据与协议说明（或链接）
├── halumem/                  # 若使用 HaluMem，可放说明与下载脚本
├── hypertension/            # DrHyper / 高血压 200+58 例说明或链接
├── diabetes/                # 糖尿病 [N] 例说明或链接
├── domain3_insurance/       # 或 domain3_*，按最终域名
├── domain4_itsupport/       # 或 domain4_*
└── pilot/                   # 试点协议与结果说明（若执行）
```

每个子目录可仅含 `README.md` 描述来源、格式、许可与使用实验列表，不提交原始数据文件。

---

## 与 experiments/ 的对应

当前 `experiments/` 仅保留与 **MOSAIC 对话记忆评测** 相关的三项实验（详见 `experiments/README.md`）：

| 实验目录 | 主要数据依赖 |
|----------|--------------|
| `01_locomo_benchmark` | LoCoMo（10 段多会话对话 + QA）；程序侧结果见 `mosaic/src/locomo results/` |
| `02_scalability` | 合成对话（S-100 … S-2000，按实验指南生成） |
| `03_ablation` | 与 Exp 1 相同 LoCoMo；在 MOSAIC 内开关组件后重跑 |
| （可选）HaluMem | HaluMem 数据集 + eval 脚本 |

若某实验需要**额外**数据集，在本 README 对应小节或该实验目录下的 `README.md` 中补充说明即可。
