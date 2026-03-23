*Thinking...*

> The user is making two key observations:
> 
> 1. The three benchmarks (HaluMem, LoCoMo, LongMemEval) already cover different domains/areas (personal conversations, clinical-style memory, long-term interactive memory), so the "four domains" requirement in the manuscript can be satisfied by these benchmarks rather than needing synthetic data generation.
> 
> 2. Some experiments can share computation - for example, the graph construction evaluation doesn't need a separate experiment because MOSAIC's graph is already constructed during the baseline experiment runs. Similarly, other measurements can be collected as byproducts of the same runs rather than requiring separate experiments.
> 
> Let me think about what can be consolidated:
> 
> 
> The core comparison experiment already generates everything I need—the graphs form naturally during MOSAIC's ingestion process, so I can evaluate their quality directly from those runs, and I can log NCS metrics and computational costs at the same time. This means the graph construction evaluation and NCS validation can both be extracted from the same baseline runs rather than requiring separate experiments.
> 
> For the mechanism validation, I need a separate dual-mode MOSAIC run to test the NCS component, but the ablation studies and graph perturbations can all happen on HaluMem-Medium, and I can compare those generated graphs against expert annotations to assess construction quality. For Experiment 2, I need to add dual-mode validation on HaluMem-Medium, run six component ablation variants, analyze graph perturbations, and evaluate the quality of auto-generated graphs from Experiment 1 through expert annotation. This consolidation brings everything down to two experiments, eliminating redundant computation while still capturing the necessary mechanism analysis. Medium, HaluMem-Long, LoCoMo, LongMemEval. Methods: MOSAIC (static), MOSAIC (evolving), Memory-Only MOSAIC, A-Mem, Letta, Mem0, Mem0-Graph, Chunked RAG, Full Context, Recency Buffer, plus published results for MemOS, Zep, Supermemory, Memobase on HaluMem.
> 
> From this single experiment I get memory-completeness gaps, full baseline comparisons, evolving versus static graph comparisons, per-category and per-type breakdowns, all computational metrics, graph statistics, and auto-generated graphs I can evaluate for quality.
> 
> For the mechanistic analysis, I need to run NCS dual-mode validation with global recompute logging on HaluMem-Medium, test six ablated MOSAIC variants, apply graph perturbations on subsampled data, and validate graph construction quality through expert annotation. Though I could theoretically capture the NCS validation during Experiment 1 with instrumentation to compute both local and global scores, I'll keep it separate for clarity. The graphs MOSAIC generates during Experiment 1 can serve as the baseline for comparing against expert annotations. The three existing benchmarks—HaluMem, LoCoMo, and LongMemEval—already cover different domains (personal assistant, social chat, and interactive memory), so I can map the manuscript's "four domains" framing to these benchmark conditions rather than creating separate evaluations. While the benchmarks don't directly include clinical intake tasks, HaluMem's persona/event/relationship structure and question types test the same underlying capabilities, so I'll need to reframe the manuscript from "four application domains" to something that better reflects what the benchmarks actually cover.
> 
> Now I'm consolidating the experimental plan into two main experiments plus an embedded computational measurement protocol. The key adjustments are that I don't need synthetic data generation since the benchmarks provide sufficient coverage, the graph construction evaluation reuses graphs built during the first experiment, the evolving versus static comparison runs both variants as separate methods within the first experiment, and NCS validation requires just one additional instrumented run as part of the second experiment rather than a standalone evaluation.

# Revised Experiment Plan for MOSAIC

---

## Overview

MOSAIC ingests multi-session conversations into a dual-graph structure (a prerequisite graph encoding logical dependencies and an association graph encoding semantic relatedness) and retrieves structurally grounded context at query time via graph traversal governed by neighbor-conditioned stability. This plan evaluates MOSAIC on three publicly available benchmarks that collectively span different conversation domains, memory operation types, and scale conditions: HaluMem (personal assistant dialogues with persona, event, and relationship memories across medium and long contexts), LoCoMo (long-horizon social conversations with multi-hop and temporal reasoning), and LongMemEval (long-term interactive memory across information extraction, multi-session reasoning, temporal reasoning, knowledge update, and abstention). HaluMem provides two scale conditions (Medium at approximately 160,000 tokens per user and Long at approximately 1,000,000 tokens per user), yielding four evaluation conditions total.

The plan contains two experiments. Experiment 1 runs every method on all four evaluation conditions in a single pass, simultaneously producing the core performance comparison, the memory-completeness gap analysis, the evolving-graph versus static-graph comparison, all per-category and per-type breakdowns, all computational timing measurements, and the automatically constructed MOSAIC graphs (which are saved as artifacts for later quality analysis). Experiment 2 performs mechanistic analysis that requires runs beyond Experiment 1: neighbor-conditioned stability dual-mode validation, component ablation, graph perturbation sensitivity, and graph construction quality evaluation against expert-annotated ground truth (using the same graphs already built during Experiment 1, with no redundant ingestion).

---

## Datasets

### HaluMem (Hallucination Evaluation for Memory Systems)

HaluMem was introduced by Chen et al. (2025) as the first operation-level hallucination evaluation benchmark tailored to agent memory systems. The dataset is publicly available on Hugging Face at https://huggingface.co/datasets/IAAR-Shanghai/HaluMem, and the evaluation code is available at https://github.com/MemTensor/HaluMem.

HaluMem provides two dataset versions sharing the same 20 synthetic users, each with a richly detailed virtual persona comprising core profile information (demographics, education, goals), dynamic state information (occupation, health, relationships), and preference information (food, music, hobbies). Each user's data includes multi-turn conversational sessions between the user and an assistant, structured memory points extracted from those sessions, and question-answer pairs for evaluation.

**HaluMem-Medium** contains 30,073 dialogues across 20 users, with an average of 70 sessions per user and an average context length of approximately 160,000 tokens per user. It includes 14,948 annotated memory points and 3,467 question-answer pairs.

**HaluMem-Long** contains 53,516 dialogues across the same 20 users, with an average of 120 sessions per user and an average context length of approximately 1,000,000 tokens per user. It uses the same 14,948 memory points and 3,467 question-answer pairs but introduces large-scale interference content (factual question-answering and math problems inserted between genuine sessions) to test robustness and hallucination resistance under noise.

Each memory point is categorized as one of three types: Persona Memory (facts about the user's identity, preferences, and traits), Event Memory (specific occurrences and activities), and Relationship Memory (connections between the user and other people). Each memory point carries metadata including a memory source field (primary, secondary, interference, or system), an is_update flag indicating whether it modifies a previously established fact, an original_memories field linking to the superseded memory when applicable, an importance score between 0 and 1, and a timestamp.

HaluMem defines three evaluation tasks. The Memory Extraction task measures how well a system identifies and stores factual information from dialogue sessions, evaluated through Recall (fraction of reference memory points captured), Weighted Recall (importance-weighted recall), Target Precision (fraction of extracted memory points corresponding to reference points), Accuracy (precision of extracted memory values), False Memory Resistance (ability to avoid storing hallucinated or irrelevant memories), and F1 score combining Recall and Target Precision. The Memory Update task measures whether the system correctly modifies existing memories when new dialogue provides updated or contradictory information, evaluated through Correct Rate, Hallucination Rate, and Omission Rate. The Memory Question Answering task measures end-to-end ability to produce accurate answers from stored memories, evaluated through Correct Rate, Hallucination Rate, and Omission Rate. Additionally, HaluMem reports typewise extraction accuracy across Event, Persona, and Relationship categories, and question-answering performance across six question types: Basic Fact Recall, Dynamic Update, Multi-hop Inference, Generalization and Application, Memory Conflict, and Memory Boundary.

HaluMem has published leaderboard results for six memory systems (MemOS, Zep, Mem0, Mem0-Graph, Supermemory, Memobase) using a standardized evaluation protocol. For these six systems, this plan uses their published numbers directly rather than re-running them, since the evaluation protocol is identical. Human annotation of HaluMem confirms high data quality: 95.70% accuracy, 9.58 out of 10 relevance, and 9.45 out of 10 consistency across a review of over 50% of HaluMem-Medium by eight annotators.

### LoCoMo (Long Conversation Memory Benchmark)

LoCoMo was introduced by Maharana et al. (2024) as a benchmark for evaluating whether systems can recall information from long, multi-session conversations. The dataset is available at https://github.com/snap-stanford/LoCoMo and on Hugging Face.

The dataset contains 10 multi-session conversations, each simulating a long-term relationship between two speakers across roughly 600 turns spanning multiple sessions with temporal gaps. Conversations cover everyday topics including work, family, hobbies, health, travel, and personal milestones. Alongside the conversations, the dataset provides over 300 question-answer pairs, each annotated with a category label and a set of evidence utterances (the specific conversation turns containing the information needed to answer the question).

Question categories are: Single-hop (recalling one explicitly stated fact), Multi-hop (combining two or more facts from different conversation segments), Temporal (reasoning about when events occurred or their ordering), Open-ended (synthesizing multiple pieces of information into a coherent response), and Adversarial (asking about things never discussed, testing correct abstention). Each question-answer pair includes explicit evidence links to specific conversation turns, enabling direct measurement of retrieval recall against ground-truth evidence.

LoCoMo is the standard benchmark for conversation memory systems. Mem0 reports 26% higher accuracy over OpenAI Memory on LoCoMo in their published paper (arXiv:2504.19413). All methods in this plan are evaluated fresh on LoCoMo using the same shared answering model to ensure fair head-to-head comparison; published numbers serve only as sanity checks.

### LongMemEval (Long-Term Memory Evaluation)

LongMemEval was introduced by Wu et al. (2024) as a benchmark for evaluating long-term interactive memory in chat assistants. The dataset is available at https://github.com/xiaowu0162/LongMemEval.

LongMemEval contains approximately 500 hand-curated test questions designed to evaluate five core long-term memory abilities: Information Extraction (recalling a single fact from a specific session), Multi-Session Reasoning (combining information stated across multiple sessions), Temporal Reasoning (questions involving time ordering, duration, or recency), Knowledge Update (correctly reflecting changed or updated information rather than outdated facts), and Abstention (correctly declining to answer when the requested information was never discussed). Each question is associated with a multi-session conversation history of varying total length, with gold-standard answers and category labels.

LongMemEval complements HaluMem and LoCoMo by testing Knowledge Update and Abstention as first-class evaluation categories and by originating from a different research group with different data generation assumptions.

### Domain Coverage Across Benchmarks

The three benchmarks collectively provide the domain diversity the manuscript requires without needing synthetic data generation. HaluMem's 20 users simulate a personal assistant scenario where the system must track evolving persona attributes, life events, and interpersonal relationships across dozens of sessions, making it structurally analogous to the manuscript's clinical intake and customer relationship domains (both involve systematic collection of structured information about a person over time, with updates and temporal evolution). LoCoMo tests social conversation memory between peers, covering the kind of episodic and relational information that arises in personal companion and informal interaction settings. LongMemEval tests general-purpose long-term memory that spans settings from factual record-keeping to preference tracking. The four evaluation conditions (HaluMem-Medium, HaluMem-Long, LoCoMo, LongMemEval) vary simultaneously in conversation length (160,000 tokens to 1,000,000 tokens), noise level (clean sessions versus interference-laden sessions in HaluMem-Long), memory operation emphasis (extraction and update in HaluMem versus pure question answering in LoCoMo and LongMemEval), and question reasoning type (six types in HaluMem, five in LoCoMo, five in LongMemEval with minimal overlap).

---

## Methods

### Shared Answering Model

All methods that involve a final answer-generation step share the same answering large language model (Qwen-2.5-32B-Instruct, temperature 0, maximum output length 512 tokens). Each method receives the same conversation input and the same query at evaluation time. The only variable across methods is the memory and retrieval mechanism.

### MOSAIC Variants

Three MOSAIC variants are run as separate "methods" within Experiment 1, sharing the same entity extraction pipeline, embedding model (BAAI/bge-large-en-v1.5), and answering model.

**MOSAIC (Evolving Graph).** The full system. Conversations are ingested sequentially. Entities are extracted and instantiated as graph nodes. Prerequisite edges encode logical dependencies; association edges encode semantic relatedness. The evolving-graph mechanism is active: when the extraction module detects information that does not map to any existing node (embedding cosine similarity below 0.85 against all existing nodes, confirmed by a language model verification call), a new node is created, prerequisite and association edges are attached (with directed acyclic graph cycle checking), and dirty flags are propagated through the neighbor-conditioned stability mechanism to the new node's neighborhood only. Community detection uses the Leiden algorithm. The scoring function combines normalized importance, precomputed PageRank centrality, and community continuity. At query time, graph traversal combines structural proximity with embedding similarity.

**MOSAIC (Static Graph).** Identical to the above except the evolving-graph mechanism is disabled. The graph topology is frozen after the initial ingestion phase. New sessions update belief distributions on existing nodes but cannot add new nodes or edges. This variant isolates the contribution of dynamic graph evolution.

**Memory-Only MOSAIC.** The dual-graph controller is disabled entirely. Entities are still extracted from conversations and stored with embeddings, but no prerequisite or association edges are constructed. At query time, retrieval is based solely on embedding similarity between the query and stored entity representations, with no graph traversal. This variant isolates the contribution of graph structure while holding the entity extraction pipeline, embedding model, and answering model constant.

During Experiment 1, all MOSAIC variants save their constructed graphs (node lists, edge lists, community assignments, belief distributions, and all metadata) as persistent JSON artifacts after ingesting each user or conversation. These saved graphs are reused in Experiment 2 for graph construction quality analysis and perturbation experiments, eliminating redundant ingestion.

### Baseline Methods

**Full Context.** The entire conversation is concatenated and passed as context to the answering model. No memory system is involved. If the conversation exceeds the model's context window (32,768 tokens for Qwen-2.5-32B-Instruct), the oldest turns are truncated. This baseline reveals degradation when conversations exceed the context window; for HaluMem-Medium (approximately 160,000 tokens per user) and HaluMem-Long (approximately 1,000,000 tokens per user), substantial truncation occurs.

**Recency Buffer.** Only the most recent 4,096 tokens of the conversation are retained as context. This represents a system with no long-term memory, providing a performance floor for questions about information from early sessions.

**Chunked Embedding Retrieval.** The conversation is divided into non-overlapping chunks of 10 turns each. Each chunk is embedded using BAAI/bge-large-en-v1.5. At query time, the 5 chunks with highest cosine similarity to the query embedding are retrieved and concatenated as context. Implemented using LangChain text splitting with FAISS as the vector index.

**Mem0.** The open-source universal memory layer for language model applications (https://github.com/mem0ai/mem0, version 1.0.7). Ingestion uses the add method with each turn sequentially, with a unique user identifier per conversation. Retrieval uses the search method. On HaluMem, published leaderboard results for Mem0 (standard) and Mem0-Graph are used directly since they follow the identical standardized protocol. On LoCoMo and LongMemEval, both variants are run from scratch using the official Python SDK with default configuration to ensure the shared answering model is used.

**A-Mem (Agentic Memory).** An agentic memory system that organizes memories using Zettelkasten-inspired principles with dynamic linking and memory evolution (https://github.com/agiresearch/A-mem). Uses ChromaDB for vector storage. Ingestion uses the add_note method with each turn's content. Retrieval uses the search_agentic method. Run from scratch on all benchmarks using the official repository with default configuration (embedding model: all-MiniLM-L6-v2, language model backend for agentic processing: gpt-4o-mini; the final answer is generated by the shared Qwen-2.5-32B-Instruct model).

**Letta (formerly MemGPT).** A framework for building stateful agents with tiered memory: core memory (always available), recall memory (searchable interaction log), and archival memory (indexed long-term store) (https://github.com/letta-ai/letta, version 0.16.6). The system autonomously promotes and demotes information across tiers. Ingestion uses the agent's message interface. At retrieval, the internal memory retrieval context is extracted and passed to the shared answering model. Run from scratch on all benchmarks using the official Python client with default configuration.

**Additional HaluMem Leaderboard Systems.** Published results for MemOS, Zep, Supermemory, and Memobase are included from the HaluMem leaderboard for the HaluMem conditions only. These systems are not re-run because their published numbers use the identical evaluation protocol. They are not evaluated on LoCoMo or LongMemEval because their public interfaces do not support the custom evaluation protocol required. Their inclusion provides additional comparison points for interpreting MOSAIC's HaluMem results.

### Summary of Methods per Evaluation Condition

| Method | HaluMem-Medium | HaluMem-Long | LoCoMo | LongMemEval | Source |
|---|---|---|---|---|---|
| MOSAIC (Evolving) | Run | Run | Run | Run | This work |
| MOSAIC (Static) | Run | Run | Run | Run | This work |
| Memory-Only MOSAIC | Run | Run | Run | Run | This work |
| Full Context | Run | Run | Run | Run | Baseline |
| Recency Buffer | Run | Run | Run | Run | Baseline |
| Chunked Embedding Retrieval | Run | Run | Run | Run | Baseline |
| Mem0 | Published | Published | Run | Run | Published + Run |
| Mem0-Graph | Published | Published | Run | Run | Published + Run |
| A-Mem | Run | Run | Run | Run | Run |
| Letta | Run | Run | Run | Run | Run |
| MemOS | Published | Published | — | — | Published |
| Zep | Published | Published | — | — | Published |
| Supermemory | Published | Published | — | — | Published |
| Memobase | Published | Published | — | — | Published |

Fresh runs on HaluMem: 8 methods × 2 dataset versions = 16 method-dataset combinations. Fresh runs on LoCoMo: 10 methods. Fresh runs on LongMemEval: 10 methods.

---

## Evaluation Metrics

### HaluMem Metrics

All HaluMem metrics follow the definitions and computation procedures in the HaluMem evaluation code exactly, ensuring comparability with published leaderboard results.

**Memory Extraction:** Recall, Weighted Recall, Target Precision, Accuracy, False Memory Resistance, F1. **Memory Update:** Correct Rate, Hallucination Rate, Omission Rate. **Memory Question Answering:** Correct Rate, Hallucination Rate, Omission Rate. **Typewise Accuracy:** Event, Persona, Relationship. **Question Type Accuracy:** Basic Fact Recall, Dynamic Update, Multi-hop Inference, Generalization and Application, Memory Conflict, Memory Boundary.

### LoCoMo Metrics

**Answer Accuracy (Language Model Judge).** GPT-4o (temperature 0) judges semantic correctness of the predicted answer against the gold answer, following the LoCoMo paper's protocol.

**Token-Level F1 Score.** Harmonic mean of token-level precision and recall between predicted and gold answers after lowercasing, punctuation removal, and stopword filtering.

**Retrieval Recall.** For methods with explicit retrieved context: fraction of gold evidence utterances covered by the retrieved context. Coverage is defined as verbatim appearance or cosine similarity above 0.85 (using BAAI/bge-large-en-v1.5).

**Per-Category Accuracy.** Accuracy broken down by Single-hop, Multi-hop, Temporal, Open-ended, and Adversarial.

### LongMemEval Metrics

Answer accuracy computed using the LongMemEval official scoring protocol. Per-category accuracy across Information Extraction, Multi-Session Reasoning, Temporal Reasoning, Knowledge Update, and Abstention.

### Computational Metrics (Collected During All Runs)

**Ingestion Latency.** Wall-clock seconds to ingest a complete conversation or user's full session history. Measured on a single NVIDIA A100 80GB GPU with 64 CPU cores and 256 gigabytes of RAM. Each measurement repeated three times; median reported.

**Query Latency.** Wall-clock milliseconds from query submission to retrieved context return, excluding the shared answering model call. Same hardware and repetition protocol.

**Peak Memory Usage.** Maximum RAM in megabytes during and after ingestion via Python's tracemalloc; GPU memory via torch.cuda.max_memory_allocated.

**Graph Statistics (MOSAIC Variants Only).** Node count, prerequisite edge count, association edge count, community count, maximum neighborhood size, mean neighborhood size. Recorded after ingestion for each user or conversation and saved to the persistent graph artifact.

---

## Experiment 1: Benchmark Evaluation

### Purpose

This single experiment produces all core quantitative results for the manuscript: the memory-completeness gap (Section 2.1), the full baseline comparison (Section 2.2), the evolving-graph versus static-graph comparison (Section 2.3), all per-category and per-type breakdowns (Sections 2.4 question-type and typewise analyses), downstream question-answering performance (Section 2.6), all computational timing and efficiency numbers (Sections 3.4 and Supplementary), and the automatically constructed graph artifacts used in Experiment 2. Running all three MOSAIC variants (Evolving, Static, Memory-Only) as separate methods within the same experiment eliminates the need for separate experiments to isolate the memory-completeness gap or the evolving-graph contribution.

### Procedure

**Step 1: HaluMem Evaluation.** For each of the 8 methods that require fresh HaluMem runs (MOSAIC Evolving, MOSAIC Static, Memory-Only MOSAIC, Full Context, Recency Buffer, Chunked Embedding Retrieval, A-Mem, Letta), execute the HaluMem evaluation pipeline on both HaluMem-Medium and HaluMem-Long. The pipeline follows HaluMem's official evaluation code: add each user's dialogue sessions sequentially through the system's ingestion interface, then run the three evaluation tasks (Memory Extraction, Memory Update, Memory Question Answering) using HaluMem's standardized evaluation scripts. For the 6 systems with published HaluMem results (MemOS, Zep, Mem0, Mem0-Graph, Supermemory, Memobase), transcribe their published numbers into the comparison tables with a footnote marking them as published results. During ingestion of each MOSAIC variant, save the complete graph artifact (nodes, edges, communities, belief distributions, metadata) as a JSON file for each user.

**Step 2: LoCoMo Evaluation.** For each of the 10 methods evaluated on LoCoMo (the 8 fresh-run methods plus Mem0 and Mem0-Graph run from scratch), ingest each of the 10 conversations independently, resetting the memory system between conversations. Submit each of the 300+ question-answer pairs to the appropriate conversation's memory system. Pass retrieved context and question to the shared answering model. Run the GPT-4o language model judge on every predicted-answer and gold-answer pair. Compute token-level F1. For methods with explicit retrieved context (Chunked Embedding Retrieval, Mem0, Mem0-Graph, A-Mem, Letta, all three MOSAIC variants), compute retrieval recall and retrieval precision against gold evidence utterances.

**Step 3: LongMemEval Evaluation.** For the same 10 methods, follow LongMemEval's official evaluation protocol: ingest each conversation history, submit each test question, generate answers with the shared answering model, and score using the official rubric.

**Step 4: Record Computational Metrics.** Throughout Steps 1 through 3, all ingestion latency, query latency, and peak memory usage measurements are recorded automatically by instrumentation wrappers around each method's ingestion and retrieval calls. For MOSAIC variants, graph statistics are extracted from the saved graph artifacts after each ingestion.

**Step 5: Statistical Analysis.** Compute 95% bootstrap confidence intervals by resampling question-answer pairs 1,000 times with replacement, separately for each evaluation condition. Perform paired bootstrap significance tests between MOSAIC (Evolving) and each other method. Compute Cohen's d effect sizes for MOSAIC (Evolving) versus Memory-Only MOSAIC (isolating graph structure), MOSAIC (Evolving) versus MOSAIC (Static) (isolating dynamic graph evolution), and MOSAIC (Evolving) versus the best-performing non-MOSAIC method (establishing state-of-the-art claim). Apply Bonferroni correction across the four evaluation conditions for each family of pairwise comparisons.

### Computational Effort

HaluMem: 8 fresh-run methods × 2 dataset versions × 20 users = 320 user-level ingestion and evaluation runs. Question-answering evaluations: 8 × 2 × 3,467 = 55,472.

LoCoMo: 10 methods × 10 conversations = 100 ingestion operations. 10 × approximately 300 question-answer pairs = approximately 3,000 evaluations. GPT-4o judge calls: approximately 3,000.

LongMemEval: 10 methods × approximately 500 questions = approximately 5,000 evaluations.

Total fresh evaluations: approximately 63,472. Total language model judge calls (GPT-4o, LoCoMo and LongMemEval only): approximately 8,000.

### Outputs

**Table 1 (Memory-Completeness Gap).** Fills manuscript Table tab_memory_gap. Rows: Memory-Only MOSAIC and MOSAIC (Evolving). Columns grouped by evaluation condition, with primary metric for each (F1 for HaluMem extraction, question-answering correct rate for HaluMem question answering, overall accuracy for LoCoMo and LongMemEval), plus 95% bootstrap confidence intervals, Cohen's d, and p-values. This table establishes that adding graph structure to the same memory pipeline substantially improves performance across all conditions.

**Table 2 (Full Baseline Comparison on HaluMem).** Fills manuscript Table tab_baselines for the HaluMem portion. All 14 methods in rows (8 fresh-run plus 6 published). Columns: Recall, Weighted Recall, Target Precision, Accuracy, False Memory Resistance, F1, Memory Update Correct/Hallucination/Omission, Question Answering Correct/Hallucination/Omission. Separate panels for HaluMem-Medium and HaluMem-Long. Best values bolded, second-best underlined.

**Table 3 (Full Baseline Comparison on LoCoMo and LongMemEval).** Fills manuscript Table tab_baselines for the non-HaluMem portion. All 10 methods evaluated on these benchmarks. Columns for LoCoMo: overall accuracy, token-level F1, retrieval recall, retrieval precision, per-category accuracy (Single-hop, Multi-hop, Temporal, Open-ended, Adversarial). Columns for LongMemEval: overall accuracy, per-category accuracy (Information Extraction, Multi-Session Reasoning, Temporal Reasoning, Knowledge Update, Abstention).

**Table 4 (Typewise Accuracy on HaluMem).** Extraction accuracy across Event, Persona, and Relationship memory types for all systems on both HaluMem-Medium and HaluMem-Long. Reveals whether MOSAIC's graph structure particularly benefits relational or event-based memories.

**Table 5 (Question-Type Performance on HaluMem).** Question-answering correct rate across the six HaluMem question types for all systems. Tests the hypothesis that MOSAIC's advantage is largest on Multi-hop Inference and Memory Conflict.

**Table 6 (Evolving Graph versus Static Graph).** Rows: MOSAIC (Evolving), MOSAIC (Static), and published results for the strongest competitors (MemOS, Mem0-Graph). Columns: Memory Update Correct Rate, Memory Update Hallucination Rate, Memory Update Omission Rate, Dynamic Update question accuracy, Memory Conflict question accuracy, and overall question-answering correct rate, separately for HaluMem-Medium and HaluMem-Long, plus LoCoMo and LongMemEval overall accuracy and LongMemEval Knowledge Update category accuracy. This table directly fills manuscript Table tab_evolving by showing where the evolving variant outperforms the static variant, particularly on update-intensive metrics.

**Table 7 (Time Consumption).** Analogous to HaluMem's published time consumption tables. All methods evaluated on HaluMem in rows. Columns: Dialogue Addition Time (minutes), Memory Retrieval Time (minutes), Total Time (minutes), for both HaluMem-Medium and HaluMem-Long. Additional columns for LoCoMo and LongMemEval: mean ingestion latency per conversation (seconds) and mean query latency (milliseconds).

**Table 8 (Graph Structural Characteristics).** Fills manuscript Table tab_graph_structure. Rows: HaluMem-Medium (mean across 20 users), HaluMem-Long (mean across 20 users), LoCoMo (mean across 10 conversations), LongMemEval (mean across conversations). Columns: node count, prerequisite edge count, association edge count, longest prerequisite chain length, maximum neighborhood size, mean neighborhood size, community count, association graph density. Values drawn from the saved graph artifacts of the MOSAIC (Evolving) runs.

**Figure 1 (Architecture and Memory-Completeness Gap).** Panel (a): MOSAIC architecture schematic showing the prerequisite graph, association graph, neighbor-conditioned stability update mechanism with the update frontier highlighted, persistent memory with confidence gating, and the answering language model, with the evolving-graph extension annotated. Panel (b): Grouped bar chart of the primary performance metric for Memory-Only MOSAIC versus MOSAIC (Evolving) across all four evaluation conditions, with 95% confidence intervals. Panel (c): For three representative HaluMem users (selected to span profile complexity), cumulative memory point acquisition over sessions under both conditions.

**Figure 2 (Baseline Comparison).** Panel (a): Grouped bar chart of question-answering correct rate across all four evaluation conditions for all methods. Panel (b): For HaluMem-Medium, cumulative memory extraction recall over sessions for each freshly run method. Panel (c): Stacked bar chart showing fraction of question-answering errors attributable to hallucination versus omission per method on HaluMem-Medium. Panel (d): Radar chart with six axes (HaluMem question types) for the top five performing methods.

**Figure 3 (Cross-Benchmark Radar Charts).** Panel (a): Radar chart with five axes (LoCoMo question categories) for all 10 LoCoMo-evaluated methods. Panel (b): Radar chart with five axes (LongMemEval ability categories) for the same methods. Panel (c): Radar chart with three axes (HaluMem Extraction F1, HaluMem Question Answering Correct Rate, LoCoMo Accuracy) for the top five methods, showing consistent versus domain-specific performance.

**Figure 4 (Evolving Graph Mechanism).** Panel (a): Walkthrough of one representative HaluMem user showing the MOSAIC graph at three time points: after initial sessions, after a session containing a major update (new or revised node highlighted, update frontier shaded), and after incorporation. Contrasted with the static-graph variant's unchanged topology. Panel (b): Bar chart comparing Evolving versus Static variants on Dynamic Update and Memory Conflict question accuracy across both HaluMem scale conditions. Panel (c): Graph node count over sessions for three users under the evolving variant, showing sparse growth that stabilizes.

**Qualitative Case Studies (Figure 7).** Three cases where MOSAIC (Evolving) answers correctly but the majority of baselines do not. For each: the question, gold answer, MOSAIC's retrieved context, and a visualization of the traversed subgraph with annotated nodes and edges. One case selected from each benchmark (HaluMem multi-hop, LoCoMo temporal, LongMemEval knowledge update) to demonstrate graph structure benefits across domains.

### Sanity Checks

MemOS's HaluMem F1 of 79.70% on Medium and 82.11% on Long are the benchmarks to beat. Mem0's published LoCoMo accuracy and HaluMem leaderboard numbers serve as reference points. If the fresh Mem0 run on LoCoMo deviates from Mem0's published numbers by more than 10 percentage points after accounting for the answering model difference (Qwen-2.5-32B-Instruct versus Mem0's original models), investigate prompt formatting or evaluation protocol discrepancies before proceeding.

---

## Experiment 2: Mechanistic Analysis and Graph Construction Quality

### Purpose

This experiment provides the mechanistic depth required for manuscript Section 2.4 (neighbor-conditioned stability validation and ablation), Section 2.5 (graph construction quality), and the perturbation robustness analysis in the Methods section. It reuses the graph artifacts and control-condition results from Experiment 1 wherever possible, running new computations only where required.

### Part A: Neighbor-Conditioned Stability Validation

This part requires one additional instrumented run of MOSAIC on HaluMem-Medium. During Experiment 1, MOSAIC ran in its default neighbor-conditioned stability mode (local re-scoring only). Here, the same 20 users are re-ingested with MOSAIC running in dual mode: at every dialogue turn, both the neighbor-conditioned stability local update and a global recompute of all node scores are performed, and both score vectors are logged.

**Procedure.** Run MOSAIC (Evolving) on HaluMem-Medium with dual-mode instrumentation enabled. At each turn, record: the full score vector under neighbor-conditioned stability mode, the full score vector under global-recompute mode, the predicted update frontier (the set of nodes whose combined neighborhoods intersect with the entities whose belief distributions changed), the actual set of nodes whose scores differ between the two turns under global-recompute mode, the wall-clock time for the score update under each mode separately, and the current graph size (node count).

**Measurements.** (1) Score match rate: fraction of turns where the predicted update frontier exactly equals the actual changed-score set. (2) Score oscillation: for each entity, the number of sign changes in its score trajectory per user, compared between modes. (3) Per-turn score update time versus graph node count for both modes. (4) Update frontier ratio: size of the update frontier divided by total graph size at each turn. (5) Overall question-answering correct rate under both modes (confirming neighbor-conditioned stability does not degrade accuracy; the dual-mode run produces answers from the neighbor-conditioned stability branch).

**Computational Effort.** 1 additional run × 20 users on HaluMem-Medium = 20 ingestion operations. The dual-mode logging adds approximately 30% overhead to per-turn computation but does not increase the number of evaluations (question-answering results for the neighbor-conditioned stability branch are confirmed to match Experiment 1 within numerical tolerance).

**Outputs.** Table 9 fills manuscript Table tab_ncs_validation. Columns: score match rate with 95% confidence interval, mean update frontier size, mean update frontier ratio, mean score oscillation per entity under neighbor-conditioned stability mode, mean score oscillation per entity under global-recompute mode, mean per-turn score update time under neighbor-conditioned stability mode (milliseconds), mean per-turn score update time under global-recompute mode (milliseconds), and overall question-answering correct rate under both modes.

Figure 5 fills manuscript Figure 4. Panel (a): Scatter plot of actual changed-score set size versus predicted update frontier size, one point per turn across all users, with points on the diagonal confirming exact match. Panel (b): Per-turn score update time versus graph node count, one line per mode, with fitted linear regression showing that neighbor-conditioned stability cost is flat while global recompute is linear. Panel (c): Histogram of update frontier ratio across all turns, showing that the median is small.

### Part B: Component Ablation

This part runs 6 modified MOSAIC variants on HaluMem-Medium. The control condition (MOSAIC Evolving on HaluMem-Medium) reuses results from Experiment 1. The MOSAIC (Static) condition also reuses results from Experiment 1.

**Ablation Conditions.**

1. **Full MOSAIC Evolving** (control). Reused from Experiment 1.

2. **MOSAIC Static** (without evolving graph). Reused from Experiment 1.

3. **Without Prerequisite Graph.** All prerequisite edges removed. The association graph and all other components remain. The frontier constraint is disabled: all unresolved entities are eligible at all times. Tests whether logical dependency ordering contributes to performance.

4. **Without Association Graph.** All association edges removed. The prerequisite graph remains. Community detection is disabled. The community continuity scoring term is zeroed. Tests whether semantic clustering and coherence-based traversal matter.

5. **Without Graph Traversal at Query Time.** The graph is constructed normally, but retrieval uses only embedding similarity between the query and individual entity nodes, without traversing edges to include neighboring context. Tests whether multi-hop traversal during retrieval adds value beyond entity-centered indexing.

6. **Without Confidence Gating.** All extracted entity values are committed to memory regardless of confidence score. The confidence threshold is set to zero. Tests whether quality filtering contributes to downstream accuracy or concordance.

7. **Without Long-Term Memory.** The persistent memory store is disabled. Only the current context window is available. The graph structure is maintained for entity scoring, but previously acquired information beyond the context window is inaccessible. Tests the memory component in isolation from graph structure.

8. **Without Entity Deduplication.** Coreference resolution and entity merging disabled. Each mention creates a separate node. Tests whether consolidation improves retrieval precision.

**Procedure.** For each of the 6 new conditions (conditions 3 through 8), run the full HaluMem-Medium evaluation pipeline for all 20 users.

**Computational Effort.** 6 conditions × 20 users = 120 ingestion and evaluation runs. 6 × 3,467 = 20,802 question-answering evaluations.

**Outputs.** Table 10 fills manuscript Table tab_ablation. Rows: all 8 conditions. Columns: Memory Extraction F1, Memory Update Correct Rate, Question Answering Correct Rate, Question Answering Hallucination Rate, and the change in each metric relative to the control. The largest drops identify the most critical components.

Figure 6: horizontal bar chart where each bar represents one ablation condition and bar length represents the drop in question-answering correct rate from the control. Bars colored by which HaluMem question type experienced the largest accuracy drop for that condition.

### Part C: Graph Perturbation Analysis

This part uses the saved graph artifacts from Experiment 1, applies perturbations, and re-evaluates question answering on the perturbed graphs without re-ingesting conversations.

**Procedure.** For each of 5 HaluMem-Medium users (selected to span the range of graph sizes from the 20 users), load the saved MOSAIC (Evolving) graph artifact. Apply five perturbation types at four severity levels (10%, 20%, 30%, 50% of edges or nodes affected): prerequisite edge deletion, prerequisite edge addition (random spurious edges with directed acyclic graph constraint maintained), prerequisite edge reversal, association edge deletion, and node deletion. For each perturbed graph, re-run only the Memory Question Answering evaluation on a subsample of 500 randomly selected questions per user. For each perturbation event, compute the neighbor-conditioned stability-predicted damage radius (number of nodes in the combined neighborhood of perturbed elements) and record which questions' accuracy actually changed.

**Computational Effort.** 5 perturbation types × 4 severity levels × 5 users × 500 questions = 50,000 question-answering evaluations. No ingestion is required (graphs are loaded from saved artifacts and perturbed in place).

**Outputs.** Degradation curves (question-answering correct rate versus perturbation percentage) for each perturbation type, used in manuscript Methods Section 5.6 and Supplementary. Overlay plots comparing the neighbor-conditioned stability-predicted damage radius against the empirically observed accuracy change radius, validating localized perturbation effects.

### Part D: Graph Construction Quality

This part evaluates the quality of the graphs that MOSAIC automatically constructed during Experiment 1, by comparing them against expert-annotated ground truth. No additional MOSAIC ingestion runs are needed for the automatic and control graphs; only the expert-graph and auto+review-graph downstream evaluations require fresh runs.

**Step 1: Expert Annotation.** For each of the 20 HaluMem-Medium users, a human annotator constructs a gold-standard entity graph using the user's persona description, session summaries, and the annotated memory points as reference. Entities correspond to memory points. Prerequisite edges are drawn where one memory point logically depends on another. Association edges are drawn between semantically related memory points. Two annotators independently construct graphs for 5 of the 20 users to compute inter-annotator agreement (edge-level Cohen's kappa for prerequisite and association edges separately). The remaining 15 users are annotated by one annotator each. Estimated effort: approximately 2 hours per user × 25 user-graph annotations (20 unique plus 5 double-annotated) = 50 person-hours.

**Step 2: Edge-Level Comparison.** For each user, compare the automatically generated graph (saved from Experiment 1) against the expert-annotated gold-standard graph at the edge level. Compute precision, recall, and F1 for prerequisite edges and association edges separately.

**Step 3: Expert Review of Automatic Graphs.** A domain expert reviews each automatically generated graph, correcting errors (adding missing edges, removing spurious edges, modifying edge directions, adding or removing nodes). Record the time spent on review (minutes per user) and the number of corrections made. The corrected graph becomes the auto+review graph.

**Step 4: Downstream Comparison.** Run MOSAIC on HaluMem-Medium using two additional graph versions for each user: the expert-constructed gold-standard graph and the auto+review graph. The automatic-graph results are reused from Experiment 1 (they are the MOSAIC Evolving results). Compare all HaluMem metrics across the three graph versions.

**Step 5: Neighbor-Conditioned Stability Locality of Corrections.** For each correction made by the expert reviewer, compute the combined neighborhood of the corrected edge or node. Measure whether the downstream performance difference between the automatic and auto+review graphs is confined to questions involving entities within the corrected neighborhood.

**Computational Effort.** 2 additional graph versions (expert, auto+review) × 20 users = 40 new ingestion and evaluation runs. 2 × 3,467 = 6,934 question-answering evaluations. Plus 50 person-hours of annotation.

**Outputs.** Table 11 fills manuscript Table tab_graph_construction. Rows: Automatic graph, Auto+Review graph, Expert graph. Columns: prerequisite edge precision, prerequisite edge recall, prerequisite edge F1, association edge precision, association edge recall, association edge F1, downstream question-answering correct rate, downstream Memory Extraction F1, mean expert review time per user (minutes), mean corrections per user. Inter-annotator Cohen's kappa for the 5 double-annotated users.

Figure 7 fills manuscript Figure 5. Panel (a): Graph construction pipeline schematic. Panel (b): Edge-level F1 for prerequisite and association edges (automatic versus auto+review). Panel (c): Downstream question-answering correct rate across the three graph versions (grouped bars with confidence intervals). Panel (d): For each expert correction, scatter plot of predicted neighborhood size versus number of questions whose accuracy changed, confirming localized correction effects.

---

## Manuscript Placeholder Mapping

**Abstract.** "four domains" → the four evaluation conditions (HaluMem-Medium, HaluMem-Long, LoCoMo, LongMemEval). "static-graph methods degrade by [X]% while DualGraph with evolving DAGs maintains [Y]%" → Experiment 1, Table 6, comparing MOSAIC (Static) versus MOSAIC (Evolving) on HaluMem Memory Update and Memory Conflict question types. "clinician-supervised pilot" → see note on prospective pilot below.

**Section 2.1 (Memory-Completeness Gap).** Table tab_memory_gap → Experiment 1, Table 1. Figure 1 → Experiment 1, Figure 1. "[X-Y]% higher entity acquisition" → difference between MOSAIC (Evolving) and Memory-Only MOSAIC across conditions. Effect sizes → Experiment 1 statistical analysis.

**Section 2.2 (Baselines).** Table tab_baselines → Experiment 1, Tables 2 and 3. Figure 2 → Experiment 1, Figure 2. Failure-mode taxonomy → Experiment 1, Figure 2 panel (c) (hallucination versus omission breakdown from HaluMem). The manuscript's original agent baselines (ReAct, Reflexion, Plan-and-Solve) should be reframed as memory system baselines (the evaluated methods already include the principal approaches: tiered memory via Letta, production memory via Mem0, agentic memory via A-Mem, retrieval-augmented generation via Chunked Embedding Retrieval, plus the HaluMem leaderboard systems). If the manuscript retains agent baselines, see the optional extension below.

**Section 2.3 (Evolving Inference DAGs).** Table tab_evolving → Experiment 1, Table 6. Figure 3 → Experiment 1, Figure 4. Evolving-graph acquisition on update-related metrics → Table 6 Dynamic Update and Memory Conflict columns. Neighbor-conditioned stability containment for evolving graph → Experiment 1 saved graph artifacts showing frontier size at each update event (logged during MOSAIC Evolving runs). Qualitative examples → Experiment 1 qualitative case studies (Figure 7).

**Section 2.4 (Mechanism).** Table tab_ncs_validation → Experiment 2 Part A, Table 9. Table tab_ablation → Experiment 2 Part B, Table 10. Figure 4 → Experiment 2 Part A, Figure 5. "Match rate ≥ 99%" → Table 9 score match rate. "Score oscillation [X]% lower" → Table 9 oscillation comparison. "Per-turn speed [A]× faster" → Table 9 timing comparison. Ablation drops → Table 10. Perturbation degradation → Experiment 2 Part C curves.

**Section 2.5 (Graph Construction).** Table tab_graph_construction → Experiment 2 Part D, Table 11. Figure 5 → Experiment 2 Part D, Figure 7. Edge-level F1 values, review time, downstream performance → Table 11. Evolving-graph compensation for incomplete graphs → compare MOSAIC (Evolving) on auto-generated graphs (Table 11 row 1) against MOSAIC (Static) on auto-generated graphs (ablation from Experiment 2 Part B condition 2, re-restricted to auto graphs if needed; alternatively note from Table 6 that the evolving variant's update capability compensates for initial graph gaps).

**Section 2.6 (Downstream Performance).** Table tab_downstream → Experiment 1, Tables 2 and 3, question-answering columns. HaluMem's three-way question-answering decomposition (Correct, Hallucination, Omission) serves as the downstream quality metric. LoCoMo and LongMemEval per-category accuracy provide additional downstream evidence.

**Section 3 (Theory).** Empirical NCS validation → Experiment 2 Part A. Convergence bound calibration → compare the theoretical bound from Theorem 1 against the empirical number of turns to achieve 90% coverage on HaluMem-Medium (extracted from Experiment 1 cumulative acquisition curves in Figure 1 panel c).

**Section 5 Methods Placeholders.** Parameter values (scoring weights, confidence threshold, hit counter maximum) → determined through hyperparameter sweep on 2 held-out HaluMem-Medium users. Graph structural characteristics → Experiment 1, Table 8. Per-turn latency breakdown → Experiment 2 Part A timing data (reported as Table 12 below). Domain specifications → Table 8 columns.

**Supplementary Placeholders.** LoCoMo table (tab_locomo) → Experiment 1, Table 3 LoCoMo columns. Full ablation → Experiment 2 Part B, Table 10 with per-condition per-question-type breakdowns. Perturbation analysis → Experiment 2 Part C. NCS proofs empirical validation → Experiment 2 Part A. Evolving graph extended analysis → Experiment 1 MOSAIC (Evolving) graph growth logs (node count over sessions, emergence detection events, frontier ratios at each update). Compute analysis → Experiment 1 Table 7 and Experiment 2 Part A Table 12. Graph construction prompts and examples → Experiment 2 Part D annotation artifacts.

**Table 12 (Per-Turn Latency Breakdown, from Experiment 2 Part A).** Rows: Score Update, Memory Retrieval, Entity Extraction, Language Model Inference, Total. Columns: MOSAIC with neighbor-conditioned stability (milliseconds), MOSAIC with global recompute (milliseconds). Fills the latency placeholder in manuscript Section 3.4.

**Table 13 (Graph Structural Characteristics, from Experiment 1).** Already described as Table 8 above. Fills manuscript Table tab_graph_structure.

---

## Note on the Prospective Pilot

The manuscript includes a prospective clinician-supervised pilot (Section 2.6.2) that requires real patient interactions and institutional review board approval. This pilot cannot be conducted through benchmark evaluation. If the pilot is not feasible within the publication timeline, the manuscript should acknowledge it as planned future work and note that the four benchmark evaluation conditions, combined with the mechanistic analysis and ablation, provide substantial empirical evidence across different conversation domains, scale conditions, and memory operation types.

---

## Optional Extension: Agent Framework Baselines

If the manuscript retains ReAct, Reflexion, and Plan-and-Solve as baselines, these can be adapted for the question-answering component of LoCoMo and LongMemEval. Each agent framework would use its reasoning strategy (thought-action-observation loop, periodic self-reflection, or upfront plan generation) to search the conversation history, retrieve relevant passages, and synthesize an answer. These agent methods are not applicable to HaluMem's extraction or update tasks. Adding them requires 3 methods × (approximately 300 LoCoMo questions + approximately 500 LongMemEval questions) = approximately 2,400 additional evaluations on LoCoMo and LongMemEval only.

---

## Complete Output Inventory

| Source | Tables | Figures |
|---|---|---|
| Experiment 1 | Table 1 (Memory-Completeness Gap), Table 2 (HaluMem Full Comparison), Table 3 (LoCoMo and LongMemEval Full Comparison), Table 4 (Typewise Accuracy), Table 5 (Question-Type Performance), Table 6 (Evolving versus Static), Table 7 (Time Consumption), Table 8 (Graph Structural Characteristics) | Figure 1 (Architecture and Gap), Figure 2 (Baseline Comparison), Figure 3 (Cross-Benchmark Radar), Figure 4 (Evolving Graph Mechanism), Figure 7 (Qualitative Case Studies) |
| Experiment 2 Part A | Table 9 (Neighbor-Conditioned Stability Validation), Table 12 (Per-Turn Latency Breakdown) | Figure 5 (Neighbor-Conditioned Stability Mechanism) |
| Experiment 2 Part B | Table 10 (Ablation Results) | Figure 6 (Component Importance) |
| Experiment 2 Part C | Perturbation degradation curves (Supplementary) | Perturbation overlay plots (Supplementary) |
| Experiment 2 Part D | Table 11 (Graph Construction Quality) | Figure 7b (Graph Construction Validation) |
| **Totals** | **12 tables** | **8 figures** |

---

## Total Computational Effort

**Experiment 1 fresh model runs:**
HaluMem: 8 methods × 2 scale conditions × 20 users × 3,467 question-answer evaluations = 55,472 evaluations, plus 320 ingestion operations.
LoCoMo: 10 methods × approximately 300 evaluations = 3,000, plus 100 ingestion operations.
LongMemEval: 10 methods × approximately 500 evaluations = 5,000, plus ingestion operations.
GPT-4o judge calls (LoCoMo + LongMemEval): approximately 8,000.

**Experiment 2 fresh model runs:**
Part A: 20 dual-mode ingestion operations (no new question-answering evaluations; answers confirmed identical to Experiment 1).
Part B: 6 ablation conditions × 20 users × 3,467 = 20,802 evaluations, plus 120 ingestion operations.
Part C: 50,000 question-answering evaluations (no ingestion; graphs loaded from artifacts).
Part D: 2 graph versions × 20 users × 3,467 = 6,934 evaluations, plus 40 ingestion operations.

**Human annotation:** 50 person-hours for expert graph construction.

**Total question-answering evaluations across both experiments:** approximately 141,208.
**Total ingestion operations:** approximately 600.
**Total GPT-4o judge calls:** approximately 8,000.

---

## Execution Timeline

**Week 1.** Set up all evaluation environments: install HaluMem evaluation code, download HaluMem-Medium and HaluMem-Long from Hugging Face, download LoCoMo, download LongMemEval. Install and configure all baseline systems (Mem0, Mem0-Graph, A-Mem, Letta). Implement MOSAIC's adapters for each benchmark's evaluation interface. Implement Memory-Only MOSAIC and MOSAIC (Static) variants. Conduct hyperparameter tuning on 2 held-out HaluMem-Medium users (these users are excluded from all reported results).

**Week 2.** Execute Experiment 1 HaluMem runs (all 8 fresh-run methods on both scale conditions). Begin expert graph annotation for Experiment 2 Part D (recruit annotators, begin annotation).

**Week 3.** Execute Experiment 1 LoCoMo and LongMemEval runs (all 10 methods). Execute Experiment 2 Part A (dual-mode neighbor-conditioned stability validation on HaluMem-Medium).

**Week 4.** Execute Experiment 2 Part B (6 ablation conditions on HaluMem-Medium). Execute Experiment 2 Part C (perturbation analysis). Complete Experiment 2 Part D (expert review of auto-generated graphs, downstream comparison runs with expert and auto+review graphs).

**Week 5.** Run all statistical analyses. Generate all 12 tables and 8 figures. Conduct qualitative case study selection and annotation for Figure 7. Verify all numbers against raw log files. Write the results narrative.

---

## Reproducibility

Every evaluation produces a structured JSON log entry containing: experiment identifier, method name, evaluation condition, user or conversation identifier, question identifier, question category or type, question text, gold answer, retrieved context (full text), predicted answer, all applicable evaluation scores (HaluMem metrics for HaluMem conditions, language model judge decision and token-level F1 for LoCoMo and LongMemEval), query latency in milliseconds, and for MOSAIC variants the graph node count, edge count, update frontier size, and community assignment of the queried entities.

Ingestion operations produce separate log entries containing: experiment identifier, method name, evaluation condition, user or conversation identifier, ingestion latency in seconds, peak memory usage in megabytes, and for MOSAIC variants the final graph node count, prerequisite edge count, association edge count, community count, maximum neighborhood size, and mean neighborhood size.

For Experiment 2 Part A, additional per-turn logs record: both score vectors (neighbor-conditioned stability and global recompute), the predicted update frontier, the actual changed-score set, per-turn score update time under both modes, and current graph size.

All MOSAIC graph artifacts are saved as JSON files (node lists with metadata, prerequisite edge lists, association edge lists with weights, community assignments, belief distributions per node) and are reused across experiments without re-ingestion.

All tables and figures are generated by deterministic post-processing scripts that read exclusively from the log files and saved graph artifacts. The random seed for all stochastic operations (bootstrap resampling, perturbation sampling, user subsampling) is fixed at 42 and recorded in each log entry. The log files, MOSAIC source code, baseline adapter code, evaluation scripts, and figure-generation scripts are included in the supplementary materials.