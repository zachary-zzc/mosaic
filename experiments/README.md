# Revised Experiment Plan for MOSAIC

---

## Overview

MOSAIC ingests multi-session conversations into a dual-graph structure (a prerequisite graph encoding logical dependencies and an association graph encoding semantic relatedness) and retrieves structurally grounded context at query time via graph traversal governed by neighbor-conditioned stability. This revised plan evaluates MOSAIC on six evaluation conditions drawn from five publicly available benchmarks spanning five distinct application domains: personal life management (HaluMem-Medium and HaluMem-Long), social relationship memory (LoCoMo), task-oriented service booking (MultiWOZ 2.4), clinical patient management (MTS-Dialog), and customer service issue resolution (ABCD). The domain diversity tests whether MOSAIC's graph-based memory generalizes beyond open-ended conversation to structured service interactions, information-dense clinical encounters, and procedural customer support, each posing qualitatively different memory organization challenges.

The plan contains two experiments. Experiment 1 runs every method on all six evaluation conditions in a single pass, simultaneously producing the core performance comparison, the memory-completeness gap analysis, the evolving-graph versus static-graph comparison, all per-category and per-type breakdowns, all computational timing measurements, and the automatically constructed MOSAIC graphs saved as artifacts. Experiment 2 performs mechanistic analysis requiring additional runs: neighbor-conditioned stability dual-mode validation, component ablation, graph perturbation sensitivity, and graph construction quality evaluation against expert-annotated ground truth using the same graphs already built during Experiment 1 with no redundant ingestion.

---

## Datasets

### Dataset 1: HaluMem (Personal Life Management Domain)

HaluMem was introduced by Chen et al. (2025) as the first operation-level hallucination evaluation benchmark tailored to agent memory systems. The dataset is publicly available on Hugging Face at https://huggingface.co/datasets/IAAR-Shanghai/HaluMem, and the evaluation code is available at https://github.com/MemTensor/HaluMem.

HaluMem provides two dataset versions sharing the same 20 synthetic users, each with a richly detailed virtual persona comprising core profile information (demographics, education, goals), dynamic state information (occupation, health, relationships), and preference information (food, music, hobbies). Each user's data includes multi-turn conversational sessions between the user and an assistant, structured memory points extracted from those sessions, and question-answer pairs for evaluation.

**HaluMem-Medium** contains 30,073 dialogues across 20 users, with an average of 70 sessions per user and an average context length of approximately 160,000 tokens per user. It includes 14,948 annotated memory points and 3,467 question-answer pairs.

**HaluMem-Long** contains 53,516 dialogues across the same 20 users, with an average of 120 sessions per user and an average context length of approximately 1,000,000 tokens per user. It uses the same 14,948 memory points and 3,467 question-answer pairs but introduces large-scale interference content (factual question-answering and math problems inserted between genuine sessions) to test robustness and hallucination resistance under noise.

Each memory point is categorized as Persona Memory, Event Memory, or Relationship Memory. Each carries metadata including a memory source field (primary, secondary, interference, or system), an is_update flag, an original_memories field linking to superseded memory when applicable, an importance score between 0 and 1, and a timestamp.

HaluMem defines three evaluation tasks. The Memory Extraction task measures how well a system identifies and stores factual information from dialogue sessions, evaluated through Recall, Weighted Recall, Target Precision, Accuracy, False Memory Resistance, and F1 score. The Memory Update task measures whether the system correctly modifies existing memories when new dialogue provides updated or contradictory information, evaluated through Correct Rate, Hallucination Rate, and Omission Rate. The Memory Question Answering task measures end-to-end ability to produce accurate answers from stored memories, evaluated through Correct Rate, Hallucination Rate, and Omission Rate. Additionally, HaluMem reports typewise extraction accuracy across Event, Persona, and Relationship categories, and question-answering performance across six question types: Basic Fact Recall, Dynamic Update, Multi-hop Inference, Generalization and Application, Memory Conflict, and Memory Boundary.

HaluMem has published leaderboard results for six memory systems (MemOS, Zep, Mem0, Mem0-Graph, Supermemory, Memobase) using a standardized evaluation protocol. For these six systems, this plan uses their published numbers directly rather than re-running them. Human annotation of HaluMem confirms high data quality: 95.70% accuracy, 9.58 out of 10 relevance, and 9.45 out of 10 consistency.

**Domain rationale.** HaluMem's personal assistant scenario requires tracking evolving persona attributes, life events, and interpersonal relationships across dozens of sessions. This maps directly to the personal life management domain described in the manuscript, testing multi-session memory with temporal evolution, information updates, and interference resistance across two scale conditions.

### Dataset 2: LoCoMo (Social Relationship Domain)

LoCoMo was introduced by Maharana et al. (2024) as a benchmark for evaluating whether systems can recall information from long, multi-session conversations. The dataset is available at https://github.com/snap-stanford/LoCoMo and on Hugging Face.

The dataset contains 10 multi-session conversations, each simulating a long-term relationship between two speakers across roughly 600 turns spanning multiple sessions with temporal gaps. Conversations cover everyday topics including work, family, hobbies, health, travel, and personal milestones. Alongside the conversations, the dataset provides over 300 question-answer pairs, each annotated with a category label and a set of evidence utterances.

Question categories are Single-hop (recalling one explicitly stated fact), Multi-hop (combining two or more facts from different conversation segments), Temporal (reasoning about when events occurred or their ordering), Open-ended (synthesizing multiple pieces of information), and Adversarial (asking about things never discussed, testing correct abstention). Each question-answer pair includes explicit evidence links to specific conversation turns, enabling direct measurement of retrieval recall against ground-truth evidence.

**Domain rationale.** LoCoMo tests social companion memory between peers, where conversations are unstructured, episodic, and span personal milestones over time. Unlike HaluMem's assistant-user dynamic, LoCoMo's peer conversations produce more implicit information and require more inference. Multi-hop and temporal reasoning demand graph traversal across loosely connected conversation segments, directly testing MOSAIC's association graph and prerequisite chain capabilities in a social context.

### Dataset 3: MultiWOZ 2.4 (Task-Oriented Service Domain)

MultiWOZ 2.4 was introduced by Ye et al. (2022) as a refined version of the Multi-Domain Wizard-of-Oz dataset for task-oriented dialogue. The dataset is available at https://github.com/smartyfh/MultiWOZ2.4.

MultiWOZ 2.4 contains 10,438 multi-turn dialogues between a user and a system spanning seven service domains: hotel, restaurant, train, taxi, attraction, hospital, and police. Each dialogue involves the user requesting services across one to five domains simultaneously. The dataset provides turn-level dialogue state annotations as slot-value pairs (for example, hotel-area=centre, restaurant-food=chinese, train-day=thursday). The test set contains 1,000 dialogues.

Each domain defines a set of informable slots (user preferences that constrain database search) and requestable slots (information the user asks the system to provide). Across all domains, there are 30 distinct domain-slot pairs. Within a single dialogue, a user may change their mind about previously stated preferences, requiring the system to update stored information. Approximately 23% of test dialogues contain at least one slot value change.

**Adaptation for memory evaluation.** We treat each dialogue as a single-session conversation to be ingested by the memory system. After ingestion of all turns, we query the system with natural language questions for each slot present in the dialogue's final annotated state. Questions are generated from templates: "What [slot description] did the user request for [domain]?" (for example, "What area did the user request for the hotel?", "What type of food did the user want at the restaurant?"). Gold answers are the annotated slot values. For dialogues containing slot value changes, we separately evaluate whether the system recalls the final (updated) value rather than the initial one. We use the full test set of 1,000 dialogues. With an average of 8.2 active slots per dialogue in the final state, this yields approximately 8,200 slot-level query evaluations.

**Slot-level question categories.** Simple Recall (slots whose values were stated once and never changed), Cross-Domain Recall (recalling slot values from a domain discussed early in the dialogue when later domains dominate), and Update Recall (slots whose values changed during the dialogue). These categories enable per-type analysis paralleling HaluMem and LoCoMo.

**Domain rationale.** Task-oriented service dialogues pose a qualitatively different memory challenge from open-ended conversation. Information is highly structured (slot-value pairs), multiple parallel information streams coexist (hotel, restaurant, train preferences tracked simultaneously), and preference updates are explicit and consequential (booking the wrong hotel is a hard failure). This tests whether MOSAIC's prerequisite graph can capture cross-domain dependencies (for example, the train arrival time must be before the restaurant reservation) and whether its evolving graph correctly processes preference updates. The structured nature of MultiWOZ also provides the tightest quantitative evaluation: slot values are unambiguous, enabling exact-match scoring without language model judges.

### Dataset 4: MTS-Dialog (Clinical Patient Management Domain)

MTS-Dialog was introduced by Ben Abacha et al. (2023) as a dataset of medical conversations paired with clinical notes for the MEDIQA-Chat 2023 shared task. The dataset is available at https://github.com/abachaa/MTS-Dialog.

MTS-Dialog contains 1,701 doctor-patient dialogues, each a multi-turn conversation covering a clinical encounter. Each dialogue is paired with an expert-written clinical note divided into standardized sections: Chief Complaint, History of Present Illness, Review of Systems, Past Medical History, Current Medications, Allergies, Physical Examination, Assessment, and Plan. The dialogues cover a broad range of medical specialties and conditions. Average dialogue length is approximately 1,200 tokens; clinical notes average approximately 350 tokens.

**Adaptation for memory evaluation.** We treat each clinical dialogue as a conversation to be ingested by the memory system. After ingestion, we query the system with seven standardized clinical questions corresponding to the major note sections: (1) "What is the patient's chief complaint?", (2) "What is the history of present illness?", (3) "What medications is the patient currently taking?", (4) "Does the patient have any known allergies?", (5) "What were the relevant findings on physical examination?", (6) "What is the clinical assessment or diagnosis?", (7) "What is the treatment plan?". Gold answers are the corresponding sections of the expert-written clinical note. Not all dialogues contain information for every section; we skip questions whose corresponding note section is empty. Across all 1,701 dialogues with an average of 5.8 non-empty sections per dialogue, this yields approximately 9,900 query evaluations.

**Medical entity extraction evaluation.** In addition to the section-level question answering, we extract medical entities from both the system's response and the gold note section using scispaCy's en_core_sci_lg model (detecting entities of types DISEASE, CHEMICAL, GENE_OR_GENE_PRODUCT, and custom patterns for symptoms, procedures, and anatomical terms). This provides an entity-level precision, recall, and F1 that complements the text-level evaluation.

**Section-level question categories.** Chief Complaint, History, Medications and Allergies (combined due to their factual-recall nature), Examination Findings, Assessment, and Plan. These map to distinct memory challenges: Chief Complaint and History test information extraction, Medications and Allergies test list recall, Examination Findings test detailed observational memory, Assessment tests inference (synthesizing symptoms and findings into a diagnosis), and Plan tests procedural memory.

**Domain rationale.** Clinical encounters are information-dense, with a natural hierarchical structure: symptoms motivate examinations, examinations inform assessments, assessments determine plans. This hierarchy maps directly to MOSAIC's prerequisite graph: the plan node logically depends on the assessment node, which depends on examination and history nodes. The clinical domain tests whether MOSAIC's graph structure improves extraction of highly structured, medically consequential information compared to flat memory. Additionally, clinical dialogues contain medical terminology and complex causal relationships, testing domain robustness. This dataset addresses the manuscript's clinical-intake domain claims directly.

### Dataset 5: ABCD (Customer Service Issue Resolution Domain)

ABCD (Action-Based Conversations Dataset) was introduced by Chen et al. (2021) at NAACL 2021 as a benchmark for building more in-depth task-oriented dialogue systems with customer service actions. The dataset is available at https://github.com/asappresearch/abcd.

ABCD contains 10,042 customer service dialogues between customers and agents across a simulated retail company. Each dialogue is annotated with the customer's intent (from 55 fine-grained intents grouped into 10 categories: order issues, account access, product questions, shipping, promotions, subscription management, and others), an action sequence (agent actions such as verify identity, search order, apply discount, process return), and key-value pairs representing customer account information referenced during the conversation (order ID, membership status, account email, product name). The test set contains 1,000 dialogues. Average dialogue length is approximately 900 tokens with an average of 22 turns.

**Adaptation for memory evaluation.** Each customer service dialogue is ingested by the memory system. After ingestion, we query the system with five standardized questions: (1) "What was the customer's primary issue or intent?", (2) "What specific product, order, or service was discussed?", (3) "What steps did the agent take to resolve the issue?", (4) "What was the final resolution or outcome?", (5) "What customer account details were referenced during the conversation?". Gold answers are derived programmatically from ABCD's annotations: question 1 uses the annotated intent label converted to natural language, question 2 uses extracted product and order entities, question 3 uses the annotated action sequence converted to natural language descriptions, question 4 uses the final action and dialogue outcome, question 5 uses the annotated key-value pairs. We use the 1,000 test set dialogues with all 5 questions per dialogue, yielding 5,000 query evaluations.

**Question categories.** Intent Identification (question 1), Entity Recall (questions 2 and 5), Procedural Recall (question 3), and Resolution Recall (question 4). These test different memory facets: intent identification requires abstracting from specific utterances to an overall purpose, entity recall requires precise factual extraction, procedural recall requires ordering and sequencing of agent actions, and resolution recall requires identifying the conversation's outcome.

**Domain rationale.** Customer service dialogues have a strong procedural structure: issue identification, customer verification, information lookup, action execution, and resolution. This sequential dependence maps naturally to MOSAIC's prerequisite graph. Customer account information (membership tier, order details, contact information) forms a parallel entity network connected by association edges. The ABCD domain tests whether MOSAIC can track both the procedural flow and the entity network in a customer relationship management context, directly addressing the manuscript's customer relationship domain claims. Additionally, customer service dialogues contain frequent references to external information (order numbers, policy details) that may not appear verbatim in the conversation, testing the system's ability to maintain contextually grounded memory.

### Domain Coverage Summary

The five datasets collectively cover five distinct application domains with qualitatively different memory challenges:

**Personal Life Management (HaluMem):** Multi-session, evolving persona, temporal updates, interference resistance. Tests long-horizon memory across 70–120 sessions per user.

**Social Relationship (LoCoMo):** Multi-session peer conversation, episodic memory, implicit information, multi-hop and temporal reasoning. Tests memory for unstructured social interactions.

**Task-Oriented Service (MultiWOZ):** Single-session multi-domain, structured slot-value tracking, parallel information streams, explicit preference updates. Tests memory for structured task parameters with unambiguous ground truth.

**Clinical Patient Management (MTS-Dialog):** Single-session clinical, hierarchical medical information, causal chains from symptoms to diagnosis to plan, domain-specific terminology. Tests memory for information-dense, medically consequential encounters.

**Customer Service (ABCD):** Single-session procedural, action sequences, customer entity tracking, issue-to-resolution flow. Tests memory for procedural workflows and customer relationship information.

The six evaluation conditions vary simultaneously across conversation length (900 tokens to 1,000,000 tokens), session structure (single-session versus multi-session), information density (sparse social chat versus dense clinical encounters), information structure (free-form versus slot-structured), update dynamics (implicit social updates versus explicit preference changes), and evaluation precision (language model judge versus exact match).

---

## Methods

### Shared Answering Model

All methods that involve a final answer-generation step share the same answering large language model (Qwen-2.5-32B-Instruct, temperature 0, maximum output length 512 tokens). Each method receives the same conversation input and the same query at evaluation time. The only variable across methods is the memory and retrieval mechanism. For MultiWOZ slot-value queries where answers are typically short phrases, the answering model is prompted to produce a concise value; exact-match evaluation supplements the language model judge.

### MOSAIC Variants

Three MOSAIC variants are run as separate methods within Experiment 1, sharing the same entity extraction pipeline, embedding model (BAAI/bge-large-en-v1.5), and answering model.

**MOSAIC (Evolving Graph).** The full system. Conversations are ingested sequentially. Entities are extracted and instantiated as graph nodes. Prerequisite edges encode logical dependencies; association edges encode semantic relatedness. The evolving-graph mechanism is active: when the extraction module detects information that does not map to any existing node (embedding cosine similarity below 0.85 against all existing nodes, confirmed by a language model verification call), a new node is created, prerequisite and association edges are attached (with directed acyclic graph cycle checking), and dirty flags are propagated through the neighbor-conditioned stability mechanism to the new node's neighborhood only. Community detection uses the Leiden algorithm. The scoring function combines normalized importance, precomputed PageRank centrality, and community continuity. At query time, graph traversal combines structural proximity with embedding similarity.

**MOSAIC (Static Graph).** Identical to the above except the evolving-graph mechanism is disabled. The graph topology is frozen after the initial ingestion phase. New sessions update belief distributions on existing nodes but cannot add new nodes or edges. This variant isolates the contribution of dynamic graph evolution.

**Memory-Only MOSAIC.** The dual-graph controller is disabled entirely. Entities are still extracted from conversations and stored with embeddings, but no prerequisite or association edges are constructed. At query time, retrieval is based solely on embedding similarity between the query and stored entity representations, with no graph traversal. This variant isolates the contribution of graph structure while holding the entity extraction pipeline, embedding model, and answering model constant.

During Experiment 1, all MOSAIC variants save their constructed graphs (node lists, edge lists, community assignments, belief distributions, and all metadata) as persistent JSON artifacts after ingesting each user, conversation, or dialogue. These saved graphs are reused in Experiment 2 for graph construction quality analysis and perturbation experiments.

### Baseline Methods

**Full Context.** The entire conversation is concatenated and passed as context to the answering model. No memory system is involved. If the conversation exceeds the model's context window (32,768 tokens for Qwen-2.5-32B-Instruct), the oldest turns are truncated. For HaluMem-Medium (approximately 160,000 tokens per user) and HaluMem-Long (approximately 1,000,000 tokens per user), substantial truncation occurs. For the shorter domain-specific dialogues (MultiWOZ, MTS-Dialog, ABCD), full context fits within the window and serves as an upper-bound reference.

**Recency Buffer.** Only the most recent 4,096 tokens of the conversation are retained as context. Provides a performance floor for questions about early-conversation information.

**Chunked Embedding Retrieval.** The conversation is divided into non-overlapping chunks of 10 turns each. Each chunk is embedded using BAAI/bge-large-en-v1.5. At query time, the 5 chunks with highest cosine similarity to the query embedding are retrieved and concatenated as context. Implemented using LangChain text splitting with FAISS as the vector index.

**Mem0.** The open-source universal memory layer (https://github.com/mem0ai/mem0, version 1.0.7). Ingestion uses the add method with each turn sequentially, with a unique user identifier per conversation or dialogue. Retrieval uses the search method. On HaluMem, published leaderboard results for Mem0 (standard) and Mem0-Graph are used directly. On all other conditions, both variants are run from scratch using the official Python SDK with default configuration to ensure the shared answering model is used.

**A-Mem (Agentic Memory).** An agentic memory system using Zettelkasten-inspired principles with dynamic linking and memory evolution (https://github.com/agiresearch/A-mem). Uses ChromaDB for vector storage. Ingestion uses the add_note method. Retrieval uses the search_agentic method. Run from scratch on all conditions using the official repository with default configuration (embedding model: all-MiniLM-L6-v2, language model backend for agentic processing: gpt-4o-mini; the final answer is generated by the shared Qwen-2.5-32B-Instruct model).

**Letta (formerly MemGPT).** A framework for building stateful agents with tiered memory: core memory, recall memory, and archival memory (https://github.com/letta-ai/letta, version 0.16.6). Run from scratch on all conditions using the official Python client with default configuration.

**Additional HaluMem Leaderboard Systems.** Published results for MemOS, Zep, Supermemory, and Memobase are included from the HaluMem leaderboard for the HaluMem conditions only. These systems are not evaluated on other conditions because their public interfaces do not support the custom evaluation protocols required.

### Summary of Methods per Evaluation Condition

| Method | HaluMem-Med | HaluMem-Long | LoCoMo | MultiWOZ | MTS-Dialog | ABCD | Source |
|---|---|---|---|---|---|---|---|
| MOSAIC (Evolving) | Run | Run | Run | Run | Run | Run | This work |
| MOSAIC (Static) | Run | Run | Run | Run | Run | Run | This work |
| Memory-Only MOSAIC | Run | Run | Run | Run | Run | Run | This work |
| Full Context | Run | Run | Run | Run | Run | Run | Baseline |
| Recency Buffer | Run | Run | Run | Run | Run | Run | Baseline |
| Chunked Embed. Retrieval | Run | Run | Run | Run | Run | Run | Baseline |
| Mem0 | Published | Published | Run | Run | Run | Run | Mixed |
| Mem0-Graph | Published | Published | Run | Run | Run | Run | Mixed |
| A-Mem | Run | Run | Run | Run | Run | Run | Run |
| Letta | Run | Run | Run | Run | Run | Run | Run |
| MemOS | Published | Published | — | — | — | — | Published |
| Zep | Published | Published | — | — | — | — | Published |
| Supermemory | Published | Published | — | — | — | — | Published |
| Memobase | Published | Published | — | — | — | — | Published |

Fresh runs: 8 methods × 2 HaluMem conditions + 10 methods × 4 non-HaluMem conditions = 56 method-condition combinations.

---

## Evaluation Metrics

### HaluMem Metrics (Both Scale Conditions)

All metrics follow the HaluMem evaluation code exactly. Memory Extraction: Recall, Weighted Recall, Target Precision, Accuracy, False Memory Resistance, F1. Memory Update: Correct Rate, Hallucination Rate, Omission Rate. Memory Question Answering: Correct Rate, Hallucination Rate, Omission Rate. Typewise Accuracy: Event, Persona, Relationship. Question Type Accuracy: Basic Fact Recall, Dynamic Update, Multi-hop Inference, Generalization and Application, Memory Conflict, Memory Boundary.

### LoCoMo Metrics

Answer Accuracy via GPT-4o language model judge (temperature 0) following the LoCoMo protocol. Token-Level F1 Score between predicted and gold answers after normalization. Retrieval Recall: fraction of gold evidence utterances covered by retrieved context (verbatim or cosine similarity above 0.85 via BAAI/bge-large-en-v1.5). Per-Category Accuracy across Single-hop, Multi-hop, Temporal, Open-ended, and Adversarial.

### MultiWOZ 2.4 Metrics

**Slot Accuracy (Exact Match).** For each query, the system's extracted value is compared against the annotated slot value after normalization (lowercasing, whitespace stripping, synonym resolution using MultiWOZ's standard value mappings). Binary correct or incorrect per slot.

**Joint Goal Accuracy.** Fraction of dialogues where all slot values in the final state are simultaneously correct.

**Slot Precision, Recall, and F1.** Precision is the fraction of system-reported slot values that match gold values; recall is the fraction of gold slot values correctly reported; F1 is the harmonic mean.

**Update Accuracy.** Among slots whose values changed during the dialogue (identified from turn-level annotations), the fraction where the system reports the final value.

**LLM Judge Accuracy.** GPT-4o judges semantic equivalence for cases where the extracted value is a paraphrase (for example, "moderate" versus "moderately priced"). Applied as a secondary metric alongside exact match.

**Per-Category Accuracy.** Simple Recall, Cross-Domain Recall, Update Recall.

### MTS-Dialog Metrics

**Section-Level ROUGE-L.** For each section query, ROUGE-L F1 between the system's response and the gold note section.

**Medical Entity F1.** Medical entities extracted from both system response and gold section using scispaCy en_core_sci_lg. Entity-level precision, recall, and F1 computed via exact string matching after lemmatization.

**LLM Judge Correctness.** GPT-4o evaluates whether the system's response (a) contains only information present in the dialogue (no hallucination) and (b) covers the key information in the gold section (completeness). Binary correct or incorrect.

**Per-Category Accuracy.** Chief Complaint, History, Medications and Allergies, Examination Findings, Assessment, Plan.

### ABCD Metrics

**Intent Accuracy.** Exact match between the system's identified intent and the annotated intent label (after mapping system free-text responses to the closest intent label via embedding similarity with a threshold of 0.90, confirmed by LLM judge for borderline cases).

**Entity F1.** Key entities (product names, order IDs, account details) extracted from the system's response compared against annotated key-value pairs. Precision, recall, F1.

**Action Sequence F1.** Actions mentioned in the system's response compared against the annotated action sequence. Token-level F1 after normalizing action descriptions.

**LLM Judge Correctness.** GPT-4o evaluates factual accuracy and completeness of each response against the gold answer derived from annotations. Binary correct or incorrect.

**Per-Category Accuracy.** Intent Identification, Entity Recall, Procedural Recall, Resolution Recall.

### Unified Primary Metric

To enable cross-domain comparison in summary tables and figures, we define a primary accuracy metric for each condition: HaluMem uses Question Answering Correct Rate; LoCoMo uses LLM Judge Accuracy; MultiWOZ uses Joint Goal Accuracy; MTS-Dialog uses LLM Judge Correctness; ABCD uses LLM Judge Correctness. All are expressed as percentages.

### Computational Metrics (Collected During All Runs)

**Ingestion Latency.** Wall-clock seconds to ingest a complete conversation, user session history, or dialogue. Measured on a single NVIDIA A100 80GB GPU with 64 CPU cores and 256 gigabytes of RAM. Each measurement repeated three times; median reported.

**Query Latency.** Wall-clock milliseconds from query submission to retrieved context return, excluding the shared answering model call. Same hardware and repetition protocol.

**Peak Memory Usage.** Maximum RAM in megabytes during and after ingestion via Python's tracemalloc; GPU memory via torch.cuda.max_memory_allocated.

**Graph Statistics (MOSAIC Variants Only).** Node count, prerequisite edge count, association edge count, community count, maximum neighborhood size, mean neighborhood size. Recorded after ingestion for each unit and saved to the persistent graph artifact.

---

## Experiment 1: Cross-Domain Benchmark Evaluation

### Purpose

This single experiment produces all core quantitative results: the memory-completeness gap (Section 2.1), the full baseline comparison (Section 2.2), the evolving-graph versus static-graph comparison (Section 2.3), all per-category and per-type breakdowns (Sections 2.4 and 2.6), downstream performance across five application domains (Section 2.6), computational efficiency (Sections 3.4 and Supplementary), and the automatically constructed graph artifacts used in Experiment 2. Running all three MOSAIC variants as separate methods within the same experiment eliminates the need for separate experiments to isolate the memory-completeness gap or the evolving-graph contribution.

### Procedure

**Step 1: HaluMem Evaluation.** For each of the 8 methods requiring fresh HaluMem runs, execute the HaluMem evaluation pipeline on both HaluMem-Medium and HaluMem-Long following HaluMem's official evaluation code: add each user's dialogue sessions sequentially through the system's ingestion interface, then run the three evaluation tasks using HaluMem's standardized scripts. For the 6 systems with published results, transcribe their published numbers into comparison tables. During ingestion of each MOSAIC variant, save the complete graph artifact as a JSON file for each user.

**Step 2: LoCoMo Evaluation.** For each of the 10 freshly run methods, ingest each of the 10 conversations independently, resetting the memory system between conversations. Submit each of the 300+ question-answer pairs. Pass retrieved context and question to the shared answering model. Run the GPT-4o judge on every predicted-gold answer pair. Compute token-level F1 and retrieval recall.

**Step 3: MultiWOZ 2.4 Evaluation.** For each of the 10 freshly run methods, process each of the 1,000 test dialogues independently. Ingest all turns sequentially. After ingestion, submit the generated slot-value questions (approximately 8.2 per dialogue). Evaluate responses via exact match against annotated values and via LLM judge for paraphrase handling. Compute all MultiWOZ metrics.

**Step 4: MTS-Dialog Evaluation.** For each of the 10 methods, process each of the 1,701 clinical dialogues. Ingest all turns. Submit the 7 section-level clinical questions (skipping empty sections). Evaluate responses via ROUGE-L against gold note sections, medical entity F1 via scispaCy, and LLM judge correctness. Compute all MTS-Dialog metrics.

**Step 5: ABCD Evaluation.** For each of the 10 methods, process each of the 1,000 test dialogues. Ingest all turns. Submit the 5 standardized customer service questions. Evaluate responses via intent exact match, entity F1, action sequence F1, and LLM judge correctness. Compute all ABCD metrics.

**Step 6: Record Computational Metrics.** Throughout Steps 1 through 5, all ingestion latency, query latency, and peak memory usage measurements are recorded automatically by instrumentation wrappers. For MOSAIC variants, graph statistics are extracted from saved graph artifacts.

**Step 7: Statistical Analysis.** Compute 95% bootstrap confidence intervals by resampling question-answer or query-answer pairs 1,000 times with replacement, separately for each evaluation condition. Perform paired bootstrap significance tests between MOSAIC (Evolving) and each other method. Compute Cohen's d effect sizes for three key comparisons: MOSAIC (Evolving) versus Memory-Only MOSAIC (graph structure contribution), MOSAIC (Evolving) versus MOSAIC (Static) (dynamic evolution contribution), and MOSAIC (Evolving) versus the best-performing non-MOSAIC method (state-of-the-art claim). Apply Bonferroni correction across the six evaluation conditions for each family of pairwise comparisons.

### Outputs

**Table 1 (Memory-Completeness Gap).** Fills manuscript placeholder tab_memory_gap. Rows: Memory-Only MOSAIC and MOSAIC (Evolving). Columns grouped by all six evaluation conditions, each showing the primary metric, 95% bootstrap confidence interval, Cohen's d, and p-value. Establishes that adding graph structure to the same memory pipeline substantially improves performance across all domains. Fills placeholders: "[X–Y]% higher entity acquisition" in Section 2.1.

**Table 2 (Full Baseline Comparison — HaluMem).** Fills manuscript placeholder tab_baselines (HaluMem portion). All 14 methods in rows. Columns: Recall, Weighted Recall, Target Precision, Accuracy, False Memory Resistance, F1, Memory Update Correct/Hallucination/Omission, QA Correct/Hallucination/Omission. Separate panels for Medium and Long. Best values bolded, second-best underlined. Fills placeholders: "[A]% F1" and "reduces hallucination to [B]%" in Section 2.2.

**Table 3 (Full Baseline Comparison — LoCoMo, MultiWOZ, MTS-Dialog, ABCD).** Fills manuscript placeholder tab_baselines (non-HaluMem portion). All 10 freshly run methods. For LoCoMo: overall accuracy, token-level F1, retrieval recall, per-category accuracy. For MultiWOZ: joint goal accuracy, slot F1, update accuracy, per-category accuracy. For MTS-Dialog: mean ROUGE-L, medical entity F1, LLM judge correctness, per-category accuracy. For ABCD: intent accuracy, entity F1, action sequence F1, LLM judge correctness, per-category accuracy. Fills placeholder: "consistent [C]–[D]% advantage across five domains" in Abstract and Section 2.2.

**Table 4 (Typewise Accuracy on HaluMem).** Extraction accuracy across Event, Persona, and Relationship memory types for all systems. Fills placeholder: "Relationship memory [E]% higher with graph" in Section 2.2.

**Table 5 (Question-Type Performance on HaluMem).** QA correct rate across the six HaluMem question types. Fills placeholder: "Multi-hop Inference advantage of [F]%" in Section 2.4.

**Table 6 (Evolving Graph versus Static Graph).** Fills manuscript placeholder tab_evolving. Rows: MOSAIC (Evolving), MOSAIC (Static), best competitor. Columns: Memory Update Correct Rate and Hallucination Rate (HaluMem), Dynamic Update and Memory Conflict question accuracy (HaluMem), LoCoMo Temporal accuracy, MultiWOZ Update Accuracy, overall primary metric for all six conditions. Fills placeholders: "static-graph methods degrade by [G]% while MOSAIC maintains [H]%" in Abstract and Section 2.3.

**Table 7 (Cross-Domain Summary).** All 10 freshly run methods in rows. Columns: primary metric for each of the six conditions, plus rank and mean rank. Shows which methods are consistently strong versus domain-specific. Fills placeholder: "MOSAIC ranks first in [I] of [J] conditions" in Abstract.

**Table 8 (Time Consumption).** All freshly run methods. Columns: ingestion time and query time per condition. Additional columns for average ingestion latency (seconds per unit) and query latency (milliseconds per query) for each domain. Fills placeholder: "computational overhead of [K]% over flat retrieval" in Section 3.4.

**Table 9 (Graph Structural Characteristics).** Fills manuscript placeholder tab_graph_structure. Rows: each evaluation condition (mean across units). Columns: node count, prerequisite edge count, association edge count, longest prerequisite chain, max neighborhood size, mean neighborhood size, community count, association graph density. Values from saved MOSAIC (Evolving) graph artifacts. Fills all structural placeholders in Section 5.

**Figure 1 (Architecture and Memory-Completeness Gap).** Panel (a): MOSAIC architecture schematic showing prerequisite graph, association graph, neighbor-conditioned stability update frontier, persistent memory with confidence gating, and answering model. Panel (b): Grouped bar chart of primary metric for Memory-Only MOSAIC versus MOSAIC (Evolving) across all six conditions with 95% confidence intervals. Panel (c): For three representative HaluMem users, cumulative memory point acquisition over sessions under both conditions.

**Figure 2 (Cross-Domain Baseline Comparison).** Panel (a): Grouped bar chart of primary metric across all six conditions for all freshly run methods. Panel (b): Radar chart with six axes (one per condition) for the top five methods, showing cross-domain consistency. Panel (c): For HaluMem-Medium, stacked bar chart showing fraction of QA errors from hallucination versus omission per method.

**Figure 3 (Per-Category Deep Dive).** Panel (a): Radar chart with six HaluMem question types for top five methods. Panel (b): Radar chart with five LoCoMo categories. Panel (c): Radar chart with three MultiWOZ categories (Simple, Cross-Domain, Update). Panel (d): Radar chart with six MTS-Dialog sections. Panel (e): Radar chart with four ABCD categories. Fills placeholder: "category-level advantages visualized in Figure [N]" in Section 2.4.

**Figure 4 (Evolving Graph Mechanism).** Panel (a): Walkthrough of one HaluMem user showing graph at three time points: initial, post-update (new/revised node highlighted, update frontier shaded), and post-incorporation. Panel (b): Bar chart comparing Evolving versus Static on update-sensitive metrics across all conditions that have update-related categories (HaluMem Dynamic Update, HaluMem Memory Conflict, MultiWOZ Update Recall). Panel (c): Graph node count over ingestion progress for representative examples from each domain.

**Figure 5 (Domain-Specific Graph Visualizations).** For one representative example from each of the five domains, a subgraph visualization showing nodes (colored by type), prerequisite edges (directed), and association edges (dashed), with the query-relevant traversal path highlighted. Demonstrates how graph topology differs across domains: HaluMem shows temporal persona chains, LoCoMo shows episodic clusters, MultiWOZ shows parallel domain subtrees, MTS-Dialog shows symptom-to-plan cascades, ABCD shows procedural action chains. Fills placeholder: "domain-adapted graph topologies in Figure [O]" in Section 5.

**Qualitative Case Studies (Figure 10).** Five cases (one per domain) where MOSAIC (Evolving) answers correctly but the majority of baselines do not. For each: question, gold answer, MOSAIC's retrieved context, and traversed subgraph visualization with annotated nodes and edges. Fills placeholder: "qualitative examples in Section [P]" in Sections 2.2 and 2.6.

### Sanity Checks

MemOS's HaluMem F1 of 79.70% (Medium) and 82.11% (Long) are the published benchmarks. Mem0's published LoCoMo accuracy serves as reference. If fresh Mem0 runs deviate from published numbers by more than 10 percentage points after accounting for the answering model difference, investigate before proceeding. For MultiWOZ, published dialogue state tracking results (approximately 55–60% joint goal accuracy for end-to-end systems) provide a ceiling reference, noting that memory system evaluation is a different protocol. For MTS-Dialog, published MEDIQA-Chat shared task results provide reference ROUGE-L ranges. For ABCD, published subflow accuracy results serve as context.

---

## Experiment 2: Mechanistic Analysis and Graph Construction Quality

### Purpose

This experiment provides the mechanistic depth required for Section 2.4 (neighbor-conditioned stability validation and ablation), Section 2.5 (graph construction quality), and perturbation robustness. It reuses graph artifacts and control-condition results from Experiment 1 wherever possible.

### Part A: Neighbor-Conditioned Stability Validation

Requires one additional instrumented run of MOSAIC on HaluMem-Medium. The same 20 users are re-ingested with MOSAIC running in dual mode: at every dialogue turn, both the neighbor-conditioned stability local update and a global recompute of all node scores are performed, and both score vectors are logged.

**Procedure.** Run MOSAIC (Evolving) on HaluMem-Medium with dual-mode instrumentation. At each turn, record: full score vector under neighbor-conditioned stability mode, full score vector under global-recompute mode, predicted update frontier, actual set of nodes whose scores changed under global recompute, wall-clock time for score update under each mode, current graph node count.

**Measurements.** (1) Score match rate: fraction of turns where predicted update frontier exactly equals actual changed-score set. (2) Score oscillation: number of sign changes in each entity's score trajectory per user, compared between modes. (3) Per-turn score update time versus graph node count. (4) Update frontier ratio: frontier size divided by total graph size at each turn. (5) QA correct rate under both modes (confirming no degradation).

**Computational Effort.** 20 user-level re-ingestion operations. Dual-mode logging adds approximately 30% overhead; no additional question-answering evaluations beyond Experiment 1 confirmation.

**Outputs.** Table 10 fills manuscript placeholder tab_ncs_validation. Columns: score match rate with 95% CI, mean update frontier size, mean frontier ratio, mean oscillation per entity under each mode, mean per-turn update time under each mode (milliseconds), QA correct rate under both modes. Fills placeholders: "match rate ≥ [L]%" and "per-turn speed [M]× faster" in Section 2.4 and Section 3.

Figure 6 fills manuscript Figure for NCS. Panel (a): Scatter of actual changed-score set size versus predicted frontier size, one point per turn across all users. Panel (b): Per-turn update time versus graph node count, one line per mode, with fitted linear regression. Panel (c): Histogram of update frontier ratio.

### Part B: Component Ablation

Six modified MOSAIC variants on HaluMem-Medium. Control condition (MOSAIC Evolving) and MOSAIC (Static) reuse Experiment 1 results.

**Ablation Conditions.**

1. Full MOSAIC Evolving (control, reused from Experiment 1).
2. MOSAIC Static (reused from Experiment 1).
3. Without Prerequisite Graph: all prerequisite edges removed; association graph and all other components remain; frontier constraint disabled.
4. Without Association Graph: all association edges removed; prerequisite graph remains; community detection disabled; community continuity scoring zeroed.
5. Without Graph Traversal at Query Time: graph constructed normally, but retrieval uses only embedding similarity to individual entity nodes without traversing edges.
6. Without Confidence Gating: all extracted entity values committed regardless of confidence score (threshold set to zero).
7. Without Long-Term Memory: persistent memory store disabled; only current context window available; graph structure maintained for entity scoring.
8. Without Entity Deduplication: coreference resolution and entity merging disabled; each mention creates a separate node.

**Procedure.** For each of the 6 new conditions (3 through 8), run the full HaluMem-Medium evaluation pipeline for all 20 users.

**Computational Effort.** 6 conditions × 20 users = 120 ingestion and evaluation operations. 6 × 3,467 = 20,802 QA evaluations.

**Cross-Domain Ablation Spot Check.** To confirm that ablation conclusions generalize beyond HaluMem, conditions 3 (without prerequisite graph), 4 (without association graph), and 5 (without graph traversal) are additionally run on MultiWOZ 2.4 (1,000 dialogues) and MTS-Dialog (1,701 dialogues). This adds 3 conditions × 2 datasets × their respective evaluations = 3 × (8,200 + 9,900) = 54,300 evaluations and 3 × (1,000 + 1,701) = 8,103 ingestion operations.

**Outputs.** Table 11 fills manuscript placeholder tab_ablation. Rows: all 8 conditions. Columns: HaluMem-Medium Memory Extraction F1, Memory Update Correct Rate, QA Correct Rate, QA Hallucination Rate, and delta from control. Additional columns for MultiWOZ Joint Goal Accuracy and MTS-Dialog LLM Judge Correctness for the three cross-domain conditions (3, 4, 5). Fills placeholder: "removing prerequisite edges degrades [Q]% on HaluMem and [R]% on MultiWOZ" in Section 2.4.

Figure 7: horizontal bar chart with bars for each ablation condition showing QA correct rate drop from control, colored by which question type suffered most. Inset panels show MultiWOZ and MTS-Dialog ablation drops for conditions 3–5.

### Part C: Graph Perturbation Analysis

Uses saved graph artifacts from Experiment 1 without re-ingesting conversations.

**Procedure.** For 5 HaluMem-Medium users (spanning graph size range) and 5 MultiWOZ dialogues (spanning domain count range), load saved MOSAIC (Evolving) graph artifacts. Apply five perturbation types at four severity levels (10%, 20%, 30%, 50%): prerequisite edge deletion, prerequisite edge addition (random spurious edges maintaining DAG constraint), prerequisite edge reversal, association edge deletion, and node deletion. For each perturbed graph, re-run only QA evaluation on a subsample (500 questions per HaluMem user, all slot queries for MultiWOZ dialogues). Record neighbor-conditioned stability predicted damage radius and actual accuracy change.

**Computational Effort.** HaluMem: 5 perturbation types × 4 severity levels × 5 users × 500 questions = 50,000 evaluations. MultiWOZ: 5 × 4 × 5 × approximately 8 slots = 800 evaluations. No ingestion required.

**Outputs.** Degradation curves (primary metric versus perturbation percentage) for each perturbation type, separately for HaluMem and MultiWOZ. Overlay plots comparing predicted damage radius versus empirical accuracy change. Fills placeholder: "perturbation of [S]% edges degrades accuracy by [T]%" in Methods Section 5.6 and Supplementary.

### Part D: Graph Construction Quality

Evaluates quality of graphs constructed during Experiment 1 against expert-annotated ground truth. No additional MOSAIC ingestion for automatic graphs.

**Step 1: Expert Annotation.** For 20 HaluMem-Medium users and 10 MultiWOZ dialogues (5 per annotator pair for agreement, remaining singly annotated), human annotators construct gold-standard entity graphs. For HaluMem, entities correspond to memory points; prerequisite edges drawn for logical dependencies; association edges drawn for semantic relatedness. For MultiWOZ, entities correspond to slot-value pairs and domain references; prerequisite edges capture cross-domain constraints; association edges capture same-domain co-occurrence. Two annotators independently construct graphs for the agreement subset; Cohen's kappa computed for prerequisite and association edges separately. Estimated effort: approximately 2 hours per HaluMem user (40 hours) plus 30 minutes per MultiWOZ dialogue (5 hours) = 45 person-hours plus 10 hours for double-annotation = 55 person-hours.

**Step 2: Edge-Level Comparison.** Compare automatically generated graphs against gold-standard graphs. Compute precision, recall, and F1 for prerequisite and association edges separately, for both HaluMem and MultiWOZ.

**Step 3: Expert Review of Automatic Graphs.** Domain expert reviews each auto-generated graph, correcting errors. Record time and number of corrections. Corrected graph becomes auto+review graph.

**Step 4: Downstream Comparison.** Run MOSAIC on HaluMem-Medium and MultiWOZ using expert and auto+review graphs in addition to the automatic graphs (reused from Experiment 1).

**Step 5: Locality of Corrections.** For each correction, compute the combined neighborhood and measure whether downstream performance differences are confined to questions involving entities within the corrected neighborhood.

**Computational Effort.** 2 graph versions × (20 HaluMem users + 10 MultiWOZ dialogues) = 60 new evaluation runs. 2 × (3,467 + approximately 82) = 7,098 QA evaluations. Plus 55 person-hours annotation.

**Outputs.** Table 12 fills manuscript placeholder tab_graph_construction. Rows: Automatic, Auto+Review, Expert. Columns: prerequisite edge P/R/F1, association edge P/R/F1, downstream primary metric, mean review time per unit (minutes), mean corrections per unit. Inter-annotator kappa. Separate panels for HaluMem and MultiWOZ. Fills placeholders: "automatic construction achieves [U]% prerequisite F1 and [V]% association F1" and "expert review improves downstream accuracy by [W] percentage points at [X] minutes per user" in Section 2.5.

Figure 8: Panel (a) graph construction pipeline schematic. Panel (b) edge-level F1 bars for automatic versus auto+review. Panel (c) downstream primary metric across three graph versions with CIs. Panel (d) scatter of predicted neighborhood size versus questions affected per correction.

---

## Complete Manuscript Placeholder Mapping

### Abstract

"five application domains" → personal life management (HaluMem), social relationship (LoCoMo), task-oriented service (MultiWOZ), clinical patient management (MTS-Dialog), customer service (ABCD).

"six evaluation conditions" → HaluMem-Medium, HaluMem-Long, LoCoMo, MultiWOZ 2.4, MTS-Dialog, ABCD.

"consistent [C]–[D]% advantage across five domains" → Experiment 1, Table 7, difference between MOSAIC (Evolving) and best non-MOSAIC method across all six conditions.

"MOSAIC ranks first in [I] of [J] conditions" → Experiment 1, Table 7, count of conditions where MOSAIC (Evolving) has highest primary metric.

"static-graph methods degrade by [G]% while MOSAIC maintains [H]%" → Experiment 1, Table 6, MOSAIC (Static) versus MOSAIC (Evolving) on update-sensitive metrics.

"clinician-supervised pilot" → partially addressed by MTS-Dialog evaluation in the clinical domain; prospective pilot with real patients acknowledged as planned future work requiring IRB approval.

### Section 2.1 (Memory-Completeness Gap)

tab_memory_gap → Experiment 1, Table 1.

Figure → Experiment 1, Figure 1.

"[X–Y]% higher entity acquisition" → Table 1, difference between MOSAIC (Evolving) and Memory-Only MOSAIC primary metric, range across six conditions.

"effect sizes" → Table 1, Cohen's d column.

"p < 0.001 after Bonferroni correction" → Table 1, p-value column.

### Section 2.2 (Baselines)

tab_baselines → Experiment 1, Tables 2 and 3.

Figure → Experiment 1, Figure 2.

"[A]% F1 on HaluMem" → Table 2, MOSAIC (Evolving) F1 on HaluMem-Medium.

"reduces hallucination to [B]%" → Table 2, MOSAIC (Evolving) QA Hallucination Rate on HaluMem-Medium.

"Relationship memory [E]% higher with graph" → Table 4, MOSAIC (Evolving) versus Memory-Only MOSAIC Relationship typewise accuracy difference.

"failure-mode taxonomy" → Figure 2 panel (c), hallucination versus omission breakdown.

### Section 2.3 (Evolving Inference DAGs)

tab_evolving → Experiment 1, Table 6.

Figure → Experiment 1, Figure 4.

"Memory Update Correct Rate [G1]% for Evolving versus [G2]% for Static on HaluMem-Medium" → Table 6.

"Dynamic Update question accuracy [H1]% versus [H2]%" → Table 6.

"Memory Conflict accuracy [I1]% versus [I2]%" → Table 6.

"MultiWOZ Update Accuracy [J1]% versus [J2]%" → Table 6, confirming evolving-graph benefit extends to task-oriented domain.

"neighbor-conditioned stability confines update frontier to [K] nodes on average" → Experiment 1 graph artifacts, logged frontier sizes at each update event.

### Section 2.4 (Mechanism — NCS and Ablation)

tab_ncs_validation → Experiment 2 Part A, Table 10.

tab_ablation → Experiment 2 Part B, Table 11.

Figure (NCS) → Experiment 2 Part A, Figure 6.

"match rate ≥ [L]%" → Table 10 score match rate.

"score oscillation [L2]% lower under NCS" → Table 10 oscillation comparison.

"per-turn speed [M]× faster" → Table 10 timing comparison.

"removing prerequisite edges degrades [Q]% on HaluMem and [R]% on MultiWOZ" → Table 11, condition 3 drops.

"removing association edges degrades [Q2]% on HaluMem and [R2]% on MTS-Dialog" → Table 11, condition 4 drops.

"without graph traversal, accuracy drops [Q3]%" → Table 11, condition 5 drop.

"without confidence gating, hallucination rate increases [Q4] percentage points" → Table 11, condition 6 hallucination change.

"without entity deduplication, precision drops [Q5]%" → Table 11, condition 8.

"cross-domain ablation confirms prerequisite edges contribute [R3]% on MultiWOZ and [R4]% on MTS-Dialog" → Table 11 cross-domain columns.

### Section 2.5 (Graph Construction)

tab_graph_construction → Experiment 2 Part D, Table 12.

Figure → Experiment 2 Part D, Figure 8.

"automatic construction achieves [U]% prerequisite F1 and [V]% association F1" → Table 12.

"expert review improves downstream accuracy by [W] percentage points at [X] minutes per user" → Table 12.

"inter-annotator Cohen's kappa of [Y] for prerequisite edges and [Z] for association edges" → Table 12.

"evolving-graph mechanism compensates for [W2]% of initial construction errors" → compare MOSAIC (Evolving) auto-graph downstream performance against MOSAIC (Static) auto-graph (from Table 6 and Table 12).

"on MultiWOZ, automatic prerequisite F1 is [U2]%, reflecting the clearer causal structure of service interactions" → Table 12 MultiWOZ panel.

### Section 2.6 (Downstream Performance Across Domains)

tab_downstream → Experiment 1, Tables 2, 3, and 7.

"HaluMem QA Correct Rate of [AA]% (Medium) and [AB]% (Long)" → Table 2.

"LoCoMo overall accuracy of [AC]%" → Table 3 LoCoMo column.

"MultiWOZ Joint Goal Accuracy of [AD]%" → Table 3 MultiWOZ column.

"MTS-Dialog clinical note correctness of [AE]%" → Table 3 MTS-Dialog column.

"ABCD customer service correctness of [AF]%" → Table 3 ABCD column.

"domain-specific category analysis reveals MOSAIC's advantage is largest on [AG] category across [AH] domains" → Table 3 per-category columns, identifying the category type where MOSAIC (Evolving) outperforms baselines by the widest margin.

"qualitative examples in Section [P]" → Experiment 1, Figure 10 qualitative case studies.

### Section 3 (Theory)

"empirical NCS validation" → Experiment 2 Part A.

"convergence bound calibration: theoretical bound from Theorem 1 predicts [AI] turns for 90% coverage; empirically observed [AJ] turns on HaluMem-Medium" → compare Theorem 1 prediction against Experiment 1 cumulative acquisition curves (Figure 1 panel c, identifying the turn number at which 90% of memory points have been acquired).

"perturbation robustness: [S]% edge deletion causes [T]% accuracy drop, consistent with Theorem 2 bound of [AK]%" → Experiment 2 Part C degradation curves.

### Section 3.4 (Computational Efficiency)

"ingestion overhead of [AL]× over flat retrieval" → Table 8, MOSAIC (Evolving) ingestion time divided by Chunked Embedding Retrieval ingestion time.

"query latency of [AM] ms, within [AN]× of embedding retrieval" → Table 8 query latency columns.

"per-turn NCS update costs [AO] ms versus [AP] ms for global recompute" → Experiment 2 Part A, Table 10 (also reported as Table 13 in Supplementary).

"memory usage peaks at [AQ] GB for HaluMem-Long" → Table 8 peak memory column.

### Section 5 (Methods — Parameter Values and Domain Specifications)

"scoring weights (importance: [AR], centrality: [AS], continuity: [AT])" → determined through hyperparameter sweep on 2 held-out HaluMem-Medium users (these users excluded from all results), confirmed on 1 held-out MultiWOZ dialogue and 1 held-out MTS-Dialog dialogue.

"confidence threshold [AU]" → same hyperparameter sweep.

"hit counter maximum [AV]" → same sweep.

"HaluMem graphs average [AW] nodes, [AX] prerequisite edges, [AY] association edges, [AZ] communities" → Table 9, HaluMem-Medium row.

"MultiWOZ graphs average [BA] nodes with [BB] prerequisite edges reflecting cross-domain dependencies" → Table 9, MultiWOZ row.

"MTS-Dialog graphs average [BC] nodes with longest prerequisite chain of [BD] hops (symptom → exam → assessment → plan)" → Table 9, MTS-Dialog row.

"ABCD graphs average [BE] nodes with [BF] prerequisite edges encoding procedural action sequences" → Table 9, ABCD row.

"LoCoMo graphs average [BG] nodes across [BH] communities, reflecting distinct conversational episodes" → Table 9, LoCoMo row.

### Supplementary Placeholders

"Full ablation with per-condition per-question-type breakdowns" → Experiment 2 Part B, extended Table 11 with per-question-type or per-category columns for each ablation condition.

"Perturbation analysis full curves" → Experiment 2 Part C, all degradation curves and overlay plots.

"NCS proofs empirical validation" → Experiment 2 Part A, all logged per-turn data.

"Evolving graph extended analysis" → Experiment 1 MOSAIC (Evolving) graph growth logs for all conditions.

"Compute analysis" → Experiment 1 Table 8 and Experiment 2 Part A Table 13.

"Graph construction prompts and examples" → Experiment 2 Part D annotation guidelines and sample annotated graphs.

"Domain-specific graph topology analysis" → Experiment 1 Figure 5 extended with quantitative topology metrics (clustering coefficient, diameter, degree distribution) for each domain.

"MultiWOZ per-domain breakdown" → Experiment 1, MultiWOZ slot accuracy broken down by domain (hotel, restaurant, train, taxi, attraction, hospital, police).

"MTS-Dialog per-specialty analysis" → Experiment 1, MTS-Dialog correctness stratified by medical specialty where available.

"ABCD per-intent-category analysis" → Experiment 1, ABCD correctness stratified by the 10 intent categories.

---

## Complete Output Inventory

| Source | Tables | Figures |
|---|---|---|
| Experiment 1 | Table 1 (Memory-Completeness Gap), Table 2 (HaluMem Full Comparison), Table 3 (LoCoMo/MultiWOZ/MTS/ABCD Full Comparison), Table 4 (Typewise Accuracy), Table 5 (Question-Type Performance), Table 6 (Evolving vs Static), Table 7 (Cross-Domain Summary), Table 8 (Time Consumption), Table 9 (Graph Structural Characteristics) | Figure 1 (Architecture & Gap), Figure 2 (Cross-Domain Baseline), Figure 3 (Per-Category Deep Dive), Figure 4 (Evolving Graph), Figure 5 (Domain Graph Visualizations), Figure 10 (Qualitative Cases) |
| Experiment 2A | Table 10 (NCS Validation), Table 13 (Per-Turn Latency) | Figure 6 (NCS Mechanism) |
| Experiment 2B | Table 11 (Ablation + Cross-Domain) | Figure 7 (Component Importance) |
| Experiment 2C | Perturbation curves (Supplementary) | Perturbation overlays (Supplementary) |
| Experiment 2D | Table 12 (Graph Construction Quality) | Figure 8 (Graph Construction Validation) |
| **Totals** | **13 tables** | **10 figures** |

---

## Total Computational Effort

### Experiment 1

**HaluMem:** 8 fresh-run methods × 2 conditions × 20 users × 3,467 QA evaluations = 55,472 evaluations plus 320 ingestion operations.

**LoCoMo:** 10 methods × approximately 300 evaluations = 3,000 evaluations plus 100 ingestion operations.

**MultiWOZ:** 10 methods × 1,000 dialogues × 8.2 slot queries = 82,000 evaluations plus 10,000 ingestion operations.

**MTS-Dialog:** 10 methods × 1,701 dialogues × 5.8 questions = 98,658 evaluations plus 17,010 ingestion operations.

**ABCD:** 10 methods × 1,000 dialogues × 5 questions = 50,000 evaluations plus 10,000 ingestion operations.

**GPT-4o judge calls:** LoCoMo (3,000) + MTS-Dialog (98,658) + ABCD (50,000) + MultiWOZ paraphrase adjudication (estimated 8,200) = approximately 159,858.

**Total Experiment 1 evaluations:** approximately 289,130.
**Total Experiment 1 ingestion operations:** approximately 37,430.

### Experiment 2

**Part A:** 20 dual-mode ingestion operations.

**Part B:** 6 ablation conditions × 20 HaluMem users × 3,467 = 20,802 evaluations plus 120 ingestion operations. Cross-domain: 3 conditions × (8,200 + 9,900) = 54,300 evaluations plus 8,103 ingestion operations.

**Part C:** 50,000 HaluMem + 800 MultiWOZ = 50,800 evaluations (no ingestion).

**Part D:** 60 evaluation runs, 7,098 evaluations plus 55 person-hours.

**Total Experiment 2 evaluations:** approximately 133,000.
**Total Experiment 2 ingestion operations:** approximately 8,303.

### Grand Totals

**Total question/slot/section evaluations across both experiments:** approximately 422,130.
**Total ingestion operations:** approximately 45,733.
**Total GPT-4o judge calls:** approximately 165,000.
**Human annotation:** 55 person-hours.

---

## Execution Timeline

**Week 1.** Set up all evaluation environments. Download and verify all five datasets. Install and configure all baseline systems. Implement MOSAIC's adapters for each dataset's evaluation interface (HaluMem adapter, LoCoMo adapter, MultiWOZ slot-query generator, MTS-Dialog section-query generator, ABCD annotation-to-query converter). Implement Memory-Only MOSAIC and MOSAIC (Static) variants. Conduct hyperparameter tuning on 2 held-out HaluMem-Medium users, 1 held-out MultiWOZ dialogue, and 1 held-out MTS-Dialog dialogue (all excluded from reported results).

**Week 2.** Execute Experiment 1 HaluMem runs (8 methods × 2 conditions × 20 users). Begin expert graph annotation for Experiment 2 Part D. Execute Experiment 1 LoCoMo runs (10 methods × 10 conversations).

**Week 3.** Execute Experiment 1 MultiWOZ runs (10 methods × 1,000 dialogues). Execute Experiment 1 MTS-Dialog runs (10 methods × 1,701 dialogues). Execute Experiment 1 ABCD runs (10 methods × 1,000 dialogues). Execute Experiment 2 Part A (dual-mode NCS validation on HaluMem-Medium).

**Week 4.** Execute Experiment 2 Part B (6 ablation conditions on HaluMem-Medium plus 3 cross-domain ablation conditions on MultiWOZ and MTS-Dialog). Execute Experiment 2 Part C (perturbation analysis). Complete Experiment 2 Part D (expert review, downstream comparison with expert and auto+review graphs).

**Week 5.** Run all statistical analyses (bootstrap CIs, significance tests, effect sizes). Generate all 13 tables and 10 figures. Conduct qualitative case study selection (one per domain). Verify all numbers against raw logs. Write results narrative and fill all manuscript placeholders with empirical values.

---

## Reproducibility

Every evaluation produces a structured JSON log entry containing: experiment identifier, method name, evaluation condition, unit identifier (user ID, conversation ID, or dialogue ID), query identifier, query category or type, query text, gold answer, retrieved context (full text), predicted answer, all applicable evaluation scores (HaluMem metrics, LLM judge decision, token-level F1, exact-match result, ROUGE-L score, entity F1, slot accuracy), query latency in milliseconds, and for MOSAIC variants the graph node count, edge count, update frontier size, and community assignment of queried entities.

Ingestion operations produce separate log entries with: experiment identifier, method name, evaluation condition, unit identifier, ingestion latency, peak memory usage, and for MOSAIC variants the final graph statistics.

For Experiment 2 Part A, additional per-turn logs record both score vectors, predicted frontier, actual changed-score set, per-turn update time under both modes, and current graph size.

All MOSAIC graph artifacts are saved as JSON files containing node lists with metadata, prerequisite edge lists, association edge lists with weights, community assignments, and belief distributions per node. These are reused across experiments without re-ingestion.

All tables and figures are generated by deterministic post-processing scripts reading exclusively from log files and saved graph artifacts. The random seed for all stochastic operations is fixed at 42 and recorded in each log entry. The MultiWOZ slot-query templates, MTS-Dialog section-query templates, and ABCD annotation-to-query conversion scripts are included in the supplementary materials alongside all source code, baseline adapters, evaluation scripts, and figure-generation scripts.

The scispaCy model version (en_core_sci_lg 0.5.4), MultiWOZ value normalization mappings, and ABCD intent-to-natural-language mappings are fixed and included in the repository to ensure exact reproducibility of entity extraction and answer matching.
