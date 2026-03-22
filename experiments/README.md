*Thinking...*

> The user is clarifying what MOSAIC actually does. Let me re-understand:
> 
> MOSAIC is a memory system that:
> 1. Takes conversations as input
> 2. Saves/organizes the conversation information into a graph structure
> 3. Provides a query interface to retrieve relevant pieces of information from the stored graph
> 
> So it's essentially a **graph-based memory system for conversations** — similar to Mem0, memg, A-mem, etc. It's NOT a dialogue agent that conducts intake interviews. It's NOT a task-planning system.
> 
> This means:
> - The experiments should evaluate MOSAIC as a **memory retrieval system**, not as a dialogue conductor
> - The previous experiment plan was wrong because it treated MOSAIC as an agent that drives conversations with task graphs, evolving DAGs, entity acquisition, etc.
> 
> 
> - The right experiments focus on how well MOSAIC stores and retrieves information from conversations compared to other memory systems like Mem0, A-mem, and MemGPT/Letta. LoCoMo is actually a primary benchmark here for long-conversation QA, and I should also consider multi-session memory retention as a key evaluation dimension.
> 
> Beyond that, I need to think about memory completeness—whether the graph captures all the important details from a conversation—and retrieval precision and recall to ensure it returns relevant information without noise. As conversations get longer, I'm curious how performance scales, and whether analyzing the actual graph structure reveals advantages over flat memory approaches. Ablation studies would help identify which components of the graph structure actually matter.
> 
> For datasets, LoCoMo with its multi-session conversations and varied question types is the obvious choice, and I should also look at Multi-Session Chat from Meta research and other conversation memory benchmarks to round out the evaluation.
> 
> Now I'm settling on the baselines to compare against: Mem0, A-mem, MemGPT, full context windows, RAG-based retrieval, recency-based filtering, and summary-based approaches. For the core experiment, I'll use LoCoMo as the primary benchmark—feeding multi-session conversations through each memory system and measuring accuracy across different question categories like single-hop, multi-hop, temporal, and open-ended queries.
> 
> Beyond that, I should test memory completeness and retrieval quality by evaluating how well each system stores and retrieves information from conversations, measure how performance scales with conversation length, run ablations to see which parts of MOSAIC's graph structure actually matter, and validate on domain-specific conversations like medical or customer service scenarios.
> 
> Actually, rereading the user's description—MOSAIC takes conversations, structures them as a graph, then retrieves relevant pieces when queried—I need to make sure my experiments are actually testing that core functionality rather than just generic memory benchmarks.
> 
> Looking at what's available, LoCoMo has QA pairs with ground truth, LongBench covers long contexts, ConvQuestions is specifically conversational, and QuALITY has question answering with extended texts. The most relevant ones for a conversation memory system would be those with conversational context and retrieval requirements.
> 
> For the MOSAIC paper, I want to demonstrate that the graph-based approach stores conversation information more completely than flat alternatives, retrieves more relevant context through graph traversal than vector similarity alone, scales better for long conversations, handles multi-hop reasoning through graph connections, and that each component of the graph structure actually contributes meaningfully. Now I'm looking at what datasets are actually available and accessible—LoCoMo is on GitHub and HuggingFace with the data I need.
> 
> Looking at the benchmark landscape, LoCoMo has around 10 multi-session dialogues with roughly 600 turns each and 300+ QA pairs covering single-hop, multi-hop, temporal, and adversarial categories. There's also Multi-Session Chat from Meta with persona-based conversations for evaluating consistency, and I should check if LongMemEval exists as another option. I'm also seeing that Mem0 and similar systems have already used LoCoMo as their primary evaluation benchmark, so that's a solid choice for comparison.
> 
> For additional evaluation, I could create synthetic conversations with specific properties, repurpose existing datasets like DailyDialog or PersonaChat by generating QA pairs, or test on domain-specific conversations. Since the user wants publicly available data with complete access, I'm thinking through other options like MultiWOZ for task-oriented dialogues where I could verify if MOSAIC correctly tracks slot-value changes, or FaithDial for faithful conversation evaluation.
> 
> The key insight is that MultiWOZ and SGD have ground-truth annotations of what information was communicated, so I could feed dialogues into MOSAIC, query for specific slots, and compare the extracted values against the annotations to measure memory completeness and accuracy.
> 
> For the experiments themselves, I'd run LoCoMo as the standard benchmark testing retrieval accuracy and multi-hop reasoning against baselines like Mem0, MemGPT, and full-context approaches. Then I'd add a task-oriented dialogue experiment using MultiWOZ or SGD to evaluate whether MOSAIC accurately captures and retrieves specific information from conversations.
> 
> I'd also test scalability by varying conversation length and measuring both retrieval quality and latency, potentially using LoCoMo at different truncation points. Finally, an ablation study would remove components of MOSAIC's graph structure to see which parts actually matter.
> 
> But wait—I need to reconsider what's actually compatible here. The user flagged that my previous plan had MOSAIC acting as an agent driving dialogues, which doesn't make sense since MOSAIC is fundamentally a memory system, not a dialogue agent. So I should focus these experiments on what MOSAIC actually does: store and retrieve information from conversations that already exist.
> 
> Now I'm thinking through the datasets I'll need. LoCoMo is the obvious choice for QA evaluation since it has conversations and questions with ground truth answers. MultiWOZ 2.4 would give me another dialogue dataset to work with.
> 
> For scalability testing, I can take LoCoMo and artificially extend it to see how the system handles larger conversation histories. The ablation study will use a subset of LoCoMo to isolate which components matter.
> 
> For baselines, I've got a solid range: Mem0 and A-mem are recent memory systems, MemGPT/Letta is the established agent memory framework, and then I need some simpler baselines like full-context retrieval, standard RAG, recency buffers, and periodic summarization to show what MOSAIC improves over.
> 
> Actually, I should reconsider which baselines make sense to implement given my constraints.
> 
> Now I'm thinking about what actually differentiates MOSAIC — it stores conversations as a graph of entities and relationships rather than flat chunks or summaries. That means I should focus on experiments that specifically test graph-based advantages: multi-hop reasoning across entities, capturing complex relationships, handling structured information, and temporal reasoning. Let me refocus the plan to be more concrete and actionable, with clear datasets, setup instructions, and specific experiment configurations that can all be run with publicly available resources. by QA type, comparing against baselines, running ablations to see which graph components matter most, and analyzing how the system scales. For a second dataset, I'm weighing options like MultiWOZ for task-oriented dialogue with structured annotations, or PersonaChat to test persona consistency across conversations—something that complements LoCoMo's strengths without duplicating it.
> 
> MultiWOZ looks promising since it has slot-value ground truth, letting me measure whether MOSAIC's graph captures all communicated information from a structured extraction angle rather than just QA performance. I could also generate synthetic conversations with controlled properties to have full control over ground truth, but for a focused paper, I think the cleaner approach is: LoCoMo as the primary benchmark with per-category breakdowns, an ablation study on a LoCoMo subset to isolate which components matter, and a scalability analysis tracking how performance and latency change with conversation length.
> 
> That gives me three solid experiments, though I'm wondering if I need a second benchmark to demonstrate generalizability. There's LongMemEval and the evaluation from the Mem0 paper that uses both LoCoMo and their own CRAG-like benchmark. For a strong paper I'd ideally want LoCoMo for direct comparison with existing work, something domain or application-specific, ablation studies, and scalability analysis—but the user wants minimal experiments, so I should see if I can consolidate these into a tighter set of experiments focused on LoCoMo with comprehensive analysis.
> 
> Let me step back and think about what experiments actually matter for a graph-based conversation memory system. The core claim is that MOSAIC outperforms existing memory systems on long conversations, so I need to design experiments that directly test that.
> 
> For the first experiment, I'm comparing against baselines on the LoCoMo benchmark. Then I can break down performance by query type to show where the graph structure helps most. The ablation removes components to validate that each piece contributes. For scalability, I'll measure how performance degrades as conversations get longer. And for the qualitative side, I want to show that the graph captures richer relational information than flat memory approaches.
> 
> Now I'm thinking about which datasets are actually available and suitable. LoCoMo is the obvious choice since it's designed for long conversations. DailyDialog and DSTC exist but have limitations—one lacks QA annotations, the other focuses on state tracking rather than memory retrieval. I should probably stick with LoCoMo as the main benchmark and maybe consider if there's a secondary dataset that would strengthen the evaluation.
> 
> For a second dataset, I could either create synthetic long conversations to test scalability in a controlled way, or adapt existing task-oriented dialogue datasets like MultiWOZ or SGD where the slot-value annotations could serve as ground truth for checking if the memory system captures the right information. Real Reddit or Ubuntu conversations are available too, but they don't have the structured annotations needed for proper evaluation.
> 
> Since MOSAIC is a general memory system rather than domain-specific, I should keep the evaluation general rather than tied to particular domains. So my plan is to run LoCoMo as the main benchmark with all baselines, then use MultiWOZ to test how well the system extracts and retains structured dialogue state information, and finally do ablations on a LoCoMo subset to understand which components matter and how performance scales. That gives me three focused experiments, each producing concrete results.
> 
> But now I'm questioning whether MultiWOZ is the right fit—it's really about tracking slot values in short conversations, not long-term memory across sessions. LoCoMo's 600-turn multi-session format is much better for testing memory retention. I could try SODA for longer social dialogues or create a synthetic evaluation with conversations of varying lengths, but that adds complexity. For now, I'll stick with LoCoMo as my primary benchmark and consider a synthetic evaluation as a secondary option if needed.
> 
> For the synthetic approach, I'd generate conversations with planted facts at different lengths—50 to 1000 turns—then query for recall and precision. It's controllable and gives perfect ground truth without needing another annotated dataset. So my plan is to run LoCoMo as the main comparison with baselines and category breakdowns, then add a synthetic scalability test to measure how performance degrades with conversation length.
> 
> Actually, I should consolidate these experiments more. I can run LoCoMo to get the main results table, per-category breakdown, accuracy curves, and qualitative examples all in one experiment. Then combine ablation and scalability analysis into a second experiment using LoCoMo sessions, which would give me the ablation table and scalability figures without duplicating effort.
> 
> But thinking about it more, three experiments feels reasonable and cleaner. Now I'm wondering whether I should compare MOSAIC against other graph-based memory systems specifically, or cast a wider net across all memory system types—buffer-based approaches like MemGPT's working memory, summary-based methods, entity-based systems like MOSAIC itself, and embedding-based retrieval approaches.
> 
> Since MOSAIC's strength is really in the graph structure, I should probably include diverse baseline types to show where that advantage comes through. Let me start drafting the experiment guide now with a practical structure covering the datasets, baselines, and three core experiments.
> 
> For the datasets, MultiWOZ actually makes sense here—I can frame it as testing whether MOSAIC can retain all the information communicated across a multi-turn task-oriented dialogue. I'd feed conversations into MOSAIC, then query for specific slot values against ground truth to evaluate structured information extraction, which is exactly what graph-based memory should excel at. SGD is another option with similar properties.
> 
> Now I'm thinking through how MOSAIC actually processes things: it ingests the conversation, builds out the graph structure with entities and relationships, then retrieves relevant information when queried. For evaluation, I need to design experiments that specifically test this pipeline and compare it meaningfully against the baselines.
> 
> I'm reconsidering the three experiments I outlined—LoCoMo for long-context QA, MultiWOZ for dialogue completeness, and an ablation study. But I want to make sure these are the right choices before finalizing.
> 
> Actually, I'm wondering if I can consolidate these. LoCoMo and MultiWOZ use different datasets and evaluation protocols, so they can't really merge. But I could fold the ablation work into the LoCoMo experiment by testing different MOSAIC configurations on the same data, then add scalability analysis there too. That would give me two clean experiments: one for main results across both datasets, and one for ablations and scaling behavior.
> 
> Wait, but the user's original plan was much more ambitious with 8 tables and 5 figures. Maybe I should stick with three experiments to keep it manageable while still covering the ground.
> 
> Now I'm thinking through which baselines I can actually run. I've got several open-source options available: Mem0 is pip-installable, A-mem is on GitHub, MemGPT/Letta is open source, and I can always do a simple full-context baseline or standard RAG. Let me figure out which ones are realistic to implement.
> 
> For the simpler baselines, I could add a recency buffer that just keeps the last N tokens, or do periodic summarization of the conversation—both are straightforward to set up. All of these are feasible to run.
> 
> One thing I need to nail down is which LLM I'm using for what. The answering LLM should be consistent across all methods so I'm only varying the memory system itself, probably something like GPT-4o or Qwen-2.5-72B. Then each memory system might use its own LLM internally for extraction or summarization, but that's separate from the main answering model.
> 
> Now I'm structuring the paper itself—I need dataset descriptions, the experiment setup, and then the actual experiment entries.
> 
> For metrics, I'm thinking accuracy broken down by category for LoCoMo, joint goal and slot accuracy for MultiWOZ, and for scalability I'll measure how accuracy and latency degrade as conversations get longer. Actually, for MultiWOZ I realize the standard evaluation is dialogue state tracking, but MOSAIC isn't a DST system. I can repurpose MultiWOZ by feeding the full dialogue into MOSAIC and querying for the final state—that becomes a memory evaluation rather than a turn-by-turn tracking task.
> 
> Better yet, I can frame this as information retrieval from dialogue: given a completed conversation, can the system answer questions about what was discussed? The slot annotations in MultiWOZ naturally give me the questions to ask, and the ground truth is already there. This is cleaner than constructing new QA pairs. The tradeoff is that MultiWOZ dialogues are short (~13 turns), so they won't stress-test long-term memory the way LoCoMo does, but they'll show how well the system extracts structured information—a different but valuable angle for a graph-based memory system.
> 
> Actually, I should prioritize LoCoMo as my main external benchmark and skip the MultiWOZ angle to avoid overcomplicating things. I'll focus instead on comprehensive analysis—ablations, scalability, graph quality—to strengthen the paper. My experiments will be: LoCoMo evaluation with full baseline comparisons and breakdowns by category, then an ablation study removing MOSAIC components to show what each part contributes.
> 
> Wait, but papers with only one benchmark often feel limited. Let me reconsider adding a second dataset—something like PersonaChat for persona consistency checks or Meta's Multi-Session Chat, which is specifically designed for multi-turn conversations where the system needs to retain information across sessions.
> 
> MSC looks promising since it has five conversation sessions between the same pair and explicit annotations for which previous utterances are relevant. I could adapt it for memory evaluation by retrieving relevant memories from sessions 1-4 for each turn in session 5, then checking if the retrieved memories match the gold-standard relevant utterances—this gives me direct retrieval precision and recall metrics.
> 
> Let me check what the Mem0 and A-mem papers actually use as benchmarks...
> 
> Both papers rely on LoCoMo as their primary benchmark, though they supplement it with custom evaluations. So I'll go with LoCoMo as my main external benchmark, then add a synthetic controlled evaluation where I generate conversations of varying lengths, plant specific facts at known positions, and measure how well the system retrieves them afterward—tracking recall, precision, and latency. This controlled setup lets me test scalability and specific capabilities in a way that most papers in this space do. I need to document the full experimental setup with dataset and baseline details, then design fresh result tables that align with the redefined memory system rather than trying to force the old paper's structure.
> 
> Generating the results tables...

# MOSAIC Experiment Guide

## System Assumption

MOSAIC is a graph-based conversation memory system. Its two core operations are **ingest** (read a conversation and organize information into a graph of entities and relationships) and **query** (given a natural language question, traverse the graph to retrieve relevant pieces of information). The answering LLM receives the retrieved context and produces a final answer. All experiments evaluate these two capabilities under different conditions.

---

## Results Map

| Output | Source |
|--------|--------|
| Table 1 — LoCoMo overall comparison | Experiment 1 |
| Table 2 — LoCoMo per-category breakdown | Experiment 1 |
| Table 3 — Retrieval precision / recall | Experiment 1 |
| Table 4 — Scalability accuracy | Experiment 2 |
| Table 5 — Efficiency (latency & memory) | Experiment 2 |
| Table 6 — Ablation results | Experiment 3 |
| Figure 1 — Architecture diagram | Manual |
| Figure 2 — LoCoMo accuracy bar chart | Experiment 1 |
| Figure 3 — Per-category radar chart | Experiment 1 |
| Figure 4 — Accuracy vs conversation length | Experiment 2 |
| Figure 5 — Latency vs conversation length | Experiment 2 |
| Figure 6 — Ablation delta chart | Experiment 3 |
| Figure 7 — Graph structure visualization | Experiment 1 (qualitative) |

---

## Datasets

### Dataset 1 — LoCoMo

LoCoMo is the established benchmark for long-conversation memory systems. It contains multi-session dialogues between pairs of speakers, along with human-annotated QA pairs that require recalling information from earlier sessions.

| Item | Detail |
|------|--------|
| Full name | Long Conversation Memory |
| GitHub | https://github.com/snap-stanford/LoCoMo |
| HuggingFace | https://huggingface.co/datasets/nicolasstucki/locomo |
| Content | 10 multi-session conversations, approximately 600 turns each |
| QA pairs | 300+ pairs with ground-truth answers |
| QA categories | Single-hop, multi-hop, temporal, open-ended, adversarial |
| Splits | Use all 10 conversations for evaluation (following prior work) |

Each QA pair consists of a question, the gold answer, the set of evidence utterances from the conversation that support the answer, and a category label. This structure allows evaluation of both end-to-end answer accuracy and intermediate retrieval quality.

### Dataset 2 — Synthetic Controlled Conversations

This dataset is generated to provide perfect ground truth and control over conversation length, fact density, and temporal distance between information and query. It complements LoCoMo by enabling scalability and efficiency analysis under controlled conditions.

#### Generation Protocol

Define a pool of 200 fact templates across 8 categories: personal information (name, age, birthday, hometown), preferences (food, music, hobby, travel), events (meetings, trips, milestones, incidents), opinions (on topics, products, policies), relationships (family members, friends, colleagues), schedules (appointments, deadlines, routines), health (conditions, medications, allergies), and work (role, projects, skills, goals). Each template has the form "[Subject] [relation] [object]" with typed slots.

For each synthetic conversation, sample K facts from the pool and generate a natural multi-turn dialogue that organically incorporates these facts, interspersed with casual chit-chat that does not contain target facts (distractor turns). Use GPT-4o with temperature 0.7 to generate the conversations, providing the fact list and a persona pair description as input. After generation, verify that each planted fact appears in the conversation by automated string matching, and regenerate any conversation where a fact is missing or distorted.

#### Configurations

| Config ID | Total turns | Facts planted | Fact density | Conversations | Purpose |
|-----------|-------------|---------------|-------------|---------------|---------|
| S-100 | 100 | 20 | 1 per 5 turns | 50 | Short baseline |
| S-300 | 300 | 40 | 1 per 7.5 turns | 50 | Medium |
| S-600 | 600 | 60 | 1 per 10 turns | 50 | LoCoMo-comparable |
| S-1000 | 1,000 | 80 | 1 per 12.5 turns | 50 | Long |
| S-2000 | 2,000 | 100 | 1 per 20 turns | 30 | Stress test |

For each planted fact, generate one direct query and one rephrased query, yielding 2 × K QA pairs per conversation. Ground truth is exact because facts are planted by design.

**Total synthetic QA pairs:** (50×40) + (50×80) + (50×120) + (50×160) + (30×200) = 2,000 + 4,000 + 6,000 + 8,000 + 6,000 = 26,000

---

## Baselines

All methods share the same answering LLM (Qwen-2.5-72B-Instruct, temperature 0) so that the only variable is the memory system. Each memory system receives the same conversation and the same query; it returns a context string; the answering LLM produces the final answer from that context.

| ID | Method | Type | Source | Notes |
|----|--------|------|--------|-------|
| B1 | Full Context | Buffer | N/A | Concatenate entire conversation into the LLM context window. Upper bound on what the LLM can do with perfect memory but limited context length. Truncate from the front if conversation exceeds context window. |
| B2 | Recency Buffer | Buffer | N/A (trivial implementation) | Keep only the most recent N tokens (N = 4096). Represents the no-memory baseline. |
| B3 | Summary Memory | Compression | N/A (implement with same LLM) | Every 20 turns, summarize the conversation so far into a running summary. At query time, provide the summary as context. |
| B4 | RAG | Retrieval | LangChain or LlamaIndex | Chunk conversation into segments of 10 turns. Embed each chunk with `BAAI/bge-large-en-v1.5`. At query time, embed the query, retrieve top-5 chunks by cosine similarity. |
| B5 | Mem0 | Hybrid | https://github.com/mem0ai/mem0 | Official SDK, default configuration. Represents the current state-of-the-art open-source memory system. |
| B6 | A-mem | Agentic | https://github.com/agiresearch/A-mem | Agentic memory with self-reflection. Follow default configuration from the repository. |
| B7 | MemGPT / Letta | Tiered | https://github.com/letta-ai/letta | Official Letta framework. Uses tiered memory (core, recall, archival). Default configuration. |
| B8 | MOSAIC | Graph | This work | Graph-based memory. Ingest conversation into entity-relationship graph, retrieve by graph traversal + embedding similarity. |

### Published Baselines (Cite Without Re-running)

Some methods have published LoCoMo numbers that can be cited for reference. However, for a fair comparison, all methods should be re-run under the identical protocol (same answering LLM, same prompts, same evaluation script).

| Method | Published overall accuracy | Source |
|--------|---------------------------|--------|
| Mem0 | 61.43% | arXiv:2504.19413 |
| memg | 60.41% | Same paper |
| ReadAgent | ~54% | LoCoMo paper |
| Full Context (GPT-4) | ~67% | LoCoMo paper |

These numbers serve as sanity checks. If re-run numbers deviate significantly, investigate prompt or evaluation differences.

---

## Evaluation Metrics

### Answer Quality Metrics

**Accuracy (Acc):** The primary metric. Use an LLM-as-judge approach (GPT-4o, temperature 0) to assess whether the generated answer is semantically equivalent to the gold answer. The judge prompt provides the question, gold answer, and predicted answer, and returns a binary correct/incorrect judgment. This follows the evaluation protocol established in the LoCoMo paper and used by Mem0.

**F1 Score:** Token-level F1 between predicted and gold answers, computed after lowercasing and removing stopwords. Serves as a secondary metric less sensitive to LLM judge variance.

**Category-level Accuracy:** Accuracy broken down by QA category (single-hop, multi-hop, temporal, open-ended, adversarial). This reveals which types of queries benefit most from graph-based memory.

### Retrieval Quality Metrics

**Retrieval Recall (R-Recall):** For LoCoMo QA pairs that include evidence utterance annotations, compute the fraction of gold evidence utterances that appear in (or are semantically covered by) the retrieved context. This measures whether the memory system finds the right information.

**Retrieval Precision (R-Precision):** Fraction of retrieved context chunks/pieces that are relevant to the query (judged by LLM or by overlap with gold evidence). This measures how much noise the memory system introduces.

### Efficiency Metrics

**Ingestion latency:** Wall-clock time to ingest a full conversation into the memory system (build the graph / index / summary).

**Query latency:** Wall-clock time from query input to context output (not including the answering LLM call, which is shared across all methods).

**Memory footprint:** Peak RAM usage of the memory system during and after ingestion.

---

## Experiment Execution

### Experiment 1 — LoCoMo Benchmark Evaluation

**Purpose:** Establish MOSAIC's performance relative to all baselines on the standard long-conversation memory benchmark.

**Produces:** Table 1 (overall comparison), Table 2 (per-category breakdown), Table 3 (retrieval precision/recall), Figure 2 (accuracy bar chart), Figure 3 (radar chart), Figure 7 (qualitative graph visualization)

#### Setup

1. Download LoCoMo dataset from the GitHub or HuggingFace link above.
2. For each of the 10 conversations, prepare a single JSON file containing the full multi-session dialogue in chronological order.
3. For each QA pair, record: question, gold answer, evidence utterances, category label.

#### Procedure

For each method B1–B8:

**Step 1 — Ingest.** Feed all 10 conversations into the memory system. Each conversation is ingested independently (the memory is reset between conversations). Record ingestion time per conversation.

**Step 2 — Query.** For each QA pair, submit the question to the memory system. The memory system returns a context string. Record the returned context and query latency.

**Step 3 — Answer.** Pass the question and retrieved context to the shared answering LLM. Record the generated answer.

**Step 4 — Evaluate.** Run the LLM-as-judge on all (predicted answer, gold answer) pairs. Compute token-level F1. For methods where retrieved context is inspectable (B4, B5, B6, B8 — not B1/B2/B3 which provide full context or summaries), compute R-Recall and R-Precision against evidence utterances.

#### Total Runs

8 methods × 10 conversations × ~30 QA pairs per conversation ≈ 2,400 query evaluations. Plus 8 × 10 = 80 ingestion operations.

#### Extracting Results

**Table 1 — Overall Comparison:** One row per method. Columns: Accuracy, F1, R-Recall, R-Precision, Ingestion latency (mean per conversation), Query latency (mean per query). For B1/B2/B3 where retrieval metrics are not applicable, mark as "—".

**Table 2 — Per-Category Breakdown:** Rows are methods, columns are (Single-hop Acc, Multi-hop Acc, Temporal Acc, Open-ended Acc, Adversarial Acc). This table reveals whether MOSAIC's graph structure specifically helps with multi-hop and temporal queries, which require connecting information across distant parts of the conversation.

**Table 3 — Retrieval Quality:** For methods B4, B5, B6, B7, B8, report R-Recall and R-Precision. Break down by category. This shows whether MOSAIC retrieves more relevant and less noisy context.

**Figure 2:** Grouped bar chart with methods on x-axis and accuracy on y-axis, one bar group per QA category plus an "Overall" group.

**Figure 3:** Radar chart with 5 axes (one per QA category). One polygon per method. Visually shows each method's strength/weakness profile.

**Figure 7:** Select 2–3 representative QA pairs where MOSAIC answers correctly but baselines fail. For each, visualize the subgraph that MOSAIC traversed to retrieve context. Annotate entities and edges that contributed to the correct retrieval. This is generated programmatically from MOSAIC's graph data structure and the query log.

#### Statistical Testing

Bootstrap 95% confidence intervals on accuracy (resample QA pairs 1,000 times). Paired bootstrap test between MOSAIC and each baseline. Report p-values in Table 1.

---

### Experiment 2 — Scalability and Efficiency Analysis

**Purpose:** Measure how each method's accuracy and latency scale as conversation length increases, under controlled conditions with perfect ground truth.

**Produces:** Table 4 (scalability accuracy), Table 5 (efficiency comparison), Figure 4 (accuracy curves), Figure 5 (latency curves)

#### Setup

1. Generate the synthetic conversations following the protocol in the Dataset 2 section above.
2. For each configuration (S-100 through S-2000), prepare conversation files and corresponding QA pairs.

#### Procedure

For each method B1–B8, for each configuration S-100 through S-2000:

**Step 1 — Ingest.** Feed conversations into the memory system. Record ingestion time and peak memory usage.

**Step 2 — Query.** Submit all QA pairs. Record retrieved context and query latency.

**Step 3 — Answer and Evaluate.** Same as Experiment 1. Since ground truth is exact (planted facts), use both LLM-as-judge and exact substring match as metrics. The exact match provides a hard lower bound; the LLM judge captures semantic equivalence for paraphrased answers.

#### Total Runs

8 methods × 5 configurations × (50 or 30 conversations) × (40–200 QA pairs per conversation). Total QA evaluations: 8 × 26,000 = 208,000. This is compute-intensive but each evaluation is a single LLM call. To reduce cost, subsample: for S-100 through S-1000, evaluate all QA pairs from 10 randomly selected conversations per configuration (instead of all 50). For S-2000, evaluate all 30 conversations. Revised total: 8 × [(10×40 + 10×80 + 10×120 + 10×160) + (30×200)] = 8 × (400+800+1200+1600+6000) = 8 × 10,000 = 80,000 evaluations.

#### Extracting Results

**Table 4 — Scalability Accuracy:** Rows are methods. Columns are configurations (S-100, S-300, S-600, S-1000, S-2000). Each cell contains mean accuracy ± std. Add a column for "Δ" showing accuracy drop from S-100 to S-2000. This reveals which methods degrade most with conversation length.

**Table 5 — Efficiency:** Rows are methods. Columns: ingestion latency at S-600 (median), query latency at S-600 (median), peak memory at S-600, ingestion latency at S-2000, query latency at S-2000, peak memory at S-2000. For B1 (full context), ingestion latency is zero but query latency includes the full context processing time by the answering LLM. Note: B1 may fail entirely at S-2000 if the conversation exceeds the context window — record this as a truncation rate.

**Figure 4:** Line chart. X-axis is conversation length (100 to 2000 turns), y-axis is accuracy. One line per method. Error bars from std across conversations. Expected patterns: B1 (full context) is strong at short lengths but drops sharply when truncation begins. B2 (recency) is poor throughout. B3 (summary) degrades as lossy compression accumulates. B4 (RAG) is stable but mediocre. MOSAIC should maintain high accuracy across all lengths due to graph structure preserving relationships regardless of conversation length.

**Figure 5:** Line chart. X-axis is conversation length, y-axis is query latency (log scale). One line per method. Expected patterns: B1 latency grows linearly. B4 (RAG) grows sub-linearly. MOSAIC's latency depends on graph traversal depth, which should grow slowly.

---

### Experiment 3 — Ablation Study

**Purpose:** Identify the contribution of each MOSAIC component to overall performance.

**Produces:** Table 6 (ablation results), Figure 6 (delta chart)

#### Ablation Conditions

Each condition removes or disables one component of MOSAIC while keeping everything else identical.

| ID | Condition | What changes |
|----|-----------|-------------|
| C0 | Full MOSAIC | Nothing removed (control) |
| C1 | − Entity extraction | Disable entity node creation; store conversation chunks as flat text nodes in the graph (edges still computed by embedding similarity) |
| C2 | − Relationship edges | Disable relationship/edge extraction between entities; graph becomes a set of isolated entity nodes; retrieval uses embedding similarity only |
| C3 | − Graph traversal | At query time, retrieve by embedding similarity over nodes only; do not traverse edges to expand context |
| C4 | − Temporal ordering | Remove temporal metadata from nodes and edges; graph has no notion of when information appeared |
| C5 | − Community structure | Disable community detection on the graph; retrieval does not consider cluster membership for context expansion |
| C6 | − Deduplication / merging | Disable entity merging; if the same entity appears multiple times across the conversation, store as separate nodes |

#### Setup and Procedure

Use the LoCoMo dataset (same as Experiment 1). For each condition C0–C6, ingest all 10 conversations and evaluate all 300+ QA pairs using the identical answering LLM and evaluation protocol.

#### Total Runs

7 conditions × 10 conversations × ~30 QA pairs ≈ 2,100 evaluations. Plus 7 × 10 = 70 ingestion operations.

Note that C0 results are identical to the B8 (MOSAIC) results from Experiment 1. No need to re-run; reuse those logs.

#### Extracting Results

**Table 6 — Ablation Results:** Rows are conditions C0–C6. Columns: Overall Acc, Single-hop Acc, Multi-hop Acc, Temporal Acc, Open-ended Acc, Adversarial Acc, Δ Overall (drop from C0). This reveals which components matter most and for which query types.

Expected patterns: C2 (no relationship edges) and C3 (no traversal) should cause the largest drops on multi-hop queries, since answering multi-hop questions requires following entity connections. C4 (no temporal ordering) should disproportionately hurt temporal queries. C1 (no entity extraction) should hurt everything by reducing the graph to a RAG-like system, confirming the value of structured extraction. C6 (no dedup) should hurt precision by creating redundant context.

**Figure 6:** Horizontal bar chart. Each bar represents one ablation condition. Bar length is the accuracy drop (Δ) from C0. Group or color-code bars by which query category is most affected. This provides an at-a-glance view of component importance.

---

## Execution Summary

| Experiment | Datasets used | Methods run | Total evaluations | Outputs |
|-----------|---------------|-------------|-------------------|---------|
| Exp 1: LoCoMo | LoCoMo (10 conv, 300+ QA) | B1–B8 (8 methods) | ~2,400 | Tables 1, 2, 3; Figures 2, 3, 7 |
| Exp 2: Scalability | Synthetic (230 conv, ~10K QA sampled) | B1–B8 (8 methods) | ~80,000 | Tables 4, 5; Figures 4, 5 |
| Exp 3: Ablation | LoCoMo (reuse) | C0–C6 (7 configs of MOSAIC) | ~2,100 (C0 reused from Exp 1) | Table 6; Figure 6 |
| **Total** | | | **~84,500** | **6 tables, 6 figures** |

---

## Reproducibility Checklist

All logs must be saved in a standardized JSON format per query evaluation:

```json
{
  "experiment": "exp1",
  "method": "mosaic",
  "conversation_id": "locomo_03",
  "qa_id": "q_142",
  "category": "multi-hop",
  "question": "...",
  "gold_answer": "...",
  "retrieved_context": "...",
  "predicted_answer": "...",
  "judge_correct": true,
  "token_f1": 0.82,
  "evidence_utterances_gold": ["u_45", "u_203"],
  "evidence_utterances_retrieved": ["u_45", "u_201", "u_203", "u_310"],
  "r_recall": 1.0,
  "r_precision": 0.5,
  "query_latency_ms": 34.2,
  "ingestion_latency_ms": null
}
```

Ingestion logs are saved separately, one per conversation per method:

```json
{
  "experiment": "exp2",
  "method": "mosaic",
  "config": "S-1000",
  "conversation_id": "syn_047",
  "ingestion_latency_ms": 12340,
  "peak_memory_mb": 842,
  "graph_nodes": 312,
  "graph_edges": 587,
  "graph_communities": 14
}
```

All post-processing scripts (table generation, figure plotting, statistical tests) read from these log files. No re-running of experiments is needed to produce any table or figure — every output is derivable from the stored logs.