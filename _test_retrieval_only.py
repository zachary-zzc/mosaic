#!/usr/bin/env python3
"""
Retrieval-only test: compare original vs patched graph's ability to retrieve
gold-answer-relevant entities for multi-hop + open-domain questions.
No LLM calls needed - purely structural verification.
"""
import json, os, sys, pickle, re
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

from src.data.graph import ClassGraph
from src.assist import load_mosaic_memory_pickle, read_to_file_json

QA_PATH = os.path.join(PROJECT_ROOT, "dataset/locomo/qa_0.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json")
ORIGINAL_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl")
PATCHED_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl")

CAT_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop", 4: "Open-domain"}


def build_text_index(memory):
    """Build entity_id -> instance text index."""
    index = {}
    for node in memory.graph.nodes:
        cid = getattr(node, "class_id", None) or ""
        for inst in getattr(node, "_instances", []) or []:
            iid = inst.get("instance_id")
            ikey = memory._instance_key(cid, iid)
            eid = memory._instance_key_to_entity_id(ikey)
            index[eid] = json.dumps(inst, ensure_ascii=False, default=str)
    return index


def find_gold_entities(text_index, gold_answer, question):
    """Find entities that contain keywords from the gold answer."""
    # Extract meaningful words from gold answer
    words = re.findall(r'\b\w{4,}\b', gold_answer)
    if not words:
        words = re.findall(r'\b\w{3,}\b', gold_answer)
    
    gold_entities = set()
    for word in words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        for eid, text in text_index.items():
            if pattern.search(text):
                gold_entities.add(eid)
    return gold_entities


def evaluate_retrieval(memory, text_index, questions, label):
    """Evaluate retrieval coverage for multi-hop + open-domain questions."""
    from src.data.dual_graph import ALL_EDGE_LEGS
    
    adj = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
    eid_to_ikey = {}
    for node in memory.graph.nodes:
        cid = getattr(node, "class_id", None) or ""
        for inst in getattr(node, "_instances", []) or []:
            iid = inst.get("instance_id")
            ikey = memory._instance_key(cid, iid)
            eid = memory._instance_key_to_entity_id(ikey)
            eid_to_ikey[eid] = ikey
    
    cat_stats = {3: [], 4: []}
    
    for q in questions:
        cat = q["category"]
        if cat not in (3, 4):
            continue
            
        question = q["question"]
        gold = q["answer"]
        
        # Run retrieval
        ctx, trace = memory._search_by_sub_hash(question)
        retrieved = set(trace.get("retrieved_entity_ids", []))
        neighbors = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))
        all_retrieved = retrieved | neighbors
        
        # Find gold entities
        gold_entities = find_gold_entities(text_index, gold, question)
        
        if not gold_entities:
            continue
        
        found = gold_entities & all_retrieved
        missed = gold_entities - all_retrieved
        
        # Check disconnected
        disconnected = 0
        for m in missed:
            m_ikey = eid_to_ikey.get(m)
            if not m_ikey or m_ikey not in adj or not adj[m_ikey]:
                disconnected += 1
        
        recall = len(found) / len(gold_entities) if gold_entities else 0
        
        cat_stats[cat].append({
            "question": question,
            "gold": gold,
            "gold_entities": len(gold_entities),
            "found": len(found),
            "missed": len(missed),
            "disconnected": disconnected,
            "recall": recall,
            "prompt_chars": trace.get("prompt_chars", 0),
        })
    
    return cat_stats


def print_results(cat_stats, label):
    print(f"\n{'='*60}")
    print(f"RETRIEVAL ANALYSIS: {label}")
    print(f"{'='*60}")
    
    for cat in [3, 4]:
        items = cat_stats[cat]
        if not items:
            continue
        avg_recall = sum(x["recall"] for x in items) / len(items) if items else 0
        avg_missed = sum(x["missed"] for x in items) / len(items) if items else 0
        avg_disconnected = sum(x["disconnected"] for x in items) / len(items) if items else 0
        total_disconnected = sum(x["disconnected"] for x in items)
        
        print(f"\n  {CAT_NAMES[cat]} ({len(items)} questions):")
        print(f"    Avg gold entity recall: {avg_recall:.1%}")
        print(f"    Avg missed entities: {avg_missed:.1f}")
        print(f"    Avg disconnected entities: {avg_disconnected:.1f}")
        print(f"    Total disconnected: {total_disconnected}")
        
        # Show worst cases
        worst = sorted(items, key=lambda x: x["recall"])[:3]
        for w in worst:
            print(f"    Worst: recall={w['recall']:.0%} missed={w['missed']} disc={w['disconnected']} | {w['question'][:60]}")


def main():
    questions = read_to_file_json(QA_PATH)
    target_qs = [q for q in questions if q.get("category") in (3, 4)]
    print(f"Target questions: {len(target_qs)} (multi-hop: {sum(1 for q in target_qs if q['category']==3)}, open-domain: {sum(1 for q in target_qs if q['category']==4)})")
    
    # Test original
    print("\nLoading original graph...")
    orig = load_mosaic_memory_pickle(ORIGINAL_GRAPH)
    orig.process_kw(TAGS_PATH)
    orig_index = build_text_index(orig)
    orig_stats = evaluate_retrieval(orig, orig_index, target_qs, "ORIGINAL")
    print_results(orig_stats, "ORIGINAL")
    
    # Test patched
    print("\nLoading patched graph...")
    patched = load_mosaic_memory_pickle(PATCHED_GRAPH)
    patched.process_kw(TAGS_PATH)
    patch_index = build_text_index(patched)
    patch_stats = evaluate_retrieval(patched, patch_index, target_qs, "PATCHED")
    print_results(patch_stats, "PATCHED")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Original vs Patched")
    print(f"{'='*60}")
    for cat in [3, 4]:
        o = orig_stats[cat]
        p = patch_stats[cat]
        if not o or not p:
            continue
        o_recall = sum(x["recall"] for x in o) / len(o)
        p_recall = sum(x["recall"] for x in p) / len(p)
        o_disc = sum(x["disconnected"] for x in o)
        p_disc = sum(x["disconnected"] for x in p)
        delta = (p_recall - o_recall) * 100
        print(f"  {CAT_NAMES[cat]}:")
        print(f"    Recall: {o_recall:.1%} -> {p_recall:.1%} ({'+' if delta>=0 else ''}{delta:.1f}pp)")
        print(f"    Disconnected: {o_disc} -> {p_disc}")


if __name__ == "__main__":
    main()
