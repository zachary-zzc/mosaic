#!/usr/bin/env python3
"""
Rebuild the patched graph using the selective sweep (min_shared_labels=2).
Overwrites graph_network_conv0_patched.pkl.
"""
import os, sys, pickle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

GRAPH_PATH = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl"
PATCHED_GRAPH = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl"

# 1. Load the fresh original graph
print("Loading original graph...")
with open(GRAPH_PATH, "rb") as f:
    memory = pickle.load(f)

from src.data.dual_graph import ALL_EDGE_LEGS
edges_before = len(getattr(memory, "edges", []) or [])
adj_before = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
nodes_before = len(adj_before)
edge_count_before = sum(len(v) for v in adj_before.values()) // 2
print(f"Before: {nodes_before} nodes, {edge_count_before} adjacency edges, {edges_before} edge records")

# 2. Apply selective sweep with min_shared_labels=2
print("\nApplying selective cross-class sweep (min_shared_labels=2)...")
stats = memory.sweep_cross_class_cooccurrence_edges(min_shared_labels=2)
print(f"Sweep stats: {stats}")

# 3. Stats after
memory._adj_cache = None
adj_after = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
nodes_after = len(adj_after)
edge_count_after = sum(len(v) for v in adj_after.values()) // 2
edges_after = len(getattr(memory, "edges", []) or [])
print(f"After:  {nodes_after} nodes, {edge_count_after} adjacency edges, {edges_after} edge records")
print(f"Delta:  +{edge_count_after - edge_count_before} adjacency edges, +{edges_after - edges_before} edge records")

# 4. Save
print(f"\nSaving patched graph to {PATCHED_GRAPH}...")
with open(PATCHED_GRAPH, "wb") as f:
    pickle.dump(memory, f)
print("Done.")
