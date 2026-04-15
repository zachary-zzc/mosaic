#!/usr/bin/env python3
"""
Re-enrich existing graph pickles with improved edge parameters.

Instead of a full graph rebuild (expensive LLM calls), this script:
1. Loads an existing ClassGraph .pkl
2. Strips all A-edges (semantic BGE edges)
3. Re-runs add_semantic_association_edges_bge with new parameters
4. Runs ensure_minimum_connectivity (P2 fix)
5. Saves the re-enriched graph to a new .pkl

Usage:
    cd mosaic
    python ../_rebuild_edges.py                          # conv3,4,5
    python ../_rebuild_edges.py --convs conv3 conv4      # specific convs
    python ../_rebuild_edges.py --min-sim 0.40           # custom threshold
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import pickle
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MOSAIC_ROOT = os.path.join(PROJECT_ROOT, "mosaic")
sys.path.insert(0, MOSAIC_ROOT)

from src.data.dual_graph import EDGE_LEG_ASSOCIATIVE, normalize_edge_leg
from src.config_loader import get_edge_construction_config
from src.graph.dual.edge_construction import (
    add_semantic_association_edges_bge,
    ensure_minimum_connectivity,
)

RUNS_DIR = os.path.join(
    PROJECT_ROOT, "experiments", "locomo", "benchmark", "runs", "runs"
)


def load_graph(pkl_path: str):
    """Load ClassGraph from pickle."""
    with open(pkl_path, "rb") as f:
        cg = pickle.load(f)
    return cg


def strip_a_edges(cg) -> int:
    """Remove all E_A (associative/semantic) edges from cg.edges and rebuild nx graphs."""
    original = len(cg.edges or [])
    cg.edges = [
        rec for rec in (cg.edges or [])
        if normalize_edge_leg(rec.get("edge_leg")) != EDGE_LEG_ASSOCIATIVE
    ]
    removed = original - len(cg.edges)
    # Clear embedding cache so it gets rebuilt
    cg._bge_embedding_cache = None
    # Rebuild nx graphs from remaining P-edges only
    cg._rebuild_dual_nx_from_edges()
    return removed


def rebuild_edges(cg, cfg):
    """Re-run A-edge construction + minimum connectivity."""
    stats_a = add_semantic_association_edges_bge(cg, cfg)
    stats_mc = ensure_minimum_connectivity(cg, cfg)
    return stats_a, stats_mc


def main():
    parser = argparse.ArgumentParser(description="Re-enrich graph edges")
    parser.add_argument(
        "--convs", nargs="*", default=["conv3", "conv4", "conv5"],
        help="Conversations to re-enrich (default: conv3 conv4 conv5)",
    )
    parser.add_argument(
        "--min-sim", type=float, default=None,
        help="Override semantic_min_similarity (default: use config)",
    )
    parser.add_argument(
        "--strategy", default="hybrid",
        choices=["hybrid", "hash_only"],
        help="Graph strategy (default: hybrid)",
    )
    parser.add_argument(
        "--suffix", default="_v2",
        help="Suffix for output pkl filename (default: _v2)",
    )
    args = parser.parse_args()

    cfg = get_edge_construction_config()
    if args.min_sim is not None:
        cfg = dataclasses.replace(cfg, semantic_min_similarity=args.min_sim)

    print(f"Edge construction config:")
    print(f"  semantic_min_similarity = {cfg.semantic_min_similarity}")
    print(f"  semantic_max_pairs      = {cfg.semantic_max_pairs}")
    print(f"  semantic_min_text_len   = {cfg.semantic_min_text_len}")
    print()

    for conv in args.convs:
        art_dir = os.path.join(RUNS_DIR, conv, "artifacts", args.strategy)
        pkl_in = os.path.join(art_dir, f"graph_network_{conv}.pkl")
        pkl_out = os.path.join(art_dir, f"graph_network_{conv}{args.suffix}.pkl")

        if not os.path.isfile(pkl_in):
            print(f"[skip] {conv}: {pkl_in} not found")
            continue

        print(f"[{conv}] Loading {pkl_in}")
        t0 = time.time()
        cg = load_graph(pkl_in)

        n_p = cg.G_p.number_of_edges()
        n_a = cg.G_a.number_of_edges()
        print(f"  Before: {n_p} P-edges, {n_a} A-edges, {len(cg.edges or [])} total records")

        removed = strip_a_edges(cg)
        print(f"  Stripped {removed} A-edge records")

        stats_a, stats_mc = rebuild_edges(cg, cfg)
        n_p2 = cg.G_p.number_of_edges()
        n_a2 = cg.G_a.number_of_edges()
        print(f"  After:  {n_p2} P-edges, {n_a2} A-edges, {len(cg.edges or [])} total records")
        print(f"  A-edges: {stats_a['edges_added']} added (was {n_a})")
        print(f"  Min-connectivity: {stats_mc['isolated_found']} isolated, {stats_mc['edges_added']} connected")

        with open(pkl_out, "wb") as f:
            pickle.dump(cg, f)
        elapsed = time.time() - t0
        print(f"  Saved to {pkl_out} ({elapsed:.1f}s)")
        print()

    print("Done. Run tests with:")
    print(f'  python ../_test_single.py 1 48 0.5 --suffix "{args.suffix}"')


if __name__ == "__main__":
    main()
