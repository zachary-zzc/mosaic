"""
手稿 DualGraph 打分与社区占位（docs/optimization.md §6 C、§10 P-3）。

完整 \(\alpha \tilde I+\beta T+\gamma C\) 需 belief 熵与 Leiden；此处提供 **PageRank on \(G_A\)** 与 **连通分量社区**（无 igraph 时的降级）。
"""
from __future__ import annotations

from typing import Any

import networkx as nx


def pagerank_on_ga(G_a: nx.Graph, *, alpha: float = 0.85) -> dict[str, float]:
    if G_a.number_of_nodes() == 0:
        return {}
    return nx.pagerank(G_a, alpha=alpha, weight="weight")


def communities_from_g_a(G_a: nx.Graph) -> dict[str, str]:
    """无 Leiden 时以无向连通分量 id 作为 ``communities`` 字段（EntityGraph 导出）。"""
    if G_a.number_of_nodes() == 0:
        return {}
    out: dict[str, str] = {}
    for i, comp in enumerate(nx.connected_components(G_a)):
        cid = f"cc_{i}"
        for n in comp:
            out[str(n)] = cid
    return out


def communities_louvain_ga(G_a: nx.Graph, *, resolution: float = 1.0, seed: int = 42) -> dict[str, str]:
    """
    在 \(G_A\) 上 **Louvain** 社区（networkx≥2.8）；失败时退回连通分量。
    手稿 Leiden 可用 igraph+leidenalg 替换；全量方案允许 Methods 写 Louvain 为工程默认。
    """
    if G_a.number_of_nodes() == 0:
        return {}
    try:
        from networkx.algorithms.community import louvain_communities
    except Exception:
        return communities_from_g_a(G_a)
    try:
        comms = louvain_communities(G_a, weight="weight", resolution=resolution, seed=seed)
    except TypeError:
        try:
            comms = louvain_communities(G_a, weight="weight", seed=seed)
        except Exception:
            return communities_from_g_a(G_a)
    except Exception:
        return communities_from_g_a(G_a)
    out: dict[str, str] = {}
    for i, c in enumerate(comms):
        for n in c:
            out[str(n)] = f"lv_{i}"
    return out


def importance_entropy_placeholder(belief: dict[str, Any] | None) -> float:
    """belief 含 entropy 则用；否则 unknown→1.0、partial→0.5、confirmed→0.1。"""
    if not belief:
        return 1.0
    h = belief.get("entropy")
    if h is not None:
        try:
            return float(h)
        except (TypeError, ValueError):
            pass
    st = str(belief.get("state") or "unknown").lower()
    if st == "confirmed":
        return 0.1
    if st == "partial":
        return 0.5
    return 1.0


def neighbors_union_gp_ga(
    G_p: nx.DiGraph,
    G_a: nx.Graph,
    delta: set[str],
) -> set[str]:
    """
    NCS 预测前沿的并集邻域（与手稿 Part A 对齐的图论定义，轨 C）：

    \(\mathcal{N}(\Delta_t) \cup \Delta_t\)，其中邻居含 \(G_P\) 上入/出邻与 \(G_A\) 上无向邻。
    """
    frontier: set[str] = set(delta)
    for u in delta:
        frontier.update(G_p.predecessors(u))
        frontier.update(G_p.successors(u))
        if G_a.has_node(u):
            frontier.update(G_a.neighbors(u))
    return frontier


def score_placeholder(
    *,
    importance_hat: float,
    pagerank_t: float,
    community_bonus: float,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> float:
    """手稿式 Score 的线性骨架（\(\tilde I,T,C\) 由调用方填入）。"""
    return alpha * importance_hat + beta * pagerank_t + gamma * community_bonus
