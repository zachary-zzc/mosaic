"""轨 C / P-3：对话控制与打分（docs/optimization.md §6、§10）。"""
from src.control.dialog_controller import DialogController, TurnTrace
from src.control.scoring import (
    communities_from_g_a,
    communities_louvain_ga,
    importance_entropy_placeholder,
    neighbors_union_gp_ga,
    pagerank_on_ga,
    score_placeholder,
)

__all__ = [
    "DialogController",
    "TurnTrace",
    "communities_from_g_a",
    "communities_louvain_ga",
    "importance_entropy_placeholder",
    "neighbors_union_gp_ga",
    "pagerank_on_ga",
    "score_placeholder",
]
