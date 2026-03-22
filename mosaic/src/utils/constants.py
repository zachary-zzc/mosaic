"""Shared constants (e.g. TF-IDF defaults) to avoid duplication."""
from __future__ import annotations

from numbers import Integral
from typing import Any


def tfidf_params_for_corpus_size(params: dict[str, Any], n_docs: int) -> dict[str, Any]:
    """
    sklearn 要求 max_doc_count >= min_doc_count（与 CountVectorizer.fit 中一致）。
    当文档很少时，默认的 float max_df（如 0.6）会使 max_df*n_docs < min_df，触发
    ValueError: max_df corresponds to < documents than min_df。此时将 max_df 放宽为 1.0。
    """
    if n_docs <= 0:
        return dict(params)
    out = dict(params)
    max_df, min_df = out["max_df"], out["min_df"]
    max_doc_count = max_df if isinstance(max_df, Integral) else max_df * n_docs
    min_doc_count = min_df if isinstance(min_df, Integral) else min_df * n_docs
    if max_doc_count < min_doc_count:
        out["max_df"] = 1.0
    return out


# Default TfidfVectorizer params used in assist and graph
DEFAULT_TFIDF_VECTORIZER_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 0.6,
    "use_idf": True,
    "smooth_idf": True,
}

# For keyword coverage (graph) uses max_df=0.7
TFIDF_KEYWORD_MAX_DF = 0.7
