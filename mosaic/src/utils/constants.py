"""Shared constants (e.g. TF-IDF defaults) to avoid duplication."""
from __future__ import annotations

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
