"""nDCG, matching the graded metric used in the setwise pilot."""

import numpy as np
import pandas as pd


def ndcg_at_k(labels_sorted_by_score, labels_ideal, k: int) -> float:
    """nDCG@k with gains 2^rel - 1.

    labels_sorted_by_score is the relevance of each document in predicted
    order, best first. labels_ideal is the relevance of the same list in
    any order; the ideal ranking is computed inside this function.
    """
    labels_sorted_by_score = np.asarray(labels_sorted_by_score, dtype=float)
    labels_ideal = np.asarray(labels_ideal, dtype=float)
    k = min(int(k), len(labels_sorted_by_score))
    if k <= 0:
        return 0.0

    predicted = labels_sorted_by_score[:k]
    ideal = np.sort(labels_ideal)[::-1][:k]
    discounts = np.log2(np.arange(1, k + 1) + 1)
    dcg = np.sum((np.power(2.0, predicted) - 1) / discounts)
    idcg = np.sum((np.power(2.0, ideal) - 1) / discounts)
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mean_ndcg(
    frame: pd.DataFrame,
    score_col: str,
    label_col: str = "relevance",
    query_col: str = "query",
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Per-query nDCG. Higher score_col is ranked first."""
    rows = []
    for query, group in frame.groupby(query_col, sort=False):
        relevance = group[label_col].to_numpy(dtype=float)
        order = np.argsort(-group[score_col].to_numpy(dtype=float), kind="mergesort")
        ranked = relevance[order]
        row = {"query": query, "num_docs": len(group)}
        for k in ks:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, relevance, k)
        rows.append(row)
    return pd.DataFrame(rows)
