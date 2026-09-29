"""Pointwise inference and nDCG against human relevance labels.

Training uses the teacher's winner. Evaluation uses the relevance column.
The student scores each document alone; the list is sorted by that score.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.dataset import query_groups
from src.utils.metrics import mean_ndcg


@dataclass
class EvalResult:
    per_query: pd.DataFrame
    means: dict[str, float]
    scored: pd.DataFrame


@torch.no_grad()
def score_query(
    student,
    query: str,
    documents: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """Pointwise scores for one query. documents are formatted texts."""
    student.eval()
    if not documents:
        return np.array([], dtype=float)

    chunks = []
    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        scores = student.score_pairs([query] * len(chunk), chunk)
        chunks.append(scores.detach().cpu().float().numpy())
    return np.concatenate(chunks)


def evaluate_pointwise(
    student,
    df: pd.DataFrame,
    format_fn,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    batch_size: int = 16,
) -> EvalResult:
    """Score every candidate independently and compute nDCG.

    format_fn must be the same formatter the teacher used, usually
    teacher.format_doc, so the student reads the text it was distilled on.
    """
    scored_rows = []
    for query, docs in query_groups(df):
        texts = [format_fn(doc) for doc in docs]
        scores = score_query(student, query, texts, batch_size=batch_size)
        for doc, text, score in zip(docs, texts, scores):
            scored_rows.append(
                {
                    "query": query,
                    "document": text,
                    "relevance": float(doc["relevance"]),
                    "score": float(score),
                }
            )

    if not scored_rows:
        empty_scores = pd.DataFrame(columns=["query", "document", "relevance", "score"])
        empty_metrics = pd.DataFrame(
            columns=["query", "num_docs", *[f"ndcg@{k}" for k in ks]]
        )
        return EvalResult(per_query=empty_metrics, means={}, scored=empty_scores)

    scored = pd.DataFrame(scored_rows)
    per_query = mean_ndcg(scored, score_col="score", ks=ks)
    means = {
        column: float(per_query[column].mean())
        for column in per_query.columns
        if column.startswith("ndcg@")
    }
    return EvalResult(per_query=per_query, means=means, scored=scored)
