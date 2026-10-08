"""Load a ranking table and cut it into sets of size k."""

import random

import pandas as pd

REQUIRED_COLUMNS = ("query", "title", "body", "relevance")


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns the teacher and the evaluator read."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    columns = list(REQUIRED_COLUMNS)
    for optional in ("court", "doc_id"):
        if optional in df.columns:
            columns.append(optional)
    return df.loc[:, columns].copy()


def query_groups(df: pd.DataFrame) -> list[tuple[str, list[dict]]]:
    """Group candidate documents by query, preserving row order."""
    frame = prepare_frame(df)
    groups = []
    for query, group in frame.groupby("query", sort=False):
        groups.append((str(query), group.to_dict(orient="records")))
    return groups


def sample_document_sets(
    docs: list[dict],
    k: int,
    n_sets: int,
    rng: random.Random,
) -> list[list[dict]]:
    """Draw n_sets subsets of size k from one query's candidates.

    Queries with fewer than k documents are skipped. A document is not
    repeated inside a single set. Sets from the same query may overlap.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(docs) < k or n_sets < 1:
        return []
    return [rng.sample(docs, k) for _ in range(n_sets)]
