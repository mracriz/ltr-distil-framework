"""Framework tests that do not require the three exercises."""

import random

import pandas as pd
import pytest

from src.config import DATASET_CONFIGS
from src.data.dataset import sample_document_sets
from src.data.schema import SetwiseExample, load_examples, save_examples
from src.engine.generator import SetwiseLabelGenerator
from src.models.teacher import SetwiseTeacher
from src.utils.metrics import mean_ndcg, ndcg_at_k


class _TeacherStub(SetwiseTeacher):
    """Skips the API client. Tests supply compare_documents_setwise."""

    def __init__(self, k_size=3, dataset_name="trec_dl", max_chars=20):
        self.api_client_type = "stub"
        self.model_name = "stub"
        self.k_size = k_size
        self.max_chars = max_chars
        self.config = DATASET_CONFIGS[dataset_name]
        self.client = None


class OracleTeacher(_TeacherStub):
    def compare_documents_setwise(self, docs, query_text):
        return max(range(len(docs)), key=lambda i: float(docs[i]["relevance"]))


class ScriptedTeacher(_TeacherStub):
    def __init__(self, winners, k_size=2):
        super().__init__(k_size=k_size)
        self._winners = list(winners)

    def compare_documents_setwise(self, docs, query_text):
        return self._winners.pop(0)


def _pair_frame():
    return pd.DataFrame(
        [
            {"query": "q", "title": "a", "body": "alpha", "relevance": 1},
            {"query": "q", "title": "b", "body": "beta", "relevance": 0},
            {"query": "other", "title": "c", "body": "gamma", "relevance": 2},
        ]
    )


def test_ndcg_perfect_and_reversed():
    relevance = [0, 1, 3, 2]
    assert ndcg_at_k([3, 2, 1, 0], relevance, 10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2, 3], relevance, 10) < 1.0
    assert ndcg_at_k([0, 0], [0, 0], 2) == 0.0


def test_mean_ndcg_sorts_by_score_descending():
    frame = pd.DataFrame(
        [
            {"query": "q", "relevance": 0, "score": 0.1},
            {"query": "q", "relevance": 3, "score": 0.9},
            {"query": "q", "relevance": 1, "score": 0.2},
        ]
    )
    metrics = mean_ndcg(frame, score_col="score", ks=(1,))
    assert metrics.loc[0, "ndcg@1"] == pytest.approx(1.0)


def test_format_doc_respects_court_field():
    legal = _TeacherStub(dataset_name="jusbrasil", max_chars=5)
    web = _TeacherStub(dataset_name="trec_dl", max_chars=5)
    row = {"title": "T", "court": "STF", "body": "abcdefghij"}

    legal_text = legal.format_doc(row)
    web_text = web.format_doc(row)

    assert "Court: STF" in legal_text
    assert "Court:" not in web_text
    assert "abcde..." in web_text
    assert "fghij" not in web_text


def test_sample_document_sets_size_and_short_queries():
    docs = [{"title": str(i)} for i in range(6)]
    rng = random.Random(0)
    drawn = sample_document_sets(docs, k=4, n_sets=3, rng=rng)
    assert len(drawn) == 3
    assert all(len(subset) == 4 for subset in drawn)
    assert all(len({doc["title"] for doc in subset}) == 4 for subset in drawn)
    assert sample_document_sets(docs[:3], k=4, n_sets=2, rng=rng) == []


def test_heapsort_orders_by_the_teacher_preference():
    docs = [{"title": str(rel), "body": "x", "relevance": rel} for rel in (1, 4, 2, 5, 3)]
    ordered, trace = OracleTeacher(k_size=3).rank(docs, "query")
    assert [doc["relevance"] for doc in ordered] == [5, 4, 3, 2, 1]
    assert trace
    assert all(example.winner_index >= 0 for example in trace)
    assert all(len(example.documents) >= 2 for example in trace)


def test_generator_keeps_winners_and_skips_invalid_replies(tmp_path):
    teacher = ScriptedTeacher([1, -1, 0], k_size=2)
    result = SetwiseLabelGenerator(teacher, sets_per_query=1, seed=0).from_samples(
        _pair_frame()
    )
    # "other" has a single document, so only query "q" produces a set.
    # Three scripted answers are more than the one set; the first answer is used.
    assert result.queries_seen == 2
    assert len(result.examples) == 1
    assert result.examples[0].winner_index == 1
    assert result.skipped == 0

    teacher = ScriptedTeacher([-1], k_size=2)
    skipped = SetwiseLabelGenerator(teacher, sets_per_query=1, seed=0).from_samples(
        _pair_frame()
    )
    assert skipped.examples == []
    assert skipped.skipped == 1

    path = tmp_path / "labels.jsonl"
    save_examples(
        [SetwiseExample(query="q", documents=["a", "b"], winner_index=0)],
        path,
    )
    loaded = load_examples(path)
    assert loaded[0].winner_index == 0
    assert loaded[0].documents == ["a", "b"]


def test_rank_trace_can_feed_the_generator():
    frame = pd.DataFrame(
        [
            {"query": "q", "title": "a", "body": "a", "relevance": 1},
            {"query": "q", "title": "b", "body": "b", "relevance": 3},
            {"query": "q", "title": "c", "body": "c", "relevance": 2},
        ]
    )
    result = SetwiseLabelGenerator(OracleTeacher(k_size=3)).from_rank_trace(frame)
    assert result.examples
    assert result.examples[0].query == "q"
