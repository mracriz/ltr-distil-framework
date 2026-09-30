"""Training loop and pointwise eval, using a stand-in student and loss."""

import torch
from torch import nn

from src.data.schema import SetwiseExample
from src.engine.evaluator import evaluate_pointwise
from src.engine.trainer import SetwiseLoader, SetwiseTrainer
from src.models.student import flatten_sets


class ToyStudent(nn.Module):
    """Scores a document by its length. Enough to exercise the loop."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def score_pairs(self, queries, documents):
        lengths = torch.tensor(
            [float(len(doc)) for doc in documents],
            dtype=torch.float32,
            device=self.weight.device,
        )
        return lengths * self.weight

    def score_sets(self, queries, document_sets):
        flat_queries, flat_documents, k = flatten_sets(queries, document_sets)
        return self.score_pairs(flat_queries, flat_documents).view(len(queries), k)


def _stand_in_loss(scores, winner_index):
    """Not the setwise loss. Only checks that a scalar can drive the optimizer."""
    return scores.pow(2).mean()


def test_trainer_step_updates_the_student(tmp_path, capsys):
    import pandas as pd

    examples = [
        SetwiseExample(query="q", documents=["aa", "bbbb"], winner_index=1),
        SetwiseExample(query="q2", documents=["c", "ddd"], winner_index=0),
    ]
    frame = pd.DataFrame(
        [
            {"query": "q", "title": "short", "body": "a", "relevance": 0},
            {"query": "q", "title": "long", "body": "aaaa", "relevance": 3},
        ]
    )
    student = ToyStudent()
    before = student.weight.detach().clone()
    trainer = SetwiseTrainer(student, _stand_in_loss, lr=0.05, device="cpu")
    log_path = tmp_path / "train_log.csv"
    history = trainer.fit(
        SetwiseLoader(examples, batch_size=2, shuffle=False),
        epochs=2,
        log_path=log_path,
        ndcg_frame=frame,
        format_fn=lambda row: row["body"],
    )
    assert [row["epoch"] for row in history] == [1, 2]
    assert history[0]["train_ndcg@10"] == 1.0
    saved = pd.read_csv(log_path)
    assert list(saved["epoch"]) == [1, 2]
    assert "train_loss" in saved.columns
    printed = capsys.readouterr().out
    assert "epoch 1" in printed
    assert "train_ndcg@10" in printed
    assert not torch.equal(before, student.weight.detach())


def test_pointwise_eval_ranks_by_student_score():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"query": "q", "title": "short", "body": "a", "relevance": 0},
            {"query": "q", "title": "long", "body": "aaaa", "relevance": 3},
        ]
    )
    student = ToyStudent()
    result = evaluate_pointwise(student, frame, format_fn=lambda row: row["body"])
    assert result.means["ndcg@1"] == 1.0
    best = result.scored.sort_values("score", ascending=False).iloc[0]
    assert best["relevance"] == 3.0
