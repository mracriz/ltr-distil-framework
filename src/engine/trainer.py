"""Train the pointwise student with a setwise loss."""

import csv
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.data.schema import SetwiseExample
from src.utils.logger import get_logger

logger = get_logger(__name__)


def collate_setwise(examples: list[SetwiseExample]) -> dict:
    """One batch. Every example must share the same set size."""
    if not examples:
        raise ValueError("empty batch")
    width = len(examples[0].documents)
    if any(len(example.documents) != width for example in examples):
        raise ValueError("mixed set sizes in one batch")
    return {
        "queries": [example.query for example in examples],
        "document_sets": [list(example.documents) for example in examples],
        "winner_index": torch.tensor(
            [example.winner_index for example in examples],
            dtype=torch.long,
        ),
    }


class SetwiseLoader:
    """Batches of SetwiseExample, grouped so each batch has one set size."""

    def __init__(
        self,
        examples: list[SetwiseExample],
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.examples = list(examples)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        grouped: dict[int, list[SetwiseExample]] = {}
        for example in self.examples:
            grouped.setdefault(len(example.documents), []).append(example)

        rng = random.Random(self.seed)
        batches: list[list[SetwiseExample]] = []
        for group in grouped.values():
            if self.shuffle:
                rng.shuffle(group)
            for start in range(0, len(group), self.batch_size):
                batches.append(group[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield collate_setwise(batch)

    def __len__(self) -> int:
        return sum(
            (count + self.batch_size - 1) // self.batch_size
            for count in self._counts_by_width().values()
        )

    def _counts_by_width(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for example in self.examples:
            width = len(example.documents)
            counts[width] = counts.get(width, 0) + 1
        return counts


class SetwiseTrainer:
    """Optimize the student so its pointwise scores pick the teacher winner.

    loss_fn is setwise_loss once you have implemented it:
    loss_fn(scores [batch, k], winner_index [batch]) -> scalar.
    """

    def __init__(self, student, loss_fn, lr: float = 2e-5, device: str | None = None):
        self.student = student
        self.loss_fn = loss_fn
        self.device = device or _default_device()
        self.student.to(self.device)
        parameters = [p for p in self.student.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError(
                "StudentRanker has no parameters. "
                "Load an encoder and a scoring head in StudentRanker.__init__."
            )
        self.optimizer = torch.optim.AdamW(parameters, lr=lr)

    def train_step(self, batch: dict) -> float:
        self.student.train()
        self.optimizer.zero_grad()
        scores = self.student.score_sets(batch["queries"], batch["document_sets"])
        winner = batch["winner_index"].to(scores.device)
        loss = self.loss_fn(scores, winner)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def fit(
        self,
        loader: SetwiseLoader,
        epochs: int = 1,
        log_path: str | Path | None = None,
        ndcg_frame=None,
        format_fn=None,
    ) -> list[dict]:
        """Train for `epochs` and record one row per epoch.

        When `ndcg_frame` and `format_fn` are both set, each row includes
        training nDCG@10 on that frame. The grades in the frame are human
        relevance labels. The loss itself is whatever `loss_fn` computes.
        """
        if (ndcg_frame is None) != (format_fn is None):
            raise ValueError("Pass ndcg_frame and format_fn together to log nDCG@10")

        history: list[dict] = []
        for epoch in range(1, epochs + 1):
            steps = tqdm(loader, total=len(loader), desc=f"epoch {epoch} train")
            losses = []
            for batch in steps:
                loss = self.train_step(batch)
                losses.append(loss)
                steps.set_postfix(loss=f"{loss:.3f}")
            mean_loss = sum(losses) / max(len(losses), 1)
            ndcg = self._train_ndcg_at_10(ndcg_frame, format_fn)
            row = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_ndcg@10": ndcg,
            }
            history.append(row)
            self._print_epoch(row)
            if log_path is not None:
                _write_epoch_log(log_path, history)
        return history

    def _train_ndcg_at_10(self, frame, format_fn) -> float | None:
        if frame is None:
            return None
        from src.engine.evaluator import evaluate_pointwise

        metrics = evaluate_pointwise(
            self.student,
            frame,
            format_fn,
            ks=(10,),
            batch_size=4,
        )
        return metrics.means.get("ndcg@10")

    def _print_epoch(self, row: dict) -> None:
        message = f"epoch {row['epoch']}  train_loss {row['train_loss']:.4f}"
        if row["train_ndcg@10"] is not None:
            message += f"  train_ndcg@10 {row['train_ndcg@10']:.4f}"
        print(message)
        logger.info(message)


def _write_epoch_log(path: str | Path, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_ndcg@10"],
        )
        writer.writeheader()
        for row in history:
            ndcg = row["train_ndcg@10"]
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": f"{row['train_loss']:.6f}",
                    "train_ndcg@10": "" if ndcg is None else f"{ndcg:.6f}",
                }
            )


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
