"""Train the student on saved teacher labels, then score validation and test.

This does not call the teacher. Generate labels in the notebook first and
keep notebooks/data/setwise_labels.jsonl.

    poetry run python scripts/train_student.py --data-dir /path/to/splits

On a GPU, raise the batch:

    poetry run python scripts/train_student.py \
        --data-dir /path/to/splits \
        --batch-size 8 --epochs 5
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.schema import load_examples  # noqa: E402
from src.engine.evaluator import evaluate_pointwise  # noqa: E402
from src.engine.trainer import SetwiseLoader, SetwiseTrainer  # noqa: E402
from src.loss.pairwise import pairwise_ranknet_loss  # noqa: E402
from src.loss.setwise import setwise_loss  # noqa: E402
from src.models.student import StudentRanker  # noqa: E402
from src.models.teacher import SetwiseTeacher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the setwise student")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with train.parquet, vali.parquet, and small_test.parquet",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "notebooks" / "data" / "setwise_labels.jsonl",
        help="Teacher labels saved from the notebook",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ROOT / "logs" / "train_log.csv",
        help="Epoch loss and train nDCG@10",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=ROOT / "logs" / "metrics.json",
        help="Validation and test nDCG written after training",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--student-model",
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--loss",
        choices=("setwise", "pairwise"),
        default="setwise",
        help="setwise is cross-entropy on the winner. pairwise is RankNet.",
    )
    return parser.parse_args()


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    if not args.labels.exists():
        raise FileNotFoundError(
            f"Missing teacher labels: {args.labels}. "
            "Run the notebook labeling cell and copy setwise_labels.jsonl."
        )

    examples = load_examples(args.labels)
    train_df = load_split(args.data_dir, "train.parquet")
    val_df = load_split(args.data_dir, "vali.parquet")
    test_df = load_split(args.data_dir, "small_test.parquet")
    print(f"labels: {len(examples)}")
    print(f"train: {train_df['query'].nunique()} queries, {len(train_df)} rows")
    print(f"val: {val_df['query'].nunique()} queries, {len(val_df)} rows")
    print(f"test: {test_df['query'].nunique()} queries, {len(test_df)} rows")

    # format_doc only. Ollama here avoids opening an API client.
    # Pairwise labels were cut at 3000 characters. Setwise labels used 1500.
    teacher = SetwiseTeacher(
        api_client_type="ollama",
        model_name="unused",
        dataset_name="jusbrasil",
        max_chars=3000 if args.loss == "pairwise" else 1500,
    )
    student = StudentRanker(args.student_model)
    loader = SetwiseLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    loss_fn = pairwise_ranknet_loss if args.loss == "pairwise" else setwise_loss
    trainer = SetwiseTrainer(student, loss_fn, lr=args.learning_rate)
    history = trainer.fit(
        loader,
        epochs=args.epochs,
        log_path=args.log_path,
        ndcg_frame=train_df,
        format_fn=teacher.format_doc,
    )

    val_metrics = evaluate_pointwise(
        student,
        val_df,
        format_fn=teacher.format_doc,
        batch_size=args.eval_batch_size,
    )
    test_metrics = evaluate_pointwise(
        student,
        test_df,
        format_fn=teacher.format_doc,
        batch_size=args.eval_batch_size,
    )
    print("val ", val_metrics.means)
    print("test", test_metrics.means)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(
            {"history": history, "val": val_metrics.means, "test": test_metrics.means},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.log_path}")
    print(f"wrote {args.metrics_path}")


if __name__ == "__main__":
    main()
