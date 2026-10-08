"""Ask a local Hugging Face teacher to label training sets.

This replaces OpenRouter and Ollama. Run it before student training.
Llama 3.1 8B fits on one A6000. Llama 3.3 70B needs four.

    poetry run python scripts/label_teacher.py \
        --data-dir /path/to/splits
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.schema import save_examples  # noqa: E402
from src.engine.generator import SetwiseLabelGenerator  # noqa: E402
from src.models.teacher import SetwiseTeacher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label sets with a local teacher")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with train.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notebooks" / "data" / "setwise_labels_hf.jsonl",
        help="Where to write the teacher winners",
    )
    parser.add_argument(
        "--teacher-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face id. 8B fits on one A6000; 70B needs four.",
    )
    parser.add_argument("--k-size", type=int, default=4)
    parser.add_argument("--sets-per-query", type=int, default=4)
    parser.add_argument("--max-doc-chars", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing split: {train_path}")

    train_df = pd.read_parquet(train_path)
    print(f"train: {train_df['query'].nunique()} queries, {len(train_df)} rows")
    print(f"teacher: {args.teacher_model}")

    teacher = SetwiseTeacher(
        api_client_type="huggingface",
        model_name=args.teacher_model,
        k_size=args.k_size,
        dataset_name="jusbrasil",
        max_chars=args.max_doc_chars,
    )
    generator = SetwiseLabelGenerator(
        teacher,
        sets_per_query=args.sets_per_query,
        seed=args.seed,
    )
    generated = generator.from_samples(train_df)
    save_examples(generated.examples, args.output)
    print(
        f"{len(generated.examples)} examples, {generated.skipped} skipped, "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
