"""Plot training loss against epoch from the CSV written by SetwiseTrainer.

Usage:
    poetry run python scripts/plot_learning_curve.py notebooks/data/train_log.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(log_path: Path, out_path: Path) -> None:
    frame = pd.read_csv(log_path)
    if "epoch" not in frame.columns or "train_loss" not in frame.columns:
        raise ValueError("log must have epoch and train_loss columns")

    figure, axis = plt.subplots()
    axis.plot(frame["epoch"], frame["train_loss"], marker="o", color="C0")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("Learning curve")

    ndcg_column = "train_ndcg@10"
    if ndcg_column in frame.columns and frame[ndcg_column].notna().any():
        twin = axis.twinx()
        twin.plot(frame["epoch"], frame[ndcg_column], marker="s", color="C1")
        twin.set_ylabel("Training nDCG@10")
        twin.set_ylim(0.0, 1.0)

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a training log CSV")
    parser.add_argument(
        "log",
        nargs="?",
        default="notebooks/data/train_log.csv",
        help="CSV written by SetwiseTrainer.fit",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="PNG path. Defaults to the CSV name with a .png suffix.",
    )
    args = parser.parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_suffix(".png")
    plot_learning_curve(log_path, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
