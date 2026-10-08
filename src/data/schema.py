"""Records passed between the teacher, the loss, and the student."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class SetwiseExample:
    """One teacher comparison.

    documents are the formatted texts the teacher judged, in prompt order.
    winner_index points at the document the teacher selected.
    The student trains on this set. It does not see a set at inference.
    """

    query: str
    documents: list[str]
    winner_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SetwiseExample":
        return cls(
            query=str(data["query"]),
            documents=[str(doc) for doc in data["documents"]],
            winner_index=int(data["winner_index"]),
        )


def save_examples(examples: list[SetwiseExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def load_examples(path: str | Path) -> list[SetwiseExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(SetwiseExample.from_dict(json.loads(line)))
    return examples
