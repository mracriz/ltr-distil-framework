"""Dataset prompts and default distillation settings."""

from dataclasses import dataclass


DATASET_CONFIGS = {
    "jusbrasil": {
        "persona": "an expert Brazilian legal information retrieval judge",
        "doc_type": "legal documents",
        "court_field": True,
    },
    "trec_dl": {
        "persona": "an expert web search quality rater",
        "doc_type": "web passages",
        "court_field": False,
    },
}


@dataclass
class DistillConfig:
    """Knobs shared by label generation, training, and evaluation.

    k_size is the number of documents in one teacher comparison.
    The student still scores documents one at a time.
    """

    dataset_name: str = "jusbrasil"
    teacher_client: str = "ollama"
    teacher_model: str = "llama3.1:8b"
    student_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    k_size: int = 4
    max_doc_chars: int = 1500
    sets_per_query: int = 4
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 2e-5
    seed: int = 0
