"""Pointwise student ranker.

The model scores one (query, document) pair at a time. Setwise structure
lives in the loss, which reads a vector of those scores. Inference sorts
by the same score and never builds a set.
"""

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def flatten_sets(
    queries: list[str],
    document_sets: list[list[str]],
) -> tuple[list[str], list[str], int]:
    """Expand a batch of sets into pairs. Every set must have the same size."""
    if not queries or len(queries) != len(document_sets):
        raise ValueError("queries and document_sets must be non-empty and aligned")
    widths = {len(docs) for docs in document_sets}
    if len(widths) != 1:
        raise ValueError("every set in a batch must have the same size")
    k = len(document_sets[0])
    if k < 2:
        raise ValueError("a setwise example needs at least 2 documents")

    flat_queries: list[str] = []
    flat_documents: list[str] = []
    for query, docs in zip(queries, document_sets):
        flat_queries.extend([query] * k)
        flat_documents.extend(docs)
    return flat_queries, flat_documents, k


class StudentRanker(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        #Load the model substituting the classification head for a linear layer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

        #If the tokenizer does not have a pad token (critical for Decoder-only models, like Llama), set it to the eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id    

    

    def score_pairs(self, queries: list[str], documents: list[str]) -> torch.Tensor:
        """Relevance logit for each pair. Shape [num_pairs]. Higher is better.

        len(queries) must equal len(documents). Put the tensor on the same
        device as the model parameters, and keep it attached to the graph
        during training.
        """
        raise NotImplementedError(
            "Implement StudentRanker.score_pairs. "
            "Return one score per (query, document), shape [num_pairs]."
        )

    def score_sets(
        self,
        queries: list[str],
        document_sets: list[list[str]],
    ) -> torch.Tensor:
        """Stack pointwise scores into [batch, k] for the setwise loss.

        This method is part of the framework. The exercise is score_pairs.
        """
        flat_queries, flat_documents, k = flatten_sets(queries, document_sets)
        scores = self.score_pairs(flat_queries, flat_documents)
        expected = len(flat_documents)
        if scores.ndim != 1 or scores.shape[0] != expected:
            raise ValueError(
                f"score_pairs must return shape [{expected}], got {tuple(scores.shape)}"
            )
        return scores.view(len(queries), k)
