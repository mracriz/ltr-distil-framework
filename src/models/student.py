"""Pointwise student ranker.

The model scores one (query, document) pair at a time. Setwise structure
lives in the loss, which reads a vector of those scores. Inference sorts
by the same score and never builds a set.
"""

import os

import torch
from torch import nn
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ollama tags are not checkpoints. These are the weights PyTorch can fine-tune.
# llama3.2:3b is the 3B Instruct model Ollama serves.
OLLAMA_STUDENT_MODELS = {
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
}


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


def resolve_student_model(model_name: str) -> str:
    """Map an Ollama tag to a Hugging Face id. Other names are used as given."""
    return OLLAMA_STUDENT_MODELS.get(model_name, model_name)


def _huggingface_token() -> str | None:
    """Read a token from the environment or the project .env file."""
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip() or os.getenv(
        "HUGGING_FACE_HUB_TOKEN", ""
    ).strip()
    return token or None


class StudentRanker(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.hf_model_name = resolve_student_model(model_name)
        token = _huggingface_token()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, token=token)

        # A new one-logit head. The pretrained language-model head is not reused.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_model_name,
            num_labels=1,
            token=token,
        )

        # A pad token equal to eos makes Llama read the wrong position.
        # Decoder models need their own pad id.
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    

    def score_pairs(self, queries: list[str], documents: list[str]) -> torch.Tensor:
        """Relevance logit for each pair. Shape [num_pairs]. Higher is better.

        len(queries) must equal len(documents). Put the tensor on the same
        device as the model parameters, and keep it attached to the graph
        during training.
        """

        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")

        # BERT-style tokenizers encode a pair. Llama has no separator token,
        # so the query and the document go in one string.
        if self.tokenizer.sep_token is None:
            texts = [
                f"Query: {query}\nDocument: {document}"
                for query, document in zip(queries, documents)
            ]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        else:
            inputs = self.tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

        #Move input tensors for the same device (CPU/GPU)
        device = next(self.model.parameters()).device
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        #Do the forward pass (without generating text, only calculating gradients/logits)
        outputs = self.model(**inputs)

        # Extract the logits. The output has the shape [batch_size, 1]
        # The squeeze(-1) removes the last dimession, return a 1D array [batch_size]
        return outputs.logits.squeeze(-1)
        

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
