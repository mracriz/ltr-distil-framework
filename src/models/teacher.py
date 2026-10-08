"""LLM teachers for distillation.

LLMTeacher owns the client, the local model, document formatting, and
completion. SetwiseTeacher and PairwiseTeacher only add the comparison
their approach needs. SetwiseTeacher.rank is the heapsort from
notebooks/jusbrasil_soc_llms_setwise.ipynb (Zhuang et al., SIGIR 2024).
"""

import os
import string
from openai import OpenAI

from src.config import DATASET_CONFIGS
from src.data.schema import SetwiseExample


def _huggingface_token() -> str | None:
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip() or os.getenv(
        "HUGGING_FACE_HUB_TOKEN", ""
    ).strip()
    return token or None


class LLMTeacher:
    def __init__(
        self,
        api_client_type: str,
        model_name: str,
        dataset_name: str = "jusbrasil",
        max_chars: int = 1500,
    ):
        self.api_client_type = api_client_type
        self.model_name = model_name
        self.max_chars = max_chars
        self.config = self._load_dataset_config(dataset_name)
        self.tokenizer = None
        self.model = None
        self.client = self._set_api_client()

    def _set_api_client(self) -> OpenAI | None:
        kind = self.api_client_type.lower()
        if kind == "huggingface":
            self._load_local_model()
            return None
        if kind == "openrouter":
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is missing. Put it in a .env file."
                )
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        if kind == "ollama":
            return OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        raise ValueError(
            f"Invalid API client: {self.api_client_type}. "
            "Use huggingface, openrouter, or ollama."
        )

    def _load_local_model(self) -> None:
        """Load a causal LM on this machine. No OpenRouter or Ollama call."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = _huggingface_token()
        on_gpu = torch.cuda.is_available()
        dtype = torch.bfloat16 if on_gpu else torch.float32
        # Leave a few GB free on each card for the attention cache.
        max_memory = None
        if on_gpu:
            gpu_count = torch.cuda.device_count()
            max_memory = {index: "40GiB" for index in range(gpu_count)}
            print(f"teacher gpus: {gpu_count}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=token,
            torch_dtype=dtype,
            device_map="auto" if on_gpu else None,
            max_memory=max_memory,
        )
        self.model.eval()

    def _load_dataset_config(self, dataset_name: str) -> dict:
        if dataset_name not in DATASET_CONFIGS:
            known = ", ".join(sorted(DATASET_CONFIGS))
            raise ValueError(f"Invalid dataset name: {dataset_name}. Known: {known}")
        return DATASET_CONFIGS[dataset_name]

    def format_doc(self, row: dict, max_chars: int | None = None) -> str:
        """Text of one document, shared by the teacher prompt and the student."""
        limit = self.max_chars if max_chars is None else max_chars
        title = str(row.get("title", "Sem título"))
        body = str(row.get("body", ""))[:limit]
        if self.config["court_field"]:
            court = str(row.get("court", "Tribunal Nao Informado"))
            return f"Title: {title}\nCourt: {court}\nText: {body}..."
        return f"Title: {title}\nText: {body}..."

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.api_client_type.lower() == "huggingface":
            return self._complete_local(system_prompt, user_prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2,
        )
        return response.choices[0].message.content.strip()

    def _complete_local(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
            )
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class SetwiseTeacher(LLMTeacher):
    def __init__(
        self,
        api_client_type: str,
        model_name: str,
        k_size: int = 4,
        dataset_name: str = "jusbrasil",
        max_chars: int = 1500,
    ):
        if k_size < 2:
            raise ValueError("k_size must be at least 2")
        super().__init__(
            api_client_type,
            model_name,
            dataset_name=dataset_name,
            max_chars=max_chars,
        )
        self.k_size = k_size

    def compare_documents_setwise(self, docs: list[dict], query_text: str) -> int:
        """Ask the LLM which document in this set is the most relevant.

        Args:
            docs: row dicts with title, body, and court when the dataset has it.
            query_text: the search query.

        Returns:
            Index into docs of the winning document.
            Return -1 when the reply is not one of the labels.

        Use self.client, self.model_name, self.format_doc, and self.config
        (persona, doc_type). The prompt you already ran is in
        notebooks/jusbrasil_soc_llms_setwise.ipynb.

        Steps to implement:
            1. Label the documents A, B, C, ... in the order they were given.
            2. Put the query and each formatted document in the user message.
            3. Ask for the single letter of the most relevant document.
            4. Call the chat API at temperature 0 with a short completion.
            5. Map that letter back to an index. Anything else is -1.
        """
        system_prompt = f"""You are {self.config["persona"]}.
        Your task is to compare multiple documents and determine which one is the most relevant to a user's search query.
        """

        labels = list(string.ascii_uppercase)[: len(docs)]

        docs_formatted = []
        for label, doc in zip(labels, docs):
            docs_formatted.append(f"Document {label}:\n{self.format_doc(doc)}")

        doc_string = "\n\n".join(docs_formatted)

        if len(labels) > 1:
            listed = ", ".join(f"'{label}'" for label in labels[:-1])
            valid_labels = f"{listed}, or '{labels[-1]}'"
        else:
            valid_labels = f"'{labels[0]}'"

        user_prompt = f"""Query: {query_text}

        {doc_string}

        Which document is the most relevant to the query? Output exactly the letter of the most relevant document ({valid_labels}). Do not provide any explanation or extra text."""

        try:
            winner_token = self._complete(system_prompt, user_prompt)
        except Exception as error:
            print(f"Error comparing documents: {error}")
            return -1

        reply = winner_token.upper()
        for character in reply:
            if character in labels:
                return labels.index(character)
        return -1

    def rank(
        self,
        docs: list[dict],
        query_text: str,
        top_n: int | None = None,
    ) -> tuple[list[dict], list[SetwiseExample]]:
        """Order docs best-first with setwise heapsort.

        k_size is the number of documents in each LLM call (the paper's c).
        Each heap step compares the parent with up to k_size - 1 children.

        The second return value is the distillation trace: one SetwiseExample
        per successful comparison. Invalid replies are used only as a sort
        fallback (index 0) and are left out of the trace.

        top_n=None ranks the full list. top_n=10 stops early, as in the paper.
        """
        arr = list(docs)
        n = len(arr)
        trace: list[SetwiseExample] = []
        if n <= 1:
            return arr, trace

        num_child = max(self.k_size - 1, 1)
        extract_n = n if top_n is None else min(top_n, n)

        def heapify(heap_size: int, index: int) -> None:
            child_start = num_child * index + 1
            if child_start >= heap_size:
                return
            child_end = min(num_child * (index + 1) + 1, heap_size)
            positions = [index, *range(child_start, child_end)]
            subset = [arr[position] for position in positions]
            winner = self.compare_documents_setwise(subset, query_text)
            if 0 <= winner < len(subset):
                trace.append(
                    SetwiseExample(
                        query=query_text,
                        documents=[self.format_doc(doc) for doc in subset],
                        winner_index=winner,
                    )
                )
                best_local = winner
            else:
                best_local = 0
            largest = positions[best_local]
            if largest != index:
                arr[index], arr[largest] = arr[largest], arr[index]
                heapify(heap_size, largest)

        for index in range(n // num_child, -1, -1):
            heapify(n, index)

        extracted = 0
        for index in range(n - 1, 0, -1):
            arr[index], arr[0] = arr[0], arr[index]
            extracted += 1
            if extracted == extract_n:
                break
            heapify(index, 0)

        if extract_n == n:
            ordered = list(reversed(arr))
        else:
            ordered = list(reversed(arr[n - extract_n :])) + arr[: n - extract_n]
        return ordered, trace


class PairwiseTeacher(LLMTeacher):
    def compare_documents_pairwise(self, doc_a: dict, doc_b: dict, query_text: str) -> int:
        """Ask which of two documents is more relevant.

        Return 0 when doc_a wins, 1 when doc_b wins, and -1 when the reply
        is neither letter. A pair stored this way is a set of size 2, so the
        same student and the same training loop can consume it.

        The notebook `notebooks/jusbrasil_soc_llms_pairwise.ipynb` uses this
        comparison as a sort key and returns 1, -1, or 0. This method returns
        an index, the same convention as compare_documents_setwise.

        Use self._complete, self.format_doc, and self.config["persona"].
        Steps:
            1. Format doc_a and doc_b.
            2. Ask for the single letter A or B at temperature 0.
            3. Map A to 0 and B to 1. Anything else is -1.
        """
        
        system_prompt  = f"""You are {self.config['persona']}.
        Your task is to compare two documents and determine which one is relevant for the user's search query."""

        user_prompt = f"""Query: {query_text}

        Document A:
        {self.format_doc(doc_a)}

        Document B:
        {self.format_doc(doc_b)}

        Which document is more relevant to the query? Output exactly 'A' if document A is more relevant, 'B' if document B is more relevant. Do not provide any explanation or extra text."""
        
        winner_token = self._complete(system_prompt, user_prompt).strip().upper()

        if winner_token == 'A':
            return 0
        elif winner_token == 'B':
            return 1
        else:
            return - 1