# Setwise distillation for learning to rank

Distil a setwise LLM teacher into a student ranker.

The teacher compares a small set of documents and names the most relevant one. Those choices become training labels. The student scores each document on its own. The training loss is setwise: inside each set, probability mass should sit on the teacher's winner. At inference there is no set and no loss. The student scores every candidate independently and the list is sorted by that score.

This follows the setwise heapsort in `notebooks/jusbrasil_soc_llms_setwise.ipynb` (Zhuang et al., SIGIR 2024). The lab notebook is `notebooks/setwise_distill.ipynb`.

## Pipeline

1. `SetwiseTeacher.compare_documents_setwise` asks the LLM which document in a set of size `k` wins.
2. `SetwiseLabelGenerator` stores those winners. Random k-subsets are the cheap source. `from_rank_trace` instead records every comparison the heapsort makes.
3. `StudentRanker.score_pairs` maps one query-document pair to one score.
4. `setwise_loss` reads a `[batch, k]` block of those scores and the winner index.
5. `evaluate_pointwise` sorts each query by the student score and reports nDCG against the human `relevance` column. Human labels are for evaluation, not for the distillation loss.

## What to implement

The scaffolding around these three pieces already runs.

| Piece | File | Contract |
| --- | --- | --- |
| Teacher comparison | `src/models/teacher.py` | `compare_documents_setwise(docs, query) -> int` |
| Setwise loss | `src/loss/setwise.py` | `setwise_loss(scores [batch, k], winner [batch]) -> scalar` |
| Student scorer | `src/models/student.py` | `score_pairs(queries, documents) -> scores [num_pairs]` |

`src/loss/listmle.py` and `src/loss/pairwise.py` are optional extra objectives. The trainer calls `setwise_loss`.

A check for the loss, once it is written: scores `[[2.0, 0.0, 0.0]]` and winner `0` should give a value near `0.2395`.

Suggested order: teacher, then generate a handful of labels, then the loss, then the student, then train and evaluate. The lab notebook follows that order.

## Layout

```
src/config.py            dataset personas and DistillConfig
src/data/                query grouping, set sampling, JSONL labels
src/models/teacher.py    LLMTeacher, plus SetwiseTeacher and PairwiseTeacher
src/models/student.py    pointwise ranker (score_pairs is yours)
src/loss/                setwise loss (yours); listmle and pairwise optional
src/engine/generator.py  teacher labels
src/engine/trainer.py    training loop
src/engine/evaluator.py  pointwise inference and nDCG
src/utils/metrics.py     nDCG, same definition as the setwise pilot
```

## Setup

```bash
poetry install
poetry run python -m ipykernel install --user --name llm-as-judge-for-ltr
poetry run jupyter lab
```

`torch` is required for the student, the loss, and the trainer. The student exercise also needs Transformers:

```bash
poetry add transformers
```

Put `OPENROUTER_API_KEY` in a `.env` file if the teacher client is `openrouter`. Ollama uses `http://localhost:11434/v1` and does not need a key.

```bash
poetry run pytest
```

## Pairwise track

Same student and the same pointwise evaluation. The teacher compares two documents instead of a set, and the loss is RankNet on that pair. A saved pair is a `SetwiseExample` with two documents, so the existing training loop can read it.

Already in place: all-pairs labeling in `PairwiseLabelGenerator.from_all_pairs`, JSONL, `scripts/label_pairwise.py`, and `scripts/train_student.py --loss pairwise`. A query with n documents produces n(n-1)/2 teacher calls.

Two pieces are yours:

| Piece | File | Contract |
| --- | --- | --- |
| Pair comparison | `src/models/teacher.py` | `PairwiseTeacher.compare_documents_pairwise(doc_a, doc_b, query) -> int` |
| RankNet loss | `src/loss/pairwise.py` | `pairwise_ranknet_loss(scores [batch, 2], winner [batch]) -> scalar` |

`compare_documents_pairwise` returns 0 when the first document wins and 1 when the second wins. The pairwise notebook returns a sort comparator instead. Use `self._complete` and `self.format_doc`. Do not rebuild the client.

A check for the loss, once it is written: scores `[[2.0, 0.0]]` and winner `0` should give a value near `0.1269`.

After both are written:

```bash
sbatch scripts/label_pairwise.sbatch
sbatch scripts/train_pairwise.sbatch
```
