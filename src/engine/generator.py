"""Turn a ranking table into setwise teacher labels."""

import random
import itertools
from dataclasses import dataclass, field

from tqdm.auto import tqdm

from src.data.dataset import query_groups, sample_document_sets
from src.data.schema import SetwiseExample
from src.models.teacher import PairwiseTeacher, SetwiseTeacher
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    examples: list[SetwiseExample] = field(default_factory=list)
    skipped: int = 0
    queries_seen: int = 0


class SetwiseLabelGenerator:
    """Call the teacher on sets and keep the winners as training targets.

    Two sources of sets:

    - from_samples draws random k-subsets. This is the cheaper way to
      build a training file.
    - from_rank_trace runs the setwise heapsort and keeps every comparison
      the sort actually made. Those are the decisions behind a full ranking.
    """

    def __init__(
        self,
        teacher: SetwiseTeacher,
        sets_per_query: int = 4,
        seed: int = 0,
    ):
        self.teacher = teacher
        self.sets_per_query = sets_per_query
        self.seed = seed

    def from_samples(
        self,
        df,
        sets_per_query: int | None = None,
        max_queries: int | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        n_sets = self.sets_per_query if sets_per_query is None else sets_per_query
        rng = random.Random(self.seed if seed is None else seed)
        groups = query_groups(df)
        if max_queries is not None:
            groups = groups[:max_queries]

        result = GenerationResult(queries_seen=len(groups))
        for query, docs in tqdm(groups, desc="Sampling setwise labels"):
            for subset in sample_document_sets(docs, self.teacher.k_size, n_sets, rng):
                self._append_comparison(result, query, subset)
        self._log(result, "sampled")
        return result

    def from_rank_trace(
        self,
        df,
        top_n: int | None = None,
        max_queries: int | None = None,
    ) -> GenerationResult:
        groups = query_groups(df)
        if max_queries is not None:
            groups = groups[:max_queries]

        result = GenerationResult(queries_seen=len(groups))
        for query, docs in tqdm(groups, desc="Tracing setwise heapsort"):
            _ordered, trace = self.teacher.rank(docs, query, top_n=top_n)
            result.examples.extend(trace)
        self._log(result, "heapsort")
        return result

    def _append_comparison(
        self,
        result: GenerationResult,
        query: str,
        subset: list[dict],
    ) -> None:
        winner = self.teacher.compare_documents_setwise(subset, query)
        if winner < 0 or winner >= len(subset):
            result.skipped += 1
            return
        result.examples.append(
            SetwiseExample(
                query=query,
                documents=[self.teacher.format_doc(doc) for doc in subset],
                winner_index=winner,
            )
        )

    def _log(self, result: GenerationResult, source: str) -> None:
        logger.info(
            "%s labels: %s examples, %s skipped, %s queries",
            source,
            len(result.examples),
            result.skipped,
            result.queries_seen,
        )


class PairwiseLabelGenerator:
    """Call the teacher on document pairs and keep the winner as the label.

    Each stored example has two formatted documents. winner_index is 0 or 1.
    The student still scores documents one at a time. The pairwise loss reads
    those two scores.
    """

    def __init__(
        self,
        teacher: PairwiseTeacher,
        pairs_per_query: int = 4,
        seed: int = 0,
    ):
        self.teacher = teacher
        self.pairs_per_query = pairs_per_query
        self.seed = seed

    def from_samples(
        self,
        df,
        pairs_per_query: int | None = None,
        max_queries: int | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        n_pairs = self.pairs_per_query if pairs_per_query is None else pairs_per_query
        rng = random.Random(self.seed if seed is None else seed)
        groups = query_groups(df)
        if max_queries is not None:
            groups = groups[:max_queries]

        result = GenerationResult(queries_seen=len(groups))
        for query, docs in tqdm(groups, desc="Sampling pairwise labels"):
            for pair in sample_document_sets(docs, 2, n_pairs, rng):
                self._append_pair(result, query, pair)
        logger.info(
            "pairwise labels: %s examples, %s skipped, %s queries",
            len(result.examples),
            result.skipped,
            result.queries_seen,
        )
        return result

    def _append_pair(
        self,
        result: GenerationResult,
        query: str,
        pair: list[dict],
    ) -> None:
        winner = self.teacher.compare_documents_pairwise(pair[0], pair[1], query)
        if winner not in (0, 1):
            result.skipped += 1
            return
        result.examples.append(
            SetwiseExample(
                query=query,
                documents=[self.teacher.format_doc(doc) for doc in pair],
                winner_index=winner,
            )
        )

    def from_all_pairs(
        self,
        df,
        max_queries: int | None = None,
    ) -> GenerationResult:
        """Compare every unordered pair of documents for each query.

        A query with n documents produces n * (n - 1) / 2 teacher calls.
        """
        groups = query_groups(df)
        if max_queries is not None:
            groups = groups[:max_queries]

        result = GenerationResult(queries_seen=len(groups))

        for query, docs in tqdm(groups, desc="Generating all-pairs labels"):
            for pair in itertools.combinations(docs, 2):
                self._append_pair(result, query, list(pair))
        logger.info(
            "all-pairs labels: %s examples, %s skipped, %s queries",
            len(result.examples),
            result.skipped,
            result.queries_seen,
        )
        return result