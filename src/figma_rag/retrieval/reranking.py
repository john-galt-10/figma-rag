"""Cross-encoder reranking for retrieved documentation chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass

from figma_rag.embeddings import load_cross_encoder

from .chroma import RetrievalResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True)
class RerankingResult:
    """Reranked chunks plus timing metadata for one reranking pass."""

    results: list[RetrievalResult]
    latency_seconds: float
    candidate_count: int
    model_name: str


class CrossEncoderReranker:
    """Rerank retrieval candidates with a Sentence Transformers cross-encoder."""

    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = load_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> RerankingResult:
        """Score query/result pairs and return the highest scoring chunks."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            raise ValueError("query must not be empty")

        start_time = time.perf_counter()
        if not results:
            latency_seconds = time.perf_counter() - start_time
            return RerankingResult(
                results=[],
                latency_seconds=latency_seconds,
                candidate_count=0,
                model_name=self.model_name,
            )

        pairs = [(query, result.text) for result in results]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scored_results = [
            (float(score), fallback_rank, result)
            for fallback_rank, (score, result) in enumerate(
                zip(scores, results),
                start=1,
            )
        ]
        ranked_results = sorted(
            scored_results,
            key=lambda item: (-item[0], item[1], item[2].chunk_id),
        )
        latency_seconds = time.perf_counter() - start_time

        return RerankingResult(
            results=[
                _with_reranking_metadata(
                    result=result,
                    rank=rank,
                    score=score,
                    model_name=self.model_name,
                    fallback_rank=fallback_rank,
                )
                for rank, (score, fallback_rank, result) in enumerate(
                    ranked_results[:top_k],
                    start=1,
                )
            ],
            latency_seconds=latency_seconds,
            candidate_count=len(results),
            model_name=self.model_name,
        )


def _with_reranking_metadata(
    result: RetrievalResult,
    rank: int,
    score: float,
    model_name: str,
    fallback_rank: int,
) -> RetrievalResult:
    """Return a result annotated with cross-encoder reranking metadata."""

    original_rank = result.rank if result.rank > 0 else fallback_rank
    metadata = dict(result.metadata)
    metadata["reranking_original_rank"] = original_rank
    metadata["reranking_original_distance"] = result.distance
    metadata["rerank_score"] = score
    metadata["reranker_model"] = model_name

    return RetrievalResult(
        rank=rank,
        chunk_id=result.chunk_id,
        text=result.text,
        distance=-score,
        title=result.title,
        section=result.section,
        source_url=result.source_url,
        metadata=metadata,
    )
