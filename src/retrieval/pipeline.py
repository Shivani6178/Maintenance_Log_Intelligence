from src.retrieval.vector_search import vector_search
from src.retrieval.bm25_search import keyword_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from typing import List, Dict


def retrieve(query: str, fusion_top_k: int = 50, final_top_k: int = 5) -> List[Dict]:
    """Full hybrid retrieval pipeline: vector + keyword search, RRF fusion, rerank."""
    vector_results = vector_search(query, top_k=fusion_top_k)
    keyword_results = keyword_search(query, top_k=fusion_top_k)

    fused = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        top_k=fusion_top_k
    )

    final_results = rerank(query, fused, top_k=final_top_k)
    return final_results