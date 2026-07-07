from typing import List, Dict


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    top_k: int = 50
) -> List[Dict]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each chunk's fused score is the sum of 1/(k + rank) across every list
    it appears in. k=60 is the standard RRF constant from the original
    paper, chosen to avoid over-weighting the very top rank of any single list.
    """
    fused_scores: Dict[str, float] = {}
    chunk_data: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = item

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    fused_results = []
    for chunk_id in ranked_ids[:top_k]:
        item = dict(chunk_data[chunk_id])
        item["fused_score"] = fused_scores[chunk_id]
        fused_results.append(item)

    return fused_results