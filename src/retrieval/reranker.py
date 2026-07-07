import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def _get_reranker():
    """Lazily load the reranker's tokenizer and model directly via transformers,
    bypassing FlagEmbedding's FlagReranker wrapper (currently incompatible with
    transformers v5's tokenizer API)."""
    global _tokenizer, _model
    if _model is None:
        logger.info("Loading BGE-reranker-v2-m3 via raw transformers")
        _tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        _model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
        _model.eval()
    return _tokenizer, _model


def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank fused candidates using a cross-encoder, return the top_k."""
    if not candidates:
        return []

    tokenizer, model = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True,
                            return_tensors='pt', max_length=512)
        logits = model(**inputs, return_dict=True).logits.view(-1).float()
        scores = torch.sigmoid(logits).tolist()

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]