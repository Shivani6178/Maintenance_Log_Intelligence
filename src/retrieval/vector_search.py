import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from src.retrieval.db import get_connection
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def _get_model():
    """Lazily load BGE-M3's tokenizer and base model directly via transformers,
    bypassing FlagEmbedding's BGEM3FlagModel wrapper (currently incompatible
    with transformers v5's from_pretrained dtype handling)."""
    global _tokenizer, _model
    if _model is None:
        logger.info("Loading BGE-M3 via raw transformers")
        _tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        _model = AutoModel.from_pretrained('BAAI/bge-m3')
        _model.eval()
    return _tokenizer, _model


def embed_query(query: str) -> List[float]:
    """Embed a single query string into a 1024-dim dense vector using
    normalized [CLS] pooling, matching BGE-M3's documented dense embedding method."""
    tokenizer, model = _get_model()

    with torch.no_grad():
        inputs = tokenizer(query, padding=True, truncation=True,
                            max_length=512, return_tensors='pt')
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0]
        normalized = F.normalize(cls_embedding, p=2, dim=1)

    return normalized[0].tolist()


def vector_search(query: str, top_k: int = 50) -> List[Dict]:
    """Retrieve the top_k chunks most semantically similar to the query."""
    query_embedding = np.array(embed_query(query), dtype=np.float32)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chunk_id, text, acn, aircraft_model, flight_phase,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))

        results = [
            {
                "chunk_id": row[0], "text": row[1], "acn": row[2],
                "aircraft_model": row[3], "flight_phase": row[4],
                "score": float(row[5]), "source": "vector"
            }
            for row in cursor.fetchall()
        ]
        return results
    finally:
        conn.close()