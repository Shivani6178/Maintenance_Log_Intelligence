from src.retrieval.pipeline import retrieve
from src.generation.llm_client import generate_answer
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def answer_query(query: str, top_k: int = 5) -> Dict:
    """Full RAG loop: retrieve relevant chunks, then generate a grounded answer."""
    logger.info(f"Processing query: {query}")

    chunks = retrieve(query, final_top_k=top_k)

    if not chunks:
        return {
            "query": query,
            "answer": "No relevant maintenance reports were found for this query.",
            "sources": []
        }

    result = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"]
    }