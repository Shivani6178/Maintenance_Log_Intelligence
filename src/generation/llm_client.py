import os
from groq import Groq
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    """Lazily initialize the Groq client once per process."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


SYSTEM_PROMPT = """You are an aviation maintenance intelligence assistant. You answer \
questions using ONLY the ASRS incident reports provided as context below. \
Every claim you make must be traceable to a specific report by its ACN number.

Rules:
- If the context does not contain enough information to answer, say so explicitly. \
Do not invent details.
- Cite the ACN number in parentheses after each claim, e.g. "engine vibration during \
climb was a common precursor to compressor stalls (ACN 2341775)."
- Be concise and factual. This is for maintenance engineers, not general readers.
- If multiple reports describe a similar pattern, synthesize them rather than listing \
each one separately.
- Cite ACN numbers precisely for each distinct claim, but do not repeat the full \
list of ACNs at the end of every sentence if they were already cited earlier in \
the same paragraph.
"""


def clean_acn(acn) -> str:
    """Strip trailing .0 from ACN values that were stored/retrieved as floats."""
    acn_str = str(acn)
    return acn_str[:-2] if acn_str.endswith('.0') else acn_str


def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[ACN {clean_acn(chunk['acn'])}] ({chunk['aircraft_model']}, {chunk['flight_phase']})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    query: str,
    context_chunks: List[Dict],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 600
) -> Dict:
    """Generate a grounded answer from retrieved chunks using Groq."""
    if not context_chunks:
        return {
            "answer": "No relevant maintenance reports were found for this query.",
            "sources": []
        }

    context = build_context(context_chunks)
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"}
            ]
        )
        answer_text = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        return {
            "answer": "Unable to generate an answer at this time. Please try again.",
            "sources": []
        }

    sources = [
        {
            "acn": clean_acn(c["acn"]),
            "aircraft_model": c["aircraft_model"],
            "flight_phase": c["flight_phase"],
            "excerpt": c["text"][:200],
            "relevance_score": c.get("rerank_score", c.get("fused_score", 0))
        }
        for c in context_chunks
    ]

    return {"answer": answer_text, "sources": sources}