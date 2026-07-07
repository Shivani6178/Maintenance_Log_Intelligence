from src.retrieval.db import get_connection
from typing import List, Dict
import re

STOPWORDS = {
    'a', 'an', 'and', 'the', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'during', 'is', 'are', 'was', 'were'
}


def build_keyword_query(text: str) -> str:
    """Convert free text into an OR-joined tsquery string, stripping stopwords."""
    words = re.findall(r'\w+', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return ' | '.join(filtered) if filtered else text.lower()


def keyword_search(query: str, top_k: int = 50) -> List[Dict]:
    """Retrieve the top_k chunks ranked by full-text keyword relevance."""
    tsquery_str = build_keyword_query(query)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chunk_id, text, acn, aircraft_model, flight_phase,
                   ts_rank(text_search_vector, tsquery) AS rank
            FROM chunks, to_tsquery('english', %s) tsquery
            WHERE text_search_vector @@ tsquery
            ORDER BY rank DESC
            LIMIT %s;
        """, (tsquery_str, top_k))

        results = [
            {
                "chunk_id": row[0], "text": row[1], "acn": row[2],
                "aircraft_model": row[3], "flight_phase": row[4],
                "score": float(row[5]), "source": "keyword"
            }
            for row in cursor.fetchall()
        ]
        return results
    finally:
        conn.close()