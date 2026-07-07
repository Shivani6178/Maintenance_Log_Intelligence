# from src.retrieval.pipeline import retrieve


# def test_retrieve_returns_results():
#     results = retrieve("engine vibration and bearing wear during climb")
#     assert len(results) > 0
#     for r in results:
#         print(f"{r['chunk_id']} (rerank={r['rerank_score']:.3f}): {r['text'][:150]}...")


# if __name__ == "__main__":
#     test_retrieve_returns_results()

from src.generation.rag_service import answer_query

result = answer_query("engine vibration and bearing wear during climb")
print(f"Answer:\n{result['answer']}\n")
print(f"Sources ({len(result['sources'])}):")
for s in result['sources']:
    print(f"  ACN {s['acn']} ({s['aircraft_model']}, {s['flight_phase']})")