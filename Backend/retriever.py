import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from confidence import assess_confidence, ConfidenceResult
from dataclasses import dataclass
from typing import List

# Same model as ingest.py — must be identical so query vectors and chunk vectors
# are in the same vector space and can be meaningfully compared
embedder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)


@dataclass
class RetrievalResult:
    chunks: List[dict]          # the actual text chunks + metadata
    confidence: ConfidenceResult  # the confidence assessment for this query


def retrieve(query: str, collection_name: str = 'docs', top_k: int = 4) -> RetrievalResult:
    """
    Full retrieval pipeline for a single user question:
    1. Embed the question using the same model used during ingestion
    2. Search Qdrant for the top_k most similar chunks
    3. Run confidence assessment on the similarity scores
    4. Return chunks + confidence together
    """

    # Step 1: Convert the user's question into a vector
    # This MUST use the same model as ingestion — if you embed with a different
    # model, the vectors won't be comparable and search results will be garbage
    query_vector = embedder.encode(query).tolist()

    # Step 2: Search Qdrant
    # with_payload=True  → return the stored text and metadata alongside the score
    # with_vectors=False → don't return the raw vector (we don't need it back)
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    results = response.points

    # Step 3: Parse results into a clean list of dicts
    chunks = [{
        'text': r.payload['text'],
        'source': r.payload['source'],
        'page': r.payload.get('page', 0),
        'score': r.score          # cosine similarity score (0.0 to 1.0)
    } for r in results]

    # Step 4: Run confidence assessment using just the scores
    similarity_scores = [r.score for r in results]
    confidence = assess_confidence(similarity_scores)

    return RetrievalResult(chunks=chunks, confidence=confidence)


if __name__ == '__main__':
    # Quick test: ask 3 questions and print what gets retrieved
    # One question the document should answer, one it should not
    test_questions = [
        'What is a spatiotemporal graph convolutional network?',
        'How does the model trace microplastic pollution sources?',
        'What is the recipe for chocolate cake?',  # should trigger knowledge gap
    ]

    for q in test_questions:
        print(f'\nQuestion: {q}')
        result = retrieve(q)
        print(f'Confidence: {result.confidence.level} (score: {result.confidence.score:.3f})')
        if result.confidence.is_knowledge_gap:
            print(f'Knowledge gap: {result.confidence.gap_reason}')
        else:
            print(f'Top chunk (score {result.chunks[0]["score"]:.3f}):')
            print(result.chunks[0]['text'][:300])
        print('-' * 60)
