import os
import requests
from retriever import retrieve

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434') + '/api/generate'
MODEL = os.getenv('OLLAMA_MODEL', 'mistral')


def ask(question: str, collection_name: str = 'docs') -> dict:
    """
    Full query pipeline for a single user question:
    1. Retrieve relevant chunks + confidence score from Qdrant
    2. Build a prompt combining the retrieved context and the question
    3. Call the local Ollama LLM to generate an answer
    4. Return the answer, confidence info, and source citations
    """

    # Step 1: Retrieve — this also runs confidence assessment internally
    retrieval = retrieve(question, collection_name)
    confidence = retrieval.confidence

    # Step 2: Build context block from retrieved chunks
    # Each chunk is labeled with its source and page so the LLM can cite them
    context_parts = []
    for i, chunk in enumerate(retrieval.chunks, 1):
        context_parts.append(
            f'[Source {i}: {chunk["source"]}, page {chunk["page"]}]\n{chunk["text"]}'
        )
    context = '\n\n'.join(context_parts)

    # Step 3: Adjust the instruction based on whether this is a knowledge gap
    # If it's a gap, we warn the LLM explicitly so it doesn't hallucinate
    # If it's not a gap, we tell it to stick to the context and cite sources
    if confidence.is_knowledge_gap:
        instruction = (
            'The retrieved context may not contain enough information to answer this question. '
            'If the context does not address the question, say explicitly that the document '
            'does not cover this topic rather than guessing.'
        )
    else:
        instruction = 'Answer using ONLY the context below. Be concise. Cite which source number you used.'

    prompt = f'{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:'

    # Step 4: Call local Ollama — no API key, no internet, runs on your machine
    # stream=False means wait for the full response before returning
    response = requests.post(OLLAMA_URL, json={
        'model': MODEL,
        'prompt': prompt,
        'stream': False
    })
    response.raise_for_status()
    answer = response.json()['response']

    # Step 5: Return everything the frontend needs
    return {
        'answer': answer,
        'confidence': {
            'level': confidence.level,
            'score': round(confidence.score, 3),
            'is_knowledge_gap': confidence.is_knowledge_gap,
            'gap_reason': confidence.gap_reason
        },
        'sources': [{
            'file': c['source'],
            'page': c['page'],
            'score': round(c['score'], 3)
        } for c in retrieval.chunks]
    }


if __name__ == '__main__':
    questions = [
        'How does the spatiotemporal graph convolutional network identify microplastic migration pathways?',
        'What is the recipe for chocolate cake?',
    ]

    for q in questions:
        print(f'\nQuestion: {q}')
        print('Thinking...')
        result = ask(q)
        conf = result['confidence']
        print(f'Confidence: {conf["level"].upper()} (score: {conf["score"]})')
        if conf['is_knowledge_gap']:
            print(f'WARNING - Knowledge gap: {conf["gap_reason"]}')
        print(f'\nAnswer:\n{result["answer"]}')
        print(f'\nSources:')
        for s in result['sources']:
            print(f'  - {s["file"]} | page {s["page"]} | relevance {s["score"]}')
        print('=' * 70)
