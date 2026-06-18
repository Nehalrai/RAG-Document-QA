from retriever import retrieve

questions = [
    'What is a spatiotemporal graph convolutional network?',
    'How does the model trace microplastic pollution sources?',
    'What dataset was used to train the model?',
    'What is the recipe for chocolate cake?',
]

for q in questions:
    result = retrieve(q)
    print(f'Q: {q}')
    print(f'Confidence: {result.confidence.level} (score: {result.confidence.score:.3f})')
    if result.confidence.is_knowledge_gap:
        print(f'Gap: {result.confidence.gap_reason}')
    else:
        top = result.chunks[0]
        print(f'Top chunk (score {top["score"]:.3f}): {top["text"][:200]}')
    print('-' * 60)
