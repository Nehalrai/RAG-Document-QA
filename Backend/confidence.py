from dataclasses import dataclass
from typing import List

# These thresholds control how the system judges answer trustworthiness.
# You can tune these later based on your evaluation results.
HIGH_CONFIDENCE_THRESHOLD = 0.55
MEDIUM_CONFIDENCE_THRESHOLD = 0.35
KNOWLEDGE_GAP_THRESHOLD = 0.20


@dataclass
class ConfidenceResult:
    level: str            # 'high', 'medium', or 'low'
    score: float          # combined score between 0.0 and 1.0
    is_knowledge_gap: bool
    gap_reason: str       # human-readable explanation shown to the user


def assess_confidence(similarity_scores: List[float]) -> ConfidenceResult:
    """
    Takes the cosine similarity scores of the top retrieved chunks and decides:
    1. Is this a knowledge gap? (document doesn't cover the topic at all)
    2. If not, how confident should we be? (high / medium / low)

    Cosine similarity scores from Qdrant range from 0.0 to 1.0:
      1.0 = the chunk is essentially asking/saying the same thing as the query
      0.5 = loosely related
      0.0 = completely unrelated
    """

    # Edge case: Qdrant returned nothing at all
    if not similarity_scores:
        return ConfidenceResult(
            level='low',
            score=0.0,
            is_knowledge_gap=True,
            gap_reason='No relevant chunks found in the document.'
        )

    best_score = max(similarity_scores)
    avg_score = sum(similarity_scores) / len(similarity_scores)

    # Knowledge gap check: if even the BEST matching chunk scores below 0.40,
    # the document almost certainly doesn't cover this topic.
    # We flag it BEFORE calling the LLM to avoid generating a hallucinated answer.
    if best_score < KNOWLEDGE_GAP_THRESHOLD:
        return ConfidenceResult(
            level='low',
            score=best_score,
            is_knowledge_gap=True,
            gap_reason=(
                f'The document does not appear to cover this topic. '
                f'Best match score was {best_score:.2f} '
                f'(threshold: {KNOWLEDGE_GAP_THRESHOLD}). '
                f'The answer below may not be reliable.'
            )
        )

    # Confidence scoring: weighted combination of best and average score.
    # We weight best_score more (0.7) because one highly relevant chunk
    # is often enough to answer a question well.
    combined = (best_score * 0.7) + (avg_score * 0.3)

    if combined >= HIGH_CONFIDENCE_THRESHOLD:
        level = 'high'
    elif combined >= MEDIUM_CONFIDENCE_THRESHOLD:
        level = 'medium'
    else:
        level = 'low'

    return ConfidenceResult(
        level=level,
        score=combined,
        is_knowledge_gap=False,
        gap_reason=''
    )
