# ============================================
# GENERATION METRICS ONLY
# ============================================

import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import flesch_reading_ease

nltk.download('punkt')

# -------------------------------------------------
# 1️⃣ BLEU Score
# -------------------------------------------------
def compute_bleu(predictions, references):
    chencherry = SmoothingFunction()
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = nltk.word_tokenize(ref)
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=chencherry.method1)
        bleu_scores.append(bleu)

    return np.mean(bleu_scores)


# -------------------------------------------------
# 2️⃣ Factual Consistency
# -------------------------------------------------
def factual_consistency(predictions, references):
    scores = []

    for pred, ref in zip(predictions, references):
        pred_words = set(nltk.word_tokenize(pred.lower()))
        ref_words = set(nltk.word_tokenize(ref.lower()))

        overlap = len(pred_words & ref_words) / len(pred_words) if len(pred_words) > 0 else 0
        scores.append(overlap)

    return np.mean(scores)


# -------------------------------------------------
# 3️⃣ Fluency & Readability
# -------------------------------------------------
def fluency_readability(predictions):
    readability_scores = [flesch_reading_ease(pred) for pred in predictions]

    sentence_counts = [len(nltk.sent_tokenize(pred)) for pred in predictions]
    fluency_scores = [
        len(nltk.word_tokenize(pred)) / max(s, 1)
        for pred, s in zip(predictions, sentence_counts)
    ]

    return {
        "Average Readability (Flesch)": np.mean(readability_scores),
        "Average Fluency (words per sentence)": np.mean(fluency_scores)
    }


# -------------------------------------------------
# 4️⃣ Diversity & Novelty
# -------------------------------------------------
def diversity_novelty(predictions):
    all_unigrams = []
    all_bigrams = []

    for pred in predictions:
        tokens = nltk.word_tokenize(pred.lower())
        all_unigrams.extend(tokens)
        all_bigrams.extend(list(nltk.bigrams(tokens)))

    distinct_unigrams = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
    distinct_bigrams = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    # Novelty
    seen_words = set()
    novelty_scores = []

    for pred in predictions:
        tokens = set(nltk.word_tokenize(pred.lower()))
        novel_count = len(tokens - seen_words)
        novelty_scores.append(novel_count / len(tokens) if len(tokens) else 0)
        seen_words.update(tokens)

    return {
        "Distinct-Unigram": distinct_unigrams,
        "Distinct-Bigram": distinct_bigrams,
        "Novelty": np.mean(novelty_scores)
    }


# ============================================
# RUN TEST
# ============================================

predictions = [
    "The cat sits on the mat.",
    "RAG systems combine retrieval and generation effectively."
]

references = [
    "A cat is sitting on the mat.",
    "Retrieval-Augmented Generation combines retrieval and generation."
]

print("==== GENERATION METRICS ====\n")

# BLEU
bleu = compute_bleu(predictions, references)
print("BLEU Score:", bleu)

# Factual Consistency
fact = factual_consistency(predictions, references)
print("Factual Consistency:", fact)

# Fluency & Readability
fluency = fluency_readability(predictions)
print("Fluency & Readability:", fluency)

# Diversity & Novelty
diversity = diversity_novelty(predictions)
print("Diversity & Novelty:", diversity)
