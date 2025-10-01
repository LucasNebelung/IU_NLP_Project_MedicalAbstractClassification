"""Implementation of a simple multinomial Naive Bayes classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class MultinomialNaiveBayes:
    """A lightweight multinomial Naive Bayes classifier."""

    alpha: float = 1.0
    class_priors: Dict[int, float] = field(default_factory=dict)
    feature_log_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)
    default_log_prob: Dict[int, float] = field(default_factory=dict)
    vocabulary_size: int = 0

    def fit(self, documents: Sequence[Dict[int, int]], labels: Sequence[int]) -> None:
        class_counts: Dict[int, int] = {}
        token_totals: Dict[int, int] = {}
        token_counts: Dict[int, Dict[int, int]] = {}

        for doc, label in zip(documents, labels):
            class_counts[label] = class_counts.get(label, 0) + 1
            token_totals[label] = token_totals.get(label, 0) + sum(doc.values())
            class_token_counts = token_counts.setdefault(label, {})
            for index, count in doc.items():
                class_token_counts[index] = class_token_counts.get(index, 0) + count

        total_documents = len(labels)
        self.class_priors = {
            label: math.log(count / total_documents)
            for label, count in class_counts.items()
        }

        vocabulary_indices = set()
        for label_counts in token_counts.values():
            vocabulary_indices.update(label_counts.keys())
        self.vocabulary_size = len(vocabulary_indices)

        self.feature_log_probs = {}
        denominator_offset = self.alpha * self.vocabulary_size
        for label, counts in token_counts.items():
            total = token_totals.get(label, 0) + denominator_offset
            self.default_log_prob[label] = math.log((self.alpha) / total) if total > 0 else float("-inf")
            log_probs: Dict[int, float] = {}
            for index in vocabulary_indices:
                count = counts.get(index, 0)
                log_prob = math.log((count + self.alpha) / total) if total > 0 else float("-inf")
                log_probs[index] = log_prob
            self.feature_log_probs[label] = log_probs

    def predict(self, documents: Sequence[Dict[int, int]]) -> List[int]:
        predictions: List[int] = []
        for doc in documents:
            scores: Dict[int, float] = {}
            for label, prior in self.class_priors.items():
                score = prior
                feature_probs = self.feature_log_probs[label]
                default_log_prob = self.default_log_prob[label]
                for index, count in doc.items():
                    score += feature_probs.get(index, default_log_prob) * count
                scores[label] = score
            predicted_label = max(scores.items(), key=lambda item: item[1])[0]
            predictions.append(predicted_label)
        return predictions

