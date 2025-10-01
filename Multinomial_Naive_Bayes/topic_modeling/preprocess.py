"""Preprocessing utilities for text normalization without external dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


_DEFAULT_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


_WORD_PATTERN = re.compile(r"[a-zA-Z']+")


def simple_lemmatize(token: str) -> str:
    """Apply lightweight lemmatization rules to a token."""

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 4:
        return token[:-3]
    if token.endswith("ed") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    if token.endswith("'s"):
        return token[:-2]
    return token


@dataclass
class TextPreprocessor:
    """Normalize text by lowercasing, removing stop words, and lemmatizing."""

    stop_words: Sequence[str] | None = None
    min_token_length: int = 2

    def __post_init__(self) -> None:
        self.stop_words = set(word.lower() for word in (self.stop_words or _DEFAULT_STOPWORDS))

    def __call__(self, text: str) -> List[str]:
        tokens = [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]
        normalized: List[str] = []
        for token in tokens:
            if token in self.stop_words:
                continue
            lemma = simple_lemmatize(token)
            if len(lemma) < self.min_token_length:
                continue
            normalized.append(lemma)
        return normalized


def build_vocabulary(documents: Iterable[List[str]]) -> List[str]:
    """Create a sorted vocabulary list from tokenized documents."""

    vocabulary = set()
    for tokens in documents:
        vocabulary.update(tokens)
    return sorted(vocabulary)


def transform_to_bow(documents: Iterable[List[str]], vocabulary: Sequence[str]) -> List[Dict[int, int]]:
    """Transform documents into bag-of-words using the provided vocabulary."""

    index_lookup = {token: idx for idx, token in enumerate(vocabulary)}
    transformed: List[Dict[int, int]] = []
    for tokens in documents:
        counts: Dict[int, int] = {}
        for token in tokens:
            if token not in index_lookup:
                continue
            index = index_lookup[token]
            counts[index] = counts.get(index, 0) + 1
        transformed.append(counts)
    return transformed

