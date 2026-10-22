"""Fold-local representations, preserving case, punctuation, and line breaks."""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CANDIDATES = ("word_lr", "char_lr", "union_lr", "union_svm", "style_svm")


class StyleFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            n = max(len(text), 1)
            words = text.split()
            rows.append([
                np.log1p(len(text)), np.log1p(len(words)),
                len(text) / max(len(words), 1),
                sum(c.isupper() for c in text) / n,
                sum(c.isdigit() for c in text) / n,
                *[text.count(p) / n for p in [".", ",", ";", ":", "!", "?", "—", "-", "\n", "\"", "'", "(", "#", "*"]],
                len(re.findall(r"\b\w+['’]\w+\b", text)) / max(len(words), 1),
            ])
        return np.asarray(rows, dtype=np.float64)


def build_model(name, seed=42):
    if name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {name}")
    word = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, strip_accents=None,
                          sublinear_tf=True, min_df=2, max_features=100_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                          sublinear_tf=True, min_df=2, max_features=150_000)
    if name == "word_lr":
        features = word
    elif name == "char_lr":
        features = char
    else:
        branches = [("word", word), ("char", char)]
        if name == "style_svm":
            branches.append(("style", Pipeline([("extract", StyleFeatures()), ("scale", StandardScaler())])))
        features = FeatureUnion(branches, n_jobs=1)
    classifier = (LogisticRegression(C=4.0, class_weight="balanced", max_iter=2500, random_state=seed)
                  if name.endswith("lr") else
                  LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=10000, random_state=seed))
    return Pipeline([("features", features), ("classifier", classifier)])
