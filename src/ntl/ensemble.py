"""Stacking with inner prompt folds; outer evaluation labels never train a blender."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def aligned_scores(model, texts, labels):
    score = model.decision_function(texts) if hasattr(model, "decision_function") else model.predict_proba(texts)
    return score[:, [list(model.classes_).index(label) for label in labels]]


def fit_grouped(model, texts, labels, prompt_ids):
    if getattr(model, "needs_prompt_groups", False):
        return model.fit(texts, labels, prompt_ids=prompt_ids)
    return model.fit(texts, labels)


class PromptStackClassifier(ClassifierMixin, BaseEstimator):
    """Fixed base families and meta regularization, with three inner group folds.

    Every meta-training row is predicted by base models that excluded its prompt.
    During outer evaluation the entire procedure is confined to outer training.
    At inference the base models and meta-classifier use only the passage.
    """
    needs_prompt_groups = True

    def __init__(self, base_names=("wide_svm", "punct_svm", "style_hgb"), inner_splits=3, meta_c=0.1, seed=42):
        self.base_names = base_names
        self.inner_splits = inner_splits
        self.meta_c = meta_c
        self.seed = seed

    def fit(self, X, y, prompt_ids=None):
        from ntl.models import build_model
        if prompt_ids is None:
            raise ValueError("Stacking requires prompt groups for inner validation")
        texts, y, group = np.asarray(X), np.asarray(y), np.asarray(prompt_ids)
        self.classes_ = np.unique(y)
        if len(set(group)) < self.inner_splits:
            raise ValueError("Too few prompt groups for stacking")
        self.inner_fold_ = np.full(len(y), -1, dtype=int)
        splitter = StratifiedGroupKFold(self.inner_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(y), len(self.classes_) * len(self.base_names)))
        for fold, (tr, va) in enumerate(splitter.split(texts, y, group)):
            if set(y[tr]) != set(self.classes_) or set(y[va]) != set(self.classes_):
                raise ValueError("Every inner fold must contain every label")
            if set(group[tr]) & set(group[va]):
                raise ValueError("Inner prompt leakage")
            self.inner_fold_[va] = fold
            for i, name in enumerate(self.base_names):
                model = build_model(name, self.seed)
                model.fit(texts[tr], y[tr])
                start = i * len(self.classes_)
                oof[va, start:start + len(self.classes_)] = aligned_scores(model, texts[va], self.classes_)
        self.meta_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.meta_c,
            class_weight="balanced", max_iter=2500, random_state=self.seed))
        self.meta_.fit(oof, y)
        self.models_ = [build_model(name, self.seed).fit(texts, y) for name in self.base_names]
        return self

    def decision_function(self, X):
        scores = np.hstack([aligned_scores(model, X, self.classes_) for model in self.models_])
        return self.meta_.decision_function(scores)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]
