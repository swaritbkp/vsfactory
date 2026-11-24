"""`text_only` probe (SPEC 11.2): TF-IDF (word 1-2 grams) + logistic regression on the question text
with its options. Detects answers predictable without looking at the image."""

from __future__ import annotations

from collections.abc import Sequence


def fit_predict(
    text_train: Sequence[str], y_train: Sequence[str], text_test: Sequence[str], seed: int
) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    classes = sorted(set(y_train))
    if len(classes) == 1:
        return [classes[0]] * len(text_test)
    vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", lowercase=True)
    xtr = vec.fit_transform(list(text_train))
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed % (2**31))
    clf.fit(xtr, list(y_train))
    return [str(v) for v in clf.predict(vec.transform(list(text_test)))]


__all__ = ["fit_predict"]
