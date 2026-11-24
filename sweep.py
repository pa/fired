"""Try classifier settings. Picks on half the test set (dev), reports the
other half (final) once, so the headline number isn't tuned on itself.

    uv run sweep.py
"""
import itertools

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline

from brain import load

Xtr, ytr = load("train")
Xte, yte = load("test")
Xdev, Xfin, ydev, yfin = train_test_split(Xte, yte, test_size=0.5, stratify=yte, random_state=0)


def build(word_ng, char_ng, C, balanced):
    parts = [("words", TfidfVectorizer(ngram_range=word_ng, sublinear_tf=True))]
    if char_ng:
        parts.append(("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng, sublinear_tf=True)))
    return make_pipeline(FeatureUnion(parts),
                         LogisticRegression(C=C, max_iter=3000,
                                            class_weight="balanced" if balanced else None))


def score(model, X, y, th=0.5):
    p = model.predict_proba(X)
    pred, conf = model.classes_[p.argmax(1)], p.max(1)
    y = np.array(y)
    return np.mean(pred == y), np.mean(conf >= th), np.mean((conf >= th) & (pred != y))


grid = itertools.product([(1, 1), (1, 2), (1, 3)], [None, (2, 4), (2, 5), (3, 5)],
                         [1, 3, 10, 30, 100], [False, True])
rows = []
for word_ng, char_ng, C, bal in grid:
    m = build(word_ng, char_ng, C, bal).fit(Xtr, ytr)
    acc, acts, wrong = score(m, Xdev, ydev)
    rows.append((acc, acts, wrong, word_ng, char_ng, C, bal))

rows.sort(key=lambda r: -r[0])
print("top 8 on dev (acc, acts@0.5, wrong@0.5, word, char, C, balanced):")
for r in rows[:8]:
    print(f"  {r[0]:.1%} {r[1]:.1%} {r[2]:.1%}  {r[3]} {r[4]} C={r[5]} bal={r[6]}")

cur = build((1, 2), (2, 5), 10, False).fit(Xtr, ytr)
best = build(*rows[0][3:]).fit(Xtr, ytr)
for name, m in (("current", cur), ("best", best)):
    acc, acts, wrong = score(m, Xfin, yfin)
    print(f"final half, {name:7}: accuracy {acc:.1%}, acts {acts:.1%}, wrong actions {wrong:.1%}")
