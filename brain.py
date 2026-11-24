"""Fired's brain: TF-IDF + logistic regression. No LLM.

    uv run brain.py train          # data/train.jsonl -> model.pkl
    uv run brain.py eval           # score on data/test.jsonl (never trained on)
    uv run brain.py ask "text"     # intent, confidence, and why
"""
import json
import pathlib
import pickle
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

HERE = pathlib.Path(__file__).parent
MODEL = HERE / "model.pkl"
THRESHOLD = 0.5  # below this, ask instead of acting


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"#?\b\d+\b", " NUM ", text)  # issue numbers
    text = re.sub(r"@\w+", " USER ", text)
    return text


def load(split: str) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in open(HERE / "data" / f"{split}.jsonl")]
    return [normalise(r["text"]) for r in rows], [r["intent"] for r in rows]


def build():
    return make_pipeline(
        FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]),
        LogisticRegression(C=10, max_iter=3000),
    )


def train() -> None:
    X, y = load("train")
    model = build()
    t = time.perf_counter()
    model.fit(X, y)
    print(f"trained on {len(X)} phrases, {len(set(y))} intents in {time.perf_counter() - t:.2f}s")
    MODEL.write_bytes(pickle.dumps(model))


def predict(model, text: str) -> tuple[str, float, list[tuple[str, float]]]:
    """Intent, confidence, and the words that pushed hardest toward it."""
    x = normalise(text)
    probs = model.predict_proba([x])[0]
    i = int(np.argmax(probs))
    union, clf = model.steps[0][1], model.steps[1][1]
    words = union.transformer_list[0][1]
    vec = words.transform([x]).tocoo()
    coef = clf.coef_[i][: len(words.vocabulary_)]  # word features come first in the union
    names = words.get_feature_names_out()
    reasons = sorted(((names[j], float(v * coef[j])) for j, v in zip(vec.col, vec.data)),
                     key=lambda r: -r[1])
    return clf.classes_[i], float(probs[i]), [r for r in reasons if r[1] > 0][:3]


def evaluate() -> None:
    model = pickle.loads(MODEL.read_bytes())
    X, y = load("test")
    probs = model.predict_proba(X)
    pred = model.classes_[probs.argmax(1)]
    conf = probs.max(1)
    t = time.perf_counter()
    for x in X:
        model.predict_proba([x])
    ms = (time.perf_counter() - t) / len(X) * 1000
    print(f"test phrases: {len(X)}  accuracy: {np.mean(pred == np.array(y)):.1%}  "
          f"latency: {ms:.2f} ms/message")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        acted = conf >= th
        wrong = np.mean(acted & (pred != np.array(y)))
        print(f"  threshold {th:.1f}: acts on {acted.mean():.1%}, wrong actions {wrong:.1%}, "
              f"no wrong change {1 - wrong:.1%}")
    misses = [(t, p, c) for t, p, c in zip(y, pred, conf) if t != p]
    by_pair: dict[tuple[str, str], int] = {}
    for t, p, _ in misses:
        by_pair[(t, p)] = by_pair.get((t, p), 0) + 1
    print("most confused (true -> predicted):")
    for (t, p), n in sorted(by_pair.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {t} -> {p}: {n}")


def ask(text: str) -> None:
    model = pickle.loads(MODEL.read_bytes())
    t = time.perf_counter()
    intent, conf, reasons = predict(model, text)
    ms = (time.perf_counter() - t) * 1000
    action = "act" if conf >= THRESHOLD else "ask"
    why = ", ".join(f"'{w}' +{s:.2f}" for w, s in reasons)
    print(f"{intent}  conf={conf:.2f}  -> {action}  ({ms:.2f} ms)  because {why}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"train": train, "eval": evaluate}.get(cmd, lambda: ask(" ".join(sys.argv[2:])))()
