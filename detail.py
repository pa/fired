"""Fired's second model: which detail of an issue is a question about?

Once the brain picks show_issue, this decides whether the message asks about the
assignee, labels, milestone, open/closed state, or nothing in particular ("none":
just show it). Same recipe as the brain: TF-IDF + logistic regression, trained on
phrases Claude Code subagents wrote, scored on phrases other subagents wrote blind.

    uv run detail.py train     # gen/detail_train.jsonl -> detail.pkl
    uv run detail.py eval      # held-out gen/detail_test.jsonl, vs the keyword rules it replaced
"""
import json
import pickle
import re
import sys

import numpy as np

from brain import HERE, THRESHOLD, build, normalise

MODEL = HERE / "detail.pkl"


def rows(name):
    return [json.loads(l) for l in open(HERE / "gen" / f"detail_{name}.jsonl") if l.strip()]


def keyword_detail(text):
    """The keyword rules this model replaced, kept here only for comparison."""
    t = text.lower()
    if re.search(r"assign|who('s| is)? (on|working|handling|owns)|\bowner", t):
        return "assignee"
    if re.search(r"\blabel|\btag", t):
        return "labels"
    if re.search(r"milestone|release", t):
        return "milestone"
    if re.search(r"\b(still )?(open|closed)\b\?|\bstatus of\b|\bis .* (open|closed|fixed|done)\b", t):
        return "state"
    return "none"


def train():
    test = {r["text"].lower() for r in rows("test")}
    data = [r for r in rows("train") if r["text"].lower() not in test]  # never train on the exam
    model = build().fit([normalise(r["text"]) for r in data], [r["detail"] for r in data])
    MODEL.write_bytes(pickle.dumps(model))
    print(f"trained on {len(data)} phrases, classes {sorted(set(r['detail'] for r in data))}")


def evaluate():
    model = pickle.loads(MODEL.read_bytes())
    test = rows("test")
    y = np.array([r["detail"] for r in test])
    p = model.predict_proba([normalise(r["text"]) for r in test])
    pred, conf = model.classes_[p.argmax(1)], p.max(1)
    # An "answer" is leading the reply with one specific detail. Below the threshold, or
    # for "none", Fired just shows the whole issue, which always contains the right detail.
    m_ans = (pred != "none") & (conf >= THRESHOLD)
    kw = np.array([keyword_detail(r["text"]) for r in test])
    k_ans = kw != "none"
    out: dict = {"messages": len(test)}
    print(f"held-out detail questions: {len(test)}")
    for label, guess, ans in (("model", np.where(m_ans, pred, "none"), m_ans), ("keyword rules", kw, k_ans)):
        right = float(np.mean(guess == y))
        wrong = float(np.mean(ans & (guess != y)))
        print(f"  {label:13} right {right:5.1%}   wrong answer given {wrong:5.1%}")
        out[label] = {"right": round(right, 3), "wrong": round(wrong, 3)}
    misses = [(r["text"], t, g) for r, t, g, a in zip(test, y, pred, m_ans) if a and t != g]
    for text, t, g in misses[:8]:
        print(f"  wrong: {text!r}  true {t}, said {g}")
    (HERE / "data" / "bench_detail.json").write_text(json.dumps(out))


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
