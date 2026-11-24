"""Does Fired ask one person less as it learns their wording?

Replays each persona's messages (shuffled) through three agents:
  frozen - never learns (baseline: is the stream just getting easier?)
  w=1    - each correction is one more training phrase
  w=5    - each correction counts 5x a synthetic phrase
A correction happens when the agent asks (conf < threshold) or acts wrongly
and the user undoes it. Reports ask and wrong-action rates, first 50 vs last 50.

    uv run simulate_persona.py
"""
import json
import random

from brain import HERE, THRESHOLD, build, load, normalise

RETRAIN = 5
Xtr, ytr = load("train")
base = build().fit(Xtr, ytr)


def run(stream, weight):
    model = base
    X, y, w = list(Xtr), list(ytr), [1.0] * len(Xtr)
    log, pending = [], 0
    for text, truth in stream:
        p = model.predict_proba([text])[0]
        guess, conf = model.classes_[p.argmax()], p.max()
        asked, wrong = conf < THRESHOLD, conf >= THRESHOLD and guess != truth
        log.append((asked, wrong))
        if weight and (asked or wrong):
            X.append(text); y.append(truth); w.append(weight); pending += 1
            if pending == RETRAIN:
                model = build().fit(X, y, logisticregression__sample_weight=w)
                pending = 0
    return log


def rates(log):
    n = len(log)
    return sum(a for a, _ in log) / n, sum(b for _, b in log) / n


for who in ("terse", "polite", "hinglish"):
    rows = [json.loads(l) for l in open(HERE / "data" / f"persona_{who}.jsonl")]
    stream = [(normalise(r["text"]), r["intent"]) for r in rows]
    random.Random(0).shuffle(stream)
    print(f"\n{who}: {len(stream)} messages")
    for label, weight in (("frozen", 0), ("w=1", 1.0), ("w=5", 5.0)):
        log = run(stream, weight)
        a1, w1 = rates(log[:50])
        a2, w2 = rates(log[-50:])
        print(f"  {label:6}  asks {a1:5.1%} -> {a2:5.1%}   wrong actions {w1:5.1%} -> {w2:5.1%}")
