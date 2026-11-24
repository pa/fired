"""Does auto-training on Jev's confident answers make the brain need Jev less?

For each stream, half the messages arrive live (shuffled). The brain handles what
it's sure of; the rest go to Jev. Jev answers at >= TEACH_MIN confidence become
training phrases, and the brain retrains every RETRAIN of them. The other half
is never trained on and scores the brain before and after: how often it can act
alone (no Jev call) and how often those actions are wrong. Averaged over shuffles.
Jev answers are cached in data/jev_cache.json so each message costs one call.

    uv run simulate_autotrain.py
"""
import json
import random
import urllib.request

import numpy as np
import yaml

from brain import HERE, THRESHOLD, build, load, normalise

JEV = "https://api.sprites.dev/v1/gateway/typesafe/SnoyU9blsmeLNMQKolqYqg/v1/systemone"
INTENTS = yaml.safe_load((HERE / "intents.yaml").read_text())
TOKENS = []
TEACH_MIN, RETRAIN, SHUFFLES = 0.9, 10, 5


def jev(text, options):
    body = {"model": "jev-latest", "state": text, "questions": {"intent": {
        "type": "choice",
        "instructions": "Which action is this Slack message asking a GitHub repo bot to do?",
        "criteria": {k: INTENTS[k]["description"] for k in options}}}}
    req = urllib.request.Request(JEV, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=60))
    TOKENS.append(out["usage"]["input_tokens"])
    a = out["answers"]["intent"]
    return a["choice"], a["confidence"]
CACHE = HERE / "data" / "jev_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def ask_jev(text):
    if text not in cache:
        cache[text] = jev(text, list(INTENTS))
    return cache[text]


def held_out(model, X, y):
    p = model.predict_proba(X)
    pred, act = model.classes_[p.argmax(1)], p.max(1) >= THRESHOLD
    return act.mean(), np.mean(act & (pred != np.array(y)))


Xtr, ytr = load("train")
base = build().fit(Xtr, ytr)

for name in ("test", "persona_terse", "persona_polite", "persona_hinglish"):
    rows = [json.loads(l) for l in open(HERE / "data" / f"{name}.jsonl")]
    before, after, taught, bad = [], [], [], []
    for seed in range(SHUFFLES):
        random.Random(seed).shuffle(rows)
        live, rest = rows[: len(rows) // 2], rows[len(rows) // 2:]
        Xe, ye = [normalise(r["text"]) for r in rest], [r["intent"] for r in rest]
        model, X, y, new, wrong_labels = base, list(Xtr), list(ytr), 0, 0
        for r in live:
            x = normalise(r["text"])
            if model.predict_proba([x])[0].max() >= THRESHOLD:
                continue
            choice, conf = ask_jev(r["text"])
            if conf >= TEACH_MIN:
                X.append(x); y.append(choice); new += 1
                wrong_labels += choice != r["intent"]
                if new % RETRAIN == 0:
                    model = build().fit(X, y)
        model = build().fit(X, y)
        before.append(held_out(base, Xe, ye)); after.append(held_out(model, Xe, ye))
        taught.append(new); bad.append(wrong_labels)
    b, a = np.mean(before, 0), np.mean(after, 0)
    print(f"\n{name}: {len(rows) // 2} live messages per shuffle, {np.mean(taught):.0f} auto-taught "
          f"({np.mean(bad):.1f} with a wrong label)")
    print(f"  held-out half  brain acts alone {b[0]:5.1%} -> {a[0]:5.1%}   "
          f"wrong actions {b[1]:4.1%} -> {a[1]:4.1%}   Jev needed {1 - b[0]:5.1%} -> {1 - a[0]:5.1%}")
    CACHE.write_text(json.dumps(cache))

print(f"\nnew jev calls: {len(TOKENS)}  cost: ${sum(TOKENS) * 0.042 / 1e6:.4f}")
