"""Personal memory test: does remembering corrections cut asks without new mistakes?

The model stays frozen. Every correction (an ask, or a wrong action the user
undoes) is stored verbatim. A new message whose TF-IDF cosine similarity to a
stored correction is >= SIM takes that correction's intent and is acted on.
No LLM involved.

    uv run simulate_memory.py
"""
import json
import random

import numpy as np

from brain import HERE, THRESHOLD, build, load, normalise

Xtr, ytr = load("train")
model = build().fit(Xtr, ytr)
features = model.steps[0][1]  # the word + char TF-IDF union


def vec(text):
    v = features.transform([text]).toarray()[0]
    return v / (np.linalg.norm(v) or 1)


def run(stream, sim):
    mem_v, mem_y, log = [], [], []
    for text, truth in stream:
        v = vec(text)
        hit = None
        if sim and mem_v:
            sims = np.array(mem_v) @ v
            if sims.max() >= sim:
                hit = mem_y[int(sims.argmax())]
        if hit:
            guess, asked = hit, False
        else:
            p = model.predict_proba([text])[0]
            guess, asked = model.classes_[p.argmax()], p.max() < THRESHOLD
        wrong = not asked and guess != truth
        log.append((asked, wrong, hit is not None))
        if asked or wrong:  # the user's answer is remembered
            mem_v.append(v); mem_y.append(truth)
    return log


def rates(log):
    n = len(log)
    return (sum(a for a, _, _ in log) / n, sum(w for _, w, _ in log) / n,
            sum(h for _, _, h in log) / n)


for who in ("terse", "polite", "hinglish"):
    rows = [json.loads(l) for l in open(HERE / "data" / f"persona_{who}.jsonl")]
    stream = [(normalise(r["text"]), r["intent"]) for r in rows]
    print(f"\n{who}: {len(stream)} messages, averaged over 5 shuffles")
    for label, sim in (("frozen", 0), ("mem 0.5", 0.5), ("mem 0.6", 0.6), ("mem 0.7", 0.7)):
        firsts, lasts, totals = [], [], []
        for seed in range(5):
            s = stream[:]
            random.Random(seed).shuffle(s)
            log = run(s, sim)
            firsts.append(rates(log[:50])); lasts.append(rates(log[-50:])); totals.append(rates(log))
        f, l, t = (np.mean(x, axis=0) for x in (firsts, lasts, totals))
        print(f"  {label:8} asks {f[0]:5.1%} -> {l[0]:5.1%}   wrong {f[1]:4.1%} -> {l[1]:4.1%}   "
              f"memory answered {l[2]:5.1%} of last 50   whole stream: asks {t[0]:5.1%} wrong {t[1]:4.1%}")
