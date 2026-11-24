"""Does Jev settle the messages the classifier is unsure about?

Every message the classifier would ask about (conf < THRESHOLD) goes to Jev
as a choice question, either among the classifier's top 3 intents or among
all intents. Reports, for those messages only, how often Jev is right and,
at each Jev confidence cut-off, how many it settles and how many it gets wrong.
Uses the held-out test set and the persona streams; nothing here is trained on.

    uv run simulate_jev.py
"""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from brain import HERE, THRESHOLD, build, load, normalise

JEV = os.environ.get("FIRED_JEV_GATEWAY",
                     "https://api.sprites.dev/v1/gateway/typesafe/SnoyU9blsmeLNMQKolqYqg")
INTENTS = yaml.safe_load((HERE / "intents.yaml").read_text())
TOKENS = []


def jev(text: str, options: list[str]) -> tuple[str, float]:
    body = {"model": "jev-latest", "state": text, "questions": {"intent": {
        "type": "choice",
        "instructions": "Which action is this Slack message asking a GitHub repo bot to do?",
        "criteria": {k: INTENTS[k]["description"] for k in options}}}}
    req = urllib.request.Request(f"{JEV}/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=60))
    TOKENS.append(out["usage"]["input_tokens"])
    a = out["answers"]["intent"]
    return a["choice"], a["confidence"]


Xtr, ytr = load("train")
model = build().fit(Xtr, ytr)

streams = {"test set": [(json.loads(l)["text"], json.loads(l)["intent"])
                        for l in open(HERE / "data" / "test.jsonl")]}
for who in ("terse", "polite", "hinglish"):
    streams[who] = [(json.loads(l)["text"], json.loads(l)["intent"])
                    for l in open(HERE / "data" / f"persona_{who}.jsonl")]

for name, rows in streams.items():
    probs = model.predict_proba([normalise(t) for t, _ in rows])
    unsure = [(t, y, list(model.classes_[np.argsort(-p)[:3]]))
              for (t, y), p in zip(rows, probs) if p.max() < THRESHOLD]
    in_top3 = np.mean([y in top for _, y, top in unsure])
    print(f"\n{name}: {len(rows)} messages, classifier unsure on {len(unsure)} "
          f"({len(unsure) / len(rows):.0%}); true intent in its top 3: {in_top3:.0%}")
    for label, opts in (("top 3", lambda top: top), ("all 16", lambda top: list(INTENTS))):
        with ThreadPoolExecutor(8) as pool:
            res = list(pool.map(lambda u: jev(u[0], opts(u[2])), unsure))
        right = np.array([c == y for (c, _), (_, y, _) in zip(res, unsure)])
        conf = np.array([c for _, c in res])
        print(f"  jev {label:6} right {right.mean():.0%}", end="")
        for cut in (0.5, 0.7, 0.9):
            acts = conf >= cut
            print(f" | conf>={cut}: settles {acts.mean():.0%}, wrong {(acts & ~right).mean():.0%}", end="")
        print()

print(f"\njev calls: {len(TOKENS)}  input tokens: {sum(TOKENS)}  "
      f"cost at $0.042/M: ${sum(TOKENS) * 0.042 / 1e6:.4f}")
