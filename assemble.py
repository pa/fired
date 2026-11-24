"""Combine the phrases Claude Code wrote (gen/) into data/ for training and evaluation.

Training = gen/train_*.jsonl + gen/contrast.jsonl + the examples in intents.yaml.
Test = gen/test_*.jsonl (written by separate agents that never saw the training files).
Personas = gen/persona_*.jsonl (evaluation only). Drops duplicates, and drops any
training phrase that also appears in the test set or a persona stream.

    uv run assemble.py
"""
import json
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
GEN, DATA = HERE / "gen", HERE / "data"
INTENTS = yaml.safe_load((HERE / "intents.yaml").read_text())


def rows(pattern):
    out, seen = [], set()
    for f in sorted(GEN.glob(pattern)):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = r["text"].strip().lower()
                if r["intent"] in INTENTS and key not in seen:
                    seen.add(key); out.append({"text": r["text"].strip(), "intent": r["intent"]})
    return out


def write(name, data):
    (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))


DATA.mkdir(exist_ok=True)
test = rows("test_*.jsonl")
personas = {w: rows(f"persona_{w}.jsonl") for w in ("terse", "polite", "hinglish")}
held = {r["text"].lower() for r in test} | {r["text"].lower() for p in personas.values() for r in p}
train = rows("train_*.jsonl") + rows("contrast.jsonl")
# Questions about one issue's details (who's on it, labels, milestone, open?) are show_issue to
# the brain. Take 25 per detail class from the detail model's training phrases, so show_issue
# learns those phrasings without swamping the other intents.
detail_rows = [json.loads(l) for l in (GEN / "detail_train.jsonl").read_text().splitlines() if l.strip()]
per_class: dict[str, int] = {}
for r in detail_rows:
    if per_class.get(r["detail"], 0) < 25:
        per_class[r["detail"]] = per_class.get(r["detail"], 0) + 1
        train.append({"text": r["text"].strip(), "intent": "show_issue"})
train += [{"text": e, "intent": k} for k, v in INTENTS.items() for e in v["examples"]]
before = len(train)
seen, train = set(), [r for r in train if r["text"].lower() not in held]
train = [r for r in train if not (r["text"].lower() in seen or seen.add(r["text"].lower()))]
write("train", train); write("test", test)
for w, p in personas.items():
    write(f"persona_{w}", p)
(DATA / "teach.json").write_text(json.dumps({
    "llm_calls": 0, "cost_usd": 0.0,
    "note": f"{len(train)} phrases written once by Claude Code subagents on a subscription; "
            "no LLM API is reachable from this Sprite"}))
print(f"train {len(train)} (dropped {before - len(train)} duplicates or held-out overlaps), "
      f"test {len(test)}, personas " + ", ".join(f"{w} {len(p)}" for w, p in personas.items()))
missing = [k for k in INTENTS if not any(r["intent"] == k for r in test)]
print("intents without test phrases:", missing or "none")
