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
