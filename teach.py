"""Teach step: the only time Fired talks to an LLM.

For every intent in intents.yaml, Claude writes training phrases and, in a
separate call that never sees them, test phrases. Writes data/train.jsonl,
data/test.jsonl and data/teach.json (how many LLM calls the teach step made).

Run on the Sprite, where the Anthropic connector's gateway holds the key:
    uv run teach.py
"""
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import anthropic
import yaml

GATEWAY = os.environ.get(
    "FIRED_ANTHROPIC_GATEWAY",
    "https://api.sprites.dev/v1/gateway/anthropic/8Cw1OkCFZhPZvNQRLbfwlQ",
)
MODEL = "claude-opus-5-5"
N_TRAIN, N_TEST = 60, 20
DATA = pathlib.Path(__file__).parent / "data"

# The gateway adds the real key; the SDK just needs a non-empty value.
USAGE: list[tuple[int, int]] = []  # (input, output) tokens per call
PRICE_IN, PRICE_OUT = 4 / 1e6, 20 / 1e6  # Opus 5.5, $ per token

client = anthropic.Anthropic(base_url=GATEWAY, api_key="via-sprites-gateway")

SCHEMA = {
    "type": "object",
    "properties": {"phrases": {"type": "array", "items": {"type": "string"}}},
    "required": ["phrases"],
    "additionalProperties": False,
}

STYLE = {
    "train": "Vary length, tone and wording. Include casual chat, terse commands, "
             "polite requests, a few typos and lowercase-only messages.",
    "test": "Write them the way a different person would: avoid the wording of the "
            "examples, use unusual verbs and sentence shapes, some indirect requests, "
            "some slang. These are a held-out exam, so do not copy the examples.",
}


def ask(name: str, spec: dict, intents: dict, split: str, n: int) -> list[str]:
    others = "\n".join(f"- {k}: {v['description']}" for k, v in intents.items() if k != name)
    prompt = (
        f"You are writing training data for a tiny text classifier inside a Slack bot "
        f"that manages one GitHub repo.\n\nWrite {n} distinct Slack messages a developer "
        f"might send that mean exactly this intent:\n\n{name}: {spec['description']}\n"
        f"Examples: {spec['examples']}\n\n{STYLE[split]}\n"
        f"Use realistic issue numbers (#1 to #40), @usernames and labels "
        f"(bug, enhancement, question, p0, stale) where they fit.\n\n"
        f"Every message must clearly NOT mean any of these other intents:\n{others}"
    )
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    USAGE.append((resp.usage.input_tokens, resp.usage.output_tokens))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{name}/{split}: refused")
    text = next(b.text for b in resp.content if b.type == "text")
    phrases = [p.strip() for p in json.loads(text)["phrases"] if p.strip()]
    print(f"{name:14} {split:5} {len(phrases)}")
    return phrases


def contrast(name: str, rival: str, intents: dict, n: int = 30) -> list[str]:
    """Extra training phrases for `name` that sit close to `rival` but aren't it."""
    a, b = intents[name], intents[rival]
    prompt = (
        f"A tiny text classifier in a GitHub Slack bot confuses two intents:\n\n"
        f"A = {name}: {a['description']}\nB = {rival}: {b['description']}\n\n"
        f"Write {n} distinct Slack messages that clearly mean A, worded so they share "
        f"vocabulary or shape with B, so the classifier learns the difference. "
        f"Mix casual, terse, polite and typo-ridden styles. Use issue numbers like #1 to #40 "
        f"and @usernames where they fit."
    )
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    USAGE.append((resp.usage.input_tokens, resp.usage.output_tokens))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{name} vs {rival}: refused")
    text = next(b.text for b in resp.content if b.type == "text")
    phrases = [p.strip() for p in json.loads(text)["phrases"] if p.strip()]
    print(f"{name:14} vs {rival:14} {len(phrases)}")
    return phrases


# Pairs the first round confused most often (true intent, what it was mistaken for).
CONFUSED = [("reopen_issue", "close_issue"), ("list_issues", "other"),
            ("list_issues", "whats_on_fire"), ("close_issue", "show_issue"),
            ("label_issue", "show_issue"), ("close_issue", "comment_issue"),
            ("close_issue", "help"), ("help", "brain_status")]


def focus():
    """Second teach round: append contrastive phrases to train.jsonl. Test set untouched."""
    intents = yaml.safe_load((pathlib.Path(__file__).parent / "intents.yaml").read_text())
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda p: (p[0], contrast(p[0], p[1], intents)), CONFUSED))
    with open(DATA / "train.jsonl", "a") as f:
        f.writelines(json.dumps({"text": t, "intent": name}) + "\n" for name, ps in results for t in ps)
    log = json.loads((DATA / "teach.json").read_text())
    tin, tout = sum(u[0] for u in USAGE), sum(u[1] for u in USAGE)
    cost = tin * PRICE_IN + tout * PRICE_OUT
    print(f"added {sum(len(ps) for _, ps in results)} phrases; llm calls: {len(USAGE)}  cost: ${cost:.2f}")
    log.update(llm_calls=log["llm_calls"] + len(USAGE), tokens_in=log["tokens_in"] + tin,
               tokens_out=log["tokens_out"] + tout, cost_usd=round(log["cost_usd"] + cost, 4))
    (DATA / "teach.json").write_text(json.dumps(log))


PERSONAS = {
    "terse": "a terse senior dev who types in lowercase with no punctuation, uses slang "
             "like 'nuke', 'yeet', 'kill', 'ship it', and abbreviations like 'pls', 'rn', 'iss'",
    "polite": "a very polite, verbose British engineer who writes full sentences, says "
              "'would you mind', 'kindly', 'cheers', and sometimes explains why",
    "hinglish": "a casual Indian developer who mixes Indian English and Hinglish, e.g. "
                "'pls do the needful', 'kar do', 'yaar', 'na', 'only', 'itself'",
}


def persona(who: str, name: str, spec: dict, n: int = 12) -> list[str]:
    prompt = (
        f"Write {n} distinct Slack messages to a GitHub repo bot, all from the same person: "
        f"{PERSONAS[who]}. Keep their habits consistent; they reuse their favourite words.\n\n"
        f"Every message must mean this intent: {name}: {spec['description']}\n"
        f"Use issue numbers like #1 to #40, @usernames and labels where they fit."
    )
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    USAGE.append((resp.usage.input_tokens, resp.usage.output_tokens))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{who}/{name}: refused")
    text = next(b.text for b in resp.content if b.type == "text")
    return [p.strip() for p in json.loads(text)["phrases"] if p.strip()]


def personas():
    """Evaluation streams from consistent individuals. Never used for training."""
    intents = yaml.safe_load((pathlib.Path(__file__).parent / "intents.yaml").read_text())
    jobs = [(who, name, spec) for who in PERSONAS for name, spec in intents.items()]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: (j[0], j[1], persona(*j)), jobs))
    for who in PERSONAS:
        rows = [{"text": t, "intent": name} for w, name, ps in results if w == who for t in ps]
        with open(DATA / f"persona_{who}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(who, len(rows))
    tin, tout = sum(u[0] for u in USAGE), sum(u[1] for u in USAGE)
    print(f"llm calls: {len(USAGE)}  cost: ${tin * PRICE_IN + tout * PRICE_OUT:.2f} (evaluation only)")


def main():
    intents = yaml.safe_load((pathlib.Path(__file__).parent / "intents.yaml").read_text())
    jobs = [(name, spec, split, n) for name, spec in intents.items()
            for split, n in (("train", N_TRAIN), ("test", N_TEST))]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: (j[0], j[2], ask(j[0], j[1], intents, j[2], j[3])), jobs))

    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        rows = [{"text": p, "intent": name} for name, s, ps in results if s == split for p in ps]
        if split == "train":  # the hand-written examples are training data too
            rows += [{"text": e, "intent": k} for k, v in intents.items() for e in v["examples"]]
        with open(DATA / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(split, len(rows))
    tin, tout = sum(u[0] for u in USAGE), sum(u[1] for u in USAGE)
    cost = tin * PRICE_IN + tout * PRICE_OUT
    print(f"llm calls: {len(USAGE)}  tokens in/out: {tin}/{tout}  cost: ${cost:.2f}")
    (DATA / "teach.json").write_text(json.dumps(
        {"llm_calls": len(USAGE), "model": MODEL, "tokens_in": tin, "tokens_out": tout,
         "cost_usd": round(cost, 4)}))


if __name__ == "__main__":
    import sys
    {"focus": focus, "personas": personas}.get(sys.argv[1] if sys.argv[1:] else "", main)()
