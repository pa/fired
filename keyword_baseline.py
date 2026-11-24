"""The 'just write if/else' baseline: keyword rules vs the brain, same held-out messages.

Rules are written from intents.yaml (descriptions and examples), the way a developer
would without seeing the test phrases. First matching rule wins; no match = ask.

    uv run keyword_baseline.py
"""
import json
import re

import numpy as np

from brain import HERE, THRESHOLD, build, load, normalise

RULES = [  # (intent, regex) in priority order
    ("undo", r"\bundo\b|revert that|take (that|it) back|undo that"),
    ("rewind_brain", r"\brewind\b|\bforget\b|\bunlearn"),
    ("brain_status", r"\bbrain\b|\bstats\b|how are you doing|your accuracy"),
    ("help", r"\bhelp\b|what can you do|\bcommands\b"),
    ("close_stale", r"\bstale\b|\bold issues\b|inactive"),
    ("triage", r"\btriage|untriaged|unlabel+ed"),
    ("whats_on_fire", r"\bfire\b|\burgent\b|\bp0\b|burning|critical|\bblockers?\b"),
    ("reopen_issue", r"\bre-?open"),
    ("close_duplicate", r"\bdup(e|licate)?s?\b|same as #?\d+"),
    ("close_not_planned", r"won'?t ?fix|not planned|not (going to|gonna) do|not doing"),
    ("close_issue", r"\bclose\b|\bshut\b|\bresolve|\bdone with\b|mark .* (done|fixed)"),
    ("search_issues", r"\bsearch|anything about|any (issues?|tickets?) (about|on|mentioning)|mentioning"),
    ("create_issue", r"\bcreate\b|new issue|open an? issue|\bfile (an? )?(issue|bug)|\breport\b"),
    ("set_milestone", r"\bmilestone|\bv\d+(\.\d+)+|\brelease\b"),
    ("remove_label", r"\b(remove|drop|strip|untag|unlabel)\b.*\b(label|tag|bug|p0|enhancement|question|stale|needs-triage)\b"),
    ("label_issue", r"\blabel|\btag\b"),
    ("unassign_issue", r"\bunassign|take .* off #?\d+|\bremove @"),
    ("assign_issue", r"\bassign|\bgive #?\d+ to\b"),
    ("comment_issue", r"\bcomment|\breply\b|add a note"),
    ("show_issue", r"\bshow\b.*#?\d+|\bdetails\b|what is #?\d+|\bopen #?\d+\b"),
    ("list_issues", r"\blist\b|\bissues\b|what'?s open"),
]


READ_ONLY = {"whats_on_fire", "list_issues", "search_issues", "show_issue", "help", "brain_status",
             "other", "ASK"}


def rules(text):
    t = text.lower()
    return next((intent for intent, rx in RULES if re.search(rx, t)), None)


def score(name, rows, model):
    y = np.array([r["intent"] for r in rows])
    kw = np.array([rules(r["text"]) or "ASK" for r in rows])
    kw_act = kw != "ASK"
    p = model.predict_proba([normalise(r["text"]) for r in rows])
    br, br_act = model.classes_[p.argmax(1)], p.max(1) >= THRESHOLD
    print(f"\n{name}: {len(rows)} messages")
    out = {"messages": len(rows)}
    for label, pred, act in (("keyword rules", kw, kw_act), ("brain", br, br_act)):
        right, acts, wrong = np.mean(pred == y), act.mean(), np.mean(act & (pred != y))
        # the wrong actions that matter most: ones that change something in GitHub
        harmful = np.mean(act & (pred != y) & ~np.isin(pred, list(READ_ONLY)))
        print(f"  {label:13} right overall {right:5.1%}   acts on {acts:5.1%}   wrong actions {wrong:5.1%}"
              f"   of which change GitHub {harmful:5.1%}")
        out[label] = {"right": round(float(right), 3), "acts": round(float(acts), 3),
                      "wrong": round(float(wrong), 3), "harmful": round(float(harmful), 3)}
    return out


if __name__ == "__main__":
    Xtr, ytr = load("train")
    model = build().fit(Xtr, ytr)
    results = {name: score(name, [json.loads(l) for l in open(HERE / "data" / f"{name}.jsonl")], model)
               for name in ("test", "persona_terse", "persona_polite", "persona_hinglish")}
    (HERE / "data" / "bench_keywords.json").write_text(json.dumps(results))
