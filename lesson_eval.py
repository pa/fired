"""How well does Fired recognise lessons, and work out what the taught word means?

Runs the held-out phrases in gen/test_3.jsonl (25 lessons, 15 look-alikes) through the real
decision path (brain, then Jev), then asks Jev which action each taught word stands for.

    uv run lesson_eval.py
"""
import json

import agent

rows = [json.loads(l) for l in open(agent.HERE / "gen" / "test_3.jsonl") if l.strip()]
caught = false = meant = 0
for r in rows:
    intent, _, _, src = agent.decide(r["text"])
    if r["intent"] != "teach_word":
        if intent == "teach_word":
            false += 1
            print(f"  FALSE LESSON: {r['text']}")
        continue
    caught += intent == "teach_word"
    word = agent.lesson_word(r["text"])
    target, conf = agent.lesson_meaning(r["text"], word) if word else (None, 0.0)
    meant += bool(word and target and conf >= agent.JEV_MIN)
    print(f"  {str(intent):16} {src:6} word={str(word)!r:14} means={str(target):18} {conf:4.0%}"
          f"  <- {r['text'][:60]}")
lessons = sum(r["intent"] == "teach_word" for r in rows)
print(f"\nlessons recognised: {caught}/{lessons}   word and meaning settled: {meant}/{lessons}"
      f"   false lessons: {false}/{len(rows) - lessons}")
print(f"jev calls {agent.stats['jev_calls']}, ${agent.stats['jev_tokens'] * 0.042 / 1e6:.4f}")
