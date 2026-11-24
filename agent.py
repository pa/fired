"""Fired: a Slack bot that runs one GitHub repo's issues with no LLM at runtime.

Each message goes: lessons you taught -> brain (TF-IDF + LR, ~1 ms) -> Jev if
the brain is unsure -> you, if Jev is unsure too. Teaching a lesson
("nuke means close") checkpoints the Sprite first, so "rewind" undoes it.

Slack pushes each message to POST /slack/events (signed with the app's signing
secret, read from ~/.fired/slack_signing_secret). Replies go out through the
Sprites Slack connector. GET / is the status page.

    uv run agent.py
"""
import collections
import hashlib
import hmac
import html
import json
import pathlib
import pickle
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import yaml

from brain import HERE, MODEL, THRESHOLD, normalise, predict
from keyword_baseline import rules as keyword_rules

GW = "https://api.sprites.dev/v1/gateway"
SLACK = f"{GW}/slack_bot/1gKljovGtTzgG7GSFH9t5A"
GITHUB = f"{GW}/github/12qaq8lhPAZk6alt0Mf50Q"
REPO = "pa/fired-demo"
GH = f"{GITHUB}/repos/{REPO}"
JEV = f"{GW}/typesafe/SnoyU9blsmeLNMQKolqYqg/v1/systemone"
CHANNEL = "C0CMEA5V2RE"
ME = "pa"  # GitHub login for "me"/"my" in Slack messages
JEV_MIN = 0.7  # Jev settles an unsure message only at this confidence or above
LESSONS = HERE / "lessons.json"
SECRET_FILE = pathlib.Path.home() / ".fired" / "slack_signing_secret"

INTENTS = yaml.safe_load((HERE / "intents.yaml").read_text())
model = pickle.loads(MODEL.read_bytes())  # written by brain.py train on this machine
lessons: dict[str, str] = json.loads(LESSONS.read_text()) if LESSONS.exists() else {}
STATS = HERE / "stats.json"  # counters survive restarts; rewind carries them via Slack metadata
stats = {"brain": {"n": 0, "ms": 0.0}, "jev": {"n": 0, "ms": 0.0}, "lesson": {"n": 0, "ms": 0.0},
         "asked": 0, "jev_calls": 0, "jev_tokens": 0, "messages": 0, "saved": 0.0,
         "since": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
recent: collections.deque = collections.deque(maxlen=15)  # latest decisions, newest first


def load_state(state):
    stats.update(state["stats"])
    recent.clear()
    recent.extend(state["recent"])


def state():
    return {"stats": stats, "recent": list(recent)}


def save():
    stats["saved"] = time.time()
    STATS.write_text(json.dumps(state()))


if STATS.exists():
    load_state(json.loads(STATS.read_text()))


def record(source, t):
    stats[source]["n"] += 1
    stats[source]["ms"] += ms(t)


def avg(s):
    return f"{s['ms'] / s['n']:.1f} ms" if s["n"] else "-"
undo_stack: list = []  # (what was done, function that reverses it)
pending: dict[str, dict] = {}  # thread ts -> what we're waiting for
events: queue.Queue = queue.Queue()  # one worker handles messages in order
seen_events: set[str] = set()  # Slack retries an event if we're slow to ack


# --- plumbing -------------------------------------------------------------

def call(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json; charset=utf-8"},
                                 method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def say(text, thread=None, why=None):
    """Post a reply. `why` goes on its own line in Slack's small grey context text."""
    # Reply in the message's thread and show it in the channel too, so it's hard to miss.
    extra = {"thread_ts": thread, "reply_broadcast": True} if thread else {}
    if why:
        extra["blocks"] = [{"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}},
                           {"type": "context", "elements": [{"type": "mrkdwn", "text": why[:2900]}]}]
    call("POST", f"{SLACK}/chat.postMessage", {"channel": CHANNEL, "text": text, **extra,
                                               "unfurl_links": False, "unfurl_media": False})


def gh(method, path, body=None):
    return call(method, f"{GH}{path}", body)


def jev(text, questions):
    out = call("POST", JEV, {"model": "jev-latest", "state": text, "questions": questions})
    stats["jev_calls"] += 1
    stats["jev_tokens"] += out["usage"]["input_tokens"]
    return out["answers"]


def ms(t):
    return (time.perf_counter() - t) * 1000


# --- checkpoints ------------------------------------------------------------

def checkpoints():
    out = subprocess.run(["sprite-env", "checkpoints", "list"], capture_output=True, text=True).stdout
    return [c for c in json.loads(out) if c["id"] != "Current"]


def checkpoint(comment):
    subprocess.run(["sprite-env", "checkpoints", "create", "--comment", comment],
                   capture_output=True, text=True, check=True)
    return max(checkpoints(), key=lambda c: c["create_time"])["id"]


# --- deciding -------------------------------------------------------------

def decide(text):
    """(intent or None, why, options to offer if None, which part decided)."""
    t = time.perf_counter()
    intent, conf, reasons = predict(model, text)
    taught = lesson_in(text)
    if taught and not (intent == "teach_word" and conf >= THRESHOLD):
        record("lesson", t)
        return lessons[taught], f"📝 *Lesson* · you taught me '{taught}' means {lessons[taught]}", [], "lesson"
    if conf >= THRESHOLD:
        record("brain", t)
        words = "  ".join(f"`{w}` +{s:.2f}" for w, s in reasons) or "overall wording"
        return intent, f"🧠 *Brain* · {conf:.0%} sure · {ms(t):.1f} ms · {words}", [], "brain"
    t = time.perf_counter()
    a = jev(text, {"intent": {
        "type": "choice",
        "instructions": "Which action is this Slack message asking a GitHub repo bot to do?",
        "criteria": {k: v["description"] for k, v in INTENTS.items()}}})["intent"]
    if a["confidence"] >= JEV_MIN:
        record("jev", t)
        return a["choice"], (f"⚖️ *Jev* · {a['confidence']:.0%} sure · {ms(t):.0f} ms · "
                             f"the brain was only {conf:.0%} sure (guessed {intent})"), [], "jev"
    stats["asked"] += 1
    probs = model.predict_proba([normalise(text)])[0]
    top = [a["choice"]] + list(model.classes_[np.argsort(-probs)[:3]])
    why = (f"🧠 Brain {conf:.0%} sure · ⚖️ Jev {a['confidence']:.0%} sure · "
           "both too unsure to act, so I'm asking")
    return None, why, list(dict.fromkeys(top))[:3], "asked"


# --- acting on GitHub -----------------------------------------------------

LABELS = ["bug", "enhancement", "needs-triage", "p0", "question", "stale"]
NEEDS_NUMBER = {"show_issue", "label_issue", "remove_label", "assign_issue", "unassign_issue",
                "comment_issue", "close_issue", "close_duplicate", "close_not_planned",
                "reopen_issue", "set_milestone"}
FILLER = set("""a an the any anything is are there issue issues ticket tickets about search find
for with mentioning mention regarding related to on in of do we have open closed pls please can
you could show me look up lookup something re anyone filed already yet""".split())


def parse(text):
    nums = [int(n) for n in re.findall(r"#?\b(\d+)\b", re.sub(r"v\d+(\.\d+)+", "", text))]
    user = re.search(r"@([\w-]+)", text)
    quote = re.search(r"[\"“](.+?)[\"”]", text) or re.search(r":\s*(.+)$", text)
    labels = [l for l in LABELS if re.search(rf"\b{re.escape(l)}\b", text.lower())]
    return (nums[0] if nums else None, nums, user.group(1) if user else None,
            quote.group(1).strip() if quote else None, labels)


def line(i):
    tags = ", ".join(l["name"] for l in i["labels"])
    return f"#{i['number']} {i['title']}" + (f"  [{tags}]" if tags else "")


def issues(state="open"):
    return [i for i in gh("GET", f"/issues?state={state}&per_page=100") if "pull_request" not in i]


def names(i):
    return {l["name"] for l in i["labels"]}


def set_state(num, state, reason):
    gh("PATCH", f"/issues/{num}", {"state": state, "state_reason": reason})


def create(title, labels):
    i = gh("POST", "/issues", {"title": title, "labels": labels})
    undo_stack.append((f"create #{i['number']}",
                       lambda n=i["number"]: set_state(n, "closed", "not_planned")))
    return f"Created {line(i)}\n{i['html_url']}"


def close_duplicate(dup, orig):
    c = gh("POST", f"/issues/{dup}/comments", {"body": f"Duplicate of #{orig}"})
    try:
        set_state(dup, "closed", "duplicate")
    except urllib.error.HTTPError:  # older API without the duplicate reason
        set_state(dup, "closed", "not_planned")
    undo_stack.append((f"close #{dup} as duplicate", lambda: (
        set_state(dup, "open", "reopened"), gh("DELETE", f"/issues/comments/{c['id']}"))))
    return f"Closed #{dup} as a duplicate of #{orig}."


DETAIL_FILE = HERE / "detail.pkl"  # second model: which detail a question asks about
detail_model = pickle.loads(DETAIL_FILE.read_bytes()) if DETAIL_FILE.exists() else None


def field_answer(text, i):
    """(answer line or None, note for the grey line). The detail model decides which
    detail the question is about; below THRESHOLD we just show the whole issue."""
    if detail_model is None:
        return None, ""
    t = time.perf_counter()
    detail, conf, reasons = predict(detail_model, text)
    words = "  ".join(f"`{w}` +{s:.2f}" for w, s in reasons) or "overall wording"
    note = f"🔎 *Detail model* · {detail} {conf:.0%} · {ms(t):.1f} ms · {words}"
    if detail == "none" or conf < THRESHOLD:
        return None, note
    n = i["number"]
    if detail == "assignee":
        people = [f"@{a['login']}" for a in i["assignees"]]
        return (f"#{n} is assigned to {', '.join(people)}." if people else f"Nobody is assigned to #{n}."), note
    if detail == "labels":
        tags = ", ".join(l["name"] for l in i["labels"])
        return (f"#{n} is labelled {tags}." if tags else f"#{n} has no labels."), note
    if detail == "milestone":
        m = (i.get("milestone") or {}).get("title")
        return (f"#{n} is in {m}." if m else f"#{n} isn't in a milestone."), note
    return f"#{n} is {i['state']}.", note


def act(intent, text, ts):
    num, nums, user, quote, labels = parse(text)
    if intent in NEEDS_NUMBER and not num:
        return "Which issue? Add its number, like #12."
    if intent == "unassign_issue" and not user and re.search(r"\b(me|myself)\b", text, re.I):
        user = ME

    if intent == "whats_on_fire":
        fire = sorted((i for i in issues() if {"p0", "bug"} & names(i)),
                      key=lambda i: "p0" not in names(i))
        return "On fire:\n" + "\n".join(map(line, fire)) if fire else "Nothing on fire."
    if intent == "list_issues":
        state = "closed" if re.search(r"\bclosed\b", text, re.I) else "open"
        who = user or (ME if re.search(r"\b(my|mine|me)\b", text, re.I) else None)
        found = [i for i in issues(state) if set(labels) <= names(i)
                 and (not who or who in {a["login"] for a in i["assignees"]})]
        return "\n".join(map(line, found)) or f"No {state} issues match."
    if intent == "search_issues":
        terms = quote or " ".join(w for w in re.findall(r"[\w.-]+", text.lower()) if w not in FILLER)
        if not terms:
            return "Search for what? Try: any issues about safari?"
        q = urllib.parse.quote(f"repo:{REPO} is:issue {terms}")
        found = call("GET", f"{GITHUB}/search/issues?q={q}&per_page=10")["items"]
        return (f"Issues matching '{terms}':\n" + "\n".join(map(line, found))) if found \
            else f"Nothing matches '{terms}'."
    if intent == "show_issue":
        i = gh("GET", f"/issues/{num}")
        who = ", ".join(a["login"] for a in i["assignees"]) or "nobody"
        ms_ = (i.get("milestone") or {}).get("title", "none")
        details = (f"{line(i)}\n{i['state']}, assigned to {who}, milestone {ms_}\n"
                   f"{(i['body'] or '')[:300]}\n{i['html_url']}")
        answer, note = field_answer(text, i)
        return (f"*{answer}*\n{details}" if answer else details), note
    if intent == "create_issue":
        if not quote:  # no quotes or colon: ask rather than guess which words are the title
            pending[ts] = {"kind": "title", "labels": labels}
            return ('What should the title be? Reply in this thread with it. '
                    '(Next time you can put it in quotes: file a bug "login broken on safari".)')
        return create(quote, labels)
    if intent == "label_issue":
        if not labels:
            return f"Which label? I know {', '.join(LABELS)}."
        gh("POST", f"/issues/{num}/labels", {"labels": labels})
        undo_stack.append((f"label #{num}", lambda: [gh("DELETE", f"/issues/{num}/labels/{l}")
                                                     for l in labels]))
        return f"Labelled #{num} {', '.join(labels)}."
    if intent == "remove_label":
        have = names(gh("GET", f"/issues/{num}"))
        drop = [l for l in labels if l in have] or (list(have) if len(have) == 1 else [])
        if not drop:
            return f"Which label? #{num} has {', '.join(sorted(have)) or 'none'}."
        for l in drop:
            gh("DELETE", f"/issues/{num}/labels/{l}")
        undo_stack.append((f"remove label from #{num}",
                           lambda: gh("POST", f"/issues/{num}/labels", {"labels": drop})))
        return f"Removed {', '.join(drop)} from #{num}."
    if intent in ("assign_issue", "unassign_issue"):
        if not user:
            user = ME if re.search(r"\b(me|myself)\b", text, re.I) else None
        if not user:
            return "Who? Use @username."
        on, off = ("POST", "DELETE") if intent == "assign_issue" else ("DELETE", "POST")
        gh(on, f"/issues/{num}/assignees", {"assignees": [user]})
        undo_stack.append((f"{intent.split('_')[0]} #{num}",
                           lambda: gh(off, f"/issues/{num}/assignees", {"assignees": [user]})))
        return f"{'Assigned' if on == 'POST' else 'Unassigned'} @{user} {'to' if on == 'POST' else 'from'} #{num}."
    if intent == "set_milestone":
        name = re.search(r"\b(v\d+(?:\.\d+)*)\b", text, re.I)
        all_ms = gh("GET", "/milestones?state=open")
        target = next((m for m in all_ms if m["title"].lower() in text.lower()), None)
        made = ""
        if not target and name:
            target = gh("POST", "/milestones", {"title": name.group(1)})
            made = f" (created milestone {target['title']})"
        if not target:
            have = ", ".join(m["title"] for m in all_ms) or "none yet"
            return f"Which milestone? Open milestones: {have}. Or name one like v2.1."
        before = (gh("GET", f"/issues/{num}").get("milestone") or {}).get("number")
        gh("PATCH", f"/issues/{num}", {"milestone": target["number"]})
        undo_stack.append((f"milestone on #{num}",
                           lambda: gh("PATCH", f"/issues/{num}", {"milestone": before})))
        return f"Put #{num} in {target['title']}{made}."
    if intent == "comment_issue":
        if not quote:
            return 'What should the comment say? Put it in quotes, like comment on #3 "fixed in main".'
        c = gh("POST", f"/issues/{num}/comments", {"body": quote})
        undo_stack.append((f"comment on #{num}",
                           lambda: gh("DELETE", f"/issues/comments/{c['id']}")))
        return f"Commented on #{num}: {quote}"
    if intent == "close_issue":
        set_state(num, "closed", "completed")
        undo_stack.append((f"close #{num}", lambda: set_state(num, "open", "reopened")))
        return f"Closed #{num} as done."
    if intent == "close_not_planned":
        set_state(num, "closed", "not_planned")
        undo_stack.append((f"close #{num}", lambda: set_state(num, "open", "reopened")))
        return f"Closed #{num} as not planned."
    if intent == "close_duplicate":
        pair = list(dict.fromkeys(nums))[:2]
        if len(pair) < 2:
            return "Duplicate of which issue? Say it like: #9 is a duplicate of #2."
        a, b = pair
        j = jev(text, {"dup": {
            "type": "choice",
            "instructions": "Which issue does this message say is the duplicate, the one to close?",
            "criteria": {f"#{a}": f"issue #{a} is the duplicate and should be closed",
                         f"#{b}": f"issue #{b} is the duplicate and should be closed"}}})["dup"]
        note = f"⚖️ *Jev* picked which one is the duplicate · {j['choice']} {j['confidence']:.0%}"
        if j["confidence"] < JEV_MIN:
            pending[ts] = {"kind": "dup", "pair": pair}
            return f"Which one should I close as the duplicate: #{a} or #{b}? Reply with the number.", note
        dup = int(j["choice"].lstrip("#"))
        return close_duplicate(dup, b if dup == a else a), note
    if intent == "reopen_issue":
        set_state(num, "open", "reopened")
        undo_stack.append((f"reopen #{num}", lambda: set_state(num, "closed", "completed")))
        return f"Reopened #{num}."
    if intent == "close_stale":
        stale = [i for i in issues() if "stale" in names(i)]
        if not stale:
            return "No stale issues."
        pending[ts] = {"kind": "confirm", "numbers": [i["number"] for i in stale]}
        return "I'd close these:\n" + "\n".join(map(line, stale)) + "\nReply *yes* to go ahead."
    if intent == "triage":
        return triage()
    if intent == "undo":
        if not undo_stack:
            return "Nothing to undo."
        what, fn = undo_stack.pop()
        fn()
        return f"Undid {what}."
    if intent == "teach_word":
        word = lesson_word(text)
        if not word:
            return "Which word should I learn? Put it in quotes, like: 'nuke' means close."
        target, conf = lesson_meaning(text, word)
        note = f"⚖️ *Jev* picked what '{word}' means · {target} {conf:.0%}"
        if conf < JEV_MIN:
            return (f"I'm not sure what '{word}' should mean. Try: '{word}' means close "
                    "(or any action from help)."), note
        return teach(word, target), note
    if intent == "rewind_brain":
        return rewind(ts)
    if intent == "brain_status":
        return status_text()
    if intent == "help":
        return (f"I run {REPO}'s issues. Try: what's on fire, my issues, any issues about safari, "
                'show #3, file a bug "login broken", comment on #4 "on it", label #6 bug, '
                "drop bug from #6, assign #2 to @pa, unassign me from #2, put #3 in v2.1, "
                "close #8, #9 is a dupe of #2, close #7 as won't fix, reopen #8, triage, "
                "close stale issues, undo.\nTeach me a word: nuke means close. Undo a lesson: rewind.")
    return "That's not something I do here. Say help to see what I can."


TRIAGE_QUESTIONS = {  # Jev can only answer these: a label from 3, and a yes/no on urgency
    "kind": {"type": "choice", "instructions": "What kind of GitHub issue is this?",
             "criteria": {"bug": "something is broken or behaves wrongly",
                          "enhancement": "a request for a new feature or improvement",
                          "question": "someone asking how to do something"}},
    "urgent": {"type": "noul", "instructions": "Is this urgent: users are blocked, money "
               "or data is at risk, or production is down?"}}


def triage():
    todo = [i for i in issues() if not names(i) - {"needs-triage"}]
    if not todo:
        return "Everything open already has a label."
    done, unsure = [], []
    for i in todo:
        a = jev(f"Issue title: {i['title']}\n\n{i['body'] or ''}", TRIAGE_QUESTIONS)
        if a["kind"]["confidence"] < JEV_MIN:
            unsure.append(f"#{i['number']} {i['title']}")
            continue
        new = [a["kind"]["choice"]] + (["p0"] if a["urgent"]["noul"] >= 0.8 else [])
        gh("POST", f"/issues/{i['number']}/labels", {"labels": new})
        done.append((i, new, a["kind"]["confidence"]))
    undo_stack.append(("triage", lambda: [gh("DELETE", f"/issues/{i['number']}/labels/{l}")
                                          for i, new, _ in done for l in new]))
    out = [f"#{i['number']} {i['title']} -> {', '.join(new)} (Jev {c:.0%})" for i, new, c in done]
    if unsure:
        out.append("Left alone, Jev wasn't sure: " + "; ".join(unsure))
    return "Triaged:\n" + "\n".join(out)


# --- lessons and rewind ---------------------------------------------------

# Once the brain says a message is teach_word, these pull out the word and its meaning.
# They only extract; they never decide whether a message is a lesson.
LESSON_SHAPES = [
    r"when i (?:say|type|write) [\"“'‘]?(.+?)[\"”'’]?,? (?:i mean|i want|that means|just|do|then)\s+(.+)",
    r"if i (?:say|type|write) [\"“'‘]?(.+?)[\"”'’]?,? (?:just |then )?(?:do |it means )?(.+)",
    r"treat [\"“'‘]?(.+?)[\"”'’]? (?:as|like) (.+)",
    r"[\"“'‘](.+?)[\"”'’]\s*(?:means|=|->|→|is short for|stands for|is|should mean)\s+(.+)",
    r"([\w-]+(?: [\w-]+)?)\s*(?:means|=|->|→|is short for|stands for|should mean)\s+(.+)",
]


def lesson_parts(text):
    t = re.sub(r"^(?:ok(?:ay)?|hey|so|fyi|note|learn this|remember|from now on)[\s,:]*", "",
               text.strip(), flags=re.I)
    t = re.sub(r"^(?:from now on|going forward)[\s,:]*", "", t, flags=re.I)
    for shape in LESSON_SHAPES:
        if m := re.search(shape, t, re.I):
            return m.group(1).strip(" '\"").lower(), m.group(2).strip()
    return None, None


LESSON_ACTIONS = {k: v["description"] for k, v in INTENTS.items()
                  if k not in ("teach_word", "other", "help")}


def lesson_word(text):
    """The word being taught: a quoted word if there is one (an exact pattern), else the
    common sentence shapes. None means ask the user."""
    if q := re.search(r"(?:^|[\s:(=])[\"“'‘]([^\"”'’]{1,30})[\"”'’]", text):
        return q.group(1).strip().lower()
    return lesson_parts(text)[0]


def lesson_meaning(text, word):
    """Jev decides which action the taught word stands for, from the action descriptions."""
    a = jev(text, {"means": {
        "type": "choice",
        "instructions": f"The user is teaching the bot that '{word}' should mean one of these "
                        "actions from now on. Which action?",
        "criteria": LESSON_ACTIONS}})["means"]
    return a["choice"], a["confidence"]


def lesson_in(text):
    """The taught word (or phrase) this message uses, if any."""
    low = text.lower()
    return next((w for w in sorted(lessons, key=len, reverse=True)
                 if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low)), None)


def teach(word, intent):
    cp = checkpoint(f"before lesson '{word.lower()}'")
    lessons[word.lower()] = intent
    LESSONS.write_text(json.dumps(lessons))
    return (f"Learned: '{word}' means {intent}. I checkpointed my brain first ({cp}), "
            "so say rewind to forget it.")


def rewind(ts):
    mine = [c for c in checkpoints() if (m := re.match(r"before lesson '(.+)'", c.get("comment", "")))
            and m.group(1) in lessons]
    if not mine:
        return "Nothing to rewind: you haven't taught me anything since my last rewind."
    cp = max(mine, key=lambda c: c["create_time"])
    save()
    call("POST", f"{SLACK}/chat.postMessage", {
        "channel": CHANNEL, "text": f"Rewinding my brain to {cp['id']} ({cp['comment']}). "
        "Back in a few seconds.",
        "metadata": {"event_type": "fired_stats", "event_payload": {"state": json.dumps(state())}}})
    subprocess.run(["sprite-env", "checkpoints", "restore", cp["id"]], check=True)
    return None  # the Sprite restarts; startup posts that we're back


# --- status ---------------------------------------------------------------

def status_text():
    n, spent = stats["messages"], stats["jev_tokens"] * 0.042 / 1e6
    code = anthropic_blocked()
    rows = [("brain", stats["brain"]["n"], avg(stats["brain"])),
            ("lesson", stats["lesson"]["n"], avg(stats["lesson"])),
            ("Jev", stats["jev"]["n"], avg(stats["jev"])),
            ("asked you", stats["asked"], "-")]
    table = "\n".join([f"{'Decided by':<12}{'Msgs':>6}{'Avg time':>11}"] +
                      [f"{s:<12}{c:>6}{a:>11}" for s, c, a in rows])
    kw = (bench("keywords") or {}).get("test")
    bench_line = (f"\nUnseen messages: brain {kw['brain']['right']:.0%} right, {kw['brain']['harmful']:.1%} "
                  f"wrong changes to GitHub; keyword rules {kw['keyword rules']['right']:.0%} right, "
                  f"{kw['keyword rules']['harmful']:.1%} wrong changes") if kw else ""
    points = [c["id"] for c in checkpoints() if c.get("comment", "").startswith("before lesson")]
    return (f"*Fired status* since {stats['since']}: {n} messages\n```{table}```\n"
            f"Claude calls at runtime: 0 (Anthropic gateway: HTTP {code})\n"
            f"Spent on decisions: ${spent:.4f} for {stats['jev_calls']} Jev calls. "
            f"A Claude agent: ~${n * CLAUDE_AGENT_PER_MESSAGE:.2f} (estimate)"
            f"{bench_line}\n"
            f"Lessons: {', '.join(f'{w} = {i}' for w, i in lessons.items()) or 'none'}. "
            f"Rewind points: {', '.join(points) or 'none'}\n"
            f"Full page: https://fired-agent-5xfj.sprites.app/")


# --- status page ------------------------------------------------------------

# A typical tool-using Claude agent for the same job: 22 tool definitions plus the message,
# two model turns. Assumed ~6,000 input + 400 output tokens on Opus 5.5 ($4 / $20 per MTok).
CLAUDE_AGENT_PER_MESSAGE = 6000 * 4 / 1e6 + 400 * 20 / 1e6
anthropic_check = {"at": 0.0, "code": None}


def anthropic_blocked():
    """Live check, cached 5 minutes: does the Anthropic gateway refuse this Sprite?"""
    if time.time() - anthropic_check["at"] > 300:
        try:
            urllib.request.urlopen(f"{GW}/anthropic/8Cw1OkCFZhPZvNQRLbfwlQ/v1/models", timeout=5)
            anthropic_check["code"] = 200
        except urllib.error.HTTPError as e:
            anthropic_check["code"] = e.code
        except Exception:
            anthropic_check["code"] = None
        anthropic_check["at"] = time.time()
    return anthropic_check["code"]


def bench(name):
    f = HERE / "data" / f"bench_{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def api_state():
    try:
        points = [{"id": c["id"], "comment": c["comment"]} for c in checkpoints()
                  if c.get("comment", "").startswith("before lesson")]
    except Exception:
        points = []
    return {"repo": REPO, "stats": stats, "recent": list(recent), "lessons": lessons,
            "points": points[-6:], "gate": anthropic_blocked(),
            "bench": {"keywords": bench("keywords"), "injection": bench("injection")},
            "claude_per_msg": CLAUDE_AGENT_PER_MESSAGE, "threshold": THRESHOLD}


def api_try(text):
    """What would Fired do with this message? Brain and keyword rules only: no Jev, no GitHub."""
    text = text[:300]
    lesson = lesson_in(text)
    t = time.perf_counter()
    intent, conf, reasons = predict(model, text)
    took = ms(t)
    probs = model.predict_proba([normalise(text)])[0]
    top = sorted(zip(model.classes_, probs), key=lambda kv: -kv[1])[:3]
    route = "lesson" if lesson else "brain" if conf >= THRESHOLD else "jev"
    return {"intent": lessons[lesson] if lesson else intent, "conf": conf, "ms": took,
            "reasons": reasons, "top": [[str(i), float(p)] for i, p in top], "route": route,
            "lesson": lesson, "rules": keyword_rules(text)}


# --- handling messages ------------------------------------------------------

def clean(text):
    text = re.sub(r"<@\w+>", "", text)
    text = re.sub(r"<(?:mailto:)?[^|>]+\|([^>]+)>", r"\1", text)
    return html.unescape(text).strip()


def run(intent, text, why, ts):
    try:
        reply = act(intent, text, ts)
    except urllib.error.HTTPError as e:
        reply = f"GitHub said {e.code}: {e.read()[:200].decode(errors='replace')}"
    note = ""
    if isinstance(reply, tuple):  # an action can add its own line to the explanation
        reply, note = reply
    if reply:
        say(reply, ts, f"`{intent}` · {why}" + (f"\n{note}" if note else ""))


def handle(text, ts):
    t = time.perf_counter()
    intent, why, options, source = decide(text)
    recent.appendleft({"at": time.strftime("%H:%M:%S"), "text": text[:90],
                       "intent": intent or "asked you", "source": source, "ms": ms(t)})
    stats["messages"] += 1
    if intent:
        return run(intent, text, why, ts)
    pending[ts] = {"kind": "ask", "text": text, "options": options}
    say("I'm not sure what you mean. Reply with a number:\n"
        + "\n".join(f"{n}. {o.replace('_', ' ')}" for n, o in enumerate(options, 1))
        + "\nor *no* to drop it.", ts, why)


def answer(thread, p, reply):
    raw = reply.strip()
    reply = raw.lower()
    del pending[thread]
    if p["kind"] == "title":
        return say(create(raw, p["labels"]), thread)
    if p["kind"] == "dup":
        n = int(reply.lstrip("#")) if reply.lstrip("#").isdigit() else None
        if n not in p["pair"]:
            return say("Okay, left both open.", thread)
        a, b = p["pair"]
        return say(close_duplicate(n, b if n == a else a), thread)
    if p["kind"] == "confirm":
        if reply not in ("yes", "y", "go", "ok"):
            return say("Okay, left them open.", thread)
        for n in p["numbers"]:
            set_state(n, "closed", "not_planned")
        undo_stack.append(("close stale", lambda ns=p["numbers"]: [
            set_state(n, "open", "reopened") for n in ns]))
        return say(f"Closed {', '.join(f'#{n}' for n in p['numbers'])}.", thread)
    pick = next((o for n, o in enumerate(p["options"], 1)
                 if reply in (str(n), o, o.replace("_", " "))), None)
    if not pick:
        return say("Dropped it.", thread)
    run(pick, p["text"], "🙋 *You* picked it", thread)


ANSWER = re.compile(r"^\s*(yes|y|yep|ok|go|no|n|nope|#?\d{1,4})\s*[.!]?\s*$", re.I)


def worker():
    while True:
        e, done = events.get()
        try:
            text, thread = clean(e.get("text", "")), e.get("thread_ts")
            if thread and thread != e["ts"]:  # a reply inside a thread
                if thread in pending:
                    answer(thread, pending[thread], text)
            elif pending and ANSWER.match(text):  # a bare "yes" or "2" answers the latest question
                latest = max(pending, key=float)
                answer(latest, pending[latest], text)
            else:
                t = time.time()
                handle(text, e["ts"])
                print(f"handled {t - float(e['ts']):.1f}s after sending, in {time.time() - t:.1f}s",
                      flush=True)
        except Exception as err:  # one bad message shouldn't kill the bot
            print("error:", repr(err), flush=True)
        finally:
            save()
            done.set()


# --- HTTP: Slack events and the status page ---------------------------------

def signed(headers, body):
    """Slack's request signature: v0=HMAC-SHA256(secret, 'v0:{timestamp}:{body}')."""
    if not SECRET_FILE.exists():
        return False
    ts = headers.get("X-Slack-Request-Timestamp", "")
    if not ts.isdigit() or abs(time.time() - int(ts)) > 300:  # refuse replays
        return False
    mac = hmac.new(SECRET_FILE.read_text().strip().encode(), f"v0:{ts}:".encode() + body,
                   hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"v0={mac}", headers.get("X-Slack-Signature", ""))


class Web(BaseHTTPRequestHandler):
    def reply(self, code, body=b"", kind="text/plain"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # Slack stopped waiting after 3 s
            pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/slack/events" or not signed(self.headers, body):
            return self.reply(403)
        msg = json.loads(body)
        if msg.get("type") == "url_verification":
            return self.reply(200, msg["challenge"].encode())
        e = msg.get("event", {})
        if (msg.get("event_id") in seen_events or e.get("type") != "message"
                or e.get("channel") != CHANNEL or e.get("bot_id") or e.get("subtype")):
            return self.reply(200)
        seen_events.add(msg["event_id"])
        # Hold the request open until the reply is posted: a Sprite pauses once it has no
        # open HTTP request, which would freeze the worker mid-message. If this takes over
        # Slack's 3 s, Slack resends the event and seen_events drops the copy.
        done = threading.Event()
        events.put((e, done))
        done.wait(60)
        self.reply(200)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/state":
            return self.reply(200, json.dumps(api_state()).encode(), "application/json")
        if url.path == "/api/try":
            q = urllib.parse.parse_qs(url.query).get("q", [""])[0]
            return self.reply(200, json.dumps(api_try(q)).encode(), "application/json")
        self.reply(200, (HERE / "status.html").read_bytes(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    try:  # after a rewind the disk holds old counters; the Rewinding message carries the latest
        for m in call("GET", f"{SLACK}/conversations.history?channel={CHANNEL}&limit=15"
                             "&include_all_metadata=true").get("messages", []):
            meta = m.get("metadata") or {}
            if meta.get("event_type") == "fired_stats":
                carried = json.loads(meta["event_payload"]["state"])
                if carried["stats"]["saved"] > stats["saved"]:
                    load_state(carried)
                    save()
                break
    except Exception as err:
        print("couldn't read carried counters:", repr(err), flush=True)
    taught = ", ".join(f"{w} = {i}" for w, i in lessons.items()) or "none"
    say(f"Fired is up. Lessons: {taught}. Say help to see what I do.")
    ThreadingHTTPServer(("", 8080), Web).serve_forever()
