"""Does the ask-and-learn loop work? Replay half the test set as live messages.

When the agent asks (confidence < threshold) or acts wrongly (the user says
"undo, I meant X"), the right answer becomes a training phrase. Retrain every
RETRAIN corrections and score on the other half, which the loop never sees.

    uv run simulate.py
"""
import numpy as np
from sklearn.model_selection import train_test_split

from brain import THRESHOLD, build, load

RETRAIN = 10

Xtr, ytr = load("train")
Xte, yte = load("test")
Xlive, Xfin, ylive, yfin = train_test_split(Xte, yte, test_size=0.5, stratify=yte, random_state=0)


def final_acc(model):
    p = model.predict_proba(Xfin)
    pred, conf = model.classes_[p.argmax(1)], p.max(1)
    y = np.array(yfin)
    return np.mean(pred == y), np.mean((conf >= THRESHOLD) & (pred != y))


model = build().fit(Xtr, ytr)
acc, wrong = final_acc(model)
print(f"messages 0    corrections 0   final-half accuracy {acc:.1%}  wrong actions {wrong:.1%}")

X, y, corrections = list(Xtr), list(ytr), 0
asks = wrongs = 0
for n, (text, truth) in enumerate(zip(Xlive, ylive), 1):
    p = model.predict_proba([text])[0]
    guess, conf = model.classes_[p.argmax()], p.max()
    if conf < THRESHOLD or guess != truth:
        asks += conf < THRESHOLD
        wrongs += conf >= THRESHOLD and guess != truth
        X.append(text); y.append(truth); corrections += 1
        if corrections % RETRAIN == 0:
            model = build().fit(X, y)
            acc, wrong = final_acc(model)
            print(f"messages {n:<4} corrections {corrections:<3} final-half accuracy {acc:.1%}  "
                  f"wrong actions {wrong:.1%}")
model = build().fit(X, y)
acc, wrong = final_acc(model)
print(f"end: {len(Xlive)} messages, {asks} asks, {wrongs} wrong actions undone -> "
      f"final-half accuracy {acc:.1%}, wrong actions {wrong:.1%}")
