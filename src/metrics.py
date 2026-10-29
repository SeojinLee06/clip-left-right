"""Left/right metrics. Input: a DataFrame with source, group, relation, pred (and optionally flip_ok)."""
import json

import pandas as pd


def summarize(df):
    df = df.assign(ok=df.pred == df.relation)
    out = {}
    for src, d in df.groupby("source"):
        m = {"n": len(d), "acc": d.ok.mean(),
             "acc_left": d[d.relation == "left"].ok.mean(),
             "acc_right": d[d.relation == "right"].ok.mean()}
        # pairs = left and right photo of the same objects (What'sUp has one of each)
        pairs = [g for _, g in d.groupby("group") if set(g.relation) == {"left", "right"}]
        if pairs:
            m["n_pairs"] = len(pairs)
            m["pair_acc"] = sum(g.ok.all() for g in pairs) / len(pairs)
            m["same_choice_rate"] = sum(g.pred.nunique() == 1 for g in pairs) / len(pairs)
        if "flip_ok" in d:
            m["flip_consistency"] = (d.ok == d.flip_ok).mean()
        out[src] = {k: float(v) for k, v in m.items()}
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def as_frame(items, preds, **extra):
    return pd.DataFrame([{"id": r["id"], "source": r["source"], "group": r["group"],
                          "relation": r["relation"], "pred": p, **{k: v[i] for k, v in extra.items()}}
                         for i, (r, p) in enumerate(zip(items, preds))])
