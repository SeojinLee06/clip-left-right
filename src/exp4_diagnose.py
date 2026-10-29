"""E4a. Where is left/right lost? Two checks on pre-trained CLIP (no training of CLIP).

T1 image side: a logistic-regression probe on CLIP's final image vector predicts the
   subject's side on What'sUp. Grouped 5-fold CV, so test object pairs are unseen.
   Near 50% means the pooled image vector has (almost) no left/right left in it.
T2 text side: is "left<->right swapped" closer to the caption than "different subject noun"?
   If yes for most captions, the text encoder treats left/right as a tiny change.

    python -m src.exp4_diagnose --models openai/clip-vit-base-patch32 openai/clip-vit-large-patch14
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .build_eval_sets import load_items
from .clip_utils import encode_images, encode_texts, load_clip
from .text import article, objects


def image_probe(model, proc, items, device, folds=5):
    out = {}
    for src in ("whatsup_A", "whatsup_B"):
        rs = [r for r in items if r["source"] == src]
        x = encode_images(model, proc, [r["image"] for r in rs], device).numpy()
        y = np.array([r["relation"] == "left" for r in rs])
        groups = np.array([r["group"] for r in rs])
        accs = [LogisticRegression(max_iter=2000).fit(x[tr], y[tr]).score(x[te], y[te])
                for tr, te in GroupKFold(folds).split(x, y, groups)]
        out[src] = {"probe_acc": float(np.mean(accs)), "fold_accs": [float(v) for v in accs], "n": len(rs)}
    return out


def text_check(model, proc, items, device):
    out = {}
    for src in sorted({r["source"] for r in items}):
        rs = [r for r in items if r["source"] == src]
        nouns = sorted({objects(r["correct"])[0] for r in rs} - {None})
        orig, swapped, other = [], [], []
        for i, r in enumerate(rs):
            subj = objects(r["correct"])[0]
            alts = [n for n in nouns if n != subj]
            if subj and alts:
                new = alts[i % len(alts)]
                orig.append(r["correct"])
                swapped.append(r["wrong"])
                other.append(re.sub(rf"^an? {re.escape(subj)}\b", f"{article(new).capitalize()} {new}",
                                    r["correct"], flags=re.I))
        t, ts, to = (encode_texts(model, proc, x, device) for x in (orig, swapped, other))
        a, b = (t * ts).sum(-1), (t * to).sum(-1)
        out[src] = {"cos_lr_swap": a.mean().item(), "cos_noun_swap": b.mean().item(),
                    "frac_lr_swap_more_similar": (a > b).float().mean().item(), "n": len(orig)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch32"])
    ap.add_argument("--out", default="results/exp4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = load_items()
    report = {}
    for name in a.models:
        model, proc = load_clip(name, device)
        report[name.split("/")[-1]] = {"T1_image_probe": image_probe(model, proc, items, device),
                                       "T2_text": text_check(model, proc, items, device)}
    json.dump(report, open(f"{a.out}/diagnosis.json", "w"), indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
