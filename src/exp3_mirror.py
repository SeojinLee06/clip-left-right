"""E3. Remove CLIP's word bias without training (control experiment).

For image I, its mirror I', and captions T_left / T_right:
    raw     I · (T_left - T_right) > 0  ->  "left"
    global  same score minus one offset (its mean on held-out COCO)
    mirror  (I - I') · (T_left - T_right) > 0
Anything that does not depend on where the objects are cancels in the mirror score.

    python -m src.exp3_mirror --models openai/clip-vit-base-patch32 openai/clip-vit-large-patch14
"""
import argparse
import json
import os

import torch

from .build_eval_sets import load_items
from .clip_utils import encode_images, encode_texts, load_clip
from .heldout import load_heldout
from .metrics import as_frame, summarize


def lr_scores(model, proc, items, device):
    left = [r["correct"] if r["relation"] == "left" else r["wrong"] for r in items]
    right = [r["wrong"] if r["relation"] == "left" else r["correct"] for r in items]
    paths = [r["image"] for r in items]
    img = encode_images(model, proc, paths, device)
    img_m = encode_images(model, proc, paths, device, mirror=True)
    t_left, t_right = encode_texts(model, proc, left, device), encode_texts(model, proc, right, device)
    d = t_left - t_right
    stats = {"cos_image_vs_mirror": (img * img_m).sum(-1).mean().item(),
             "cos_left_vs_right_caption": (t_left * t_right).sum(-1).mean().item()}
    return (img * d).sum(-1), (img_m * d).sum(-1), stats


def run(name, items, heldout, device, out):
    model, proc = load_clip(name, device)
    s_held, s_held_m, stats_held = lr_scores(model, proc, heldout, device)
    s, s_m, stats_test = lr_scores(model, proc, items, device)
    offset = s_held.mean().item()

    metrics = {}
    for method, (a, b) in {"raw": (s, s_held), "global": (s - offset, s_held - offset),
                           "mirror": (s - s_m, s_held - s_held_m)}.items():
        pred = lambda x: ["left" if v > 0 else "right" for v in x.tolist()]  # noqa: E731
        df = as_frame(items + heldout, pred(a) + pred(b))
        metrics[method] = summarize(df)
    tag = name.split("/")[-1]
    json.dump({"model": name, "global_offset": offset, "n_heldout": len(heldout),
               "stats": {"heldout": stats_held, "test": stats_test}, "metrics": metrics},
              open(f"{out}/metrics_{tag}.json", "w"), indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch32"])
    ap.add_argument("--out", default="results/exp3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, heldout = load_items(), load_heldout()
    for name in a.models:
        m = run(name, items, heldout, device, a.out)
        for method, per_set in m.items():
            print(name, method, {k: round(100 * v["acc"], 1) for k, v in per_set.items()})


if __name__ == "__main__":
    main()
