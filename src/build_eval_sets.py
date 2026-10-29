"""Left/right test items from What'sUp A/B and COCO-spatial (one/two objects).

Uses the loaders of the official repo (github.com/amitakamath/whatsup_vlms, MIT).
Every item becomes a 2-way choice: correct caption vs. the same caption with left/right swapped.

    python -m src.build_eval_sets --repo whatsup_vlms --out data/eval_lr
"""
import argparse
import json
import os
import sys

from .metrics import read_jsonl, write_jsonl
from .text import group_key, has_lr, relation, swap_lr

SOURCES = [("whatsup_A", "Controlled_Images", "A"), ("whatsup_B", "Controlled_Images", "B"),
           ("coco_one", "COCO_QA", "one"), ("coco_two", "COCO_QA", "two")]


def build(repo, out):
    sys.path.insert(0, os.path.abspath(repo))
    from dataset_zoo import aro_datasets

    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    items, counts = [], {}
    for name, cls, subset in SOURCES:
        ds = getattr(aro_datasets, cls)(image_preprocess=None, download=True,
                                       root_dir=os.path.abspath("data"), subset=subset)
        kept = 0
        for i in range(len(ds)):
            item = ds[i]
            correct = item["caption_options"][0]  # index 0 is the correct caption
            if not has_lr(correct):
                continue
            path = f"{out}/images/{name}_{i:05d}.png"
            if not os.path.exists(path):
                img = item["image_options"]
                (img[0] if isinstance(img, (list, tuple)) else img).convert("RGB").save(path)
            items.append({"id": f"{name}_{i:05d}", "source": name, "image": path,
                          "correct": correct, "wrong": swap_lr(correct),
                          "relation": relation(correct), "group": f"{name}|{group_key(correct)}"})
            kept += 1
        counts[name] = {"total": len(ds), "left_right": kept}
        print(name, counts[name])
    write_jsonl(f"{out}/items.jsonl", items)
    json.dump(counts, open(f"{out}/summary.json", "w"), indent=2)


def load_items(path="data/eval_lr/items.jsonl"):
    return read_jsonl(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="whatsup_vlms")
    ap.add_argument("--out", default="data/eval_lr")
    a = ap.parse_args()
    build(a.repo, a.out)
