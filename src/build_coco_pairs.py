"""Left/right image-caption pairs from COCO 2017 train.

An image is kept if two object classes appear exactly once, each covers >= min_area of the
image, and one box is entirely left of the other with a larger horizontal than vertical gap.
The caption names the pair in random order, so "left" and "right" are both used.

    # training pool (E1, E2)
    python -m src.build_coco_pairs --ann coco/annotations/instances_train2017.json --out data/coco_lr
    # held-out set for choosing settings in E3/E4 (no overlap with the training pool)
    python -m src.build_coco_pairs --ann ... --out data/coco_heldout --heldout --max-images 400

COCO val2017 is not used: COCO-spatial (a test set) is built from it.
"""
import argparse
import json
import os
import random
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .metrics import write_jsonl
from .text import make_caption

URL = "http://images.cocodataset.org/train2017/{}"


def find_pairs(ann_path, min_area, seed=0):
    coco = json.load(open(ann_path))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    rng = random.Random(seed)
    pairs = []
    for img_id, anns in by_image.items():
        if any(a.get("iscrowd") for a in anns):
            continue
        im = images[img_id]
        area = im["width"] * im["height"]
        count = Counter(a["category_id"] for a in anns)
        single = [a for a in anns if count[a["category_id"]] == 1
                  and a["bbox"][2] * a["bbox"][3] / area >= min_area]
        options = []
        for a in single:
            for b in single:
                (xa, ya, wa, ha), (xb, yb, wb, hb) = a["bbox"], b["bbox"]
                dx = (xb + wb / 2) - (xa + wa / 2)
                dy = abs((yb + hb / 2) - (ya + ha / 2))
                if a is not b and xa + wa <= xb and dx > dy:
                    options.append((names[a["category_id"]], names[b["category_id"]]))
        if not options:
            continue
        left, right = rng.choice(options)
        if rng.random() < 0.5:
            caption, rel = make_caption(left, "left", right), "left"
        else:
            caption, rel = make_caption(right, "right", left), "right"
        pairs.append({"image_id": img_id, "file_name": im["file_name"], "caption": caption,
                      "relation": rel, "objects": sorted([left, right])})
    return pairs


def download(pairs, folder, workers=16):
    os.makedirs(folder, exist_ok=True)

    def fetch(p):
        path = os.path.join(folder, p["file_name"])
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(URL.format(p["file_name"]), path)
            return True
        except OSError:
            return False

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(fetch, pairs))
    return [p for p, k in zip(pairs, ok) if k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", default="data/coco_lr")
    ap.add_argument("--max-images", type=int, default=10000)
    ap.add_argument("--min-area", type=float, default=0.03)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--heldout", action="store_true",
                    help="images that pass min_area=0.02 but not the training filter (0.03)")
    a = ap.parse_args()

    if a.heldout:
        train_ids = {p["image_id"] for p in find_pairs(a.ann, a.min_area)}
        pairs = [p for p in find_pairs(a.ann, 0.02, seed=1) if p["image_id"] not in train_ids]
    else:
        pairs = find_pairs(a.ann, a.min_area)
    random.Random(0).shuffle(pairs)
    pairs = download(pairs[:a.max_images], os.path.join(a.out, "images"))

    if a.heldout:
        write_jsonl(f"{a.out}/heldout.jsonl", pairs)
    else:
        n_val = int(len(pairs) * a.val_frac)  # split by image, so an image and its mirror stay together
        write_jsonl(f"{a.out}/val.jsonl", pairs[:n_val])
        write_jsonl(f"{a.out}/train.jsonl", pairs[n_val:])
    print(f"{len(pairs)} images, left {sum(p['relation'] == 'left' for p in pairs)}")


if __name__ == "__main__":
    main()
