"""Held-out COCO items (src.build_coco_pairs --heldout) in the same format as the test items."""
import os

from .metrics import read_jsonl
from .text import swap_lr


def load_heldout(path="data/coco_heldout/heldout.jsonl", folder="data/coco_heldout/images"):
    items = []
    for p in read_jsonl(path):
        image = os.path.join(folder, p["file_name"])
        if os.path.exists(image):
            items.append({"id": str(p["image_id"]), "source": "coco_heldout", "group": f"coco|{p['image_id']}",
                          "image": image, "correct": p["caption"], "wrong": swap_lr(p["caption"]),
                          "relation": p["relation"]})
    return items
