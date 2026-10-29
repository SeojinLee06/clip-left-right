"""Standard CLIP evaluation: pick the caption with the higher image-text similarity.

The image encoder is never trained here, so image embeddings (original and mirrored)
are computed once and cached. On the mirrored image the swapped caption is the right one,
which gives flip_consistency.
"""
import os

import torch

from .clip_utils import encode_images, encode_texts
from .metrics import as_frame, summarize


def image_embeddings(model, proc, items, device, cache):
    ids = [r["id"] for r in items]
    if os.path.exists(cache):
        d = torch.load(cache)
        if d["ids"] == ids:
            return d["orig"], d["flip"]
    paths = [r["image"] for r in items]
    orig = encode_images(model, proc, paths, device)
    flip = encode_images(model, proc, paths, device, mirror=True)
    torch.save({"ids": ids, "orig": orig, "flip": flip}, cache)
    return orig, flip


def choose(model, proc, items, img, img_flip, device):
    t_right = encode_texts(model, proc, [r["correct"] for r in items], device)
    t_wrong = encode_texts(model, proc, [r["wrong"] for r in items], device)
    ok = (img * t_right).sum(-1) > (img * t_wrong).sum(-1)
    flip_ok = (img_flip * t_wrong).sum(-1) > (img_flip * t_right).sum(-1)
    other = {"left": "right", "right": "left"}
    preds = [r["relation"] if k else other[r["relation"]] for r, k in zip(items, ok.tolist())]
    return as_frame(items, preds, flip_ok=flip_ok.tolist())


def evaluate(model, proc, items, device, cache="data/eval_lr/img_emb.pt"):
    img, img_flip = image_embeddings(model, proc, items, device, cache)
    df = choose(model, proc, items, img, img_flip, device)
    return summarize(df), df
