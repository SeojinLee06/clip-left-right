"""E1. Fine-tune the text encoder on COCO left/right pairs (image encoder frozen).

    baseline          no training
    text_neg          swapped caption added as a hard negative for each image
    text_neg_flipaug  text_neg, but each image is replaced by its mirror (and the swapped
                      caption) with p=0.5. Control: sees mirrored images, never as a pair.
    flip_pairs        image and its mirror in the same batch, each with its own caption

All trained conditions use the same steps, batch size and 2N captions per step.

    python -m src.exp1_finetune --condition flip_pairs --seed 0
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
import yaml

from .build_eval_sets import load_items
from .caption_eval import evaluate
from .clip_utils import encode_images, encode_texts, load_clip, train_text_only
from .metrics import read_jsonl, write_jsonl
from .text import swap_lr

CONDITIONS = ["baseline", "text_neg", "text_neg_flipaug", "flip_pairs"]


def coco_embeddings(model, proc, pairs, folder, device, cache):
    if os.path.exists(cache):
        return torch.load(cache)
    paths = [os.path.join(folder, p["file_name"]) for p in pairs]
    emb = (encode_images(model, proc, paths, device), encode_images(model, proc, paths, device, mirror=True))
    torch.save(emb, cache)
    return emb


def loss_fn(condition, model, proc, caps, img, img_flip, device, scale, gen):
    n = len(caps)
    swapped = [swap_lr(c) for c in caps]
    if condition == "text_neg_flipaug":
        use_flip = torch.rand(n, generator=gen) < 0.5
        img = torch.where(use_flip[:, None], img_flip, img)
        caps, swapped = ([s if f else c for c, s, f in zip(caps, swapped, use_flip.tolist())],
                         [c if f else s for c, s, f in zip(caps, swapped, use_flip.tolist())])
    text = encode_texts(model, proc, caps + swapped, device, grad=True)

    if condition == "flip_pairs":
        img = torch.cat([img, img_flip]).to(device)
        logits = scale * img @ text.T
        target = torch.arange(2 * n, device=device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2

    img = img.to(device)
    target = torch.arange(n, device=device)
    i2t = F.cross_entropy(scale * img @ text.T, target)        # swapped captions act as negatives
    t2i = F.cross_entropy(scale * text[:n] @ img.T, target)
    return (i2t + t2i) / 2


@torch.no_grad()
def val_acc(model, proc, pairs, img, img_flip, device):
    caps = [p["caption"] for p in pairs]
    t = encode_texts(model, proc, caps, device)
    ts = encode_texts(model, proc, [swap_lr(c) for c in caps], device)
    orig = ((img * t).sum(-1) > (img * ts).sum(-1)).float().mean().item()
    flip = ((img_flip * ts).sum(-1) > (img_flip * t).sum(-1)).float().mean().item()
    return orig, flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = a.condition if a.condition == "baseline" else f"{a.condition}_seed{a.seed}"
    out = os.path.join(cfg["results_dir"], "exp1", run)
    os.makedirs(out, exist_ok=True)
    model, proc = load_clip(cfg["model"], device)
    items = load_items(cfg["eval_items"])

    log = []
    if a.condition != "baseline":
        d = cfg["coco_dir"]
        train, val = read_jsonl(f"{d}/train.jsonl"), read_jsonl(f"{d}/val.jsonl")
        tr_img, tr_flip = coco_embeddings(model, proc, train, f"{d}/images", device, f"{d}/train_emb.pt")
        va_img, va_flip = coco_embeddings(model, proc, val, f"{d}/images", device, f"{d}/val_emb.pt")

        params = train_text_only(model)
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps, warm = cfg["steps"], cfg["warmup"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        scale = model.logit_scale.exp().detach()
        model.train()
        for step in range(1, steps + 1):
            idx = torch.randperm(len(train), generator=gen)[:cfg["batch_size"]]
            caps = [train[i]["caption"] for i in idx.tolist()]
            loss = loss_fn(a.condition, model, proc, caps, tr_img[idx], tr_flip[idx], device, scale, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % cfg["log_every"] == 0:
                model.eval()
                orig, flip = val_acc(model, proc, val, va_img, va_flip, device)
                model.train()
                log.append({"step": step, "loss": loss.item(), "val_acc": orig, "val_acc_mirrored": flip})
                print(f"[{run}] step {step} loss {loss.item():.3f} val {orig:.3f} / mirrored {flip:.3f}")
        model.eval()
        write_jsonl(f"{out}/train_log.jsonl", log)

    metrics, df = evaluate(model, proc, items, device)
    df.to_csv(f"{out}/predictions.csv", index=False)
    json.dump({"run": run, "config": cfg, "metrics": metrics}, open(f"{out}/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
