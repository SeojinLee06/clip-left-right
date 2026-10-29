"""E2. Same as E1 flip_pairs, but optionally also train the last vision transformer block.

The only difference between the two arms is whether the last block is trainable.
For speed, the input to the last block is cached once (float32) and shared by both arms;
the cached path is checked to reproduce CLIP's own output.

    python -m src.exp2_lastblock --arm unfrozen --seed 0
"""
import argparse
import inspect
import json
import math
import os
import random

import torch
import torch.nn.functional as F
import yaml

from .build_eval_sets import load_items
from .caption_eval import choose
from .clip_utils import encode_texts, load_clip, load_images, pooled, train_text_only
from .metrics import read_jsonl, summarize, write_jsonl
from .text import swap_lr


def last_block(model, h):
    block = model.vision_model.encoder.layers[-1]
    accepted = inspect.signature(block.forward).parameters
    out = block(h, **{k: None for k in ("attention_mask", "causal_attention_mask") if k in accepted})
    out = out[0] if isinstance(out, (tuple, list)) else out
    out = model.vision_model.post_layernorm(out[:, 0, :])
    return F.normalize(model.visual_projection(out).float(), dim=-1)


@torch.no_grad()
def cache_block_input(model, proc, paths, device, cache, bs=96):
    if os.path.exists(cache):
        return torch.load(cache)
    captured = []
    hook = model.vision_model.encoder.layers[-1].register_forward_pre_hook(
        lambda m, args, kwargs: captured.append((args[0] if args else kwargs["hidden_states"]).float().cpu()),
        with_kwargs=True)
    out = []
    for mirror in (False, True):
        chunks = []
        for i in range(0, len(paths), bs):
            px = proc(images=load_images(paths[i:i + bs], mirror), return_tensors="pt")["pixel_values"].to(device)
            direct = F.normalize(pooled(model.get_image_features(pixel_values=px)).float(), dim=-1)
            h = captured.pop()
            if i == 0:
                assert (direct - last_block(model, h.to(device))).abs().max() < 1e-5
            chunks.append(h)
        out.append(torch.cat(chunks))
    hook.remove()
    torch.save(out, cache)
    return out


@torch.no_grad()
def embed(model, h, device, bs=128):
    return torch.cat([last_block(model, h[i:i + bs].to(device)).cpu() for i in range(0, len(h), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arm", choices=["frozen", "unfrozen"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    device = "cuda"

    model, proc = load_clip(cfg["model"], device)
    d = cfg["coco_dir"]
    train, val = read_jsonl(f"{d}/train.jsonl"), read_jsonl(f"{d}/val.jsonl")
    items = load_items(cfg["eval_items"])
    os.makedirs("data/prefix", exist_ok=True)
    tr = cache_block_input(model, proc, [f"{d}/images/{p['file_name']}" for p in train], device, "data/prefix/train.pt")
    va = cache_block_input(model, proc, [f"{d}/images/{p['file_name']}" for p in val], device, "data/prefix/val.pt")
    ev = cache_block_input(model, proc, [r["image"] for r in items], device, "data/prefix/eval.pt")

    params = train_text_only(model)
    if a.arm == "unfrozen":
        block = list(model.vision_model.encoder.layers[-1].parameters())
        for p in block:
            p.requires_grad = True
        params += block
    fixed = None if a.arm == "unfrozen" else (embed(model, tr[0], device), embed(model, tr[1], device))

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps, warm = cfg["steps"], cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scale = model.logit_scale.exp().detach()
    run = f"{a.arm}_seed{a.seed}"
    out = os.path.join(cfg["results_dir"], "exp2", run)
    os.makedirs(out, exist_ok=True)

    log = []
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randperm(len(train), generator=gen)[:cfg["batch_size"]]
        caps = [train[i]["caption"] for i in idx.tolist()]
        text = encode_texts(model, proc, caps + [swap_lr(c) for c in caps], device, grad=True)
        if fixed is None:
            img = last_block(model, torch.cat([tr[0][idx], tr[1][idx]]).to(device))
        else:
            img = torch.cat([fixed[0][idx], fixed[1][idx]]).to(device)
        logits = scale * img @ text.T
        target = torch.arange(2 * len(caps), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0:
            model.eval()
            with torch.no_grad():
                t = encode_texts(model, proc, [p["caption"] for p in val], device)
                ts = encode_texts(model, proc, [swap_lr(p["caption"]) for p in val], device)
                vo, vf = embed(model, va[0], device), embed(model, va[1], device)
            log.append({"step": step, "loss": loss.item(),
                        "val_acc": ((vo * t).sum(-1) > (vo * ts).sum(-1)).float().mean().item(),
                        "val_acc_mirrored": ((vf * ts).sum(-1) > (vf * t).sum(-1)).float().mean().item()})
            print(run, log[-1])

    model.eval()
    df = choose(model, proc, items, embed(model, ev[0], device), embed(model, ev[1], device), device)
    write_jsonl(f"{out}/train_log.jsonl", log)
    df.to_csv(f"{out}/predictions.csv", index=False)
    json.dump({"run": run, "trainable_params": sum(p.numel() for p in params), "metrics": summarize(df)},
              open(f"{out}/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
