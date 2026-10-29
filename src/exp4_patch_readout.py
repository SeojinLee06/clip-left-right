"""E4b. Read left/right from patch features, before CLIP pools the image (no training).

1. Get one feature per image patch in CLIP's joint image-text space.
2. Assign each patch softly to one of the two object names in the caption.
3. Compare the objects' mean x positions: x(subject) < x(object) -> "left".
   With one object ("A dog on the left") its position is compared with the image centre.

Two ways to get patch features are compared on held-out COCO only, and the better one is used:
    plain  last-layer patch tokens
    value  value path of the last attention layer (MaskCLIP, Zhou et al., ECCV 2022)

    python -m src.exp4_patch_readout --models openai/clip-vit-base-patch32 openai/clip-vit-large-patch14
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from .build_eval_sets import load_items
from .clip_utils import encode_texts, load_clip, load_images
from .heldout import load_heldout
from .metrics import as_frame, summarize
from .text import objects


@torch.no_grad()
def patch_features(model, proc, paths, device, variant, bs=64):
    vision = model.vision_model
    out = []
    for i in range(0, len(paths), bs):
        px = proc(images=load_images(paths[i:i + bs]), return_tensors="pt")["pixel_values"].to(device)
        hidden = vision(pixel_values=px, output_hidden_states=True).hidden_states
        if variant == "plain":
            x = hidden[-1]
        else:
            last = vision.encoder.layers[-1]
            x = last.self_attn.out_proj(last.self_attn.v_proj(last.layer_norm1(hidden[-2])))
        x = F.normalize(model.visual_projection(vision.post_layernorm(x[:, 1:])).float(), dim=-1)
        side = int(round(x.shape[1] ** 0.5))
        out.append(x.view(len(x), side, side, -1).cpu())
    return torch.cat(out)


def x_positions(patches, names, text_emb, temp=100.0):
    _, h, w, d = patches.shape
    xs = torch.linspace(0, 1, w)
    out = []
    for p, (subj, obj) in zip(patches, names):
        sim = p.reshape(-1, d) @ text_emb.T
        if obj is not None:
            weight = torch.softmax(temp * sim[:, [subj, obj]], dim=-1).view(h, w, 2)
            col = weight.sum(0)                                     # (w, 2) mass per column
            out.append(((col[:, 0] * xs).sum() / col[:, 0].sum()).item() <
                       ((col[:, 1] * xs).sum() / col[:, 1].sum()).item())
        else:
            weight = torch.softmax(temp * sim[:, subj], dim=0).view(h, w)
            out.append((weight.sum(0) * xs).sum().item() < 0.5)
    return ["left" if is_left else "right" for is_left in out]


def predict(model, proc, items, device, variant):
    names = [objects(r["correct"]) for r in items]
    vocab = sorted({n for pair in names for n in pair if n})
    index = {n: i for i, n in enumerate(vocab)}
    text_emb = encode_texts(model, proc, [f"a photo of a {n}." for n in vocab], device)
    patches = patch_features(model, proc, [r["image"] for r in items], device, variant)
    ids = [(index[s], index[o] if o else None) for s, o in names]
    return as_frame(items, x_positions(patches, ids, text_emb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch32"])
    ap.add_argument("--out", default="results/exp4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, heldout = load_items(), load_heldout()
    assert all(objects(r["correct"])[0] for r in items + heldout), "unparsed caption"

    report = {}
    for name in a.models:
        tag = name.split("/")[-1]
        model, proc = load_clip(name, device)
        selection = {v: summarize(predict(model, proc, heldout, device, v))["coco_heldout"]["acc"]
                     for v in ("plain", "value")}
        best = max(selection, key=selection.get)
        df = predict(model, proc, items, device, best)
        df.to_csv(f"{a.out}/predictions_{tag}.csv", index=False)
        report[tag] = {"selection_on_heldout": selection, "chosen": best, "test": summarize(df)}
        print(tag, best, {k: round(100 * v["acc"], 1) for k, v in report[tag]["test"].items()})
    json.dump(report, open(f"{a.out}/patch_readout.json", "w"), indent=2)


if __name__ == "__main__":
    main()
