import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


def load_clip(name="openai/clip-vit-base-patch32", device="cuda"):
    return CLIPModel.from_pretrained(name).to(device).eval(), CLIPProcessor.from_pretrained(name)


def pooled(out):
    # newer transformers versions return an output object instead of a tensor
    return out if torch.is_tensor(out) else out.pooler_output


def train_text_only(model):
    """Freeze everything except the text encoder and text projection."""
    for p in model.parameters():
        p.requires_grad = False
    params = list(model.text_model.parameters()) + list(model.text_projection.parameters())
    for p in params:
        p.requires_grad = True
    return params


def load_images(paths, mirror=False):
    images = [Image.open(p).convert("RGB") for p in paths]
    return [ImageOps.mirror(im) for im in images] if mirror else images


@torch.no_grad()
def encode_images(model, proc, paths, device, mirror=False, bs=128):
    out = []
    for i in range(0, len(paths), bs):
        px = proc(images=load_images(paths[i:i + bs], mirror), return_tensors="pt")["pixel_values"]
        f = pooled(model.get_image_features(pixel_values=px.to(device)))
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


def encode_texts(model, proc, texts, device, grad=False, bs=256):
    out = []
    with torch.set_grad_enabled(grad):
        for i in range(0, len(texts), bs):
            tok = proc(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True)
            f = pooled(model.get_text_features(**tok.to(device)))
            out.append(F.normalize(f.float(), dim=-1))
    out = torch.cat(out)
    return out if grad else out.cpu()
