"""Caption helpers. Left/right is always from the viewer's side, as in What'sUp."""
import re

LR = re.compile(r"\b(left|right)\b", re.I)
OBJECTS = re.compile(
    r"^an? (.+?) (?:is )?(?:to the (?:left|right) of an? (.+)|on the (?:left|right).*)$", re.I)


def has_lr(caption):
    return bool(LR.search(caption))


def relation(caption):
    m = LR.search(caption)
    return m.group(1).lower() if m else ""


def swap_lr(caption):
    def flip(m):
        w = "right" if m.group(0).lower() == "left" else "left"
        return w.capitalize() if m.group(0)[0].isupper() else w
    return LR.sub(flip, caption)


def group_key(caption):
    """Same key for captions that differ only in left/right."""
    return LR.sub("<lr>", caption.lower()).strip()


def objects(caption):
    """'A mug to the left of a table' -> ('mug', 'table'); 'A dog on the left' -> ('dog', None)."""
    m = OBJECTS.match(caption.strip().rstrip("."))
    return (m.group(1), m.group(2)) if m else (None, None)


def article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def make_caption(subj, rel, obj):
    return f"{article(subj).capitalize()} {subj} to the {rel} of {article(obj)} {obj}"
