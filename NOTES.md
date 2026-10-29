# Notes

What I tried, in order. Numbers are left/right accuracy (%), from `results/`.

## E1 · Fine-tune the text encoder
**Why.** In What'sUp, adding swapped captions as negatives did not work. The image always matched one of the two captions, so I added the mirrored image, which matches the other one.
**Setup.** ViT-B/32, image encoder frozen, 1,000 steps, 3 seeds. Swapped-caption negatives / plus mirror as plain augmentation (control) / image and mirror in the same batch.
**Result.** All at chance on What'sUp A (49.5, 50.5, 49.5). Training only moved the preference from one word to the other (What'sUp B with swapped negatives: left 29%, right 72%).
**Takeaway.** The setups I tried don't teach left/right. They don't tell me why.

## E2 · Also train the last image block
**Why.** With a frozen image encoder the image side cannot change.
**Result.** +1.1 points on What'sUp A and B, but not consistent across seeds (A: +3.4, +0.5, −0.5).
**Takeaway.** Not enough. I stopped guessing at fixes and decided to measure where the information goes.

## E3 · Subtract the mirrored image
**Why.** CLIP almost always picked the same word for both photos of a pair (94% on What'sUp A). Subtracting the score of the mirrored image cancels any fixed word preference.
**Result.** Same-answer rate dropped to 53–64%, accuracy stayed at chance (49.5 / 51.0). Pair accuracy (17–31%) is no better than two independent guesses (25%).
**Takeaway.** The bias is real, but removing it reveals no left/right signal in the final vectors.
**Mistake.** I first read raw similarity values (0.98, 0.99) as evidence with nothing to compare them to. For E4 I added a reference and fixed the decision rules before running.

## E4 · Diagnose, then read out from patches
**Diagnosis.**
- Linear probe on the final image vector, test object pairs unseen: 60–62%. Little survives pooling.
- Caption vs. its left/right swap is closer than caption vs. a different object, for 100% of captions (0.98–0.99 vs 0.78–0.90).

**Fix.** Before pooling, each patch still has a position. Match patches to the two object names, compare their mean x position. The patch variant was chosen on held-out COCO (plain 26%, value path 74–77%).
**Result.** What'sUp A 49.0 → 74.3 (B/32), 48.5 → 82.5 (L/14). What'sUp B → 99.5 / 100. COCO-spatial → 82–90.
**Takeaway.** Left/right is present in CLIP's patch features and can be read out without training. This does not make CLIP's caption matching itself spatially aware; the comparison rule is mine.
**Open question.** Plain last-layer patch tokens scored 26% on held-out COCO, clearly *below* chance. They carry position, but the name matching reads them the wrong way round. I did not look into this.
