# Where does CLIP lose left and right?

CLIP cannot tell *"a mug to the left of a table"* from *"a mug to the right of a table"*. What'sUp (Kamath et al., EMNLP 2023) showed this and also found that fine-tuning with left/right-swapped captions does not fix it.

I started from that result. I first tried to fix it with training, which did not work. I then checked where the left/right information disappears, and read it out from the part of CLIP where it still exists. That reached **74–100%** on left/right questions where CLIP is at chance, with no training.

## Task

Two captions that differ only in *left* / *right*. Pick the one that matches the image (chance = 50%).
Test sets: the left/right items of What'sUp A, What'sUp B and COCO-spatial (1 or 2 objects), 1,829 images in total.

## 1. Training did not help (E1, E2)

| | What'sUp A | What'sUp B |
|---|---|---|
| CLIP ViT-B/32 | 49.0 | 52.9 |
| + text fine-tuning, swapped caption as negative | 49.5 ± 0.8 | 50.7 ± 1.5 |
| + image and its mirror in the same batch | 49.5 ± 0.5 | 48.9 ± 1.6 |
| + also train the last image block | 50.6 ± 1.6 | 50.0 ± 0.5 |

Mean ± std over 3 seeds. Training only changed which word CLIP prefers. Accuracy also stayed near 50% on held-out COCO images from the training distribution, so this is not just a domain gap.

## 2. Where is it lost? (E4)

| Check | B/32 | L/14 |
|---|---|---|
| Linear probe on CLIP's final **image vector**, unseen object pairs (chance 50%) | 62 / 60 | 60 / 59 |
| Similarity of a caption to its **left/right-swapped** version | 0.99 | 0.98 |
| Similarity of a caption to a **different-object** version | 0.82–0.90 | 0.78–0.85 |

(probe: What'sUp A / B)

- The final image vector keeps very little left/right information.
- For every caption, the text encoder places the left/right swap closer than a change of object.

Both vectors that CLIP compares have lost most of the signal. A ViT, however, holds one feature **per image patch** before it pools them, and every patch has a fixed position.

## 3. Read it out before pooling (E4)

1. Take CLIP's patch features (value path of the last attention layer, as in MaskCLIP).
2. Match each patch to the two object names in the caption.
3. Compare the objects' average x position.

The patch-feature variant was chosen on 400 held-out COCO images (plain tokens 26%, value path 74–77%). The test sets were evaluated once.

| Accuracy (%) | CLIP B/32 | **Read-out B/32** | CLIP L/14 | **Read-out L/14** |
|---|---|---|---|---|
| What'sUp A | 49.0 | **74.3** | 48.5 | **82.5** |
| What'sUp B | 52.9 | **99.5** | 52.9 | **100.0** |
| COCO-spatial, 1 object | 50.9 | **83.7** | 51.0 | **90.4** |
| COCO-spatial, 2 objects | 48.7 | **81.7** | 47.3 | **85.7** |
| What'sUp A, both photos of a pair right | 1.9 | **50.5** | 1.9 | **67.0** |

**Control (E3):** subtracting each image's mirrored version removes the word bias but leaves accuracy at chance (What'sUp A: 49.5 B/32, 51.0 L/14). Removing bias alone is not what helps.

## Limitations

- The final decision (compare x positions of two named objects) is a hand-written rule, so it only covers left/right and needs both object names. CLIP's own caption matching is unchanged.
- The patch read-out comes from MaskCLIP. This project adds the diagnosis that led to it and its evaluation for left/right.
- 2-way scores on the left/right subset. They are not comparable to the 4-way scores in the What'sUp paper.
- Two CLIP models and four test sets. Other model families were not tested.

## Run

```bash
bash scripts/run_all.sh        # one GPU; E3 + E4 take about 20 min on a Colab T4
```

| File | |
|---|---|
| `src/build_eval_sets.py`, `src/build_coco_pairs.py` | test sets, COCO training pairs, held-out set |
| `src/exp1_finetune.py`, `src/exp2_lastblock.py` | training experiments |
| `src/exp3_mirror.py` | mirror-subtraction control |
| `src/exp4_diagnose.py`, `src/exp4_patch_readout.py` | diagnosis and read-out |
| `results/` | metrics of every run · [`NOTES.md`](NOTES.md): what I tried and why |

## References

- A. Kamath, J. Hessel, K.-W. Chang. *What's "up" with vision-language models? Investigating their struggle with spatial reasoning.* EMNLP 2023. [code](https://github.com/amitakamath/whatsup_vlms)
- C. Zhou, C. C. Loy, B. Dai. *Extract Free Dense Labels from CLIP.* ECCV 2022.
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
