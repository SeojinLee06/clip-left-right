#!/usr/bin/env bash
# Everything, from the repository root, on one GPU (a Colab T4 is enough).
set -e
pip install -q -r requirements.txt

# test sets: What'sUp A/B and COCO-spatial, left/right items only
git clone -q https://github.com/amitakamath/whatsup_vlms.git || true
python -m src.build_eval_sets --repo whatsup_vlms

# COCO 2017 train: training pool (E1, E2) and a separate held-out set (E3, E4)
mkdir -p coco && (cd coco && wget -q -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  && unzip -q -n annotations_trainval2017.zip)
ANN=coco/annotations/instances_train2017.json
python -m src.build_coco_pairs --ann $ANN --out data/coco_lr
python -m src.build_coco_pairs --ann $ANN --out data/coco_heldout --heldout --max-images 400

MODELS="openai/clip-vit-base-patch32 openai/clip-vit-large-patch14"

# E1: text fine-tuning
python -m src.exp1_finetune --condition baseline
for c in text_neg text_neg_flipaug flip_pairs; do
  for s in 0 1 2; do python -m src.exp1_finetune --condition $c --seed $s; done
done

# E2: also train the last image block
for arm in frozen unfrozen; do
  for s in 0 1 2; do python -m src.exp2_lastblock --arm $arm --seed $s; done
done
python -m src.summarize results/exp1 results/exp2

# E3: mirror subtraction (no training)
python -m src.exp3_mirror --models $MODELS

# E4: diagnosis and patch read-out (no training)
python -m src.exp4_diagnose --models $MODELS
python -m src.exp4_patch_readout --models $MODELS
