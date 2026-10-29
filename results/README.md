Metrics from the runs reported in the main README (accuracies as fractions, 0–1).
The JSON files were reorganised into the layout the current scripts write; the numbers are unchanged.
Per-image prediction CSVs are not committed. `bash scripts/run_all.sh` regenerates them.

| Folder | Script |
|---|---|
| `exp1/` | `src/exp1_finetune.py` |
| `exp2/` | `src/exp2_lastblock.py` |
| `exp3/` | `src/exp3_mirror.py` |
| `exp4/` | `src/exp4_diagnose.py`, `src/exp4_patch_readout.py` |
