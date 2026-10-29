"""Mean ± std over seeds for E1 and E2 (What'sUp and COCO-spatial accuracy, %).

    python -m src.summarize results/exp1 results/exp2
"""
import glob
import json
import re
import sys

import pandas as pd


def table(folder):
    rows = []
    for path in glob.glob(f"{folder}/*/metrics.json"):
        d = json.load(open(path))
        cond = re.sub(r"_seed\d+$", "", d["run"])
        for src, m in d["metrics"].items():
            rows.append({"condition": cond, "source": src, "acc": 100 * m["acc"], "pair_acc": 100 * m.get("pair_acc", float("nan"))})
    df = pd.DataFrame(rows)
    g = df.groupby(["condition", "source"])
    t = (g.mean().round(1).astype(str) + " ± " + g.std().fillna(0).round(1).astype(str)).unstack("source")
    t.columns = [f"{src} {metric}" for metric, src in t.columns]
    return t[sorted(t.columns)]


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        print(f"\n{folder}\n{table(folder).to_markdown()}")
