from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path("results/tables")
    files = list(metrics_dir.glob(f"*_{args.dataset}_metrics.json"))

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["source_file"] = fp.name
        if "model_name" not in row:
            row["model_name"] = fp.name.replace(f"_{args.dataset}_metrics.json", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = metrics_dir / f"model_comparison_{args.dataset}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(df[["model_name", "accuracy", "precision_macro", "recall_macro", "f1_macro"]])


if __name__ == "__main__":
    main()
