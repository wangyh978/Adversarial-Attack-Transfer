from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = Path("results/tables")
    files = list(table_dir.glob(f"surrogate_eval_{args.dataset}_{args.target_model}_*.json"))

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["source_file"] = fp.name
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = table_dir / f"surrogate_summary_{args.dataset}_{args.target_model}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(df.head())


if __name__ == "__main__":
    main()
