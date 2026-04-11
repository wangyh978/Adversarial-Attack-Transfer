from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
import re


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def parse_meta_from_name(file_name: str) -> dict:
    m = re.search(r"seed(\d+)_a([0-9.]+)_d(\d+)", file_name)
    if not m:
        return {}
    return {"seed_size": int(m.group(1)), "alpha": float(m.group(2)), "depth": int(m.group(3))}


def main() -> None:
    args = parse_args()
    table_dir = Path("results/tables")
    files = list(table_dir.glob(f"surrogate_eval_{args.dataset}_{args.target_model}_*.json"))
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            row = json.load(f)
        row.update(parse_meta_from_name(fp.name))
        row["source_file"] = fp.name
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = table_dir / f"surrogate_batch_eval_{args.dataset}_{args.target_model}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(df.head())


if __name__ == "__main__":
    main()
