from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from src.data.merge_seed_query_results import merge_query_results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = merge_query_results(args.dataset, args.target_model)
    df["is_match"] = df["label_id"] == df["blackbox_label"]

    summary = df.groupby("source_file").agg(
        num_rows=("is_match", "size"),
        match_rate=("is_match", "mean"),
    ).reset_index()

    out_path = Path("results/tables") / f"{args.target_model}_{args.dataset}_seed_query_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
