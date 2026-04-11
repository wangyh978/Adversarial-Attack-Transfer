from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from src.evaluation.surrogate_ranking import rank_surrogates


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path("results/tables") / f"surrogate_batch_eval_{args.dataset}_{args.target_model}.csv"
    df = pd.read_csv(path)
    ranked = rank_surrogates(df)
    out_path = Path("results/tables") / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.csv"
    ranked.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(ranked.head(10))


if __name__ == "__main__":
    main()
