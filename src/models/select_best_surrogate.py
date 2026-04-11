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
    in_path = Path("results/tables") / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.csv"
    df = pd.read_csv(in_path)
    ranked = rank_surrogates(df)
    best = ranked.iloc[:1]

    out_path = Path("results/tables") / f"best_surrogate_{args.dataset}_{args.target_model}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(best.T)


if __name__ == "__main__":
    main()
