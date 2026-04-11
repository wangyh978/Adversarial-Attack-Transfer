from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--seed_size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path("data/seeds") / args.dataset / f"seed_{args.seed_size}.parquet"
    df = pd.read_parquet(path)
    print("shape:", df.shape)
    print("duplicates:", int(df.duplicated().sum()))
    print("missing:", int(df.isna().sum().sum()))
    print("label distribution:")
    print(df["label_id"].value_counts().sort_index())


if __name__ == "__main__":
    main()
