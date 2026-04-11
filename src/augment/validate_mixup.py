from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mixup_file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(Path(args.mixup_file))
    print("shape:", df.shape)
    print("missing:", int(df.isna().sum().sum()))
    print("lam min:", float(df["lam"].min()))
    print("lam max:", float(df["lam"].max()))
    print(df[["y_a", "y_b", "lam"]].head())


if __name__ == "__main__":
    main()
