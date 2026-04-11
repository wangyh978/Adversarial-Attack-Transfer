from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path("data") / args.dataset / "processed" / f"{args.dataset}_labeled.parquet"
    df = pd.read_parquet(in_path)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label_id"],
    )

    val_ratio_in_train = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio_in_train,
        random_state=args.random_state,
        stratify=train_df["label_id"],
    )

    out_dir = ensure_dir(Path("data") / args.dataset / "processed")
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    print("train:", train_df.shape, train_df["label_id"].value_counts().to_dict())
    print("val:", val_df.shape, val_df["label_id"].value_counts().to_dict())
    print("test:", test_df.shape, test_df["label_id"].value_counts().to_dict())


if __name__ == "__main__":
    main()
