from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir


def split_nsl_kdd(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"],
    )
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val["label_id"],
    )
    return train_df, val_df, test_df


def split_unsw_nb15(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split_source" in df.columns and set(df["split_source"].unique()) & {"train_official", "test_official"}:
        train_pool = df[df["split_source"] == "train_official"].copy()
        test_df = df[df["split_source"] == "test_official"].copy()

        if train_pool.empty or test_df.empty:
            raise ValueError("UNSW-NB15 官方 train/test 切分为空，请检查原始文件名或 split_source 列")

        train_df, val_df = train_test_split(
            train_pool,
            test_size=val_size,
            random_state=random_state,
            stratify=train_pool["label_id"],
        )
        return train_df, val_df, test_df

    # fallback
    return split_nsl_kdd(df, test_size=0.2, val_size=0.1, random_state=random_state)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path("data") / args.dataset / "processed" / f"{args.dataset}_labeled.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {in_path}")

    df = pd.read_parquet(in_path)

    if args.dataset == "nsl_kdd":
        train_df, val_df, test_df = split_nsl_kdd(
            df,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    else:
        train_df, val_df, test_df = split_unsw_nb15(
            df,
            val_size=args.val_size,
            random_state=args.random_state,
        )

    out_dir = ensure_dir(Path("data") / args.dataset / "processed")
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("saved:", train_path, train_df.shape)
    print("saved:", val_path, val_df.shape)
    print("saved:", test_path, test_df.shape)


if __name__ == "__main__":
    main()
