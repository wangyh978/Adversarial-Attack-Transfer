from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser(
        description="Build surrogate training data using hard labels for both queried seed and paper-style mixup samples."
    )
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--mixup_path",
        type=str,
        default=None,
        help="Optional override for mixup parquet path.",
    )
    return parser.parse_args()


def _normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    if "target_label" in df.columns:
        return df
    if "blackbox_label" in df.columns:
        return df.rename(columns={"blackbox_label": "target_label"})
    raise ValueError("Expected target_label or blackbox_label in dataframe.")


def main() -> None:
    args = parse_args()

    seed_path = (
        Path("data/seeds")
        / args.dataset
        / "queried"
        / f"{args.target_model}_seed_{args.seed_size}_queried.parquet"
    )
    mix_path = (
        Path(args.mixup_path)
        if args.mixup_path
        else Path("data/mixup")
        / args.dataset
        / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}.parquet"
    )

    seed_df = _normalize_label_column(pd.read_parquet(seed_path))
    mix_df = _normalize_label_column(pd.read_parquet(mix_path))

    feature_cols = [c for c in seed_df.columns if str(c).startswith("f_")]
    keep_cols = feature_cols + ["target_label"]

    seed_train = seed_df[keep_cols].copy()
    seed_train["data_source"] = "seed"

    mix_train = mix_df[keep_cols].copy()
    if "data_source" not in mix_df.columns:
        mix_train["data_source"] = "mixup_blackbox"
    else:
        mix_train["data_source"] = mix_df["data_source"].values

    union_train = pd.concat([seed_train, mix_train], axis=0, ignore_index=True)

    out_dir = ensure_dir(Path("data/surrogate_train") / args.dataset)

    seed_out = out_dir / f"{args.target_model}_seed_{args.seed_size}_seed_only.parquet"
    mix_out = out_dir / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}_mixup.parquet"
    union_out = out_dir / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}_paper_union.parquet"

    seed_train.to_parquet(seed_out, index=False)
    mix_train.to_parquet(mix_out, index=False)
    union_train.to_parquet(union_out, index=False)

    print("saved:", seed_out)
    print("saved:", mix_out)
    print("saved:", union_out)
    print("rows:", {"seed": len(seed_train), "mixup": len(mix_train), "union": len(union_train)})


if __name__ == "__main__":
    main()
