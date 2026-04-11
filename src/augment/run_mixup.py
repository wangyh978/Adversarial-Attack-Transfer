from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np

from src.augment.mixup import mixup_features
from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path("data/seeds") / args.dataset / "queried" / f"{args.target_model}_seed_{args.seed_size}_queried.parquet"
    df = pd.read_parquet(in_path)
    feature_cols = [c for c in df.columns if str(c).startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["blackbox_label"].to_numpy()

    X_mix, y_a, y_b, lam = mixup_features(X, y, alpha=args.alpha)

    mix_df = pd.DataFrame(X_mix, columns=feature_cols)
    mix_df["y_a"] = y_a
    mix_df["y_b"] = y_b
    mix_df["lam"] = lam
    mix_df["seed_size"] = args.seed_size
    mix_df["alpha"] = args.alpha
    mix_df["target_model"] = args.target_model

    out_dir = ensure_dir(Path("data/mixup") / args.dataset)
    out_path = out_dir / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}.parquet"
    mix_df.to_parquet(out_path, index=False)

    print("saved:", out_path)
    print(mix_df.head())


if __name__ == "__main__":
    main()
