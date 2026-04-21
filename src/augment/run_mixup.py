from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.blackbox.query_api import BlackBoxModel
from src.blackbox.query_batch import batched_predict_label
from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser(
        description="Paper-style mixup augmentation for NIDS: generate X~ and relabel by black-box A.predict(X~)."
    )
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional parquet input for iterative MSM. Defaults to queried seed set.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output parquet path. Defaults to data/mixup/{dataset}/...",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def _resolve_input_path(args) -> Path:
    if args.input_path:
        return Path(args.input_path)
    return (
        Path("data/seeds")
        / args.dataset
        / "queried"
        / f"{args.target_model}_seed_{args.seed_size}_queried.parquet"
    )


def _resolve_output_path(args) -> Path:
    if args.output_path:
        return Path(args.output_path)
    out_dir = ensure_dir(Path("data/mixup") / args.dataset)
    return out_dir / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}.parquet"


def _pick_label_column(df: pd.DataFrame) -> str:
    for candidate in ("target_label", "blackbox_label", "label_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No usable label column found. Expected one of: target_label, blackbox_label, label_id")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    in_path = _resolve_input_path(args)
    out_path = _resolve_output_path(args)

    df = pd.read_parquet(in_path)
    feature_cols = [c for c in df.columns if str(c).startswith("f_")]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {in_path}")

    label_col = _pick_label_column(df)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy()

    n = len(df)
    idx_a = rng.integers(0, n, size=n)
    idx_b = rng.integers(0, n, size=n)
    lam = rng.beta(args.alpha, args.alpha, size=n).astype(np.float32)

    X_a = X[idx_a]
    X_b = X[idx_b]
    X_mix = lam[:, None] * X_a + (1.0 - lam[:, None]) * X_b

    # Paper method: use hard label from black-box model instead of mixup soft labels.
    blackbox = BlackBoxModel(args.dataset, args.target_model)
    y_mix = batched_predict_label(blackbox, X_mix, batch_size=args.batch_size)

    mix_df = pd.DataFrame(X_mix, columns=feature_cols)
    mix_df["target_label"] = y_mix
    mix_df["data_source"] = "mixup_blackbox"
    mix_df["parent_idx_a"] = idx_a
    mix_df["parent_idx_b"] = idx_b
    mix_df["parent_label_a"] = y[idx_a]
    mix_df["parent_label_b"] = y[idx_b]
    mix_df["lam"] = lam

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mix_df.to_parquet(out_path, index=False)

    print(f"loaded: {in_path}")
    print(f"saved:  {out_path}")
    print(mix_df.head())


if __name__ == "__main__":
    main()
