from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np

from src.blackbox.query_api import BlackBoxModel
from src.blackbox.query_batch import batched_predict_label
from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_path = Path("data/seeds") / args.dataset / f"seed_{args.seed_size}.parquet"
    df = pd.read_parquet(seed_path)

    feature_cols = [c for c in df.columns if str(c).startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    blackbox = BlackBoxModel(args.dataset, args.target_model)
    y_blackbox = batched_predict_label(blackbox, X, batch_size=256)

    out = df.copy()
    out["blackbox_label"] = y_blackbox

    out_dir = ensure_dir(Path("data/seeds") / args.dataset / "queried")
    out_path = out_dir / f"{args.target_model}_seed_{args.seed_size}_queried.parquet"
    out.to_parquet(out_path, index=False)

    print(out[["label_id", "blackbox_label"]].head())
    print("saved:", out_path)


if __name__ == "__main__":
    main()
