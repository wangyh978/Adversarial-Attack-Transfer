from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from src.data.sample_by_class import sample_by_class
from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--seed_size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_features_path = Path("data") / args.dataset / "processed" / "train_features.parquet"
    if not train_features_path.exists():
        raise FileNotFoundError(f"{train_features_path} 不存在，请先运行预处理流水线")

    df = pd.read_parquet(train_features_path)
    seed_df = sample_by_class(df, label_col="label_id", total_size=args.seed_size, random_state=42)

    out_dir = ensure_dir(Path("data/seeds") / args.dataset)
    out_path = out_dir / f"seed_{args.seed_size}.parquet"
    seed_df.to_parquet(out_path, index=False)

    print(seed_df["label_id"].value_counts().sort_index())
    print("saved:", out_path)


if __name__ == "__main__":
    main()
