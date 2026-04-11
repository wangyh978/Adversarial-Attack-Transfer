from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from src.data.schema import summarize_schema
from src.utils.io import ensure_dir


def profile_dataframe(df: pd.DataFrame) -> dict:
    return {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "num_duplicates": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path("data") / args.dataset / "processed" / f"{args.dataset}_labeled.parquet"
    df = pd.read_parquet(in_path)

    out_dir = ensure_dir(Path("results/tables"))
    summarize_schema(df).to_csv(out_dir / f"{args.dataset}_schema.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([profile_dataframe(df)]).to_csv(
        out_dir / f"{args.dataset}_profile.csv", index=False, encoding="utf-8-sig"
    )

    print(profile_dataframe(df))


if __name__ == "__main__":
    main()
