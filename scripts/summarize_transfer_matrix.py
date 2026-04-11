from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, default="nsl_kdd")
    return parser.parse_args()


def infer_tsr(df: pd.DataFrame) -> float:
    if "transfer_success_rate" in df.columns:
        return float(df["transfer_success_rate"].iloc[0])

    for col in ["is_transfer_success", "success", "transfer_success"]:
        if col in df.columns:
            return float(df[col].astype(float).mean())

    raise ValueError(f"Cannot infer transfer_success_rate from columns: {list(df.columns)}")


def main():
    args = parse_args()
    results_dir = Path("results/tables")
    pattern = re.compile(
        rf"transfer_(?P<attack>[a-zA-Z0-9_]+)_{re.escape(args.dataset)}_(?P<target>[a-zA-Z0-9_]+)\.csv$"
    )

    rows = []
    for path in results_dir.glob(f"transfer_*_{args.dataset}_*.csv"):
        m = pattern.match(path.name)
        if not m:
            continue

        attack = m.group("attack").upper()
        target = m.group("target")

        target_display = {
            "tabnet": "TabNet",
            "xgb": "XGB",
            "gbdt": "GBDT",
        }.get(target.lower(), target)

        df = pd.read_csv(path)
        tsr = infer_tsr(df)

        rows.append(
            {
                "attack": attack,
                "target_model": target_display,
                "transfer_success_rate": round(tsr, 4),
            }
        )

    if not rows:
        raise FileNotFoundError("No transfer result files found under results/tables")

    out_df = pd.DataFrame(rows).sort_values(["attack", "target_model"]).reset_index(drop=True)

    out_csv = results_dir / f"final_transfer_matrix_{args.dataset}.csv"
    out_md = results_dir / f"final_transfer_matrix_{args.dataset}.md"

    out_df.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(out_df.to_markdown(index=False))

    print(out_df)
    print(f"saved: {out_csv}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
