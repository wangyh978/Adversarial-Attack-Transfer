from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_report_table(in_csv: str | Path, out_csv: str | Path) -> None:
    df = pd.read_csv(in_csv)
    keep_cols = [
        "seed_size",
        "alpha",
        "depth",
        "accuracy",
        "f1_macro",
        "target_agreement",
        "composite_score",
        "rank",
    ]
    out_df = df[keep_cols].copy().sort_values(by="rank", ascending=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
