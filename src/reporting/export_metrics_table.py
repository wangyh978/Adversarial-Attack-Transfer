from __future__ import annotations

from pathlib import Path
import pandas as pd


def export_pretty_metrics_table(comparison_csv: str | Path, save_csv: str | Path) -> None:
    df = pd.read_csv(comparison_csv)
    keep_cols = ["model_name", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    out_df = df[keep_cols].copy().sort_values(by="f1_macro", ascending=False)

    save_csv = Path(save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_csv, index=False, encoding="utf-8-sig")
