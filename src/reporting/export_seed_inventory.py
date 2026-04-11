from __future__ import annotations

from pathlib import Path
import pandas as pd


def export_seed_inventory(dataset: str, save_path: str | Path) -> None:
    seed_dir = Path("data/seeds") / dataset
    rows = []
    for fp in sorted(seed_dir.glob("seed_*.parquet")):
        df = pd.read_parquet(fp)
        rows.append({
            "file_name": fp.name,
            "num_rows": len(df),
            "num_cols": df.shape[1],
            "duplicates": int(df.duplicated().sum()),
            "missing": int(df.isna().sum().sum()),
        })

    out_df = pd.DataFrame(rows)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_path, index=False, encoding="utf-8-sig")
