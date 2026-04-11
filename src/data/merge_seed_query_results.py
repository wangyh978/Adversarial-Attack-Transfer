from __future__ import annotations

from pathlib import Path
import pandas as pd


def merge_query_results(dataset: str, target_model: str) -> pd.DataFrame:
    query_dir = Path("data/seeds") / dataset / "queried"
    files = sorted(query_dir.glob(f"{target_model}_seed_*_queried.parquet"))
    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        df["source_file"] = fp.name
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No queried seed files found.")
    return pd.concat(dfs, axis=0, ignore_index=True)
