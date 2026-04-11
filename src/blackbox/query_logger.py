from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


def append_query_log(dataset: str, model_name: str, query_size: int, save_path: str | Path = "logs/query_log.csv") -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "model_name": model_name,
        "query_size": int(query_size),
    }
    df_new = pd.DataFrame([row])

    if save_path.exists():
        df_old = pd.read_csv(save_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
