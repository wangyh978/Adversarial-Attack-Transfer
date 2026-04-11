from __future__ import annotations

from pathlib import Path
import pandas as pd
import json


def export_best_config(best_csv: str | Path, save_json: str | Path) -> None:
    df = pd.read_csv(best_csv)
    row = df.iloc[0].to_dict()

    save_json = Path(save_json)
    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
