from __future__ import annotations

import pandas as pd


def compare_by_setting(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    metrics = ["accuracy", "f1_macro", "target_agreement"]
    return df.groupby(setting)[metrics].mean().reset_index()
