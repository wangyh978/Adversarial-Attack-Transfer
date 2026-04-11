from __future__ import annotations

import pandas as pd


def rank_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["composite_score"] = (
        0.6 * out["target_agreement"] +
        0.3 * out["f1_macro"] +
        0.1 * out["accuracy"]
    )
    out = out.sort_values(
        by=["composite_score", "target_agreement", "f1_macro"],
        ascending=False,
    ).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out
