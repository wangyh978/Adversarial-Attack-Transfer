from __future__ import annotations

import pandas as pd


DEFAULT_WEIGHTS = {
    "target_agreement": 0.5,
    "f1_macro": 0.3,
    "accuracy": 0.2,
}


def rank_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No surrogate evaluation rows were found.")

    required_cols = {
        "accuracy",
        "f1_macro",
        "target_agreement",
        "seed_size",
        "alpha",
        "depth",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for surrogate ranking: {missing}")

    out = df.copy()

    for col in ["accuracy", "f1_macro", "target_agreement", "seed_size", "alpha", "depth"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["composite_score"] = (
        DEFAULT_WEIGHTS["target_agreement"] * out["target_agreement"]
        + DEFAULT_WEIGHTS["f1_macro"] * out["f1_macro"]
        + DEFAULT_WEIGHTS["accuracy"] * out["accuracy"]
    )

    # Formal ranking policy:
    # 1) composite_score
    # 2) target_agreement
    # 3) f1_macro
    # 4) accuracy
    # 5) seed_size
    # 6) depth
    out = out.sort_values(
        by=[
            "composite_score",
            "target_agreement",
            "f1_macro",
            "accuracy",
            "seed_size",
            "depth",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    out["rank"] = range(1, len(out) + 1)
    return out
