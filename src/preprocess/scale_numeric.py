from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_numeric_scaler(train_df: pd.DataFrame, numeric_cols: list[str]) -> StandardScaler | None:
    if not numeric_cols:
        return None
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler


def transform_numeric(
    scaler: StandardScaler | None,
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(index=df.index)
    out = scaler.transform(df[numeric_cols])
    return pd.DataFrame(out, columns=numeric_cols, index=df.index)
