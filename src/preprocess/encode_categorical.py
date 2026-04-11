from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def fit_categorical_encoder(train_df: pd.DataFrame, categorical_cols: list[str]) -> OrdinalEncoder | None:
    if not categorical_cols:
        return None
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[categorical_cols].astype(str))
    return enc


def transform_categorical(
    encoder: OrdinalEncoder | None,
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    if not categorical_cols:
        return pd.DataFrame(index=df.index)
    out = encoder.transform(df[categorical_cols].astype(str))
    return pd.DataFrame(out, columns=categorical_cols, index=df.index)
