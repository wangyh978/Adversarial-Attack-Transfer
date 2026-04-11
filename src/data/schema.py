from __future__ import annotations

import pandas as pd


def summarize_schema(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": [str(c) for c in df.columns],
        "dtype": [str(t) for t in df.dtypes],
        "non_null": df.notna().sum().values,
        "null_count": df.isna().sum().values,
        "nunique": df.nunique(dropna=False).values,
    })
