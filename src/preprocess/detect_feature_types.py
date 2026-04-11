from __future__ import annotations

import pandas as pd


EXCLUDE_COLS = {
    "label_raw", "label_clean", "label_id", "dataset_name", "difficulty",
    "blackbox_label", "target_label", "data_source", "source_file",
}


def detect_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    categorical_cols = []
    numeric_cols = []
    drop_cols = []

    for col in df.columns:
        if col in EXCLUDE_COLS:
            drop_cols.append(col)
            continue

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            drop_cols.append(col)

    return categorical_cols, numeric_cols, drop_cols
