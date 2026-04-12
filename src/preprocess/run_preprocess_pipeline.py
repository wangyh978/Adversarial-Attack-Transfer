from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.io import ensure_dir


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def dataset_schema(dataset: str) -> tuple[list[str], list[str]]:
    if dataset == "nsl_kdd":
        categorical_cols = ["protocol_type", "service", "flag"]
        drop_cols = [
            "label_raw", "label_clean", "label_id", "difficulty", "dataset_name", "split_source"
        ]
    elif dataset == "unsw_nb15":
        categorical_cols = ["proto", "service", "state"]
        drop_cols = [
            "id", "label", "attack_cat", "label_raw", "label_clean",
            "label_id", "dataset_name", "split_source"
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return categorical_cols, drop_cols


def build_feature_columns(df: pd.DataFrame, categorical_cols: list[str], drop_cols: list[str]) -> tuple[list[str], list[str]]:
    existing_categorical = [c for c in categorical_cols if c in df.columns]
    exclude = set(drop_cols) | set(existing_categorical)

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return existing_categorical, numeric_cols


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_cols),
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def transform_to_df(preprocessor, df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = preprocessor.transform(df[categorical_cols + numeric_cols])
    y = df["label_id"].to_numpy()

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]

    norm_feature_names = [f"f_{i}" for i in range(len(feature_names))]
    feat_df = pd.DataFrame(X, columns=norm_feature_names, index=df.index)
    feat_df["label_id"] = y
    if "label_clean" in df.columns:
        feat_df["label_clean"] = df["label_clean"].values
    if "split_source" in df.columns:
        feat_df["split_source"] = df["split_source"].values
    return feat_df, np.asarray(X), y


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proc_dir = Path("data") / args.dataset / "processed"

    train_df = pd.read_parquet(proc_dir / "train.parquet")
    val_df = pd.read_parquet(proc_dir / "val.parquet")
    test_df = pd.read_parquet(proc_dir / "test.parquet")

    categorical_cols, drop_cols = dataset_schema(args.dataset)
    categorical_cols, numeric_cols = build_feature_columns(train_df, categorical_cols, drop_cols)

    preprocessor = make_preprocessor(categorical_cols, numeric_cols)
    preprocessor.fit(train_df[categorical_cols + numeric_cols])

    train_feat_df, X_train, y_train = transform_to_df(preprocessor, train_df, categorical_cols, numeric_cols)
    val_feat_df, X_val, y_val = transform_to_df(preprocessor, val_df, categorical_cols, numeric_cols)
    test_feat_df, X_test, y_test = transform_to_df(preprocessor, test_df, categorical_cols, numeric_cols)

    train_feat_df.to_parquet(proc_dir / "train_features.parquet", index=False)
    val_feat_df.to_parquet(proc_dir / "val_features.parquet", index=False)
    test_feat_df.to_parquet(proc_dir / "test_features.parquet", index=False)

    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "y_val.npy", y_val)
    np.save(proc_dir / "y_test.npy", y_test)

    ensure_dir(Path("artifacts/preprocessors"))
    joblib.dump(preprocessor, Path("artifacts/preprocessors") / f"{args.dataset}_preprocessor.joblib")

    info = {
        "dataset": args.dataset,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "n_features_out": int(X_train.shape[1]),
        "feature_columns": [c for c in train_feat_df.columns if c.startswith("f_")],
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "num_classes": int(len(sorted(train_df["label_id"].unique()))),
        "label_ids": sorted(train_df["label_id"].unique().tolist()),
    }

    with open(Path("artifacts/preprocessors") / f"{args.dataset}_feature_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("saved:", proc_dir / "train_features.parquet", train_feat_df.shape)
    print("saved:", proc_dir / "val_features.parquet", val_feat_df.shape)
    print("saved:", proc_dir / "test_features.parquet", test_feat_df.shape)
    print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)


if __name__ == "__main__":
    main()
