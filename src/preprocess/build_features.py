from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.preprocess.detect_feature_types import detect_feature_types
from src.preprocess.encode_categorical import fit_categorical_encoder, transform_categorical
from src.preprocess.scale_numeric import fit_numeric_scaler, transform_numeric
from src.utils.io import ensure_dir, save_json


def build_feature_frame(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    encoder,
    scaler,
) -> pd.DataFrame:
    cat_df = transform_categorical(encoder, df, categorical_cols)
    num_df = transform_numeric(scaler, df, numeric_cols)
    feat_df = pd.concat([cat_df, num_df], axis=1)
    feat_df = feat_df.loc[:, sorted(feat_df.columns)]
    feat_df.columns = [f"f_{i}" for i in range(feat_df.shape[1])]
    return feat_df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path("data") / args.dataset / "processed"

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    categorical_cols, numeric_cols, drop_cols = detect_feature_types(train_df)

    encoder = fit_categorical_encoder(train_df, categorical_cols)
    scaler = fit_numeric_scaler(train_df, numeric_cols)

    X_train_df = build_feature_frame(train_df, categorical_cols, numeric_cols, encoder, scaler)
    X_val_df = build_feature_frame(val_df, categorical_cols, numeric_cols, encoder, scaler)
    X_test_df = build_feature_frame(test_df, categorical_cols, numeric_cols, encoder, scaler)

    y_train = train_df["label_id"].to_numpy()
    y_val = val_df["label_id"].to_numpy()
    y_test = test_df["label_id"].to_numpy()

    bundle = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "drop_cols": drop_cols,
        "feature_names": X_train_df.columns.tolist(),
        "num_classes": int(train_df["label_id"].nunique()),
    }

    pre_dir = ensure_dir(Path("artifacts/preprocessors"))
    joblib.dump(
        {"encoder": encoder, "scaler": scaler, "bundle": bundle},
        pre_dir / f"{args.dataset}_preprocessor.joblib",
    )

    feature_info = {
        "feature_names": X_train_df.columns.tolist(),
        "feature_min": X_train_df.min(axis=0).astype(float).tolist(),
        "feature_max": X_train_df.max(axis=0).astype(float).tolist(),
        "num_classes": int(train_df["label_id"].nunique()),
    }
    save_json(feature_info, pre_dir / f"{args.dataset}_feature_info.json")

    # Save npy
    np.save(processed_dir / "X_train.npy", X_train_df.to_numpy(dtype=np.float32))
    np.save(processed_dir / "y_train.npy", y_train.astype(np.int64))
    np.save(processed_dir / "X_val.npy", X_val_df.to_numpy(dtype=np.float32))
    np.save(processed_dir / "y_val.npy", y_val.astype(np.int64))
    np.save(processed_dir / "X_test.npy", X_test_df.to_numpy(dtype=np.float32))
    np.save(processed_dir / "y_test.npy", y_test.astype(np.int64))

    # Save feature parquet with labels
    train_feat = X_train_df.copy()
    train_feat["label_id"] = y_train
    train_feat["label_clean"] = train_df["label_clean"].values
    train_feat.to_parquet(processed_dir / "train_features.parquet", index=False)

    val_feat = X_val_df.copy()
    val_feat["label_id"] = y_val
    val_feat["label_clean"] = val_df["label_clean"].values
    val_feat.to_parquet(processed_dir / "val_features.parquet", index=False)

    test_feat = X_test_df.copy()
    test_feat["label_id"] = y_test
    test_feat["label_clean"] = test_df["label_clean"].values
    test_feat.to_parquet(processed_dir / "test_features.parquet", index=False)

    print("train features:", X_train_df.shape)
    print("val features:", X_val_df.shape)
    print("test features:", X_test_df.shape)


if __name__ == "__main__":
    main()
