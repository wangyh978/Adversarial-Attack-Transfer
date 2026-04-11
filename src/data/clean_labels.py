from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from src.data.load_raw import load_nsl_kdd, load_unsw_nb15
from src.data.label_maps import map_nsl_label_to_5class
from src.utils.io import ensure_dir, save_json


def build_label_id_map(labels: pd.Series) -> dict[str, int]:
    unique = sorted(labels.dropna().astype(str).unique().tolist())
    return {name: idx for idx, name in enumerate(unique)}


def clean_nsl_labels(df: pd.DataFrame, mode: str = "5class") -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    label_col = "label_raw" if "label_raw" in out.columns else out.columns[-1]
    out["label_raw"] = out[label_col].astype(str).str.strip().str.lower()

    if mode == "5class":
        out["label_clean"] = out["label_raw"].apply(map_nsl_label_to_5class)
    elif mode == "binary":
        out["label_clean"] = out["label_raw"].apply(lambda x: "Normal" if x == "normal" else "Attack")
    else:
        raise ValueError(f"Unsupported mode for NSL-KDD: {mode}")

    label_map = build_label_id_map(out["label_clean"])
    out["label_id"] = out["label_clean"].map(label_map)
    return out, label_map


def find_unsw_label_col(df: pd.DataFrame) -> str:
    for col in ["attack_cat", "Attack_cat", "attack", "label", "Label"]:
        if col in df.columns:
            return col
    raise ValueError("UNSW-NB15 数据中未找到标签列，请检查原始文件")


def clean_unsw_labels(df: pd.DataFrame, mode: str = "multiclass") -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    label_col = find_unsw_label_col(out)

    out["label_raw"] = out[label_col].astype(str).fillna("Unknown").str.strip()
    if mode == "multiclass":
        out["label_clean"] = out["label_raw"]
    elif mode == "binary":
        out["label_clean"] = out["label_raw"].apply(lambda x: "Normal" if x.lower() == "normal" else "Attack")
    else:
        raise ValueError(f"Unsupported mode for UNSW-NB15: {mode}")

    label_map = build_label_id_map(out["label_clean"])
    out["label_id"] = out["label_clean"].map(label_map)
    return out, label_map


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "nsl_kdd":
        df = load_nsl_kdd("data/nsl_kdd/raw")
        mode = args.mode or "5class"
        out, label_map = clean_nsl_labels(df, mode=mode)
    else:
        df = load_unsw_nb15("data/unsw_nb15/raw")
        mode = args.mode or "multiclass"
        out, label_map = clean_unsw_labels(df, mode=mode)

    out_dir = ensure_dir(Path("data") / args.dataset / "processed")
    out_path = out_dir / f"{args.dataset}_labeled.parquet"
    out.to_parquet(out_path, index=False)
    save_json(label_map, Path("artifacts/metadata") / f"{args.dataset}_label_map.json")

    print(out[["label_raw", "label_clean", "label_id"]].head())
    print("saved:", out_path)


if __name__ == "__main__":
    main()
