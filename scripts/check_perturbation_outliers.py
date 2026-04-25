from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return sorted(cols, key=lambda x: int(str(x).split("_")[-1]))


def main():
    parser = argparse.ArgumentParser(description="Inspect perturbation outliers in generated adversarial parquet files.")
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target-model", required=True, choices=["xgb", "gbdt", "tabnet"])
    parser.add_argument("--attack", required=True)
    parser.add_argument("--seed-size", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    path = (
        Path("data/adversarial")
        / args.dataset
        / f"{args.attack}_{args.target_model}_seed{args.seed_size}_a{args.alpha}_d{args.depth}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    f_cols = _cols(df, "f_")
    o_cols = _cols(df, "orig_f_")
    if not o_cols:
        raise ValueError("This adversarial file does not contain orig_f_* columns. Regenerate it first.")

    X_adv = df[f_cols].to_numpy(dtype=np.float32)
    X_org = df[o_cols].to_numpy(dtype=np.float32)
    diff = X_adv - X_org

    l2 = np.linalg.norm(diff, axis=1)
    linf = np.max(np.abs(diff), axis=1)

    print("file:", path)
    print("shape:", X_adv.shape)
    print("mean_l2:", float(l2.mean()), "max_l2:", float(l2.max()))
    print("mean_linf:", float(linf.mean()), "max_linf:", float(linf.max()))
    print("num_linf_gt_1:", int((linf > 1).sum()))
    print("num_l2_gt_5:", int((l2 > 5).sum()))

    idx = np.argsort(-linf)[: args.top]
    rows = []
    for i in idx:
        j = int(np.argmax(np.abs(diff[i])))
        rows.append(
            {
                "rank": len(rows) + 1,
                "sample_id": int(df["sample_id"].iloc[i]) if "sample_id" in df.columns else int(i),
                "label_true": int(df["label_true"].iloc[i]),
                "l2": float(l2[i]),
                "linf": float(linf[i]),
                "max_feature_index": j,
                "orig_value": float(X_org[i, j]),
                "adv_value": float(X_adv[i, j]),
                "delta": float(diff[i, j]),
            }
        )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
