from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd

from src.evaluation.surrogate_ranking import rank_surrogates


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path("results/tables") / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing surrogate summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    ranked = rank_surrogates(df)
    best = ranked.iloc[0].to_dict()

    seed_size = int(best["seed_size"])
    alpha = float(best["alpha"])
    depth = int(best["depth"])

    model_path = Path("artifacts/models") / (
        f"surrogate_{args.dataset}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}.pt"
    )
    eval_json_path = Path("results/tables") / (
        f"surrogate_eval_{args.dataset}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Best surrogate model file not found: {model_path}")

    out_tables = Path("results/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_meta = Path("artifacts/metadata")
    out_meta.mkdir(parents=True, exist_ok=True)

    best_df = pd.DataFrame([best])
    best_csv_path = out_tables / f"best_surrogate_{args.dataset}_{args.target_model}.csv"
    best_df.to_csv(best_csv_path, index=False, encoding="utf-8-sig")

    best_payload = {
        "dataset": args.dataset,
        "target_model": args.target_model,
        "seed_size": seed_size,
        "alpha": alpha,
        "depth": depth,
        "accuracy": float(best["accuracy"]),
        "f1_macro": float(best["f1_macro"]),
        "target_agreement": float(best["target_agreement"]),
        "composite_score": float(best["composite_score"]),
        "rank": int(best["rank"]),
        "model_path": str(model_path),
        "evaluation_json": str(eval_json_path),
        "summary_csv": str(summary_path),
        "selection_policy": {
            "weights": {
                "target_agreement": 0.5,
                "f1_macro": 0.3,
                "accuracy": 0.2,
            },
            "tie_break": [
                "target_agreement",
                "f1_macro",
                "accuracy",
                "seed_size",
                "depth",
            ],
        },
    }

    best_json_path = out_meta / f"best_surrogate_{args.dataset}_{args.target_model}.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    print(pd.Series(best_payload))
    print(f"[saved] {best_csv_path}")
    print(f"[saved] {best_json_path}")


if __name__ == "__main__":
    main()
