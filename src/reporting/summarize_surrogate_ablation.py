from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.evaluation.surrogate_ranking import rank_surrogates


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> None:
    args = parse_args()

    in_path = Path("results/tables") / f"surrogate_batch_eval_{args.dataset}_{args.target_model}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing batch evaluation file: {in_path}")

    df = pd.read_csv(in_path)
    ranked = rank_surrogates(df)

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.csv"
    md_path = out_dir / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.md"

    ranked.to_csv(csv_path, index=False, encoding="utf-8-sig")

    preview_cols = [
        "rank",
        "seed_size",
        "alpha",
        "depth",
        "accuracy",
        "f1_macro",
        "target_agreement",
        "composite_score",
    ]
    preview = ranked[preview_cols]
    md_text = "# Surrogate Ablation Summary\n\n" + to_markdown_safe(preview)
    md_path.write_text(md_text, encoding="utf-8")

    print(preview.head(10))
    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
