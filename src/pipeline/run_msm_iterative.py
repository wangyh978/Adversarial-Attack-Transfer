from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def parse_args():
    parser = ArgumentParser(
        description="Iterative MSM pipeline driver for paper method. It generates D_{i+1} by mixup + black-box relabel."
    )
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir = ensure_dir(Path("data/msm_rounds") / args.dataset / args.target_model)
    initial_seed = (
        Path("data/seeds")
        / args.dataset
        / "queried"
        / f"{args.target_model}_seed_{args.seed_size}_queried.parquet"
    )

    current_path = initial_seed
    print(f"[MSM] start from: {current_path}")

    for round_idx in range(1, args.rounds + 1):
        round_dir = ensure_dir(work_dir / f"round_{round_idx}")
        mix_path = round_dir / f"mixup_round_{round_idx}.parquet"
        union_path = round_dir / f"train_union_round_{round_idx}.parquet"

        # Shell command examples for the user; this file focuses on dataset bookkeeping.
        print("=" * 72)
        print(f"[MSM] ROUND {round_idx}")
        print(f"Input set : {current_path}")
        print(f"Mixup out : {mix_path}")
        print(f"Union out : {union_path}")
        print(
            "Run next:\n"
            f"  python -m src.augment.run_mixup --dataset {args.dataset} --target_model {args.target_model} "
            f"--seed_size {args.seed_size} --alpha {args.alpha} --input_path \"{current_path}\" --output_path \"{mix_path}\"\n"
        )

        current_df = pd.read_parquet(current_path)
        if current_df.empty:
            raise ValueError(f"Empty dataset at {current_path}")

        # The actual union file can be built after run_mixup has completed.
        # This script intentionally prints the exact paths needed for each round,
        # keeping the iterative process reproducible and easy to debug.
        current_path = union_path

    print("=" * 72)
    print("[MSM] After each round's mixup file is generated, concatenate it with the previous round input")
    print("      and train the surrogate on the resulting union dataset.")
    print("      This preserves the paper's D_i -> D_{i+1} iterative structure.")


if __name__ == "__main__":
    main()
