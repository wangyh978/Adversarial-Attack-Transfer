from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.models.build_surrogate_grid import build_surrogate_grid
from src.models.train_surrogate_job import SurrogateJob


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = build_surrogate_grid(args.dataset, args.target_model)

    out_path = Path("results/tables") / f"surrogate_grid_{args.dataset}_{args.target_model}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_path, index=False)

    for _, row in grid.iterrows():
        job = SurrogateJob(
            dataset=row["dataset"],
            target_model=row["target_model"],
            seed_size=int(row["seed_size"]),
            alpha=float(row["alpha"]),
            depth=int(row["depth"]),
        )
        job.train()
        job.evaluate()


if __name__ == "__main__":
    main()
