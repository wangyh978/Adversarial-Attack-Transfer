from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.models.build_surrogate_grid import build_surrogate_grid
from src.models.train_surrogate_job import SurrogateJob


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def model_path(dataset: str, target_model: str, seed_size: int, alpha: float, depth: int) -> Path:
    return Path("artifacts/models") / (
        f"surrogate_{dataset}_{target_model}_seed{seed_size}_a{alpha}_d{depth}.pt"
    )


def eval_path(dataset: str, target_model: str, seed_size: int, alpha: float, depth: int) -> Path:
    return Path("results/tables") / (
        f"surrogate_eval_{dataset}_{target_model}_seed{seed_size}_a{alpha}_d{depth}.json"
    )


def main() -> None:
    args = parse_args()

    grid = build_surrogate_grid(args.dataset, args.target_model)
    out_path = Path("results/tables") / f"surrogate_grid_{args.dataset}_{args.target_model}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_path, index=False, encoding="utf-8-sig")

    for _, row in grid.iterrows():
        dataset = row["dataset"]
        target_model = row["target_model"]
        seed_size = int(row["seed_size"])
        alpha = float(row["alpha"])
        depth = int(row["depth"])

        mp = model_path(dataset, target_model, seed_size, alpha, depth)
        ep = eval_path(dataset, target_model, seed_size, alpha, depth)

        if args.skip_existing and mp.exists() and ep.exists():
            print(f"[skip] {mp.name} and {ep.name}")
            continue

        print(
            f"[run] dataset={dataset} target={target_model} "
            f"seed={seed_size} alpha={alpha} depth={depth}"
        )
        job = SurrogateJob(
            dataset=dataset,
            target_model=target_model,
            seed_size=seed_size,
            alpha=alpha,
            depth=depth,
        )
        job.train()
        job.evaluate()

    print(f"[done] saved grid -> {out_path}")


if __name__ == "__main__":
    main()
