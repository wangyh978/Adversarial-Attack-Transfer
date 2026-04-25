from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def run_command(cmd: list[str], dry_run: bool = False) -> int:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def parse_list(values: list[str], cast):
    out = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            if item:
                out.append(cast(item))
    return out


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_path(dataset: str, target: str, attack: str) -> Path:
    return Path("results/tables") / f"transfer_{attack}_{dataset}_{target}_metrics.json"


def surrogate_meta_path(dataset: str, target: str, seed_size: int, alpha: float, depth: int) -> Path:
    # Most repositories save metadata in this naming style. If absent, we still summarize transfer metrics.
    return Path("artifacts/metadata") / f"surrogate_{dataset}_{target}_seed{seed_size}_a{alpha}_d{depth}_metrics.json"


def append_rows(
    out_csv: Path,
    dataset: str,
    target: str,
    seed_size: int,
    alpha: float,
    depth: int,
    attacks: list[str],
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "target_model",
        "seed_size",
        "alpha",
        "depth",
        "attack",
        "transfer_success_rate",
        "legacy_misclassification_rate",
        "clean_accuracy",
        "adversarial_accuracy",
        "accuracy_drop",
        "clean_macro_f1",
        "adversarial_macro_f1",
        "macro_f1_drop",
        "mean_l2_perturbation",
        "mean_linf_perturbation",
        "surrogate_accuracy",
        "surrogate_macro_f1",
        "score",
    ]

    rows = []
    surrogate_meta = read_json(surrogate_meta_path(dataset, target, seed_size, alpha, depth))
    surrogate_accuracy = surrogate_meta.get("accuracy", surrogate_meta.get("surrogate_accuracy"))
    surrogate_macro_f1 = surrogate_meta.get("macro_f1", surrogate_meta.get("surrogate_macro_f1"))

    for attack in attacks:
        metrics = read_json(metric_path(dataset, target, attack))
        if not metrics:
            continue

        transfer = float(metrics.get("transfer_success_rate", 0.0) or 0.0)
        f1_drop = float(metrics.get("macro_f1_drop", 0.0) or 0.0)
        l2 = float(metrics.get("mean_l2_perturbation", 0.0) or 0.0)

        # A simple ranking score: prefer high transfer and F1 degradation, penalize large perturbation.
        score = transfer + 0.25 * f1_drop - 0.02 * l2

        rows.append(
            {
                "dataset": dataset,
                "target_model": target,
                "seed_size": seed_size,
                "alpha": alpha,
                "depth": depth,
                "attack": attack,
                "transfer_success_rate": metrics.get("transfer_success_rate"),
                "legacy_misclassification_rate": metrics.get("legacy_misclassification_rate"),
                "clean_accuracy": metrics.get("clean_accuracy"),
                "adversarial_accuracy": metrics.get("adversarial_accuracy"),
                "accuracy_drop": metrics.get("accuracy_drop"),
                "clean_macro_f1": metrics.get("clean_macro_f1"),
                "adversarial_macro_f1": metrics.get("adversarial_macro_f1"),
                "macro_f1_drop": metrics.get("macro_f1_drop"),
                "mean_l2_perturbation": metrics.get("mean_l2_perturbation"),
                "mean_linf_perturbation": metrics.get("mean_linf_perturbation"),
                "surrogate_accuracy": surrogate_accuracy,
                "surrogate_macro_f1": surrogate_macro_f1,
                "score": score,
            }
        )

    exists = out_csv.exists()
    with out_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_best_config(summary_csv: Path, dataset: str, target: str):
    if not summary_csv.exists():
        return

    import pandas as pd

    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    # Average over attacks so one surrogate config is selected per target.
    grouped = (
        df.groupby(["dataset", "target_model", "seed_size", "alpha", "depth"], as_index=False)
        .agg(
            mean_transfer_success_rate=("transfer_success_rate", "mean"),
            mean_macro_f1_drop=("macro_f1_drop", "mean"),
            mean_l2_perturbation=("mean_l2_perturbation", "mean"),
            mean_score=("score", "mean"),
        )
        .sort_values("mean_score", ascending=False)
    )

    grouped_path = summary_csv.with_name(summary_csv.stem + "_grouped.csv")
    grouped.to_csv(grouped_path, index=False, encoding="utf-8-sig")

    best = grouped.iloc[0].to_dict()
    out_dir = Path("artifacts/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"best_surrogate_sweep_{dataset}_{target}.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\nBest config written:", best_path)
    print(best)
    print("Grouped summary:", grouped_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run MSM/surrogate hyperparameter sweep and summarize transfer metrics."
    )
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--targets", nargs="+", default=["xgb", "gbdt", "tabnet"])
    parser.add_argument("--seed-sizes", nargs="+", default=["500", "1000", "2000"])
    parser.add_argument("--alphas", nargs="+", default=["0.05", "0.1", "0.2"])
    parser.add_argument("--depths", nargs="+", default=["3", "4", "5"])
    parser.add_argument("--attacks", nargs="+", default=["fgm", "pgd", "slide"])
    parser.add_argument(
        "--stage",
        default="full_attack_matrix",
        choices=["full_attack_matrix", "reuse_artifacts"],
        help="Use full_attack_matrix for fresh surrogate construction; use reuse_artifacts only after models exist.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run a smaller recommended sweep around seed/alpha/depth instead of full grid.",
    )
    args = parser.parse_args()

    seed_sizes = parse_list(args.seed_sizes, int)
    alphas = parse_list(args.alphas, float)
    depths = parse_list(args.depths, int)
    attacks = parse_list(args.attacks, str)

    if args.core_only:
        # Seven runs: baseline + isolated changes. Much faster than full 27-combination grid.
        combos = [
            (1000, 0.1, 3),
            (500, 0.1, 3),
            (2000, 0.1, 3),
            (1000, 0.05, 3),
            (1000, 0.2, 3),
            (1000, 0.1, 4),
            (1000, 0.1, 5),
        ]
    else:
        combos = [(s, a, d) for s in seed_sizes for a in alphas for d in depths]

    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path("results/tables") / f"surrogate_sweep_{args.dataset}.csv"
    )
    if summary_csv.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup = summary_csv.with_name(summary_csv.stem + f"_{timestamp}.bak.csv")
        summary_csv.rename(backup)
        print("Existing summary moved to:", backup)

    for target in args.targets:
        for seed_size, alpha, depth in combos:
            cmd = [
                args.python,
                "main.py",
                "nsl" if args.dataset == "nsl_kdd" else "unsw",
                "--stage",
                args.stage,
                "--targets",
                target,
                "--seed-size",
                str(seed_size),
                "--alpha",
                str(alpha),
                "--depth",
                str(depth),
                "--attacks",
                *attacks,
            ]

            ret = run_command(cmd, dry_run=args.dry_run)
            if ret != 0:
                print(f"[ERROR] command failed with code {ret}: target={target}, seed={seed_size}, alpha={alpha}, depth={depth}")
                if args.stop_on_error:
                    sys.exit(ret)
                continue

            if not args.dry_run:
                append_rows(summary_csv, args.dataset, target, seed_size, alpha, depth, attacks)

        if not args.dry_run:
            target_csv = Path("results/tables") / f"surrogate_sweep_{args.dataset}_{target}.csv"
            # Extract target-specific rows for easier reading.
            try:
                import pandas as pd

                df = pd.read_csv(summary_csv)
                df[df["target_model"] == target].to_csv(target_csv, index=False, encoding="utf-8-sig")
                write_best_config(target_csv, args.dataset, target)
            except Exception as exc:
                print("[WARN] failed to write target summary:", exc)

    if not args.dry_run:
        write_best_config(summary_csv, args.dataset, "all_targets")
        print("\nSweep summary:", summary_csv)


if __name__ == "__main__":
    main()
