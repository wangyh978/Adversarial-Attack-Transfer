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


def candidate_surrogate_meta_paths(dataset: str, target: str, seed_size: int, alpha: float, depth: int) -> list[Path]:
    tag = f"{dataset}_{target}_seed{seed_size}_a{alpha}_d{depth}"
    return [
        Path("artifacts/metadata") / f"surrogate_{tag}_metrics.json",
        Path("results/tables") / f"surrogate_{tag}_metrics.json",
        Path("results/tables") / f"evaluate_surrogate_{tag}.json",
        Path("results/tables") / f"surrogate_metrics_{tag}.json",
    ]


def read_surrogate_meta(dataset: str, target: str, seed_size: int, alpha: float, depth: int) -> dict:
    for path in candidate_surrogate_meta_paths(dataset, target, seed_size, alpha, depth):
        data = read_json(path)
        if data:
            return data
    return {}


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
        "transfer_success_count",
        "legacy_misclassification_rate",
        "clean_accuracy",
        "adversarial_accuracy",
        "accuracy_drop",
        "clean_macro_f1",
        "adversarial_macro_f1",
        "macro_f1_drop",
        "mean_l2_perturbation",
        "mean_linf_perturbation",
        "l2_q0.999",
        "linf_q0.999",
        "num_linf_gt_1",
        "num_l2_gt_5",
        "surrogate_accuracy",
        "surrogate_macro_f1",
        "surrogate_agreement",
        "score",
    ]

    rows = []
    surrogate_meta = read_surrogate_meta(dataset, target, seed_size, alpha, depth)
    surrogate_accuracy = surrogate_meta.get("accuracy", surrogate_meta.get("surrogate_accuracy"))
    surrogate_macro_f1 = surrogate_meta.get("macro_f1", surrogate_meta.get("surrogate_macro_f1"))
    surrogate_agreement = surrogate_meta.get("agreement", surrogate_meta.get("target_agreement"))

    for attack in attacks:
        metrics = read_json(metric_path(dataset, target, attack))
        if not metrics:
            print(f"[WARN] missing metrics: {metric_path(dataset, target, attack)}")
            continue

        transfer = float(metrics.get("transfer_success_rate", 0.0) or 0.0)
        f1_drop = float(metrics.get("macro_f1_drop", 0.0) or 0.0)
        l2 = float(metrics.get("mean_l2_perturbation", 0.0) or 0.0)
        linf999 = float(metrics.get("linf_q0.999", 0.0) or 0.0)
        anomaly_penalty = 0.02 * float(metrics.get("num_linf_gt_1", 0.0) or 0.0) / max(float(metrics.get("num_samples", 1.0) or 1.0), 1.0)

        # Prefer high transfer and macro-F1 degradation, while mildly penalizing
        # large perturbations and obvious outliers.
        score = transfer + 0.25 * f1_drop - 0.02 * l2 - 0.05 * linf999 - anomaly_penalty

        rows.append(
            {
                "dataset": dataset,
                "target_model": target,
                "seed_size": seed_size,
                "alpha": alpha,
                "depth": depth,
                "attack": attack,
                "transfer_success_rate": metrics.get("transfer_success_rate"),
                "transfer_success_count": metrics.get("transfer_success_count"),
                "legacy_misclassification_rate": metrics.get("legacy_misclassification_rate"),
                "clean_accuracy": metrics.get("clean_accuracy"),
                "adversarial_accuracy": metrics.get("adversarial_accuracy"),
                "accuracy_drop": metrics.get("accuracy_drop"),
                "clean_macro_f1": metrics.get("clean_macro_f1"),
                "adversarial_macro_f1": metrics.get("adversarial_macro_f1"),
                "macro_f1_drop": metrics.get("macro_f1_drop"),
                "mean_l2_perturbation": metrics.get("mean_l2_perturbation"),
                "mean_linf_perturbation": metrics.get("mean_linf_perturbation"),
                "l2_q0.999": metrics.get("l2_q0.999"),
                "linf_q0.999": metrics.get("linf_q0.999"),
                "num_linf_gt_1": metrics.get("num_linf_gt_1"),
                "num_l2_gt_5": metrics.get("num_l2_gt_5"),
                "surrogate_accuracy": surrogate_accuracy,
                "surrogate_macro_f1": surrogate_macro_f1,
                "surrogate_agreement": surrogate_agreement,
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

    grouped = (
        df.groupby(["dataset", "target_model", "seed_size", "alpha", "depth"], as_index=False)
        .agg(
            mean_transfer_success_rate=("transfer_success_rate", "mean"),
            max_transfer_success_rate=("transfer_success_rate", "max"),
            mean_macro_f1_drop=("macro_f1_drop", "mean"),
            mean_l2_perturbation=("mean_l2_perturbation", "mean"),
            mean_linf_999=("linf_q0.999", "mean"),
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


def build_combos(seed_sizes: list[int], alphas: list[float], depths: list[int], core_only: bool):
    if core_only:
        # Recommended compact search:
        # baseline + seed-size changes + alpha changes + depth changes.
        return [
            (1000, 0.1, 3),
            (500, 0.1, 3),
            (2000, 0.1, 3),
            (1000, 0.05, 3),
            (1000, 0.2, 3),
            (1000, 0.1, 4),
            (1000, 0.1, 5),
        ]
    return [(s, a, d) for s in seed_sizes for a in alphas for d in depths]


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
        default="min_transfer",
        choices=["min_transfer", "full_attack_matrix", "reuse_artifacts"],
        help="Recommended: min_transfer. It rebuilds surrogate for each parameter combo.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run compact recommended sweep instead of full grid.",
    )
    parser.add_argument(
        "--retrain-targets",
        action="store_true",
        help="Retrain target models inside each sweep run. Usually keep this off.",
    )
    parser.add_argument(
        "--run-report",
        action="store_true",
        help="Run scripts/build_result_report.py after sweep.",
    )

    args = parser.parse_args()

    seed_sizes = parse_list(args.seed_sizes, int)
    alphas = parse_list(args.alphas, float)
    depths = parse_list(args.depths, int)
    attacks = parse_list(args.attacks, str)

    combos = build_combos(seed_sizes, alphas, depths, args.core_only)

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

    dataset_arg = "nsl" if args.dataset == "nsl_kdd" else "unsw"

    for target in args.targets:
        for seed_size, alpha, depth in combos:
            cmd = [
                args.python,
                "main.py",
                dataset_arg,
                "--stage",
                args.stage,
                "--target",
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

            # For min_transfer, this skips repeated target retraining but still rebuilds surrogate.
            if not args.retrain_targets and args.stage == "min_transfer":
                cmd.append("--reuse-existing-artifacts")

            ret = run_command(cmd, dry_run=args.dry_run)
            if ret != 0:
                print(
                    f"[ERROR] command failed with code {ret}: "
                    f"target={target}, seed={seed_size}, alpha={alpha}, depth={depth}"
                )
                if args.stop_on_error:
                    sys.exit(ret)
                continue

            if not args.dry_run:
                append_rows(summary_csv, args.dataset, target, seed_size, alpha, depth, attacks)

        if not args.dry_run:
            target_csv = Path("results/tables") / f"surrogate_sweep_{args.dataset}_{target}.csv"
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

    if args.run_report and not args.dry_run:
        ret = run_command([args.python, "scripts/build_result_report.py"], dry_run=False)
        if ret != 0 and args.stop_on_error:
            sys.exit(ret)


if __name__ == "__main__":
    main()
