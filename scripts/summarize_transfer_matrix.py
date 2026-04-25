from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


KNOWN_DATASETS = ("nsl_kdd", "unsw_nb15")


def parse_metrics_filename(path: Path) -> Optional[tuple[str, str, str]]:
    """
    Parse names like:
        transfer_pgd_nsl_kdd_xgb_metrics.json
        transfer_slide_unsw_nb15_tabnet_metrics.json

    The old parser was fragile around dataset names containing underscores.
    """
    name = path.name
    if not (name.startswith("transfer_") and name.endswith("_metrics.json")):
        return None

    body = name[len("transfer_") : -len("_metrics.json")]

    for dataset in KNOWN_DATASETS:
        marker = f"_{dataset}_"
        if marker in body:
            attack, target = body.split(marker, 1)
            if attack and target:
                return attack, dataset, target
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarize transfer metrics into csv/md.")
    parser.add_argument("--dataset", required=True, choices=KNOWN_DATASETS)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--results-dir", default="results/tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_files = sorted(results_dir.glob("transfer_*_metrics.json"))

    rows = []
    for path in metrics_files:
        parsed = parse_metrics_filename(path)
        if parsed is None:
            continue

        attack, dataset, target_model = parsed
        if dataset != args.dataset:
            continue
        if args.target_model and target_model != args.target_model:
            continue

        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append(
            {
                "dataset": dataset,
                "target_model": target_model,
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
            }
        )

    if not rows:
        print("No transfer metrics found.")
        return

    df = pd.DataFrame(rows).sort_values(["target_model", "attack"]).reset_index(drop=True)

    # Dataset-level, attack-level generalization across target structures.
    if not args.target_model:
        grouped = (
            df.groupby("attack")["transfer_success_rate"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "perturbation_generalization", "std": "rate_std"})
        )
        grouped["rate_std"] = grouped["rate_std"].fillna(0.0)
        grouped["structural_robustness"] = 1.0 - grouped["rate_std"]
        df = df.merge(
            grouped[["attack", "perturbation_generalization", "structural_robustness"]],
            on="attack",
            how="left",
        )
    else:
        df["perturbation_generalization"] = df["transfer_success_rate"]
        df["structural_robustness"] = 1.0

    target_suffix = args.target_model if args.target_model else "all_targets"
    csv_path = results_dir / f"final_transfer_matrix_{args.dataset}_{target_suffix}.csv"
    md_path = results_dir / f"final_transfer_matrix_{args.dataset}_{target_suffix}.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Transfer summary for {args.dataset}\n\n")
        if args.target_model:
            f.write(f"Target model: `{args.target_model}`\n\n")
        f.write(
            "Metric note: `transfer_success_rate` uses the strict denominator "
            "`clean_correct and adv_wrong / clean_correct`. "
            "`legacy_misclassification_rate` is kept only for comparison with old outputs.\n\n"
        )
        f.write(df.to_markdown(index=False))

    print(df)
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
