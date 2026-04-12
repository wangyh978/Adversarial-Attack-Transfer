from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_metrics_filename(path: Path):
    m = re.match(r"^transfer_(?P<attack>[^_]+)_(?P<body>.+)_metrics\.json$", path.name)
    if not m:
        return None
    attack = m.group("attack")
    body = m.group("body")
    dataset, _, target = body.rpartition("_")
    if not dataset or not target:
        return None
    return attack, dataset, target


def main():
    parser = argparse.ArgumentParser(description="Summarize transfer metrics into csv/md.")
    parser.add_argument("--dataset", required=True)
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
                "perturbation_generalization": metrics.get("perturbation_generalization"),
                "structural_robustness": metrics.get("structural_robustness"),
            }
        )

    if not rows:
        print("No transfer metrics found.")
        return

    df = pd.DataFrame(rows).sort_values(["target_model", "attack"]).reset_index(drop=True)

    target_suffix = args.target_model if args.target_model else "all_targets"
    csv_path = results_dir / f"final_transfer_matrix_{args.dataset}_{target_suffix}.csv"
    md_path = results_dir / f"final_transfer_matrix_{args.dataset}_{target_suffix}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Transfer summary for {args.dataset}\n\n")
        if args.target_model:
            f.write(f"Target model: `{args.target_model}`\n\n")
        f.write(df.to_markdown(index=False))

    print(df)
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
