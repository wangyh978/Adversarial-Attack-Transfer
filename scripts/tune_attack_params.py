from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.registry import SUPPORTED_ATTACKS
from src.transfer.experiment import resolve_surrogate_config, transfer_results_dir
from src.utils.io import ensure_dir


DEFAULT_ATTACK_GRIDS = {
    "mim": [
        {"epsilon": 0.5, "steps": 10, "step_size": 0.1, "decay": 1.0},
        {"epsilon": 0.6, "steps": 12, "step_size": 0.08, "decay": 1.0},
        {"epsilon": 0.75, "steps": 15, "step_size": 0.06, "decay": 1.1},
    ],
    "ti": [
        {"epsilon": 0.5, "steps": 12, "step_size": 0.08, "decay": 1.0, "kernel_size": 5, "kernel_sigma": 1.0},
        {"epsilon": 0.6, "steps": 14, "step_size": 0.07, "decay": 1.0, "kernel_size": 7, "kernel_sigma": 1.2},
        {"epsilon": 0.75, "steps": 16, "step_size": 0.06, "decay": 1.1, "kernel_size": 7, "kernel_sigma": 1.4},
    ],
    "cw": [
        {"c_const": 0.01, "steps": 20, "attack_lr": 0.005, "confidence": 0.0, "binary_search_steps": 3},
        {"c_const": 0.02, "steps": 24, "attack_lr": 0.005, "confidence": 0.0, "binary_search_steps": 4},
        {"c_const": 0.05, "steps": 30, "attack_lr": 0.003, "confidence": 0.25, "binary_search_steps": 4},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep MIM/TI/CW parameters on saved surrogates.")
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--targets", nargs="+", default=["xgb", "gbdt", "tabnet"])
    parser.add_argument("--attacks", nargs="+", default=["mim", "ti", "cw"], choices=SUPPORTED_ATTACKS)
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified subset size for faster sweeps.")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--run-tag-prefix", type=str, default="attack_sweep")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _fmt_value(value) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def profile_tag(attack: str, params: dict) -> str:
    parts = [attack]
    for key in sorted(params):
        parts.append(f"{key[:3]}{_fmt_value(params[key])}")
    return "_".join(parts)


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def load_metrics(run_tag: str, attack: str, dataset: str, target: str) -> dict:
    metrics_path = transfer_results_dir(run_tag) / f"transfer_{attack}_{dataset}_{target}_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    out_root = ensure_dir(Path("results/tables/attack_sweeps"))

    for target in args.targets:
        config = resolve_surrogate_config(args.dataset, target)

        for attack in args.attacks:
            grid = DEFAULT_ATTACK_GRIDS.get(attack)
            if not grid:
                print(f"[skip] no built-in grid for attack={attack}")
                continue

            rows = []
            for idx, params in enumerate(grid, start=1):
                run_tag = f"{args.run_tag_prefix}_{args.dataset}_{target}_{profile_tag(attack, params)}_{idx:02d}"

                gen_cmd = [
                    sys.executable,
                    "-m",
                    "src.transfer.generate_from_surrogate",
                    "--dataset",
                    args.dataset,
                    "--target_model",
                    target,
                    "--attack",
                    attack,
                    "--seed_size",
                    str(config.seed_size),
                    "--alpha",
                    str(config.alpha),
                    "--depth",
                    str(config.depth),
                    "--run_tag",
                    run_tag,
                ]
                if args.sample_size is not None:
                    gen_cmd.extend(["--sample_size", str(args.sample_size), "--sample_seed", str(args.sample_seed)])
                for key, value in params.items():
                    gen_cmd.extend([f"--{key}", str(value)])

                eval_cmd = [
                    sys.executable,
                    "-m",
                    "src.transfer.attack_target",
                    "--dataset",
                    args.dataset,
                    "--target_model",
                    target,
                    "--attack",
                    attack,
                    "--seed_size",
                    str(config.seed_size),
                    "--alpha",
                    str(config.alpha),
                    "--depth",
                    str(config.depth),
                    "--run_tag",
                    run_tag,
                ]

                try:
                    run_cmd(gen_cmd)
                    run_cmd(eval_cmd)
                except Exception as exc:
                    print(f"[error] {exc}")
                    if args.stop_on_error:
                        raise
                    continue

                metrics = load_metrics(run_tag, attack, args.dataset, target)
                row = {
                    "dataset": args.dataset,
                    "target_model": target,
                    "attack": attack,
                    "run_tag": run_tag,
                    "seed_size": config.seed_size,
                    "alpha": config.alpha,
                    "depth": config.depth,
                    "transfer_success_rate": metrics.get("transfer_success_rate"),
                    "accuracy_drop": metrics.get("accuracy_drop"),
                    "macro_f1_drop": metrics.get("macro_f1_drop"),
                    "mean_l2_perturbation": metrics.get("mean_l2_perturbation"),
                    "linf_q0.999": metrics.get("linf_q0.999"),
                    "num_linf_gt_1": metrics.get("num_linf_gt_1"),
                    "num_l2_gt_5": metrics.get("num_l2_gt_5"),
                }
                row.update(params)
                rows.append(row)

            if not rows:
                continue

            rows.sort(
                key=lambda row: (
                    row.get("transfer_success_rate", 0.0) or 0.0,
                    -(row.get("mean_l2_perturbation", 1e9) or 1e9),
                ),
                reverse=True,
            )

            out_csv = out_root / f"attack_sweep_{args.dataset}_{target}_{attack}.csv"
            with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            best_json = out_root / f"best_attack_sweep_{args.dataset}_{target}_{attack}.json"
            best_payload = rows[0]
            best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            print("[saved]", out_csv)
            print("[saved]", best_json)
            print("[best]", best_payload)


if __name__ == "__main__":
    main()
