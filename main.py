#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified experiment entrypoint for Adversarial-Attack-Transfer.

Includes:
- dataset preparation
- target model training
- MSM hard-label mixup surrogate pipeline
- transfer attack generation and evaluation
- multi-target full attack matrix
- iterative MSM workflow
- surrogate hyperparameter sweep
- result report generation
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent

DATASET_ALIASES = {
    "nsl": "nsl_kdd",
    "nsl_kdd": "nsl_kdd",
    "unsw": "unsw_nb15",
    "unsw_nb15": "unsw_nb15",
}

DEFAULT_TARGETS = {
    "nsl_kdd": ["tabnet", "xgb", "gbdt"],
    "unsw_nb15": ["xgb", "gbdt", "tabnet"],
}

DEFAULT_ATTACKS = ["fgm", "pgd", "slide"]


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"\n>> {' '.join(shlex.quote(x) for x in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_module(module: str, *args: str) -> None:
    run_cmd([sys.executable, "-m", module, *args], cwd=ROOT)


def run_script(script_relpath: str, *args: str) -> None:
    script_path = ROOT / script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    run_cmd([sys.executable, str(script_path), *args], cwd=ROOT)


def normalize_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise SystemExit(f"Unsupported dataset: {name}")
    return DATASET_ALIASES[key]


def dataset_alias_for(dataset: str) -> str:
    return "nsl" if dataset == "nsl_kdd" else "unsw"


def label_mode_for(dataset: str) -> str:
    return "5class" if dataset == "nsl_kdd" else "multiclass"


def default_targets_for(dataset: str) -> List[str]:
    return list(DEFAULT_TARGETS[dataset])


def resolve_targets(dataset: str, args: argparse.Namespace) -> List[str]:
    if args.target:
        return [args.target]
    if args.targets:
        return list(args.targets)
    return default_targets_for(dataset)


def resolve_attacks(args: argparse.Namespace) -> List[str]:
    return list(args.attacks) if args.attacks else list(DEFAULT_ATTACKS)


def prepare_dataset(dataset: str) -> None:
    print_header(f"Prepare dataset: {dataset}")
    mode = label_mode_for(dataset)
    run_module("src.data.load_raw", "--dataset", dataset)
    run_module("src.data.clean_labels", "--dataset", dataset, "--mode", mode)
    run_module("src.data.split_data", "--dataset", dataset)
    run_module("src.preprocess.run_preprocess_pipeline", "--dataset", dataset)


def train_target_model(dataset: str, target: str) -> None:
    module_map = {
        "xgb": "src.models.train_xgb",
        "gbdt": "src.models.train_gbdt",
        "tabnet": "src.models.train_tabnet",
    }
    if target not in module_map:
        raise SystemExit(f"Unsupported target model: {target}")
    print_header(f"Train target model: dataset={dataset} target={target}")
    run_module(module_map[target], "--dataset", dataset)


def compare_baseline(dataset: str) -> None:
    print_header(f"Compare baseline models: {dataset}")
    run_module("src.reporting.compare_models", "--dataset", dataset)


def build_msm_surrogate(
    dataset: str,
    target: str,
    seed_size: int,
    alpha: float,
    depth: int,
    *,
    include_eval: bool = True,
) -> None:
    print_header(
        f"Build MSM surrogate: dataset={dataset} target={target} "
        f"seed_size={seed_size} alpha={alpha} depth={depth}"
    )
    run_module(
        "src.data.build_seed_set",
        "--dataset", dataset,
        "--seed_size", str(seed_size),
    )
    run_module(
        "src.data.query_seed_labels",
        "--dataset", dataset,
        "--target_model", target,
        "--seed_size", str(seed_size),
    )
    run_module(
        "src.augment.run_mixup",
        "--dataset", dataset,
        "--target_model", target,
        "--seed_size", str(seed_size),
        "--alpha", str(alpha),
    )
    run_module(
        "src.data.build_surrogate_trainset",
        "--dataset", dataset,
        "--target_model", target,
        "--seed_size", str(seed_size),
        "--alpha", str(alpha),
    )
    run_module(
        "src.models.train_surrogate_mlp",
        "--dataset", dataset,
        "--target_model", target,
        "--seed_size", str(seed_size),
        "--alpha", str(alpha),
        "--depth", str(depth),
    )
    if include_eval:
        run_module(
            "src.evaluation.evaluate_surrogate",
            "--dataset", dataset,
            "--target_model", target,
            "--seed_size", str(seed_size),
            "--alpha", str(alpha),
            "--depth", str(depth),
        )


def generate_attack_from_surrogate(
    dataset: str,
    target: str,
    attack: str,
    seed_size: int,
    alpha: float,
    depth: int,
) -> None:
    print_header(
        f"Generate adversarial samples from surrogate | "
        f"dataset={dataset} target={target} attack={attack}"
    )
    run_module(
        "src.transfer.generate_from_surrogate",
        "--dataset", dataset,
        "--target_model", target,
        "--seed_size", str(seed_size),
        "--alpha", str(alpha),
        "--depth", str(depth),
        "--attack", attack,
    )


def evaluate_attack_on_target(
    dataset: str,
    target: str,
    attack: str,
    seed_size: int,
    alpha: float,
    depth: int,
) -> None:
    print_header(
        f"Evaluate transfer on target | dataset={dataset} "
        f"target={target} attack={attack}"
    )
    run_module(
        "src.transfer.attack_target",
        "--dataset", dataset,
        "--target_model", target,
        "--seed_size", str(seed_size),
        "--alpha", str(alpha),
        "--depth", str(depth),
        "--attack", attack,
    )


def run_attack_pair(
    dataset: str,
    target: str,
    attack: str,
    seed_size: int,
    alpha: float,
    depth: int,
) -> None:
    generate_attack_from_surrogate(
        dataset=dataset,
        target=target,
        attack=attack,
        seed_size=seed_size,
        alpha=alpha,
        depth=depth,
    )
    evaluate_attack_on_target(
        dataset=dataset,
        target=target,
        attack=attack,
        seed_size=seed_size,
        alpha=alpha,
        depth=depth,
    )


def summarize_target(dataset: str, target: str) -> None:
    print_header(f"Summarize transfer matrix: dataset={dataset} target={target}")
    run_script(
        "scripts/summarize_transfer_matrix.py",
        "--dataset", dataset,
        "--target-model", target,
    )


def surrogate_only(
    dataset: str,
    target: str,
    seed_size: int,
    alpha: float,
    depth: int,
    *,
    include_prepare: bool = False,
    include_target_training: bool = False,
) -> None:
    if include_prepare:
        prepare_dataset(dataset)
    if include_target_training:
        train_target_model(dataset, target)
    build_msm_surrogate(
        dataset=dataset,
        target=target,
        seed_size=seed_size,
        alpha=alpha,
        depth=depth,
        include_eval=True,
    )


def min_transfer(
    dataset: str,
    target: str,
    attacks: Iterable[str],
    seed_size: int,
    alpha: float,
    depth: int,
    *,
    include_prepare: bool = False,
    include_target_training: bool = False,
) -> None:
    if include_prepare:
        prepare_dataset(dataset)
    if include_target_training:
        train_target_model(dataset, target)

    build_msm_surrogate(
        dataset=dataset,
        target=target,
        seed_size=seed_size,
        alpha=alpha,
        depth=depth,
        include_eval=True,
    )
    for attack in attacks:
        run_attack_pair(dataset, target, attack, seed_size, alpha, depth)
    summarize_target(dataset, target)


def transfer_only(
    dataset: str,
    target: str,
    attacks: Iterable[str],
    seed_size: int,
    alpha: float,
    depth: int,
) -> None:
    for attack in attacks:
        run_attack_pair(dataset, target, attack, seed_size, alpha, depth)
    summarize_target(dataset, target)


def full_attack_matrix(
    dataset: str,
    targets: Iterable[str],
    attacks: Iterable[str],
    seed_size: int,
    alpha: float,
    depth: int,
    *,
    reuse_existing_artifacts: bool,
    include_prepare: bool = False,
) -> None:
    if include_prepare:
        prepare_dataset(dataset)

    for target in targets:
        if not reuse_existing_artifacts:
            train_target_model(dataset, target)
            build_msm_surrogate(
                dataset=dataset,
                target=target,
                seed_size=seed_size,
                alpha=alpha,
                depth=depth,
                include_eval=True,
            )
        else:
            print_header(f"Reuse existing artifacts: dataset={dataset} target={target}")

        for attack in attacks:
            run_attack_pair(dataset, target, attack, seed_size, alpha, depth)
        summarize_target(dataset, target)


def msm_iterative(
    dataset: str,
    target: str,
    attacks: Iterable[str],
    seed_size: int,
    alpha: float,
    depth: int,
    rounds: int,
    *,
    include_prepare: bool = False,
    include_target_training: bool = False,
) -> None:
    if rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    if include_prepare:
        prepare_dataset(dataset)
    if include_target_training:
        train_target_model(dataset, target)

    for i in range(1, rounds + 1):
        print_header(
            f"MSM iterative round {i}/{rounds}: dataset={dataset} target={target}"
        )
        build_msm_surrogate(
            dataset=dataset,
            target=target,
            seed_size=seed_size,
            alpha=alpha,
            depth=depth,
            include_eval=True,
        )
        for attack in attacks:
            run_attack_pair(dataset, target, attack, seed_size, alpha, depth)
        summarize_target(dataset, target)


def run_surrogate_sweep_stage(
    dataset: str,
    targets: List[str],
    attacks: List[str],
    args: argparse.Namespace,
) -> None:
    cmd = [
        "--dataset", dataset,
        "--targets", *targets,
        "--seed-sizes", *[str(x) for x in args.seed_sizes],
        "--alphas", *[str(x) for x in args.alphas],
        "--depths", *[str(x) for x in args.depths],
        "--attacks", *attacks,
        "--stage", args.sweep_stage,
    ]

    if args.core_only:
        cmd.append("--core-only")
    if args.summary_csv:
        cmd.extend(["--summary-csv", args.summary_csv])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    if args.retrain_targets:
        cmd.append("--retrain-targets")
    if args.run_report:
        cmd.append("--run-report")

    run_script("scripts/run_surrogate_sweep.py", *cmd)


def run_report_stage() -> None:
    run_script("scripts/build_result_report.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for NIDS adversarial transfer experiments."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="nsl_kdd",
        choices=["nsl", "nsl_kdd", "unsw", "unsw_nb15"],
        help="Dataset alias or full dataset name.",
    )
    parser.add_argument(
        "--stage",
        default="min_transfer",
        choices=[
            "prepare",
            "baseline",
            "compare_baseline",
            "surrogate",
            "generate_attack",
            "attack_target",
            "transfer_only",
            "min_transfer",
            "full_attack_matrix",
            "full_pipeline",
            "reuse_artifacts",
            "msm_iterative",
            "surrogate_sweep",
            "report",
        ],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--target",
        choices=["xgb", "gbdt", "tabnet"],
        help="Single target model for single-target stages.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["xgb", "gbdt", "tabnet"],
        help="Multiple target models for matrix/full/sweep stages.",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=["fgm", "pgd", "slide", "mim", "ti", "cw"],
        default=None,
        help="Attack methods to run. Default: fgm pgd slide.",
    )
    parser.add_argument("--seed-size", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help="Reuse saved target/surrogate artifacts instead of retraining in selected stages.",
    )

    # surrogate_sweep options
    parser.add_argument("--seed-sizes", nargs="+", default=["500", "1000", "2000"])
    parser.add_argument("--alphas", nargs="+", default=["0.05", "0.1", "0.2"])
    parser.add_argument("--depths", nargs="+", default=["3", "4", "5"])
    parser.add_argument(
        "--sweep-stage",
        default="min_transfer",
        choices=["min_transfer", "full_attack_matrix", "reuse_artifacts"],
        help="Internal stage used by surrogate_sweep. Default is min_transfer.",
    )
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--retrain-targets",
        action="store_true",
        help="During surrogate_sweep, retrain target models for every config. Usually not recommended.",
    )
    parser.add_argument(
        "--run-report",
        action="store_true",
        help="Run scripts/build_result_report.py after surrogate_sweep.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = normalize_dataset(args.dataset)
    targets = resolve_targets(dataset, args)
    attacks = resolve_attacks(args)

    print_header("Resolved configuration")
    print(f"dataset = {dataset}")
    print(f"stage = {args.stage}")
    print(f"targets = {targets}")
    print(f"seed_size = {args.seed_size}")
    print(f"alpha = {args.alpha}")
    print(f"depth = {args.depth}")
    print(f"attacks = {attacks}")
    print(f"rounds = {args.rounds}")
    print(f"reuse_existing_artifacts = {args.reuse_existing_artifacts}")

    if args.stage == "prepare":
        prepare_dataset(dataset)
        return

    if args.stage == "baseline":
        for target in targets:
            train_target_model(dataset, target)
        return

    if args.stage == "compare_baseline":
        compare_baseline(dataset)
        return

    if args.stage == "surrogate":
        surrogate_only(
            dataset=dataset,
            target=targets[0],
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
            include_prepare=False,
            include_target_training=not args.reuse_existing_artifacts,
        )
        return

    if args.stage == "generate_attack":
        for attack in attacks:
            generate_attack_from_surrogate(
                dataset=dataset,
                target=targets[0],
                attack=attack,
                seed_size=args.seed_size,
                alpha=args.alpha,
                depth=args.depth,
            )
        return

    if args.stage == "attack_target":
        for attack in attacks:
            evaluate_attack_on_target(
                dataset=dataset,
                target=targets[0],
                attack=attack,
                seed_size=args.seed_size,
                alpha=args.alpha,
                depth=args.depth,
            )
        summarize_target(dataset, targets[0])
        return

    if args.stage == "transfer_only":
        transfer_only(
            dataset=dataset,
            target=targets[0],
            attacks=attacks,
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
        )
        return

    if args.stage == "min_transfer":
        min_transfer(
            dataset=dataset,
            target=targets[0],
            attacks=attacks,
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
            include_prepare=False,
            include_target_training=not args.reuse_existing_artifacts,
        )
        return

    if args.stage == "full_attack_matrix":
        full_attack_matrix(
            dataset=dataset,
            targets=targets,
            attacks=attacks,
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
            reuse_existing_artifacts=args.reuse_existing_artifacts,
            include_prepare=False,
        )
        return

    if args.stage == "full_pipeline":
        full_attack_matrix(
            dataset=dataset,
            targets=targets,
            attacks=attacks,
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
            reuse_existing_artifacts=False,
            include_prepare=True,
        )
        return

    if args.stage == "reuse_artifacts":
        full_attack_matrix(
            dataset=dataset,
            targets=targets,
            attacks=attacks,
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
            reuse_existing_artifacts=True,
            include_prepare=False,
        )
        return

    if args.stage == "msm_iterative":
        msm_iterative(
            dataset=dataset,
            target=targets[0],
            attacks=attacks,
            seed_size=args.seed_size,
            alpha=args.alpha,
            depth=args.depth,
            rounds=args.rounds,
            include_prepare=False,
            include_target_training=not args.reuse_existing_artifacts,
        )
        return

    if args.stage == "surrogate_sweep":
        run_surrogate_sweep_stage(dataset, targets, attacks, args)
        return

    if args.stage == "report":
        run_report_stage()
        return

    raise SystemExit(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
