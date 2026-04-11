from __future__ import annotations

from argparse import ArgumentParser
import subprocess
import sys


TASK_TO_SCRIPT = {
    "check_env": "src/utils/check_env.py",
    "load_data": "src/data/load_raw.py",
    "clean_labels": "src/data/clean_labels.py",
    "split_data": "src/data/split_data.py",
    "preprocess": "src/preprocess/run_preprocess_pipeline.py",
    "train_sklearn": "src/models/train_sklearn_baseline.py",
    "train_xgb": "src/models/train_xgb.py",
    "train_gbdt": "src/models/train_gbdt.py",
    "train_tabnet": "src/models/train_tabnet.py",
    "query_api": "src/blackbox/query_api.py",
    "build_seed_set": "src/data/build_seed_set.py",
    "query_seed_labels": "src/data/query_seed_labels.py",
    "run_mixup": "src/augment/run_mixup.py",
    "train_surrogate": "src/models/train_surrogate_mlp.py",
    "eval_surrogate": "src/evaluation/evaluate_surrogate.py",
    "run_surrogate_ablation": "src/models/run_surrogate_ablation.py",
    "compare_models": "src/reporting/compare_models.py",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="NIDS adversarial robustness project")
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_SCRIPT))
    parser.add_argument("--dataset", default="nsl_kdd", choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--target_model", default="tabnet")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--seed_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script = TASK_TO_SCRIPT[args.task]

    cmd = [sys.executable, script, "--dataset", args.dataset]

    if args.model:
        cmd += ["--model", args.model]
    if args.mode:
        cmd += ["--mode", args.mode]
    if args.target_model:
        cmd += ["--target_model", args.target_model]
    if args.seed_size is not None:
        cmd += ["--seed_size", str(args.seed_size)]
    if args.alpha is not None:
        cmd += ["--alpha", str(args.alpha)]
    if args.depth is not None:
        cmd += ["--depth", str(args.depth)]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
