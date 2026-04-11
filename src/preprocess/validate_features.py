from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path("data") / args.dataset / "processed"

    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")

    print("train:", X_train.shape, "nan:", np.isnan(X_train).sum(), "inf:", np.isinf(X_train).sum())
    print("val:", X_val.shape, "nan:", np.isnan(X_val).sum(), "inf:", np.isinf(X_val).sum())
    print("test:", X_test.shape, "nan:", np.isnan(X_test).sum(), "inf:", np.isinf(X_test).sum())


if __name__ == "__main__":
    main()
