import sys
import subprocess
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.preprocess.build_features",
        "--dataset",
        args.dataset,
    ]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
