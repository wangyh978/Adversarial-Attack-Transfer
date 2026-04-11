from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_model", required=True)
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, x_col: str, metric: str, save_path: str | Path) -> None:
    grouped = df.groupby(x_col)[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped[x_col].astype(str), grouped[metric])
    ax.set_title(f"{metric} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(Path("results/tables") / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.csv")
    plot_metric(df, "seed_size", "target_agreement",
                Path("results/figures") / f"surrogate_seed_agreement_{args.dataset}_{args.target_model}.png")
    plot_metric(df, "depth", "f1_macro",
                Path("results/figures") / f"surrogate_depth_f1_{args.dataset}_{args.target_model}.png")


if __name__ == "__main__":
    main()
