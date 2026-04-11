from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_alpha_curve(csv_path: str | Path, save_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    grouped = df.groupby("alpha")["target_agreement"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["alpha"], grouped["target_agreement"], marker="o")
    ax.set_title("Target agreement vs alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("target_agreement")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
