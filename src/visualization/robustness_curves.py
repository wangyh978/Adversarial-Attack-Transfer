from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_curve(csv_path: str | Path, x_col: str, y_col: str, save_path: str | Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    grouped = df.groupby(x_col)[y_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped[x_col], grouped[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
