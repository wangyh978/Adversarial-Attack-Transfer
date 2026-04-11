from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_bar(csv_path: str | Path, metric: str, save_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    labels = df["model_name"].tolist() if "model_name" in df.columns else df["source_file"].tolist()
    values = df[metric].tolist()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(f"Model comparison: {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
