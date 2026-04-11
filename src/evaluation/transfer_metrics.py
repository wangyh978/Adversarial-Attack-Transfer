from __future__ import annotations

import pandas as pd


def compute_transfer_metrics(df: pd.DataFrame) -> dict[str, float]:
    if "is_transfer_success" not in df.columns:
        raise ValueError("transfer results 缺少 is_transfer_success 列")

    transfer_success_rate = float(df["is_transfer_success"].mean())

    result = {
        "transfer_success_rate": transfer_success_rate,
    }

    if "target_model" in df.columns:
        model_rates = df.groupby("target_model")["is_transfer_success"].mean()
        result["perturbation_generalization"] = float(model_rates.mean())
        result["structural_robustness"] = float(1.0 - model_rates.std(ddof=0)) if len(model_rates) > 1 else 1.0

    return result
