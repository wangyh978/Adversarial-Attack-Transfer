from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover
    f1_score = None


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if f1_score is None:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def compute_transfer_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute transfer-attack metrics with a strict denominator.

    Important fix:
    Older code counted every sample whose adversarial prediction differed from
    the true label:
        pred_adv_target != label_true
    That overestimates attack success because samples already misclassified by
    the clean target model are counted as successful attacks.

    Strict transfer success is now:
        clean_correct AND pred_adv_target != label_true
    and the rate denominator is the number of clean-correct samples.
    """

    required = {"label_true", "pred_clean_target", "pred_adv_target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"transfer results missing required columns: {sorted(missing)}")

    y_true = df["label_true"].to_numpy()
    y_clean = df["pred_clean_target"].to_numpy()
    y_adv = df["pred_adv_target"].to_numpy()

    clean_correct = y_clean == y_true
    adv_correct = y_adv == y_true
    strict_success = clean_correct & (~adv_correct)
    legacy_success = y_adv != y_true

    clean_correct_count = int(clean_correct.sum())
    total_count = int(len(df))

    transfer_success_rate = (
        float(strict_success.sum() / clean_correct_count)
        if clean_correct_count > 0
        else 0.0
    )

    result: dict[str, Any] = {
        "num_samples": total_count,
        "num_clean_correct": clean_correct_count,
        "clean_accuracy": float(clean_correct.mean()) if total_count else 0.0,
        "adversarial_accuracy": float(adv_correct.mean()) if total_count else 0.0,
        "accuracy_drop": float(clean_correct.mean() - adv_correct.mean()) if total_count else 0.0,
        "clean_macro_f1": _macro_f1(y_true, y_clean),
        "adversarial_macro_f1": _macro_f1(y_true, y_adv),
        "macro_f1_drop": _macro_f1(y_true, y_clean) - _macro_f1(y_true, y_adv),
        "transfer_success_rate": transfer_success_rate,
        "transfer_success_count": int(strict_success.sum()),
        "legacy_misclassification_rate": float(legacy_success.mean()) if total_count else 0.0,
        "legacy_misclassification_count": int(legacy_success.sum()),
        "metric_definition": (
            "transfer_success_rate = count(clean_correct and adv_wrong) / count(clean_correct)"
        ),
    }

    if "l2_perturbation" in df.columns:
        result["mean_l2_perturbation"] = float(df["l2_perturbation"].mean())
        result["median_l2_perturbation"] = float(df["l2_perturbation"].median())
        result["max_l2_perturbation"] = float(df["l2_perturbation"].max())

    if "linf_perturbation" in df.columns:
        result["mean_linf_perturbation"] = float(df["linf_perturbation"].mean())
        result["max_linf_perturbation"] = float(df["linf_perturbation"].max())

    per_class: dict[str, dict[str, float | int]] = {}
    for label, group in df.groupby("label_true", sort=True):
        g_clean = group["pred_clean_target"].to_numpy() == group["label_true"].to_numpy()
        g_adv = group["pred_adv_target"].to_numpy() == group["label_true"].to_numpy()
        denom = int(g_clean.sum())
        per_class[str(label)] = {
            "num_samples": int(len(group)),
            "num_clean_correct": denom,
            "clean_accuracy": float(g_clean.mean()) if len(group) else 0.0,
            "adversarial_accuracy": float(g_adv.mean()) if len(group) else 0.0,
            "transfer_success_rate": float(((g_clean) & (~g_adv)).sum() / denom) if denom else 0.0,
        }
    result["per_class"] = per_class

    if "target_model" in df.columns:
        model_rates = []
        for _, group in df.groupby("target_model"):
            g_clean = group["pred_clean_target"].to_numpy() == group["label_true"].to_numpy()
            g_adv = group["pred_adv_target"].to_numpy() == group["label_true"].to_numpy()
            denom = int(g_clean.sum())
            rate = float(((g_clean) & (~g_adv)).sum() / denom) if denom else 0.0
            model_rates.append(rate)

        if model_rates:
            model_rates_arr = np.array(model_rates, dtype=float)
            result["perturbation_generalization"] = float(model_rates_arr.mean())
            result["structural_robustness"] = (
                float(1.0 - model_rates_arr.std(ddof=0))
                if len(model_rates_arr) > 1
                else 1.0
            )

    return result
