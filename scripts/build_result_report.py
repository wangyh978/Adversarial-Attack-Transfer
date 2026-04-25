from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_DIR = RESULTS_DIR / "summary"
PLOTS_DIR = SUMMARY_DIR / "plots"


def ensure_dirs():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_final_matrices():
    files = sorted(TABLES_DIR.glob("final_transfer_matrix_*.csv"))
    if not files:
        raise FileNotFoundError("No final_transfer_matrix_*.csv found in results/tables")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_metric_jsons():
    files = sorted(TABLES_DIR.glob("transfer_*_metrics.json"))
    if not files:
        raise FileNotFoundError("No transfer_*_metrics.json found in results/tables")

    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        rows.append({
            "dataset": data.get("dataset"),
            "target_model": data.get("target_model"),
            "attack": data.get("attack"),
            "num_samples": data.get("num_samples"),
            "num_clean_correct": data.get("num_clean_correct"),
            "clean_accuracy": data.get("clean_accuracy"),
            "adversarial_accuracy": data.get("adversarial_accuracy"),
            "accuracy_drop": data.get("accuracy_drop"),
            "clean_macro_f1": data.get("clean_macro_f1"),
            "adversarial_macro_f1": data.get("adversarial_macro_f1"),
            "macro_f1_drop": data.get("macro_f1_drop"),
            "transfer_success_rate": data.get("transfer_success_rate"),
            "transfer_success_count": data.get("transfer_success_count"),
            "legacy_misclassification_rate": data.get("legacy_misclassification_rate"),
            "legacy_misclassification_count": data.get("legacy_misclassification_count"),
            "mean_l2_perturbation": data.get("mean_l2_perturbation"),
            "median_l2_perturbation": data.get("median_l2_perturbation"),
            "max_l2_perturbation": data.get("max_l2_perturbation"),
            "mean_linf_perturbation": data.get("mean_linf_perturbation"),
            "max_linf_perturbation": data.get("max_linf_perturbation"),
            "l2_q0.5": data.get("l2_q0.5"),
            "linf_q0.5": data.get("linf_q0.5"),
            "l2_q0.9": data.get("l2_q0.9"),
            "linf_q0.9": data.get("linf_q0.9"),
            "l2_q0.95": data.get("l2_q0.95"),
            "linf_q0.95": data.get("linf_q0.95"),
            "l2_q0.99": data.get("l2_q0.99"),
            "linf_q0.99": data.get("linf_q0.99"),
            "l2_q0.999": data.get("l2_q0.999"),
            "linf_q0.999": data.get("linf_q0.999"),
            "num_linf_gt_1": data.get("num_linf_gt_1"),
            "num_l2_gt_5": data.get("num_l2_gt_5"),
            "metric_definition": data.get("metric_definition"),
            "metric_file": f.name,
        })

    return pd.DataFrame(rows)


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def save_markdown_table(df, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))


def plot_bar(df, value_col, title, ylabel, filename):
    plot_df = df.copy()
    plot_df["group"] = (
        plot_df["dataset"].astype(str)
        + "\n"
        + plot_df["target_model"].astype(str)
        + "-"
        + plot_df["attack"].astype(str)
    )

    plt.figure(figsize=(14, 6))
    plt.bar(plot_df["group"], plot_df[value_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close()


def plot_grouped_by_dataset_attack(df, value_col, filename, title):
    pivot = df.pivot_table(
        index=["dataset", "target_model"],
        columns="attack",
        values=value_col,
        aggfunc="mean",
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.set_xlabel("Dataset / Target Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close()


def plot_heatmap_like(df, value_col, filename, title):
    pivot = df.pivot_table(
        index=["dataset", "target_model"],
        columns="attack",
        values=value_col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a}/{b}" for a, b in pivot.index])
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close()


def build_summary_text(df):
    df = df.copy()
    df["transfer_success_rate"] = pd.to_numeric(df["transfer_success_rate"], errors="coerce")
    df["accuracy_drop"] = pd.to_numeric(df["accuracy_drop"], errors="coerce")
    df["macro_f1_drop"] = pd.to_numeric(df["macro_f1_drop"], errors="coerce")

    best_attack = df.sort_values("transfer_success_rate", ascending=False).iloc[0]
    best_by_dataset = (
        df.sort_values("transfer_success_rate", ascending=False)
        .groupby("dataset", dropna=False)
        .head(1)
    )

    best_by_target = (
        df.sort_values("transfer_success_rate", ascending=False)
        .groupby(["dataset", "target_model"], dropna=False)
        .head(1)
    )

    attack_avg = (
        df.groupby("attack", dropna=False)["transfer_success_rate"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    anomaly_df = df[
        (pd.to_numeric(df["num_linf_gt_1"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(df["num_l2_gt_5"], errors="coerce").fillna(0) > 0)
    ].copy()

    lines = []
    lines.append("# Transfer Attack Result Summary\n")
    lines.append("## Overall conclusion\n")
    lines.append(
        f"- Strongest setting: `{best_attack['dataset']} / {best_attack['target_model']} / {best_attack['attack']}`, "
        f"transfer success rate = `{safe_float(best_attack['transfer_success_rate']):.4f}`."
    )
    lines.append(
        f"- Average strongest attack type: `{attack_avg.iloc[0]['attack']}`, "
        f"mean transfer success rate = `{safe_float(attack_avg.iloc[0]['transfer_success_rate']):.4f}`."
    )

    lines.append("\n## Best attack per dataset\n")
    for _, r in best_by_dataset.iterrows():
        lines.append(
            f"- `{r['dataset']}`: `{r['attack']}` on `{r['target_model']}`, "
            f"transfer success rate = `{safe_float(r['transfer_success_rate']):.4f}`."
        )

    lines.append("\n## Best attack per target model\n")
    for _, r in best_by_target.iterrows():
        lines.append(
            f"- `{r['dataset']} / {r['target_model']}`: best attack = `{r['attack']}`, "
            f"transfer success rate = `{safe_float(r['transfer_success_rate']):.4f}`, "
            f"accuracy drop = `{safe_float(r['accuracy_drop']):.4f}`, "
            f"macro-F1 drop = `{safe_float(r['macro_f1_drop']):.4f}`."
        )

    lines.append("\n## Attack average ranking\n")
    lines.append(attack_avg.to_markdown(index=False))

    lines.append("\n## Perturbation anomaly check\n")
    if anomaly_df.empty:
        lines.append("- No perturbation anomaly detected under `linf > 1` or `l2 > 5`.")
    else:
        lines.append(
            "- Some adversarial files contain extreme perturbation samples. "
            "Because the 99.9% quantiles are much smaller than the maximum values, "
            "the anomaly is concentrated in a small number of samples rather than the whole attack set."
        )
        lines.append("")
        keep_cols = [
            "dataset",
            "target_model",
            "attack",
            "max_l2_perturbation",
            "max_linf_perturbation",
            "l2_q0.999",
            "linf_q0.999",
            "num_linf_gt_1",
            "num_l2_gt_5",
        ]
        lines.append(anomaly_df[keep_cols].to_markdown(index=False))

    lines.append("\n## Suggested wording for paper/report\n")
    lines.append(
        "> Most adversarial samples are constrained within a reasonable perturbation range. "
        "However, a small number of samples show unusually large maximum L2/Linf perturbations, "
        "which may be caused by feature normalization boundaries, inverse-scaling artifacts, or extreme original feature values. "
        "Therefore, this study reports both maximum perturbation and high-quantile perturbation statistics "
        "to avoid overestimating the global perturbation magnitude."
    )

    lines.append("\n## Generated files\n")
    lines.append("- `results/summary/all_transfer_matrix.csv`")
    lines.append("- `results/summary/all_transfer_matrix.md`")
    lines.append("- `results/summary/all_metrics_detail.csv`")
    lines.append("- `results/summary/result_summary.md`")
    lines.append("- `results/summary/plots/transfer_success_rate_bar.png`")
    lines.append("- `results/summary/plots/accuracy_drop_bar.png`")
    lines.append("- `results/summary/plots/macro_f1_drop_bar.png`")
    lines.append("- `results/summary/plots/transfer_success_rate_grouped.png`")
    lines.append("- `results/summary/plots/transfer_success_rate_heatmap.png`")
    lines.append("- `results/summary/plots/perturbation_linf_999.png`")

    return "\n".join(lines)


def main():
    ensure_dirs()

    _ = load_final_matrices()
    metrics_df = load_metric_jsons()

    metrics_df = metrics_df.sort_values(
        ["dataset", "target_model", "attack"]
    ).reset_index(drop=True)

    all_csv = SUMMARY_DIR / "all_transfer_matrix.csv"
    all_md = SUMMARY_DIR / "all_transfer_matrix.md"
    metrics_csv = SUMMARY_DIR / "all_metrics_detail.csv"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    save_markdown_table(metrics_df, all_md)

    plot_bar(
        metrics_df,
        "transfer_success_rate",
        "Transfer Success Rate",
        "Transfer Success Rate",
        "transfer_success_rate_bar.png",
    )

    plot_bar(
        metrics_df,
        "accuracy_drop",
        "Accuracy Drop",
        "Accuracy Drop",
        "accuracy_drop_bar.png",
    )

    plot_bar(
        metrics_df,
        "macro_f1_drop",
        "Macro-F1 Drop",
        "Macro-F1 Drop",
        "macro_f1_drop_bar.png",
    )

    plot_bar(
        metrics_df,
        "linf_q0.999",
        "99.9% Linf Perturbation",
        "Linf q0.999",
        "perturbation_linf_999.png",
    )

    plot_grouped_by_dataset_attack(
        metrics_df,
        "transfer_success_rate",
        "transfer_success_rate_grouped.png",
        "Transfer Success Rate by Dataset and Target Model",
    )

    plot_heatmap_like(
        metrics_df,
        "transfer_success_rate",
        "transfer_success_rate_heatmap.png",
        "Transfer Success Rate Heatmap",
    )

    summary_text = build_summary_text(metrics_df)
    summary_path = SUMMARY_DIR / "result_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"saved: {all_csv}")
    print(f"saved: {all_md}")
    print(f"saved: {metrics_csv}")
    print(f"saved: {summary_path}")
    print(f"saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
