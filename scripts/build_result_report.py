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
            "l2_q0.99": data.get("l2_q0.99"),
            "linf_q0.99": data.get("linf_q0.99"),
            "l2_q0.999": data.get("l2_q0.999"),
            "linf_q0.999": data.get("linf_q0.999"),
            "num_linf_gt_1": data.get("num_linf_gt_1"),
            "num_l2_gt_5": data.get("num_l2_gt_5"),
            "metric_file": f.name,
        })

    return pd.DataFrame(rows)


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


def plot_heatmap(df, value_col, filename, title):
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
    numeric_cols = [
        "transfer_success_rate", "accuracy_drop", "macro_f1_drop",
        "mean_l2_perturbation", "linf_q0.999", "num_linf_gt_1", "num_l2_gt_5"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    best = df.sort_values("transfer_success_rate", ascending=False).iloc[0]
    best_by_dataset = df.sort_values("transfer_success_rate", ascending=False).groupby("dataset").head(1)
    best_by_target = df.sort_values("transfer_success_rate", ascending=False).groupby(["dataset", "target_model"]).head(1)
    attack_rank = df.groupby("attack", as_index=False)["transfer_success_rate"].mean().sort_values("transfer_success_rate", ascending=False)

    anomaly_df = df[(df["num_linf_gt_1"].fillna(0) > 0) | (df["num_l2_gt_5"].fillna(0) > 0)]

    lines = []
    lines.append("# MSM Transfer Attack Result Summary\n")
    lines.append("## Overall conclusion\n")
    lines.append(
        f"- Best result: `{best['dataset']} / {best['target_model']} / {best['attack']}`, "
        f"transfer_success_rate = `{best['transfer_success_rate']:.4f}`."
    )

    lines.append("\n## Best attack per dataset\n")
    for _, r in best_by_dataset.iterrows():
        lines.append(f"- `{r['dataset']}`: `{r['target_model']} / {r['attack']}` = `{r['transfer_success_rate']:.4f}`")

    lines.append("\n## Best attack per target model\n")
    for _, r in best_by_target.iterrows():
        lines.append(
            f"- `{r['dataset']} / {r['target_model']}`: `{r['attack']}`, "
            f"transfer = `{r['transfer_success_rate']:.4f}`, "
            f"accuracy_drop = `{r['accuracy_drop']:.4f}`, "
            f"macro_f1_drop = `{r['macro_f1_drop']:.4f}`"
        )

    lines.append("\n## Attack average ranking\n")
    lines.append(attack_rank.to_markdown(index=False))

    lines.append("\n## Perturbation anomaly check\n")
    if anomaly_df.empty:
        lines.append("- No anomaly found under `linf > 1` or `l2 > 5`.")
    else:
        keep = [
            "dataset", "target_model", "attack",
            "max_l2_perturbation", "max_linf_perturbation",
            "l2_q0.999", "linf_q0.999", "num_linf_gt_1", "num_l2_gt_5",
        ]
        lines.append(
            "- Some extreme perturbation samples exist. Since high quantiles are much smaller than maxima, "
            "these are concentrated outliers rather than global perturbation inflation."
        )
        lines.append("")
        lines.append(anomaly_df[keep].to_markdown(index=False))

    lines.append("\n## Suggested report wording\n")
    lines.append(
        "> Most adversarial samples are constrained within a reasonable perturbation range. "
        "A small number of samples show unusually large maximum L2/Linf perturbations, "
        "likely caused by normalization boundaries, inverse-scaling artifacts, or extreme original feature values. "
        "Therefore, both maximum perturbation and high-quantile perturbation statistics are reported."
    )

    return "\n".join(lines)


def main():
    ensure_dirs()
    df = load_metric_jsons()
    df = df.sort_values(["dataset", "target_model", "attack"]).reset_index(drop=True)

    all_csv = SUMMARY_DIR / "all_transfer_matrix.csv"
    all_md = SUMMARY_DIR / "all_transfer_matrix.md"
    summary_md = SUMMARY_DIR / "result_summary.md"

    df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    save_markdown_table(df, all_md)

    plot_bar(df, "transfer_success_rate", "Transfer Success Rate", "Transfer Success Rate", "transfer_success_rate_bar.png")
    plot_bar(df, "accuracy_drop", "Accuracy Drop", "Accuracy Drop", "accuracy_drop_bar.png")
    plot_bar(df, "macro_f1_drop", "Macro-F1 Drop", "Macro-F1 Drop", "macro_f1_drop_bar.png")
    plot_bar(df, "linf_q0.999", "99.9% Linf Perturbation", "Linf q0.999", "perturbation_linf_999.png")
    plot_heatmap(df, "transfer_success_rate", "transfer_success_rate_heatmap.png", "Transfer Success Rate Heatmap")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(build_summary_text(df))

    print(f"saved: {all_csv}")
    print(f"saved: {all_md}")
    print(f"saved: {summary_md}")
    print(f"saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
