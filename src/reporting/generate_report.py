from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_model", default="tabnet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lines = []
    lines.append(f"# 实验报告：{args.dataset}")
    lines.append("")
    lines.append("## 1. 基线模型")
    model_cmp = Path("results/tables") / f"model_comparison_{args.dataset}.csv"
    if model_cmp.exists():
        df = pd.read_csv(model_cmp)
        lines.append(df.to_markdown(index=False))
    else:
        lines.append("尚未生成基线模型比较表。")

    lines.append("")
    lines.append("## 2. surrogate")
    surrogate_sum = Path("results/tables") / f"surrogate_ablation_summary_{args.dataset}_{args.target_model}.csv"
    if surrogate_sum.exists():
        df = pd.read_csv(surrogate_sum).head(10)
        lines.append(df.to_markdown(index=False))
    else:
        lines.append("尚未生成 surrogate 消融汇总表。")

    out_path = Path("results/reports") / f"report_{args.dataset}_{args.target_model}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
