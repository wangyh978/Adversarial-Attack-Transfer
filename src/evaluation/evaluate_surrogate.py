from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import torch

from src.blackbox.query_api import BlackBoxModel
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.agreement_metrics import compute_agreement
from src.models.mlp_surrogate import MLPSurrogate
from src.utils.io import load_json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_dir = Path("data") / args.dataset / "processed"
    X_test = np.load(processed_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(processed_dir / "y_test.npy")

    feature_info = load_json(Path("artifacts/preprocessors") / f"{args.dataset}_feature_info.json")
    num_classes = int(feature_info["num_classes"])

    model = MLPSurrogate(
        input_dim=X_test.shape[1],
        num_classes=num_classes,
        depth=args.depth,
    )
    model_path = Path("artifacts/models") / f"surrogate_{args.dataset}_{args.target_model}_seed{args.seed_size}_a{args.alpha}_d{args.depth}.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    cls_metrics = compute_classification_metrics(y_test, y_pred)

    blackbox = BlackBoxModel(args.dataset, args.target_model)
    y_target = blackbox.predict_label(X_test)
    agreement_metrics = compute_agreement(y_target, y_pred)

    metrics = {**cls_metrics, **agreement_metrics}

    out_path = Path("results/tables") / f"surrogate_eval_{args.dataset}_{args.target_model}_seed{args.seed_size}_a{args.alpha}_d{args.depth}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
