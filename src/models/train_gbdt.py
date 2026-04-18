from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.evaluation.classification_metrics import (
    compute_classification_metrics,
    save_metrics,
)
from src.models.predict import predict_labels
from src.models.save_model_metadata import save_model_metadata


def load_feature_bundle(dataset: str) -> dict:
    feature_dir = Path("data") / dataset / "processed"
    return {
        "X_train": np.load(feature_dir / "X_train.npy"),
        "y_train": np.load(feature_dir / "y_train.npy"),
        "X_test": np.load(feature_dir / "X_test.npy"),
        "y_test": np.load(feature_dir / "y_test.npy"),
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_feature_bundle(args.dataset)

    print(f"[GBDT] dataset={args.dataset}", flush=True)
    print(f"[GBDT] X_train={data['X_train'].shape}, y_train={data['y_train'].shape}", flush=True)
    print("[GBDT] training started...", flush=True)

    model = GradientBoostingClassifier(random_state=42)

    start = time.time()
    model.fit(data["X_train"], data["y_train"])
    train_time = time.time() - start

    print(f"[GBDT] training finished in {train_time:.2f}s", flush=True)
    print("[GBDT] evaluating...", flush=True)

    y_pred = predict_labels(model, data["X_test"])
    metrics = compute_classification_metrics(data["y_test"], y_pred)
    metrics["model_name"] = "gbdt"
    metrics["train_time_sec"] = train_time

    model_path = Path("artifacts/models") / f"gbdt_{args.dataset}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = Path("results/tables") / f"gbdt_{args.dataset}_metrics.json"
    save_metrics(metrics, metrics_path)

    save_model_metadata(
        {
            "model_name": "gbdt",
            "dataset": args.dataset,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        },
        Path("artifacts/metadata") / f"gbdt_{args.dataset}_meta.json",
    )

    print(f"[GBDT] accuracy={metrics['accuracy']:.6f} f1_macro={metrics['f1_macro']:.6f}", flush=True)


if __name__ == "__main__":
    main()
