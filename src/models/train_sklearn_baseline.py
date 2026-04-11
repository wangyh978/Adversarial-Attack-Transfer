from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.classification_metrics import compute_classification_metrics, save_metrics
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


def build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=42)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--model", required=True, choices=["random_forest", "decision_tree", "logistic_regression"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_feature_bundle(args.dataset)
    model = build_model(args.model)

    model.fit(data["X_train"], data["y_train"])
    y_pred = predict_labels(model, data["X_test"])
    metrics = compute_classification_metrics(data["y_test"], y_pred)
    metrics["model_name"] = args.model

    model_path = Path("artifacts/models") / f"{args.model}_{args.dataset}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = Path("results/tables") / f"{args.model}_{args.dataset}_metrics.json"
    save_metrics(metrics, metrics_path)

    save_model_metadata(
        {"model_name": args.model, "dataset": args.dataset, "model_path": str(model_path), "metrics_path": str(metrics_path)},
        Path("artifacts/metadata") / f"{args.model}_{args.dataset}_meta.json",
    )

    print(metrics["accuracy"], metrics["f1_macro"])


if __name__ == "__main__":
    main()
