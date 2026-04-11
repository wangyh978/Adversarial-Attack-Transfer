from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier

from src.evaluation.classification_metrics import compute_classification_metrics, save_metrics
from src.models.save_model_metadata import save_model_metadata


def load_feature_bundle(dataset: str) -> dict:
    feature_dir = Path("data") / dataset / "processed"
    return {
        "X_train": np.load(feature_dir / "X_train.npy"),
        "y_train": np.load(feature_dir / "y_train.npy"),
        "X_val": np.load(feature_dir / "X_val.npy"),
        "y_val": np.load(feature_dir / "y_val.npy"),
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

    clf = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        seed=42,
        verbose=1,
    )

    clf.fit(
        X_train=data["X_train"],
        y_train=data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
    )

    y_pred = clf.predict(data["X_test"])
    metrics = compute_classification_metrics(data["y_test"], y_pred)
    metrics["model_name"] = "tabnet"

    save_dir = Path("artifacts/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = save_dir / f"tabnet_{args.dataset}"
    clf.save_model(str(model_prefix))

    metrics_path = Path("results/tables") / f"tabnet_{args.dataset}_metrics.json"
    save_metrics(metrics, metrics_path)
    save_model_metadata(
        {"model_name": "tabnet", "dataset": args.dataset, "model_path": str(model_prefix) + ".zip", "metrics_path": str(metrics_path)},
        Path("artifacts/metadata") / f"tabnet_{args.dataset}_meta.json",
    )

    print(metrics["accuracy"], metrics["f1_macro"])


if __name__ == "__main__":
    main()
