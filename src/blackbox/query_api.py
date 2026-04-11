from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from src.blackbox.load_target_model import load_target_model
from src.blackbox.query_logger import append_query_log


class BlackBoxModel:
    def __init__(self, dataset: str, model_name: str):
        self.dataset = dataset
        self.model_name = model_name
        self.model = load_target_model(dataset, model_name)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        append_query_log(self.dataset, self.model_name, len(X))
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray):
        append_query_log(self.dataset, self.model_name, len(X))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt", "random_forest", "decision_tree", "logistic_regression"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bb = BlackBoxModel(args.dataset, args.target_model)

    feature_dir = Path("data") / args.dataset / "processed"
    X_test = np.load(feature_dir / "X_test.npy")[:8]

    y_pred = bb.predict_label(X_test)
    y_prob = bb.predict_proba(X_test)

    print("pred labels:", y_pred)
    if y_prob is not None:
        print("prob shape:", y_prob.shape)


if __name__ == "__main__":
    main()
