from __future__ import annotations

from pathlib import Path
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier


def load_sklearn_like_model(model_path: str | Path):
    return joblib.load(model_path)


def load_tabnet_model(model_path: str | Path) -> TabNetClassifier:
    clf = TabNetClassifier()
    clf.load_model(str(model_path))
    return clf


def load_target_model(dataset: str, model_name: str):
    artifacts_dir = Path("artifacts/models")

    if model_name == "tabnet":
        return load_tabnet_model(artifacts_dir / f"tabnet_{dataset}.zip")
    if model_name == "xgb":
        return load_sklearn_like_model(artifacts_dir / f"xgb_{dataset}.pkl")
    if model_name == "gbdt":
        return load_sklearn_like_model(artifacts_dir / f"gbdt_{dataset}.pkl")
    if model_name in {"random_forest", "decision_tree", "logistic_regression"}:
        return load_sklearn_like_model(artifacts_dir / f"{model_name}_{dataset}.pkl")

    raise ValueError(f"Unsupported target model: {model_name}")
