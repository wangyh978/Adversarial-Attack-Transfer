from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp_surrogate import MLPSurrogate
from src.models.save_surrogate_metadata import save_surrogate_metadata
from src.utils.io import load_json
from src.utils.seed import set_global_seed


def parse_args():
    parser = ArgumentParser(
        description="Train surrogate using paper-style hard labels from black-box relabelled mixup samples."
    )
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--trainset_path",
        type=str,
        default=None,
        help="Optional override. Defaults to *_paper_union.parquet if present, else *_mixup.parquet.",
    )
    return parser.parse_args()


def _resolve_trainset_path(args) -> Path:
    if args.trainset_path:
        return Path(args.trainset_path)

    out_dir = Path("data/surrogate_train") / args.dataset
    preferred = out_dir / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}_paper_union.parquet"
    fallback = out_dir / f"{args.target_model}_seed_{args.seed_size}_alpha_{args.alpha}_mixup.parquet"

    if preferred.exists():
        return preferred
    return fallback


def main() -> None:
    args = parse_args()
    set_global_seed(42)

    trainset_path = _resolve_trainset_path(args)
    df = pd.read_parquet(trainset_path)

    feature_cols = [c for c in df.columns if str(c).startswith("f_")]
    if "target_label" not in df.columns:
        raise ValueError(f"Expected target_label in {trainset_path}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target_label"].to_numpy(dtype=np.int64)

    feature_info = load_json(Path("artifacts/preprocessors") / f"{args.dataset}_feature_info.json")
    num_classes = int(feature_info["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPSurrogate(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_seen += int(batch_y.numel())

        avg_loss = total_loss / max(1, len(loader))
        acc = total_correct / max(1, total_seen)
        print(f"epoch={epoch+1} loss={avg_loss:.6f} train_acc={acc:.6f}")

    out_dir = Path("artifacts/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"surrogate_{args.dataset}_{args.target_model}_seed{args.seed_size}_a{args.alpha}_d{args.depth}.pt"
    torch.save(model.state_dict(), model_path)

    save_surrogate_metadata(
        {
            "dataset": args.dataset,
            "target_model": args.target_model,
            "seed_size": args.seed_size,
            "alpha": args.alpha,
            "depth": args.depth,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "model_path": str(model_path),
            "num_features": int(X.shape[1]),
            "num_classes": num_classes,
            "trainset_path": str(trainset_path),
            "label_style": "paper_hard_label_from_blackbox_mixup",
        },
        Path("artifacts/metadata") / f"surrogate_{args.dataset}_{args.target_model}_seed{args.seed_size}_a{args.alpha}_d{args.depth}.json",
    )
    print("saved:", model_path)


if __name__ == "__main__":
    main()
