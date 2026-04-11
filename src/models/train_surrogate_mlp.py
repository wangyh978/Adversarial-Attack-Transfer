from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.data.prepare_surrogate_features import load_mixup_dataset
from src.models.mlp_surrogate import MLPSurrogate
from src.models.save_surrogate_metadata import save_surrogate_metadata
from src.utils.seed import set_global_seed
from src.utils.io import load_json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--seed_size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def mixup_ce_loss(logits, y_a, y_b, lam):
    loss_a = F.cross_entropy(logits, y_a, reduction="none")
    loss_b = F.cross_entropy(logits, y_b, reduction="none")
    return (lam * loss_a + (1.0 - lam) * loss_b).mean()


def main() -> None:
    args = parse_args()
    set_global_seed(42)

    X, y_a, y_b, lam, feature_cols = load_mixup_dataset(args.dataset, args.target_model, args.seed_size, args.alpha)

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
        torch.tensor(y_a, dtype=torch.long),
        torch.tensor(y_b, dtype=torch.long),
        torch.tensor(lam, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_x, batch_y_a, batch_y_b, batch_lam in loader:
            batch_x = batch_x.to(device)
            batch_y_a = batch_y_a.to(device)
            batch_y_b = batch_y_b.to(device)
            batch_lam = batch_lam.to(device)

            logits = model(batch_x)
            loss = mixup_ce_loss(logits, batch_y_a, batch_y_b, batch_lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(f"epoch={epoch+1} loss={total_loss/len(loader):.6f}")

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
            "model_path": str(model_path),
            "num_features": X.shape[1],
            "num_classes": num_classes,
        },
        Path("artifacts/metadata") / f"surrogate_{args.dataset}_{args.target_model}_seed{args.seed_size}_a{args.alpha}_d{args.depth}.json",
    )

    print("saved:", model_path)


if __name__ == "__main__":
    main()
