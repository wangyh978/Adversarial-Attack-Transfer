from __future__ import annotations

import pandas as pd


def sample_by_class(
    df: pd.DataFrame,
    label_col: str,
    total_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    classes = sorted(df[label_col].unique().tolist())
    per_class = max(total_size // len(classes), 1)

    sampled = []
    used_idx = []
    for cls in classes:
        part = df[df[label_col] == cls]
        n = min(len(part), per_class)
        pick = part.sample(n=n, random_state=random_state, replace=False)
        sampled.append(pick)
        used_idx.extend(pick.index.tolist())

    out = pd.concat(sampled, axis=0)

    if len(out) < total_size:
        remain = df.loc[~df.index.isin(used_idx)]
        gap = total_size - len(out)
        if len(remain) > 0:
            extra = remain.sample(n=min(gap, len(remain)), random_state=random_state, replace=False)
            out = pd.concat([out, extra], axis=0)

    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out
