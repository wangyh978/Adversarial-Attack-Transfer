from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.io import ensure_dir

logger = get_logger("load_raw", "logs/load_raw.log")

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label_raw", "difficulty",
]


def _find_files(raw_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(raw_dir.glob(pat)))
    return files


def load_nsl_kdd(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = _find_files(raw_dir, ["*.txt", "*.csv", "*.data"])

    if not files:
        raise FileNotFoundError(f"未在 {raw_dir} 下找到 NSL-KDD 原始文件")

    file_path = files[0]
    df = pd.read_csv(file_path, header=None)

    if df.shape[1] == 43:
        df.columns = NSL_KDD_COLUMNS
    elif df.shape[1] == 42:
        df.columns = NSL_KDD_COLUMNS[:-1]
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    df["dataset_name"] = "nsl_kdd"
    logger.info("Loaded NSL-KDD from %s shape=%s", file_path, df.shape)
    return df


def load_unsw_nb15(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)

    preferred_train = raw_dir / "UNSW_NB15_training-set.csv"
    preferred_test = raw_dir / "UNSW_NB15_testing-set.csv"

    def _smart_read_csv(fp: Path) -> pd.DataFrame:
        # 先尝试默认逗号
        try:
            df = pd.read_csv(fp)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="latin1")

        joined_cols = " ".join(map(str, df.columns))

        # 注意这里必须是 "\t"，不是 "\\t"
        if df.shape[1] <= 3 and "\t" in joined_cols:
            try:
                df = pd.read_csv(fp, sep="\t")
            except UnicodeDecodeError:
                df = pd.read_csv(fp, sep="\t", encoding="latin1")

        return df


    dfs: list[pd.DataFrame] = []

    if preferred_train.exists() and preferred_test.exists():
        train_df = _smart_read_csv(preferred_train)
        test_df = _smart_read_csv(preferred_test)

        train_df["split_source"] = "train_official"
        test_df["split_source"] = "test_official"
        dfs.extend([train_df, test_df])

        logger.info(
            "Loaded UNSW-NB15 official split from %s and %s",
            preferred_train,
            preferred_test,
        )
    else:
        csv_files = _find_files(raw_dir, ["*.csv"])
        if not csv_files:
            raise FileNotFoundError(f"未在 {raw_dir} 下找到 UNSW-NB15 原始 CSV 文件")

        for fp in csv_files:
            df_part = _smart_read_csv(fp)

            lower_name = fp.name.lower()
            if "training" in lower_name or "train" in lower_name:
                df_part["split_source"] = "train_official"
            elif "testing" in lower_name or "test" in lower_name:
                df_part["split_source"] = "test_official"
            else:
                df_part["split_source"] = "unknown"
            dfs.append(df_part)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["dataset_name"] = "unsw_nb15"

    logger.info("Loaded UNSW-NB15 shape=%s", df.shape)
    return df



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--save", action="store_true", help="是否将原始拼接结果保存到 processed/raw_loaded.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "nsl_kdd":
        raw_dir = args.raw_dir or "data/nsl_kdd/raw"
        df = load_nsl_kdd(raw_dir)
    else:
        raw_dir = args.raw_dir or "data/unsw_nb15/raw"
        df = load_unsw_nb15(raw_dir)

    print(df.head())
    print("shape:", df.shape)

    if args.save:
        out_dir = ensure_dir(Path("data") / args.dataset / "processed")
        out_path = out_dir / "raw_loaded.parquet"
        df.to_parquet(out_path, index=False)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
