# UNSW-NB15 第二数据集接入说明

这是一份按当前仓库结构整理的 **UNSW-NB15 增量覆盖包说明**。

## 1. 放置原始数据

将两个官方切分文件放到：

```text
data/unsw_nb15/raw/
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

## 2. 覆盖本增量包

把压缩包解压到项目根目录，允许覆盖同名文件：

- `src/data/label_maps.py`
- `src/data/load_raw.py`
- `src/data/clean_labels.py`
- `src/data/split_data.py`
- `src/preprocess/run_preprocess_pipeline.py`
- `scripts/run_unsw_prepare.ps1`
- `scripts/run_unsw_baselines.ps1`
- `scripts/run_surrogate_ablation_unsw_formal.ps1`
- `scripts/run_full_attack_matrix_unsw.ps1`

## 3. 先跑数据层

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_unsw_prepare.ps1
```

预期输出：

- `data/unsw_nb15/processed/raw_loaded.parquet`
- `data/unsw_nb15/processed/unsw_nb15_labeled.parquet`
- `data/unsw_nb15/processed/train.parquet`
- `data/unsw_nb15/processed/val.parquet`
- `data/unsw_nb15/processed/test.parquet`
- `data/unsw_nb15/processed/train_features.parquet`
- `data/unsw_nb15/processed/val_features.parquet`
- `data/unsw_nb15/processed/test_features.parquet`
- `data/unsw_nb15/processed/X_train.npy`
- `data/unsw_nb15/processed/y_train.npy`
- `artifacts/metadata/unsw_nb15_label_map.json`
- `artifacts/preprocessors/unsw_nb15_preprocessor.joblib`

## 4. 再跑 baseline

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_unsw_baselines.ps1
```

## 5. 正式 surrogate 消融

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_surrogate_ablation_unsw_formal.ps1
```

## 6. 再跑完整迁移矩阵

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_attack_matrix_unsw.ps1
```

## 7. 说明

- 本覆盖包优先适配你当前仓库的 **目录结构、命令命名和数据流**
- 预处理输出保持 `f_0, f_1, ...` 风格，兼容你当前 mixup / surrogate 主线
- `split_data.py` 对 `UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv` 优先采用官方 train/test 切分，只从官方训练集再切一部分做 `val`
