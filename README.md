# NIDS Project Full Code

基于黑盒迁移的 AI 对抗攻击研究项目，面向网络入侵检测系统（NIDS）的对抗鲁棒性评估与黑盒迁移攻击实验。

本项目围绕 **NSL-KDD** 数据集，构建了从数据预处理、基线模型训练、黑盒查询、替代模型（surrogate）训练、白盒攻击生成到跨模型迁移评估的完整实验链路。当前已完成 **最小完整攻击矩阵** 与 **完整攻击矩阵** 的首轮实验，并形成可复现的结果表。

---

## 1. Project Overview

本项目的目标是：

1. 训练多种异构 NIDS 基线模型；
2. 在黑盒设定下，利用少量种子样本 + 数据增强训练 surrogate 模型；
3. 在 surrogate 上生成白盒对抗样本；
4. 将对抗样本迁移到目标黑盒模型上，量化其鲁棒性；
5. 形成标准化的对抗鲁棒性评估流程与结果表。

当前重点围绕以下目标模型展开：

- TabNet
- XGBoost
- GBDT
- Random Forest（基线对比）

当前重点围绕以下攻击方法展开：

- FGM
- PGD
- SLIDE

---

## 2. Current Progress

### 2.1 Completed modules

当前已经完成并验证可运行的模块包括：

#### Data pipeline

- `src.data.load_raw`
- `src.data.clean_labels`
- `src.data.split_data`
- `src.preprocess.run_preprocess_pipeline`

#### Baseline models

- `src.models.train_xgb`
- `src.models.train_tabnet`
- `src.models.train_gbdt`
- `src.models.train_sklearn_baseline --model random_forest`
- `src.reporting.compare_models`

#### Black-box query and surrogate pipeline

- `src.blackbox.query_api`
- `src.data.build_seed_set`
- `src.data.query_seed_labels`
- `src.augment.run_mixup`
- `src.data.build_surrogate_trainset`
- `src.models.train_surrogate_mlp`
- `src.evaluation.evaluate_surrogate`

#### Transfer attack pipeline

- `src.transfer.generate_from_surrogate`
- `src.transfer.attack_target`

#### Batch experiment and reporting

- `scripts/run_min_transfer_matrix.ps1`
- `scripts/run_full_attack_matrix.ps1`
- `scripts/summarize_transfer_matrix.py`

---

## 3. Repository Structure

项目主要目录说明如下：

```text
artifacts/
  metadata/        # 标签映射、模型元信息、best surrogate 配置等
  models/          # 训练后的模型文件（xgb/tabnet/surrogate 等）
  preprocessors/   # 预处理器与特征信息

data/
  nsl_kdd/
    raw/           # 原始数据
    processed/     # 清洗、划分、预处理后的数据
  seeds/           # 黑盒查询种子集与 queried 种子集
  mixup/           # mixup 增强结果
  surrogate_train/ # surrogate 训练集
  adversarial/     # surrogate 生成的对抗样本

results/
  tables/          # 指标表、迁移矩阵、消融汇总表等

logs/              # 训练/处理日志

src/
  attacks/         # 攻击实现
  augment/         # 数据增强
  blackbox/        # 黑盒查询接口
  data/            # 数据处理
  evaluation/      # 评估逻辑
  models/          # 模型训练
  preprocess/      # 预处理流水线
  reporting/       # 结果汇总
  transfer/        # 迁移攻击
  visualization/   # 可视化
```

---

## 4. Environment

建议使用独立虚拟环境运行：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果导出 Markdown 表格时遇到 `to_markdown()` 相关问题，请额外安装：

```bash
pip install tabulate
```

---

## 5. End-to-End Pipeline

## 5.1 Data preparation

```bash
python -m src.data.load_raw --dataset nsl_kdd
python -m src.data.clean_labels --dataset nsl_kdd --mode 5class
python -m src.data.split_data --dataset nsl_kdd
python -m src.preprocess.run_preprocess_pipeline --dataset nsl_kdd
```

### Expected outputs

```text
data/nsl_kdd/processed/
  nsl_kdd_labeled.parquet
  train.parquet
  val.parquet
  test.parquet
  train_features.parquet
  val_features.parquet
  test_features.parquet
  X_train.npy
  X_val.npy
  X_test.npy
  y_train.npy
  y_val.npy
  y_test.npy

artifacts/preprocessors/
  nsl_kdd_preprocessor.joblib
  nsl_kdd_feature_info.json

artifacts/metadata/
  nsl_kdd_label_map.json
```

---

## 5.2 Baseline model training

```bash
python -m src.models.train_xgb --dataset nsl_kdd
python -m src.models.train_tabnet --dataset nsl_kdd
python -m src.models.train_gbdt --dataset nsl_kdd
python -m src.models.train_sklearn_baseline --dataset nsl_kdd --model random_forest
python -m src.reporting.compare_models --dataset nsl_kdd
```

### Expected outputs

```text
artifacts/models/
  xgb_nsl_kdd.pkl
  tabnet_nsl_kdd.zip
  gbdt_nsl_kdd.pkl
  random_forest_nsl_kdd.pkl

artifacts/metadata/
  xgb_nsl_kdd_meta.json
  tabnet_nsl_kdd_meta.json
  gbdt_nsl_kdd_meta.json
  random_forest_nsl_kdd_meta.json

results/tables/
  xgb_nsl_kdd_metrics.json
  tabnet_nsl_kdd_metrics.json
  gbdt_nsl_kdd_metrics.json
  random_forest_nsl_kdd_metrics.json
  model_comparison_nsl_kdd.csv
```

---

## 5.3 Black-box seed query

```bash
python -m src.blackbox.query_api --dataset nsl_kdd --target_model tabnet

python -m src.data.build_seed_set --dataset nsl_kdd --seed_size 500
python -m src.data.query_seed_labels --dataset nsl_kdd --target_model tabnet --seed_size 500
```

### Expected outputs

```text
data/seeds/nsl_kdd/
  seed_500.parquet
  queried/
    tabnet_seed_500_queried.parquet
```

---

## 5.4 Mixup and surrogate dataset construction

```bash
python -m src.augment.run_mixup --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1
python -m src.data.build_surrogate_trainset --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1
```

### Expected outputs

```text
data/mixup/nsl_kdd/
  tabnet_seed_500_alpha_0.1.parquet

data/surrogate_train/nsl_kdd/
  tabnet_seed_500_seed_only.parquet
  tabnet_seed_500_alpha_0.1_mixup.parquet
```

---

## 5.5 Surrogate training and evaluation

```bash
python -m src.models.train_surrogate_mlp --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1 --depth 7
python -m src.evaluation.evaluate_surrogate --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1 --depth 7
```

### Expected outputs

```text
artifacts/models/
  surrogate_nsl_kdd_tabnet_seed500_a0.1_d7.pt

results/tables/
  surrogate_eval_nsl_kdd_tabnet_seed500_a0.1_d7.json
```

---

## 5.6 Transfer attack

```bash
python -m src.transfer.generate_from_surrogate --dataset nsl_kdd --target_model tabnet --attack pgd
python -m src.transfer.attack_target --dataset nsl_kdd --target_model tabnet --attack pgd
```

### Expected outputs

```text
data/adversarial/nsl_kdd/
  pgd_tabnet_seed500_a0.1_d7.parquet
  pgd_tabnet_seed500_a0.1_d7_meta.json

results/tables/
  transfer_pgd_nsl_kdd_tabnet.csv
```

---

## 6. Minimal complete matrix

为了先快速获得可分析结果，项目提供了最小完整矩阵脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_min_transfer_matrix.ps1
```

该脚本默认运行：

- dataset: `nsl_kdd`
- seed size: `500`
- alpha: `0.1`
- depth: `7`
- targets: `tabnet`, `xgb`, `gbdt`
- attacks: `fgm`, `pgd`, `slide`

最终自动汇总：

```text
results/tables/final_transfer_matrix_nsl_kdd.csv
results/tables/final_transfer_matrix_nsl_kdd.md
```

---

## 7. Full attack matrix

当前项目也支持首轮完整攻击矩阵脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_attack_matrix.ps1
```

该脚本用于：

- 补齐 seed / queried seed / mixup / surrogate trainset
- 批量执行 surrogate 消融与选优
- 对多个目标模型执行多攻击方法迁移评估
- 汇总最终矩阵

> 注意：完整脚本运行时间较长，建议先跑最小矩阵验证流程无误，再运行 full matrix。

---

## 8. Current experimental results

### 8.1 Minimal matrix (first valid run)

| attack | target_model | transfer_success_rate |
| ------ | ------------ | ---------------------:|
| FGM    | GBDT         | 0.5423                |
| FGM    | TabNet       | 0.1171                |
| FGM    | XGB          | 0.3448                |
| PGD    | GBDT         | 0.6591                |
| PGD    | TabNet       | 0.1703                |
| PGD    | XGB          | 0.3732                |
| SLIDE  | GBDT         | 0.6591                |
| SLIDE  | TabNet       | 0.1703                |
| SLIDE  | XGB          | 0.3732                |

### 8.2 Full attack matrix (current)

由 `.\scripts\run_full_attack_matrix.ps1` 得到的当前结果如下：

| attack | target_model | transfer_success_rate |
| ------ | ------------ | ---------------------:|
| FGM    | GBDT         | 0.4870                |
| FGM    | TabNet       | 0.1156                |
| FGM    | XGB          | 0.3971                |
| PGD    | GBDT         | 0.5308                |
| PGD    | TabNet       | 0.1733                |
| PGD    | XGB          | 0.3933                |
| SLIDE  | GBDT         | 0.5308                |
| SLIDE  | TabNet       | 0.1733                |
| SLIDE  | XGB          | 0.3933                |

### 8.3 Preliminary observations

当前结果显示：

1. **GBDT 最脆弱**  
   在三种攻击下，GBDT 的迁移成功率均最高。

2. **TabNet 最稳健**  
   TabNet 的迁移成功率显著低于 GBDT 和 XGB。

3. **XGB 居中**  
   XGB 的鲁棒性位于 TabNet 与 GBDT 之间。

4. **PGD / SLIDE 强于 FGM**  
   迭代型攻击整体迁移效果高于单步攻击。

5. **PGD 与 SLIDE 当前结果一致**  
   需要进一步检查 `slide` 的实现与参数设置，确认是否与 `pgd` 高度重合。

---

## 9. What is completed vs. what remains

### Completed

- NSL-KDD 数据准备与预处理
- 多基线模型训练与比较
- 黑盒查询接口
- seed / queried seed / mixup / surrogate trainset
- surrogate 训练与评估
- FGM / PGD / SLIDE 三类迁移攻击
- TabNet / XGB / GBDT 三类目标模型迁移矩阵
- 最小矩阵与首轮 full matrix 汇总脚本

### In progress / remaining

- 完整 surrogate 消融与正式 best surrogate 选优
- C&W 攻击补齐
- 第二数据集（如 UNSW-NB15）复现
- 更完整评估指标：扰动泛化度、结构鲁棒性、保真度
- t-SNE 可视化与类别敏感度分析
- 防御基线
- 原型软件与自动报告输出

---

## 10. Recommended next steps

建议后续按以下顺序推进：

1. 完成 surrogate 消融与正式 best surrogate 选择；
2. 补齐 C&W；
3. 在 `unsw_nb15` 上复现实验；
4. 增加可视化与保真度分析；
5. 构建标准化报告输出；
6. 进一步撰写论文与结题材料。

---

## 11. Acknowledgement

本项目当前技术路线与实验设计参考了入侵检测黑盒迁移攻击相关研究，并结合课题申请书目标进行了工程化实现与阶段性扩展。
