# 🔥 NIDS Adversarial Attack Transfer

## 1. 项目简介

本项目围绕：

> **“面向入侵检测的黑盒对抗攻击迁移性研究”**

构建一套**完整可复现的 NIDS 对抗鲁棒性评估框架**，核心目标是：

- 在**黑盒场景**下评估入侵检测模型的安全性
- 利用少量查询数据构建 surrogate 模型
- 研究不同攻击方法在**异构模型之间的迁移能力**
- 对比不同数据集（NSL-KDD / UNSW-NB15）的迁移差异

---

## 2. 核心方法

```text
Raw Data
   ↓
Preprocess
   ↓
Target Model（黑盒）
   ↓
Seed Query
   ↓
Mixup Augmentation
   ↓
Surrogate Model
   ↓
FGM / PGD / SLIDE
   ↓
Transfer Attack
   ↓
Robustness Evaluation
```

---

## 3. 当前项目进度

### ✅ 已完成

### （1）NSL-KDD 主线（已完整）

- 数据清洗（5分类）
- 特征工程
- 多模型训练（TabNet / XGB / GBDT）
- surrogate（mixup）训练
- **FGM / PGD / SLIDE 全攻击完成**
- **Full Attack Matrix 完整跑通**

---

### （2）UNSW-NB15 主线（已打通）

- 数据清洗（多分类）
- baseline 对比（RF / XGB / GBDT / TabNet）
- target = XGB（当前主模型）
- surrogate 训练完成
- **FGM / PGD / SLIDE 全攻击完成**
- Full pipeline 跑通

---

### ❗ 未完成

- UNSW 多 target（GBDT / TabNet）
- 鲁棒性指标（accuracy drop / F1 drop）
- 可视化（图表）
- 自动报告

---

## 4. 项目结构

```text
.
├── src/                      # 核心代码
│   ├── attacks/             # 攻击算法（FGM / PGD / SLIDE）
│   ├── augment/             # mixup 数据增强
│   ├── blackbox/            # 黑盒查询接口
│   ├── data/                # 数据构建（seed / surrogate）
│   ├── evaluation/          # 指标计算
│   ├── models/              # target + surrogate 模型
│   ├── preprocess/          # 特征工程
│   ├── reporting/           # 结果汇总
│   ├── transfer/            # 迁移攻击核心逻辑
│   ├── utils/               # IO / 工具函数
│   └── visualization/       # 可视化（后续使用）
│
├── scripts/                 # ⭐ 一键运行入口（强烈建议用这个）
│   ├── run_min_transfer_matrix.ps1
│   ├── run_full_attack_matrix.ps1
│   ├── run_nsl_kdd_pipeline.ps1
│   └── run_unsw_nb15_pipeline.ps1
│
├── data/
│   ├── nsl_kdd/
│   │   ├── raw/
│   │   └── processed/
│   ├── unsw_nb15/
│   ├── seeds/               # 查询数据
│   ├── mixup/               # 增强数据
│   ├── surrogate_train/
│   └── adversarial/         # 对抗样本
│
├── artifacts/
│   ├── models/              # 训练好的模型
│   ├── preprocessors/       # scaler / feature info
│   └── metadata/
│
├── results/
│   ├── tables/              # CSV / metrics
│   ├── figures/             # 图表
│   └── reports/             # 报告
│
├── configs/                 # 参数配置（可扩展）
├── logs/                    # 运行日志
├── README.md
└── requirements.txt
```

---

## 5. 运行方式

### ⭐ 推荐：脚本一键运行

### NSL-KDD（完整攻击矩阵）
```powershell
.\scripts\run_nsl_kdd_pipeline.ps1 -Stage FullAttackMatrix
```

### UNSW-NB15（完整攻击矩阵）
```powershell
.\scripts\run_unsw_nb15_pipeline.ps1 -Stage FullAttackMatrix
```

### 只跑快速验证（推荐调试）
```powershell
.\scripts\run_nsl_kdd_pipeline.ps1 -Stage MinTransfer
```

---

## 6. 手动运行顺序

### NSL-KDD
```powershell
python -m src.data.load_raw --dataset nsl_kdd
python -m src.data.clean_labels --dataset nsl_kdd --mode 5class
python -m src.data.split_data --dataset nsl_kdd
python -m src.preprocess.run_preprocess_pipeline --dataset nsl_kdd

python -m src.models.train_xgb --dataset nsl_kdd
python -m src.models.train_gbdt --dataset nsl_kdd
python -m src.models.train_tabnet --dataset nsl_kdd
python -m src.reporting.compare_models --dataset nsl_kdd

python -m src.data.build_seed_set --dataset nsl_kdd --seed_size 1000
python -m src.data.query_seed_labels --dataset nsl_kdd --target_model tabnet --seed_size 1000
python -m src.augment.run_mixup --dataset nsl_kdd --target_model tabnet --seed_size 1000 --alpha 0.1
python -m src.data.build_surrogate_trainset --dataset nsl_kdd --target_model tabnet --seed_size 1000 --alpha 0.1
python -m src.models.train_surrogate_mlp --dataset nsl_kdd --target_model tabnet --seed_size 1000 --alpha 0.1 --depth 3
python -m src.evaluation.evaluate_surrogate --dataset nsl_kdd --target_model tabnet --seed_size 1000 --alpha 0.1 --depth 3

python -m src.transfer.generate_from_surrogate --dataset nsl_kdd --target_model tabnet --seed_size 1000 --alpha 0.1 --depth 3 --attack fgm
python -m src.transfer.attack_target --dataset nsl_kdd --target_model tabnet --seed_size 1000 --alpha 0.1 --depth 3 --attack fgm
```

### UNSW-NB15
```powershell
python -m src.data.load_raw --dataset unsw_nb15
python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
python -m src.data.split_data --dataset unsw_nb15
python -m src.preprocess.run_preprocess_pipeline --dataset unsw_nb15

python -m src.models.train_xgb --dataset unsw_nb15
python -m src.models.train_gbdt --dataset unsw_nb15
python -m src.models.train_tabnet --dataset unsw_nb15
python -m src.reporting.compare_models --dataset unsw_nb15

python -m src.data.build_seed_set --dataset unsw_nb15 --seed_size 1000
python -m src.data.query_seed_labels --dataset unsw_nb15 --target_model xgb --seed_size 1000
python -m src.augment.run_mixup --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1
python -m src.data.build_surrogate_trainset --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1
python -m src.models.train_surrogate_mlp --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1 --depth 3
python -m src.evaluation.evaluate_surrogate --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1 --depth 3

python -m src.transfer.generate_from_surrogate --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1 --depth 3 --attack fgm
python -m src.transfer.attack_target --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1 --depth 3 --attack fgm

python -m src.transfer.generate_from_surrogate --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1 --depth 3 --attack pgd
python -m src.transfer.attack_target --dataset unsw_nb15 --target_model xgb --seed_size 1000 --alpha 0.1 --depth 3 --attack pgd
```

---

## 7. 实验结果（更新后的核心部分）

## 🔹 NSL-KDD（Full Attack Matrix）

### target = TabNet（已有结果）

| attack | transfer |
|--------|---------:|
| FGM    | 0.061 |
| PGD    | 0.402 |
| SLIDE  | 0.402 |

### target = GBDT（本次新结果）

```text
FGM   = 0.2578
PGD   = 0.3040
SLIDE = 0.3040
```

👉 现象：

- GBDT 比 TabNet 更容易被攻击
- FGM 有一定效果（不像 TabNet 完全失效）
- PGD / SLIDE 表现接近

---

## 🔹 UNSW-NB15（Full Attack Matrix）

### target = XGB

```text
FGM   = 0.6250
PGD   = 0.6277
SLIDE = 0.6354
```

👉 现象：

- 整体迁移成功率显著高于 NSL-KDD
- 三种攻击均有效
- **SLIDE 最强**

---

## 8. 关键实验结论

### 1️⃣ 数据集差异极大

| 数据集 | 迁移水平 |
|--------|---------|
| NSL-KDD | 低 ~ 中 |
| UNSW    | 高 |

👉 UNSW 更容易被攻击

### 2️⃣ 模型结构影响显著

| 模型 | 迁移性 |
|------|-------|
| TabNet | 最难攻击 |
| GBDT   | 中等 |
| XGB    | 较易攻击 |

### 3️⃣ 攻击方法排序不固定

| 数据集 | 最强攻击 |
|--------|----------|
| NSL-KDD（TabNet） | PGD ≈ SLIDE |
| NSL-KDD（GBDT）   | PGD ≈ SLIDE |
| UNSW              | SLIDE |

👉 说明：

> **攻击方法的优劣依赖于数据分布 + 模型结构**

### 4️⃣ FGM 不稳定

- NSL-KDD + TabNet：几乎无效
- NSL-KDD + GBDT：中等
- UNSW：有效

👉 说明：

> 单步攻击迁移性不稳定

### 5️⃣ 当前最强组合

```text
UNSW-NB15 + XGB + SLIDE ≈ 0.635
```

👉 可以作为展示结果

---

## 9. 下一步计划

### 优先级 1（必须做）
- UNSW → 加 GBDT / TabNet

### 优先级 2（论文需要）
- accuracy drop
- F1 drop

### 优先级 3（展示）
- 柱状图
- transfer matrix 图

---

## 10. 一句话总结

> 本项目已完成 NSL-KDD 全攻击矩阵评估，并成功将完整对抗迁移流程扩展到 UNSW-NB15，实验表明不同数据集与模型结构对迁移攻击效果具有显著影响，目前已进入结果分析与论文整理阶段。
