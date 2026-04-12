# NIDS 对抗鲁棒性评估项目

本项目用于实现**网络入侵检测系统（NIDS）的黑盒迁移对抗攻击与鲁棒性评估**，覆盖从数据准备、预处理、异构基线模型、黑盒查询、种子集构造、mixup 数据增强、surrogate 训练，到白盒攻击生成、黑盒迁移评估和结果汇总的完整工程骨架。

当前项目已经完成：

- NSL-KDD 数据链路打通
- 基线目标模型训练与保存
- 黑盒查询接口
- 种子集构建与标签查询
- mixup 增强与 surrogate 训练
- surrogate 消融与 formal best surrogate selection
- 基础白盒攻击（FGM / PGD / SLIDE）
- full attack matrix 迁移评估

---

## 目标

- 使用 NSL-KDD / UNSW-NB15 等公开数据集
- 训练异构基线模型：TabNet、XGBoost、GBDT、sklearn baseline
- 将目标模型封装为黑盒接口
- 基于少量种子样本和 mixup 训练 surrogate 模型
- 在 surrogate 上生成对抗样本，并评估其对黑盒目标模型的迁移效果
- 构建面向 NIDS 的对抗鲁棒性评估实验流程

---

## 当前进度概览

当前主实验已经进入 **formal surrogate + full transfer evaluation** 阶段。

已完成主线：

- `NSL-KDD` 数据预处理与特征管线
- `TabNet / XGB / GBDT` 黑盒目标模型训练
- `seed_size = 500 / 1000 / 2000` 种子集构建
- `alpha = 0.1 / 0.2 / 0.5` 的 mixup 数据增强
- surrogate MLP 深度消融：`depth = 3 / 5 / 7`
- 正式 surrogate 选优与 metadata 固化
- 基于正式 best surrogate 的 full attack matrix 重跑

当前推荐作为正式结果使用的部分：

- formal best surrogate selection
- 基于 formal best surrogate 的 transfer success rate 结果
- 后续 README、报告、论文初稿建议都以这套结果为主

---

## 环境

- Python 3.10 或 3.11
- pip >= 23

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 目录结构

```text
.
├─ README.md
├─ requirements.txt
├─ main.py
├─ configs/
├─ data/
│  ├─ nsl_kdd/
│  │  ├─ raw/
│  │  └─ processed/
│  ├─ unsw_nb15/
│  │  ├─ raw/
│  │  └─ processed/
│  ├─ seeds/
│  ├─ mixup/
│  └─ surrogate_train/
├─ artifacts/
│  ├─ models/
│  ├─ preprocessors/
│  └─ metadata/
├─ logs/
├─ results/
│  ├─ tables/
│  ├─ figures/
│  └─ reports/
├─ docs/
└─ src/
```

---

## 推荐运行顺序

### 1. 检查环境

```bash
python src/utils/check_env.py
```

### 2. 读取与清洗数据

```bash
python -m src.data.load_raw --dataset nsl_kdd
python -m src.data.clean_labels --dataset nsl_kdd --mode 5class

python -m src.data.load_raw --dataset unsw_nb15
python -m src.data.clean_labels --dataset unsw_nb15 --mode multiclass
```

### 3. 数据划分与预处理

```bash
python -m src.data.split_data --dataset nsl_kdd
python -m src.preprocess.run_preprocess_pipeline --dataset nsl_kdd
```

### 4. 训练基线模型

```bash
python -m src.models.train_sklearn_baseline --dataset nsl_kdd --model random_forest
python -m src.models.train_xgb --dataset nsl_kdd
python -m src.models.train_gbdt --dataset nsl_kdd
python -m src.models.train_tabnet --dataset nsl_kdd
python -m src.reporting.compare_models --dataset nsl_kdd
```

### 5. 黑盒接口与种子集

```bash
python -m src.blackbox.query_api --dataset nsl_kdd --target_model tabnet
python -m src.data.build_seed_set --dataset nsl_kdd --seed_size 500
python -m src.data.query_seed_labels --dataset nsl_kdd --target_model tabnet --seed_size 500
```

### 6. mixup 与 surrogate

```bash
python -m src.augment.run_mixup --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1
python -m src.data.build_surrogate_trainset --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1
python -m src.models.train_surrogate_mlp --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1 --depth 3
python -m src.evaluation.evaluate_surrogate --dataset nsl_kdd --target_model tabnet --seed_size 500 --alpha 0.1 --depth 3
```

### 7. surrogate 消融与正式选优

```bash
python -m src.models.run_surrogate_ablation --dataset nsl_kdd --target_model tabnet
python -m src.evaluation.evaluate_surrogate_batch --dataset nsl_kdd --target_model tabnet
python -m src.reporting.summarize_surrogate_ablation --dataset nsl_kdd --target_model tabnet
python -m src.models.select_best_surrogate --dataset nsl_kdd --target_model tabnet
```

正式批处理方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_surrogate_ablation_formal.ps1
```

### 8. 白盒攻击与迁移评估

```bash
python -m src.transfer.generate_from_surrogate --dataset nsl_kdd --target_model tabnet --attack pgd
python -m src.transfer.attack_target --dataset nsl_kdd --target_model tabnet --attack pgd
```

完整矩阵运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_attack_matrix.ps1
```

---

## 数据目录说明

将原始数据放入：

```text
data/nsl_kdd/raw/
data/unsw_nb15/raw/
```

当前主实验优先使用：

```text
data/nsl_kdd/
```

---

## 结果输出

- `artifacts/models/`：保存目标模型与 surrogate 模型
- `artifacts/preprocessors/`：保存预处理器和特征信息
- `artifacts/metadata/`：保存 formal best surrogate 选择结果
- `results/tables/`：保存指标表、汇总表、迁移矩阵
- `results/figures/`：保存图
- `results/reports/`：保存报告

---

## 当前默认最小可行实验

用于快速检查工程链路是否打通的最小配置：

- dataset：`nsl_kdd`
- target model：`tabnet`
- seed size：`500`
- mixup alpha：`0.1`
- surrogate depth：`3`

该配置主要用于：

- 环境检查
- 脚本联调
- 最小闭环验证

> 注意：该配置不是当前正式实验的最终 surrogate 配置。  
> 正式迁移攻击实验请参考下方“正式 surrogate 选优结果”。

---

## 正式 surrogate 选优结果

在完成 `surrogate ablation` 后，项目已从“临时 fallback surrogate”切换为“formal best surrogate selection”。

当前在 `NSL-KDD` 数据集上的正式最优 surrogate 配置如下：

| target model | seed_size | mixup alpha | surrogate depth |
|:--|--:|--:|--:|
| TabNet | 2000 | 0.1 | 7 |
| XGB | 2000 | 0.1 | 3 |
| GBDT | 2000 | 0.1 | 5 |

对应输出文件：

```text
artifacts/metadata/best_surrogate_nsl_kdd_tabnet.json
artifacts/metadata/best_surrogate_nsl_kdd_xgb.json
artifacts/metadata/best_surrogate_nsl_kdd_gbdt.json
```

对应汇总文件：

```text
results/tables/surrogate_ablation_summary_nsl_kdd_tabnet.csv
results/tables/surrogate_ablation_summary_nsl_kdd_xgb.csv
results/tables/surrogate_ablation_summary_nsl_kdd_gbdt.csv
```

对应 best 结果文件：

```text
results/tables/best_surrogate_nsl_kdd_tabnet.csv
results/tables/best_surrogate_nsl_kdd_xgb.csv
results/tables/best_surrogate_nsl_kdd_gbdt.csv
```

### 选优结论

- 在当前 NSL-KDD 场景下，三类目标模型的最优 surrogate 均偏向较大的种子集规模（`seed_size=2000`）
- `mixup alpha=0.1` 在三类目标模型上都表现最稳定，说明轻度 mixup 更适合当前网络流量表格特征场景
- 不同目标模型对应的最优 surrogate 深度不同：
  - TabNet 更适合较深的 surrogate（7 层）
  - XGB 更适合较浅的 surrogate（3 层）
  - GBDT 更适合中等深度 surrogate（5 层）

这说明 surrogate 选择应采用 **target-aware** 策略，而不应固定单一深度配置。

---

## 当前 full attack matrix 结果

在完成 formal best surrogate selection 之后，重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_attack_matrix.ps1
```

当前得到的迁移攻击结果如下：

| attack | target_model | transfer_success_rate |
|:--|:--|--:|
| FGM | GBDT | 0.5050 |
| FGM | TabNet | 0.1886 |
| FGM | XGB | 0.4317 |
| PGD | GBDT | 0.5213 |
| PGD | TabNet | 0.3253 |
| PGD | XGB | 0.5461 |
| SLIDE | GBDT | 0.5213 |
| SLIDE | TabNet | 0.3253 |
| SLIDE | XGB | 0.5461 |

### 当前结果的直接观察

- `PGD` 与 `SLIDE` 的迁移效果整体优于 `FGM`
- `XGB` 是当前最容易被迁移攻击影响的目标模型
- `GBDT` 次之
- `TabNet` 的迁移成功率最低，说明其在当前设置下具有更强的迁移鲁棒性

### 按目标模型汇总的最高迁移成功率

| target_model | best_attack | best_transfer_success_rate |
|:--|:--|--:|
| TabNet | PGD / SLIDE | 0.3253 |
| XGB | PGD / SLIDE | 0.5461 |
| GBDT | PGD / SLIDE | 0.5213 |

---

## 实验结果分析

从正式 surrogate 消融结果来看，当前 NSL-KDD 场景下最优配置具有较明显规律：三类目标模型的最佳 surrogate 都选择了较大的种子集规模（`seed_size=2000`），并统一偏向较小的 mixup 系数（`alpha=0.1`）。这说明在黑盒迁移攻击任务中，扩大初始查询样本规模能够更稳定地提升 surrogate 对目标模型决策边界的拟合能力，而轻度 mixup 更适合当前网络流量表格特征场景。与此同时，不同目标模型的最优 surrogate 深度并不一致：TabNet 对应 7 层、XGB 对应 3 层、GBDT 对应 5 层，表明 surrogate 选择具有明显的目标模型依赖性，不能简单固定为单一深度。

从重新运行 `run_full_attack_matrix.ps1` 的结果来看，当前 formal best surrogate 配置下，`PGD` 和 `SLIDE` 的迁移攻击效果整体优于 `FGM`。其中，`XGB` 上的最高迁移成功率达到 `0.5461`，`GBDT` 上达到 `0.5213`，而 `TabNet` 上最高为 `0.3253`。这说明在当前实验设定中，TabNet 相比 XGB 和 GBDT 具有更强的迁移鲁棒性；而 XGB 与 GBDT 更容易受到 surrogate 生成对抗样本的影响。另一方面，`PGD` 与 `SLIDE` 在当前结果中表现一致，说明二者在当前实现、参数设置或表格特征约束下呈现出接近的迁移行为，这一现象后续可继续通过更细粒度参数实验进一步分析。

从项目工程角度看，formal best surrogate selection 的完成意味着迁移攻击阶段已经从“临时默认 surrogate”过渡到“正式选优 surrogate”，因此当前结果更适合作为后续 README、阶段报告、论文初稿与结题材料中的主结果。

---

## 推荐的正式实验流程

如果希望直接复现实验主线，建议按下面顺序执行：

### 1. 数据预处理

```bash
python -m src.data.load_raw --dataset nsl_kdd
python -m src.data.clean_labels --dataset nsl_kdd --mode 5class
python -m src.data.split_data --dataset nsl_kdd
python -m src.preprocess.run_preprocess_pipeline --dataset nsl_kdd
```

### 2. 训练目标模型

```bash
python -m src.models.train_tabnet --dataset nsl_kdd
python -m src.models.train_xgb --dataset nsl_kdd
python -m src.models.train_gbdt --dataset nsl_kdd
```

### 3. 正式 surrogate 选优

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_surrogate_ablation_formal.ps1
```

### 4. 运行完整迁移攻击矩阵

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_attack_matrix.ps1
```

### 5. 查看结果

重点查看：

```text
results/tables/surrogate_ablation_summary_*.csv
results/tables/best_surrogate_*.csv
artifacts/metadata/best_surrogate_*.json
results/tables/*transfer*.csv
```

---

## 当前已完成事项

- NSL-KDD 原始数据读取与标签清洗
- 预处理管线构建
- TabNet / XGB / GBDT 黑盒目标模型
- 种子集构建与查询标签
- mixup 数据增强
- surrogate 训练与评估
- surrogate 消融与正式 best surrogate 选择
- FGM / PGD / SLIDE 攻击生成
- full attack matrix 跑通
- README 与实验主线文档化

---

## 当前待完成事项

- UNSW-NB15 第二数据集完整复现
- C&W 攻击正式纳入 full attack matrix
- 更细粒度攻击参数实验
- t-SNE 可视化
- 鲁棒性曲线绘制
- 自动报告导出
- 与申请书任务对照的阶段总结文档
- 论文体例结果整理

---

## 常用输出文件说明

### surrogate 相关

```text
results/tables/surrogate_grid_nsl_kdd_<target>.csv
results/tables/surrogate_ablation_summary_nsl_kdd_<target>.csv
results/tables/surrogate_ablation_summary_nsl_kdd_<target>.md
results/tables/best_surrogate_nsl_kdd_<target>.csv
artifacts/metadata/best_surrogate_nsl_kdd_<target>.json
```

### 模型相关

```text
artifacts/models/tabnet_nsl_kdd.zip
artifacts/models/xgb_nsl_kdd.pkl
artifacts/models/gbdt_nsl_kdd.pkl
artifacts/models/surrogate_*.pt
```

### 数据相关

```text
data/seeds/nsl_kdd/
data/mixup/nsl_kdd/
data/surrogate_train/nsl_kdd/
```

---

## 注意事项

1. 建议统一从项目根目录执行命令
2. 后续多个脚本默认特征列命名为 `f_0, f_1, ...`
3. Parquet 读写依赖 `pyarrow`
4. TabNet 模型文件默认保存为 `artifacts/models/tabnet_<dataset>.zip`
5. 部分高级攻击在表格特征场景下采用工程化近似实现，后续可按论文进一步细化
6. `run_surrogate_ablation_formal.ps1` 若带 `--skip_existing`，会跳过已有 surrogate 结果；当训练逻辑改动后，建议清理旧结果或关闭 skip 再重跑
7. 当前 `PGD` 与 `SLIDE` 在结果中表现一致，后续可继续检查是否为参数设置一致、实现近似或样本裁剪策略导致
8. TabNet 运行在 CPU 时较慢，正式实验建议固定版本与随机种子，避免结果抖动
9. 如果看到 `NumPy array is not writable` 的 warning，建议在进入 torch 前显式 `.copy()`

---

## 项目现阶段一句话总结

> 当前项目已经完成 NSL-KDD 场景下的 formal surrogate selection 与基于正式 best surrogate 的 full transfer evaluation。现阶段主结论是：轻度 mixup（`alpha=0.1`）和较大 seed set（`2000`）更有利于 surrogate 拟合目标模型，而在当前实验设定下，`XGB` 与 `GBDT` 相比 `TabNet` 更容易受到迁移攻击影响。

---

## 后续建议

当前最推荐的后续推进顺序：

1. 将 `UNSW-NB15` 复用现有主线完整跑通
2. 将 `C&W` 纳入正式 full attack matrix
3. 做攻击强度与 surrogate 深度的更细粒度消融
4. 补 t-SNE 与鲁棒性曲线
5. 把当前 README 结果整理成论文/结题报告中的“实验结果”章节

---
