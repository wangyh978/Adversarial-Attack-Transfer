# NIDS 对抗鲁棒性评估项目

本项目围绕“网络入侵检测中基于黑盒迁移的 AI 模型对抗攻击研究”展开，目标是构建一套可复现、可扩展的 NIDS 对抗鲁棒性评估实验流程。当前代码已经覆盖从原始数据读取、标签清洗、预处理、异构基线模型训练，到黑盒查询、种子集构建、mixup 数据增强、surrogate 训练、白盒攻击生成、迁移攻击评估与结果汇总的主要链路。

本项目直接服务于开放性课题申请书中的两条主线：

1. 异构模型间对抗样本迁移性机理研究  
2. 面向对抗鲁棒性评估的黑盒迁移攻击机制研究  

同时参考《面向入侵检测的黑盒对抗攻击迁移性研究》的 MSM（Mixup & Surrogate Model）思路，将 TabNet、XGBoost、GBDT 等模型统一纳入实验框架，并支持在 NSL-KDD 与 UNSW-NB15 上复用完整流程。

---

## 1. 项目简介

### 1.1 研究目标

本项目希望解决以下问题：

- 如何在黑盒约束下评估 NIDS 的对抗脆弱性
- 如何利用少量种子样本、黑盒输出和 mixup 训练高质量 surrogate
- 如何比较不同白盒攻击方法在黑盒目标模型上的迁移效果
- 如何形成标准化、可批量运行的鲁棒性评估主线

### 1.2 当前支持的数据集

- NSL-KDD
- UNSW-NB15

### 1.3 当前支持的目标模型

- TabNet
- XGBoost
- GBDT
- sklearn baseline（如 RandomForest）

### 1.4 当前支持的攻击方法

- FGM
- PGD
- SLIDE

---

## 2. 当前项目进度

### 2.1 已完成

#### NSL-KDD 主线
- 原始数据读取与标签清洗
- 数据划分与预处理
- TabNet / XGBoost / GBDT 基线模型训练与比较
- 黑盒查询接口
- 种子集构建与标签查询
- mixup 增强与 surrogate 训练
- surrogate 消融与 best surrogate 选优
- FGM / PGD / SLIDE 白盒生成与黑盒迁移
- full attack matrix 结果汇总

#### UNSW-NB15 主线
- 原始数据读取与标签清洗
- 数据划分与预处理
- TabNet / XGBoost / GBDT / RandomForest 基线比较
- 目标模型选择（当前推荐 XGBoost）
- surrogate 训练链路打通
- FGM / PGD 迁移攻击跑通
- 结果汇总脚本可复用

### 2.2 当前推荐作为正式结果使用的部分

#### NSL-KDD
- target model：TabNet
- full attack matrix：FGM / PGD / SLIDE

#### UNSW-NB15
- target model：XGBoost
- surrogate：seed_size=1000, alpha=0.1, depth=3
- 当前已验证攻击：FGM / PGD
- 如果后续补充 SLIDE / C&W，可继续沿用同一套 surrogate

### 2.3 尚未完成

- 将 UNSW-NB15 扩展为与 NSL-KDD 同等完整的 full attack matrix（建议补 GBDT / TabNet 目标模型）
- 增加目标模型 clean vs adversarial 性能下降分析
- 增加更系统的可视化输出（鲁棒性对比图、迁移矩阵图、t-SNE）
- 将“结构鲁棒性”从单目标值扩展为多目标异构评估
- 补齐论文/结题所需的结果解读与自动报告导出
- 清理仓库中的派生文件，让 Git 仓库更轻、更干净

### 2.4 现在只需要继续做什么

如果你的目标是“尽快形成可交付版本”，当前最应该继续做的是：

1. 把 UNSW-NB15 再补一个或两个 target model  
   推荐顺序：GBDT -> TabNet

2. 统一汇总 NSL-KDD / UNSW-NB15 的迁移结果  
   建议统一输出：
   - transfer_success_rate
   - perturbation_generalization
   - structural_robustness
   - accuracy_drop
   - macro_f1_drop

3. 增加图表与报告导出  
   至少补：
   - 攻击方法对比柱状图
   - 不同数据集对比图
   - 最终 Markdown 报告

---

## 3. 项目使用说明

## 3.1 环境准备

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3.2 原始数据目录

请将原始数据放到以下目录：

```text
data/
├─ nsl_kdd/
│  └─ raw/
└─ unsw_nb15/
   └─ raw/
```

---

## 3.3 推荐入口

### NSL-KDD：最小迁移验证
```powershell
.\scripts\run_nsl_kdd_pipeline.ps1 -Stage MinTransfer
```

### NSL-KDD：完整 full attack matrix
```powershell
.\scripts\run_nsl_kdd_pipeline.ps1 -Stage FullAttackMatrix
```

### UNSW-NB15：最小迁移验证
```powershell
.\scripts\run_unsw_nb15_pipeline.ps1 -Stage MinTransfer
```

### UNSW-NB15：当前推荐配置
```powershell
.\scripts\run_unsw_nb15_pipeline.ps1 `
  -Stage FullAttackMatrix `
  -TargetModels xgb `
  -SeedSize 1000 `
  -Alpha 0.1 `
  -Depth 3 `
  -Attacks fgm,pgd
```

---

## 3.4 手工运行顺序

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

## 4. 推荐项目结构

建议把仓库收敛为下面这套结构：

```text
.
├─ README.md
├─ requirements.txt
├─ main.py
├─ configs/
├─ scripts/
│  ├─ run_min_transfer_matrix.ps1
│  ├─ run_full_attack_matrix.ps1
│  ├─ run_nsl_kdd_pipeline.ps1
│  ├─ run_unsw_nb15_pipeline.ps1
│  └─ summarize_transfer_matrix.py
├─ src/
│  ├─ attacks/
│  ├─ augment/
│  ├─ blackbox/
│  ├─ data/
│  ├─ evaluation/
│  ├─ models/
│  ├─ preprocess/
│  ├─ reporting/
│  ├─ transfer/
│  ├─ utils/
│  └─ visualization/
├─ docs/
│  ├─ reports/
│  ├─ notes/
│  └─ archive/
├─ data/
│  ├─ nsl_kdd/raw/
│  ├─ nsl_kdd/processed/
│  ├─ unsw_nb15/raw/
│  ├─ unsw_nb15/processed/
│  ├─ seeds/
│  ├─ mixup/
│  ├─ surrogate_train/
│  └─ adversarial/
├─ artifacts/
│  ├─ models/
│  ├─ preprocessors/
│  └─ metadata/
└─ results/
   ├─ tables/
   ├─ figures/
   └─ reports/
```

---

## 5. 仓库清理建议

### 建议保留
- `src/`
- `scripts/`
- `configs/`
- `README.md`
- `requirements.txt`
- `main.py`

### 建议从 Git 跟踪中移除
以下内容属于派生文件，不建议长期直接提交到 Git 仓库：

- `__pycache__/`
- `artifacts/models/`
- `artifacts/preprocessors/`
- `data/*/processed/`
- `data/seeds/`
- `data/mixup/`
- `data/surrogate_train/`
- `data/adversarial/`
- `logs/`
- `results/tables/`
- `results/figures/`
- `results/reports/`

更合适的做法是：
- 保留目录结构与 `.gitkeep`
- 使用 `.gitignore` 忽略运行产物
- 只在 `docs/reports/` 中保留少量最终结果快照

### 建议新增
- `.gitignore`
- `docs/archive/`
- `docs/reports/`
- `docs/notes/`
- `results/reports/summary_latest.md`

---

## 6. 下一阶段里程碑

### 阶段 1：补齐 UNSW-NB15 的 full matrix
- 增加 target=`gbdt`
- 增加 target=`tabnet`
- 复用同一套 `seed=1000, alpha=0.1, depth=3`
- 统一导出最终 transfer matrix

### 阶段 2：补完整的鲁棒性报告
- clean vs adversarial 性能下降
- 攻击方法对比图
- 数据集对比图
- Markdown / PDF 报告

### 阶段 3：论文与结题输出
- 结果分析
- 方法章节
- 实验章节
- 总结与不足
- 结题材料与答辩用图表

---

## 7. 当前一句话总结

目前项目已经从“代码骨架阶段”进入“结果整合阶段”。  
NSL-KDD 主线已经较完整，UNSW-NB15 主线已经跑通 baseline、surrogate 与 FGM/PGD 迁移攻击，下一步重点是补齐异构目标模型与最终报告导出。
