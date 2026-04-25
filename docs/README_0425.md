# Adversarial-Attack-Transfer

基于 **MSM hard-label mixup** 的网络入侵检测黑盒迁移对抗攻击研究框架。

研究目标是构建一套可复现、可扩展的 **NIDS（Network Intrusion Detection System）对抗鲁棒性评估流程**，用于系统分析不同目标模型在黑盒迁移攻击下的脆弱性，并为正式部署前的安全评估提供实验依据。

---

## 1. 项目背景

在真实网络安全场景中，基于机器学习与深度学习的网络入侵检测系统通常以黑盒形式部署。攻击者难以直接获得目标模型结构、参数与训练集，但可以通过有限样本、模型查询与代理模型训练，构建迁移型黑盒攻击链路。

围绕这一问题，本项目当前采用：

- **目标模型（black-box target）**：`XGBoost`、`GBDT`、`TabNet`
- **代理模型（surrogate）**：`MLP`
- **代理训练方案**：**MSM hard-label mixup**
- **白盒攻击**：`FGM`、`PGD`、`SLIDE`
- **评估数据集**：`NSL-KDD`、`UNSW-NB15`

当前代码主线已经从早期 soft-label mixup 方案收敛到 **MSM hard-label mixup** 路线，即：

1. 从受限原始样本构建 seed set  
2. 查询黑盒目标模型标签  
3. 对 seed 数据做 mixup 增强  
4. 对 mixup 样本重新使用目标模型赋予 **hard label**  
5. 训练 surrogate MLP  
6. 在 surrogate 上实施白盒攻击  
7. 将对抗样本迁移到目标模型  
8. 统计迁移成功率等指标  

---

## 2. 项目目标

本项目的核心目标包括：

- 构建一套面向 NIDS 的黑盒迁移攻击复现实验框架
- 分析异构目标模型之间的对抗迁移差异
- 评估 MSM hard-label mixup 对代理模型训练与攻击迁移能力的影响
- 形成标准化的鲁棒性评估流程与实验结果汇总机制
- 为后续研究报告、论文、图表与自动化评测提供基础

---

## 3. 当前采用的 MSM hard-label mixup 主线

当前仓库默认推荐使用的主线如下：

```text
build_seed_set
-> query_seed_labels
-> run_mixup
-> build_surrogate_trainset
-> train_surrogate_mlp
-> evaluate_surrogate
-> generate_from_surrogate
-> attack_target
-> summarize_transfer_matrix
```

其中关键变化是：

- **不再使用旧 soft-label mixup 训练路线**
- `run_mixup` 生成的新样本会通过目标模型重新打标签
- `build_surrogate_trainset` 构造的是 **paper_union / hard-label** 训练集
- `train_surrogate_mlp` 是当前默认 surrogate 训练入口

---

## 4. 仓库结构（核心目录）

```text
.
├─ main.py
├─ README.md
├─ requirements.txt
├─ configs/
├─ scripts/
├─ src/
│  ├─ augment/
│  ├─ attacks/
│  ├─ blackbox/
│  ├─ data/
│  ├─ evaluation/
│  ├─ models/
│  ├─ preprocess/
│  ├─ reporting/
│  └─ transfer/
├─ data/
├─ artifacts/
├─ results/
└─ logs/
```

---

## 5. 从拉取仓库到环境搭建

### 5.1 拉取仓库

```bash
git clone https://github.com/wangyh978/Adversarial-Attack-Transfer.git
cd Adversarial-Attack-Transfer
```

### 5.2 创建虚拟环境

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5.3 安装依赖

```bash
pip install -r requirements.txt
```

如果个别环境中 `requirements.txt` 未完全覆盖依赖，可按需补装：

```bash
pip install numpy pandas scikit-learn xgboost torch pytorch-tabnet pyarrow matplotlib
```

---

## 6. 数据准备与预处理

首次运行前建议先完成数据准备：

```bash
python main.py nsl --stage prepare
python main.py unsw --stage prepare
```

这一步会执行：

- 原始数据读取
- 标签清洗
- 训练/测试划分
- 特征预处理
- 生成 `data/<dataset>/processed/` 下的标准化产物

如果出现旧版 `.npy` 文件与当前代码不兼容的情况，建议删除旧的 `processed/` 目录后重新执行 `prepare`。

---

## 7. 当前 main.py 的推荐用法

### 7.1 单目标完整迁移流程

```bash
python main.py nsl --stage min_transfer --target xgb --seed-size 1000 --alpha 0.1 --depth 3 --attacks fgm pgd slide
```

### 7.2 多目标完整攻击矩阵

```bash
python main.py nsl --stage full_attack_matrix --targets xgb gbdt tabnet --seed-size 1000 --alpha 0.1 --depth 3 --attacks fgm pgd slide
```

```bash
python main.py unsw --stage full_attack_matrix --targets xgb gbdt tabnet --seed-size 1000 --alpha 0.1 --depth 3 --attacks fgm pgd slide
```

### 7.3 全流程（含 prepare）

```bash
python main.py nsl --stage full_pipeline --targets xgb gbdt tabnet --seed-size 1000 --alpha 0.1 --depth 3 --attacks fgm pgd slide
```

### 7.4 仅训练 surrogate

```bash
python main.py nsl --stage surrogate --target xgb --seed-size 1000 --alpha 0.1 --depth 3
```

### 7.5 迭代式 MSM

```bash
python main.py nsl --stage msm_iterative --target xgb --seed-size 1000 --alpha 0.1 --depth 3 --rounds 3 --attacks fgm pgd slide
```

---

## 8. 当前已完成内容

结合目前代码进度、申请书任务要求以及最新实验运行结果，当前已经完成：

### 8.1 框架与流程层面

- [x] NSL-KDD 与 UNSW-NB15 两个数据集的主线接通
- [x] `XGB / GBDT / TabNet` 三类目标模型训练与评估
- [x] `FGM / PGD / SLIDE` 三类白盒攻击在 surrogate 上生成对抗样本
- [x] 迁移攻击评估流程打通
- [x] `main.py` 已整合为统一入口
- [x] 仓库当前默认主线已经迁移为 **MSM hard-label mixup**
- [x] `full_attack_matrix` 已能完整跑通 `NSL-KDD` 与 `UNSW-NB15` 的多 target 实验

### 8.2 与申请书目标对应的完成情况

申请书提出要在 2025.11-2026.10 周期内，逐步完成研究环境与实验数据准备、异构模型基线构建、黑盒 surrogate 训练、迁移攻击评估以及标准化鲁棒性评估框架集成。相关研究内容强调：在黑盒场景下通过数据增强与替代模型训练，量化不同模型在多种攻击方法下的鲁棒性表现，并以迁移成功率、扰动泛化度、结构鲁棒性为核心评价指标。fileciteturn1file0

截至目前，上述目标中的 1-3 已基本落地，4 已具备雏形：

- 多目标模型：已覆盖 `xgb / gbdt / tabnet`
- 数据增强：已采用 `MSM hard-label mixup`
- 迁移攻击：已覆盖 `FGM / PGD / SLIDE`
- 评估输出：已自动生成 `transfer_*.csv` 与 `final_transfer_matrix_*.csv/.md`

---

## 9. 当前实验结果（MSM hard-label mixup）

以下结果来自当前项目代码实际运行结果，均基于：

- surrogate: `MLP`
- seed size: `1000`
- alpha: `0.1`
- depth: `3`
- attacks: `FGM / PGD / SLIDE`

### 9.1 NSL-KDD

#### 目标模型：XGB

- FGM: `0.4406`
- PGD: `0.5393`
- SLIDE: `0.5393`

#### 目标模型：GBDT

- FGM: `0.2942`
- PGD: `0.3507`
- SLIDE: `0.3507`

#### 目标模型：TabNet

- FGM: `0.0674`
- PGD: `0.2694`
- SLIDE: `0.2694`

#### NSL-KDD 现象总结

- 在当前 **MSM hard-label mixup** 方案下，`XGB` 是 NSL-KDD 上最易被迁移攻击击中的目标模型
- `TabNet` 依然相对更稳健，尤其在 `FGM` 下迁移成功率最低
- `PGD` 与 `SLIDE` 在三个目标模型上均强于或不弱于 `FGM`
- 当前实现中，`PGD` 与 `SLIDE` 的结果非常接近，后续需要继续检查是否存在攻击参数设置过于相近的问题

### 9.2 UNSW-NB15

#### 目标模型：XGB

- FGM: `0.6316`
- PGD: `0.6473`
- SLIDE: `0.6473`

#### 目标模型：GBDT

- FGM: `0.4633`
- PGD: `0.5588`
- SLIDE: `0.5588`

#### 目标模型：TabNet

- FGM: `0.4336`
- PGD: `0.6492`
- SLIDE: `0.6492`

#### UNSW-NB15 现象总结

- `UNSW-NB15` 上整体迁移成功率显著高于 `NSL-KDD`
- 在当前配置下，`TabNet` 与 `XGB` 在 `PGD / SLIDE` 下都呈现出较高迁移成功率
- `FGM` 仍然偏弱，但在 UNSW 上也已具备明显有效性
- 当前最佳迁移结果出现在：
  - `UNSW-NB15 + TabNet + PGD/SLIDE = 0.6492`
  - `UNSW-NB15 + XGB + PGD/SLIDE = 0.6473`

---

## 10. 与论文技术路线的对应关系

论文摘要与第 3、4 章已经明确提出：研究主线包括以 TabNet、XGBoost、GBDT 作为黑盒目标模型，引入 mixup 辅助代理模型生成，设计 MSM 黑盒威胁框架，并在 NSL-KDD 与 UNSW-NB15 上使用 FGM、PGD、SLIDE 等攻击评估跨模型迁移表现。论文同时指出，迁移成功率、攻击成功率和保真度分析是实验评价重点。fileciteturn1file1

当前仓库与实验结果已经与上述技术路线实现了较高一致性：

- 数据集一致：`NSL-KDD / UNSW-NB15`
- 目标模型一致：`TabNet / XGB / GBDT`
- MSM 路线一致：`mixup -> surrogate -> transfer`
- 攻击方法一致：`FGM / PGD / SLIDE`
- 结果输出形式已具备标准化基础

---

## 11. 当前 surrogate 表现概况

### NSL-KDD surrogate

- 对 `xgb` 的 surrogate agreement 约 `0.9548`
- 对 `gbdt` 的 surrogate agreement 约 `0.9506`
- 对 `tabnet` 的 surrogate agreement 约 `0.9710`

说明在 NSL-KDD 上，当前 surrogate 已能较好逼近目标模型决策边界。

### UNSW-NB15 surrogate

- 对 `xgb` 的 surrogate agreement 约 `0.8096`
- 对 `gbdt` 的 surrogate agreement 约 `0.8266`
- 对 `tabnet` 的 surrogate agreement 约 `0.8661`

说明在 UNSW-NB15 上 surrogate 仍有较大优化空间，尤其是：

- 多分类不均衡问题更明显
- surrogate 的 macro-F1 仍偏低
- 但在黑盒迁移攻击上已经能产生较强效果

---

## 12. 当前未完成内容

虽然主线已经打通，但按申请书“标准化评估框架 + 报告输出 + 研究总结”的要求，当前仍有几部分没有完成：

### 12.1 指标层面尚不完整

目前已稳定输出：

- transfer_success_rate
- perturbation_generalization
- structural_robustness

但仍建议补充：

- clean accuracy
- adversarial accuracy
- macro-F1 drop
- per-class robustness degradation
- L2 / L-infinity 扰动统计
- attack generation time

### 12.2 图表与报告尚未系统化

目前已能输出结果表格，但还缺：

- 攻击方法对比柱状图
- 数据集间迁移对比图
- 目标模型对比图
- transfer matrix 热力图
- 自动化 Markdown / PDF 报告

### 12.3 MSM 迭代机制仍需进一步强化

当前代码已提供 `msm_iterative` 入口，但距离申请书和论文中更完整的“多轮边界探测与迭代增强”仍有差距，仍需要：

- 更清晰的轮次产物命名
- 每轮 surrogate 演化结果保存
- 不同轮次的性能对比
- 迭代终止条件设计

### 12.4 旧 soft-label 路线仍需进一步清理

虽然主线已转向 **MSM hard-label mixup**，但仓库中仍建议继续清理旧思路残留文件与注释，确保 README 与代码入口完全一致。

---

## 13. 下一步优化建议

结合当前运行结果与申请书目标，后续最值得优先推进的内容如下。

### 优先级 1：补足标准化鲁棒性评估输出

建议在 `attack_target` 与 `reporting` 中补充：

- clean vs adversarial accuracy
- macro-F1 drop
- class-wise attack sensitivity
- 结果统一导出为 `.csv + .md + 图表`

### 优先级 2：优化 UNSW surrogate

由于 UNSW 上 surrogate 的 macro-F1 和 target agreement 仍低于 NSL-KDD，建议重点优化：

- surrogate 网络深度
- seed size
- class balancing
- attack parameter grid
- mixup alpha 搜索

### 优先级 3：检查 PGD 与 SLIDE 参数差异

当前多个实验中 `PGD` 与 `SLIDE` 的结果完全相同或非常接近，建议重点检查：

- `epsilon / step_size / iters`
- 是否存在共同默认参数
- 是否存在实现复用导致的攻击等价现象

### 优先级 4：整理图表与论文写作材料

建议尽快补以下材料：

- NSL-KDD 与 UNSW-NB15 结果总表
- surrogate agreement 对比表
- 目标模型 clean baseline 表
- 各攻击迁移成功率对比图
- 方法流程图与目录结构图

---

## 14. 输出文件说明

当前实验运行后，通常会生成：

### 中间产物

```text
data/seeds/<dataset>/
data/mixup/<dataset>/
data/surrogate_train/<dataset>/
data/adversarial/<dataset>/
artifacts/models/
```

### 结果文件

```text
results/tables/transfer_<attack>_<dataset>_<target>.csv
results/tables/final_transfer_matrix_<dataset>_<target>.csv
results/tables/final_transfer_matrix_<dataset>_<target>.md
```

---

## 15. 当前项目状态总结

当前项目已经不再处于“代码是否能跑通”的早期阶段，而是进入了：

> **主线已打通，正在向结果整理、指标完善、图表生成和结题/论文输出阶段过渡。**

从当前进度看，可以这样概括：

### 已完成

- NSL-KDD 多目标完整迁移攻击矩阵
- UNSW-NB15 多目标完整迁移攻击矩阵
- MSM hard-label mixup 主线接入完成
- main.py 统一入口整合完成
- FGM / PGD / SLIDE 已全部接回完整 pipeline
- 可自动输出 final transfer matrix

### 正在进行

- README 与仓库结构整理
- 旧 soft-label 路线清理
- surrogate 训练链统一到 MSM hard-label mixup
- 结果归档与复现实验命令整理

### 下一步重点

- 完善鲁棒性指标体系
- 补全图表与自动报告
- 优化 UNSW surrogate
- 深化 MSM iterative 版本
- 形成研究报告 / 结题材料 / 论文正文支撑结果

---

## 16. 研究价值与预期成果衔接

申请书明确提出，项目预期形成：

- 一套 “NIDS 对抗鲁棒性标准化评估框架” 原型软件
- 相关技术发明专利
- 一套针对网络流量特征的对抗攻击与防御算法方案
- 高水平学术论文与完整研究总结报告等成果形式。fileciteturn1file0

按照当前项目进度，前两项尚处于中后期推进阶段，但：

- 评估框架原型已经具备核心功能
- 迁移攻击实验矩阵已经形成
- README、结果表、自动汇总脚本已具备结题材料雏形
- 后续只需继续补足指标、图表和报告生成层，即可更顺利对接项目结题、论文整理与阶段汇报

---

## 17. 一句话总结

本项目当前已经初步实现了一个面向 NIDS 的、基于 **MSM hard-label mixup** 的黑盒迁移攻击评估框架，并在 `NSL-KDD` 与 `UNSW-NB15` 上完成了多目标模型的系统实验，为后续的鲁棒性分析、论文写作与结题验收提供了可复现的技术基础。
