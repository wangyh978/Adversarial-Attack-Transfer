基于网络入侵检测的对抗攻击迁移研究（Adversarial Attack Transfer）
==============================================

本项目实现了针对表格数据（网络入侵检测）的**黑盒对抗攻击迁移攻击框架**，支持 NSL-KDD 与 UNSW-NB15 数据集，并提供完整实验流水线。

* * *

📦 一、环境配置
=========

1. 克隆项目

-------

    git clone https://github.com/wangyh978/Adversarial-Attack-Transfer.git
    cd Adversarial-Attack-Transfer

2. 创建虚拟环境

---------

    python -m venv .venv
    .venv\Scripts\activate   # Windows
    # source .venv/bin/activate  # Linux / Mac

3. 安装依赖

-------

    pip install -r requirements.txt

* * *

📊 二、数据集说明
==========

| 数据集       | 类别数 | 训练集大小 | 特征维度 |
| --------- | --- | ----- | ---- |
| NSL-KDD   | 5类  | 15780 | 116  |
| UNSW-NB15 | 10类 | 69982 | 190  |

* * *

⚙️ 三、运行方式
=========

1. NSL-KDD 全流程运行

----------------

    .\scripts\run_nsl_kdd_pipeline.ps1 -Stage FullPipeline

包含：

* 数据加载与清洗

* 特征工程

* 目标模型训练

* surrogate模型训练

* 对抗样本生成

* 迁移攻击评估

* * *

2. UNSW-NB15 攻击矩阵实验

-------------------

    .\scripts\run_unsw_nb15_pipeline.ps1 -Stage FullAttackMatrix

* * *

🧠 四、模型说明
=========

目标模型（Target Models）
-------------------

* TabNet（深度模型）

* XGBoost（高性能）

* GBDT（稳定但较慢）

替代模型（Surrogate）
---------------

* 多层感知机（MLP，深度可调）

* * *

📈 五、实验结果
=========

* * *

1️⃣ NSL-KDD
-----------

### （1）目标模型性能

| 模型     | Accuracy | F1 Macro |
| ------ | -------- | -------- |
| TabNet | 0.9752   | 0.9535   |
| XGB    | 0.9864   | 0.9698   |
| GBDT   | 0.9864   | 0.9668   |

* * *

### （2）迁移攻击结果

#### TabNet

| 攻击    | 迁移成功率  |
| ----- | ------ |
| FGM   | 0.0585 |
| PGD   | 0.3510 |
| SLIDE | 0.3510 |

#### XGB

| 攻击    | 迁移成功率  |
| ----- | ------ |
| FGM   | 0.3749 |
| PGD   | 0.4042 |
| SLIDE | 0.4042 |

#### GBDT

| 攻击    | 迁移成功率  |
| ----- | ------ |
| FGM   | 0.3096 |
| PGD   | 0.3832 |
| SLIDE | 0.3832 |

* * *

2️⃣ UNSW-NB15
-------------

### （1）目标模型性能

| 模型     | Accuracy | F1 Macro |
| ------ | -------- | -------- |
| TabNet | 0.7519   | 0.4268   |
| XGB    | 0.7652   | 0.5188   |
| GBDT   | 0.7539   | 0.4942   |

* * *

### （2）迁移攻击结果

#### XGB

| 攻击    | 迁移成功率  |
| ----- | ------ |
| FGM   | 0.6030 |
| PGD   | 0.5894 |
| SLIDE | 0.5894 |

#### GBDT

| 攻击    | 迁移成功率  |
| ----- | ------ |
| FGM   | 0.3298 |
| PGD   | 0.4154 |
| SLIDE | 0.4154 |

#### TabNet

| 攻击    | 迁移成功率  |
| ----- | ------ |
| FGM   | 0.4629 |
| PGD   | 0.5375 |
| SLIDE | 0.5375 |

* * *

🔍 六、关键结论
=========

* XGB 在两个数据集上表现最佳

* GBDT 在 UNSW 上训练耗时极长（约 1400s）

* NSL-KDD 迁移性较低（约 0.3~0.4）

* UNSW-NB15 迁移性明显更高（最高 0.6）

* PGD 与 SLIDE 明显优于 FGM

* surrogate 模型表现：
  
  * NSL-KDD：≈ 0.95
  
  * UNSW：≈ 0.67~0.70

* * *

⚠️ 七、注意事项
=========

* GBDT 在大规模数据上训练较慢

* TabNet 默认使用 CPU（建议 GPU 加速）

* numpy writable warning 可忽略

* * *

🚧 八、项目进度
=========

| 模块          | 状态    |
| ----------- | ----- |
| NSL-KDD 全流程 | ✅ 已完成 |
| UNSW XGB    | ✅ 已完成 |
| UNSW GBDT   | ✅ 已完成 |
| UNSW TabNet | ✅ 已完成 |
| 攻击迁移矩阵      | ✅ 已完成 |
| 实验结果导出      | ✅ 已完成 |

* * *

📁 九、输出目录结构
===========

    artifacts/
      models/
    
    data/
      seeds/
      mixup/
      surrogate_train/
      adversarial/
    
    results/
      tables/

* * *

🧪 十、实验参数
=========

    SeedSize = 1000
    Alpha = 0.1
    Depth = 3
    Attacks = FGM, PGD, SLIDE

* * *

📌 十一、未来工作
==========

* 使用 LightGBM 替代 sklearn GBDT（提升速度）

* 提升 UNSW surrogate 质量

* 加入对抗训练（防御）

* 支持 GPU 加速

* * *

⭐ 十二、说明
=======

本项目已完成完整攻击迁移实验流程，可直接复现实验结果，适用于：

* 对抗攻击研究

* 网络安全方向实验

* 表格数据鲁棒性分析

---


