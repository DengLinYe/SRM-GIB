# SR-GIB：面向社交网络的鲁棒图信息瓶颈框架  

> - 在线社交网络智能分析与对抗课程结课大作业论文，方向为：**其他社交网络分析与对抗**。
>
> - **主要工作**：
>
>   1. 复现RM-GIB框架。
>
>      > Dai E, Cui L, Wang Z, et al. A unified framework of graph information bottleneck for robustness and membership privacy[C]//Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023: 368-379.
>
>   2. 设计了一种图注入攻击方法。
>
>   3. 优化了RM-GIB框架，更换主干为GraphSAGE，并重新设计自监督器。
>
>   4. 在Twitch（EN）数据集上完成测试。

## 一、总体架构

### 1.1 目录结构

```
SR-GIB/
├── run_ablation.py          # 防御实验脚本
├── run_attack_curve.py      # 攻击实验脚本
├── requirements.txt         
│
├── data/                    # 数据处理
│   ├── data_preparation.py  # 从 SNAP twitch.zip 生成干净图与划分掩码
│   ├── topology_attack.py   # 图注入攻击方法
│   └── graph_utils.py       # 特征归一化等图工具函数
│
├── modal/                   # 模型定义
│   ├── gcn.py               # Vanilla GCN 基线
│   ├── rm_gib.py            # RM-GIB
│   └── srm_gib.py           # SR-GIB
│
└── utils/                   # 工具配置
    ├── config.py            # 脚本配置
    ├── data_io.py           # 数据加载
    ├── split_utils.py       # train/val/test、unlabeled、original_test 掩码
    ├── supervision.py       # 伪标签与类别权重
    ├── topology_priors.py   # 预计算边 Jaccard 相似度
    ├── gcn_train.py         # GCN 训练与评估
    └── gib_train.py         # GIB 系列训练与评估
```

### 1.2 架构简述

根目录上的两个脚本完成实验部分：`run_ablation.py`包括了防御实验、消融实验；`run_attack_curve.py`是测试不同攻击预算下的攻击效果。

`\data`目录下是数据处理模块，主要是读取数据源并处理，以及数据投毒（攻击方法）

`\modal`就是模型文件，分别是两个基线（GCN、RM-GIB）和主要方法（SR-GIB）的模型定义。

`\utils`是一些公用函数等。



## 二、环境与运行

### 2.1 实验环境

实验基于`Windows11`平台以及`Python 3.12.8`。具体库需求详见`\requirements.txt `。

此外，如要运行，还需要准备`SNAP Twitch`[数据集](https://snap.stanford.edu/data/twitch.zip)，并放置为`dataset/twitch_zip/twitch.zip`。



### 2.2 快速运行

```powershell
git clone https://github.com/DengLinYe/SRM-GIB.git
cd SRM-GIB

python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 数据准备
python .\data\data_preparation.py
python .\data\topology_attack.py

# 攻击实验
python .\run_attack_curve.py

# 防御实验
python .\run_ablation.py
```

