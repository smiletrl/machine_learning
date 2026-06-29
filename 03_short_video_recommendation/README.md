# 短视频推荐系统：从底层推导到工业落地路线图

本项目旨在用纯 NumPy 撕开现代短视频推荐系统的底层黑盒，主打“底层推导与工程实战”。我们不仅要实现数学上的前向与反向传播，还要构建一个完整的从零开始的端到端推荐流水线。

> 🚧 **当前状态：研发中 (Work in Progress)** > 本文档为该模块的核心设计蓝图与 TODO 追踪列表。

结合工业界痛点，本模块不再一味贪多复现论文，而是将 **80% 的精力集中于 Attention 的底层推导**，并补全数据、召回、评估与推理的工程地基。建议按以下四大阶段推进：

## 阶段一：工程地基 —— 真实工业数据接入与召回塔

摒弃写代码模拟假数据的玩具思维，直接让模型在真实的工业恶劣环境中（极度稀疏、长尾分布）进行试炼。

* [ ] **1.1 腾讯 Tenrec 数据集接入 (锁定 QK 场景):** * 明确指定仅使用 Tenrec 中的 **QK (QQ看点/短视频)** 场景数据，避免多场景混合。

  * **核心契合点：** 该数据集主要提供脱敏的离散 ID，没有任何花哨的文本特征，完美倒逼硬核实现“ID -> Embedding”的底层映射逻辑。

* [ ] **1.2 双塔召回模型 (Two-Tower Retrieval):** * 在进入复杂的注意力机制之前，先用纯 NumPy 实现经典的 User Tower 和 Item Tower。

  * 证明通过简单的内积 (Dot-Product)，就能极速从 Tenrec 的海量物品库中“粗筛”出 Top-100 候选集。

## 阶段二：降维与优化 —— Embedding 与负采样

解决离散 ID 到稠密向量的映射，以及随之而来的计算量爆炸问题。

* [ ] **2.1 Embedding 查表与更新:** * 实现权重矩阵 $W_{emb} \in \mathbb{R}^{V \times D}$ 的前向索引 (`W_emb[history_ids]`)。

  * 手推并证明这个 Embedding 矩阵可以通过后续网络传回的梯度进行反向更新。

* [ ] **2.2 负采样 (Negative Sampling):** * **工程精髓：** 面对 Tenrec 中海量的 Item ID，必须在 NumPy 中实现 Sampled Softmax 或 BPR Loss，避免全量 Softmax 瞬间撑爆内存。

## 阶段三：核心攻坚 (占 80% 精力) —— Attention 矩阵与工程闭环

把全部精力砸在这里，死磕推荐系统的心脏，并打通评估与推理的最后一公里。

* [ ] **3.1 Scaled Dot-Product 前向与反向 (核心试金石):** * 用纯 NumPy 实现 Attention 公式，处理 Masking 掩码机制（Padding 处 $-\infty$）。

  * 手推 Softmax 的雅可比矩阵，实现 Loss 对 $Q, K, V$ 及 $W_{emb}$ 的偏导数，并严格进行**数值梯度校验 (Numerical Gradient)**。

* [ ] **3.2 模型评估指标 (Metrics):** * 拒绝只看 Loss 下降。纯手工实现 **Recall@50** (评估双塔召回能力) 和 **NDCG@10** (评估 Attention 排序精度)。

* [ ] **3.3 权重持久化与推理 (Checkpoint & Inference):**

  * 引入 `np.savez` 保存训练好的 $W_{emb}$ 和 Attention 权重。

  * 编写 `predict.py`，模拟线上真实推断：输入 User ID，毫秒级走完“召回 -> 排序”全链路，输出 Top-10 视频。

## 阶段四：工业模型对标 —— 阿里 DIN 的目标注意力

在手写完标准 Attention 后，用最精简的代码展示其在工业界最成功的变形体。

* [ ] **白盒实现 DIN (仅限前向推断 Forward Pass):** * 摒弃标准点积，用纯 NumPy 实现 DIN 的“外积拼接 + MLP”局部激活单元。

  * **⚠️ 架构师注记：** DIN 的核心在于改变 Score 的计算方式，其 MLP 部分的训练反向传播原理与前置项目一致。本阶段**仅展示其最具特色的“局部激活”前向逻辑**，证明其能精准捕捉用户动态兴趣即可。

## 📚 附录：核心参考论文 (理论演进脉络)

### 1. 硅谷奠基篇：架构与数学的起源

* **现代深度推荐开山作 —— Google YouTube DNN (2016):** 确立了“召回 + 排序”的现代工业架构。

* **自注意力序列推荐先驱 —— UCSD & Pinterest SASRec (2018):** 将 Transformer 机制完美应用于用户行为序列预测的海外神作，手撕 Attention 的绝佳蓝本。

* **工程架构的极致 —— Meta (Facebook) DLRM (2019):** 探讨 Embedding 内存墙问题与特征交叉工程。

### 2. 国内大厂深化篇：极限压榨用户序列

* **目标注意力机制 —— 阿里 DIN (2018):** 提出“局部激活”，精准捕捉与当前候选视频相关的历史兴趣。

* **Transformer 的电商化 —— 阿里 BST (2019):** 完整的 Transformer 架构在真实海量数据场景的应用。

## 💻 Repo 目录结构规划

```text
03_short_video_recommendation/
├── utils/                       # 核心工程地基
│   ├── metrics.py               # 评估指标：Recall@K, NDCG@K
│   └── checkpoint.py            # 权重持久化：基于 np.savez 的模型存取
├── 01_tenrec_data_loader/       # 腾讯 Tenrec 真实工业数据集接入
│   ├── download_and_preprocess.py # 下载与脱敏 ID 预处理 (仅限 QK 视频场景)
│   └── data_iterator.py         # 构建 Batch 数据迭代器
├── 02_retrieval_two_tower/      # 召回阶段：双塔模型基础
│   └── user_item_dot_product.py
├── 03_embeddings/               # 离散ID向量化与计算优化
│   ├── embedding_layer.py       # 纯NumPy实现 W_emb 查表与反向梯度
│   ├── negative_sampling.py     # 核心：负采样/Sampled Softmax 逻辑
│   └── test_embedding_grad.py   # 数值梯度检验
├── 04_attention_mechanism/      # 核心攻坚 (投入主要精力)
│   ├── scaled_dot_product.py    # 带 Mask 的纯 NumPy Attention 前向
│   ├── attention_backward.py    # 手推 Jacobian，实现 backward() 函数
│   └── test_attention_grad.py   # 梯度校验 (本模块的核心试金石)
├── 05_industrial_models/        # 工业界网络结构落地
│   ├── din_target_attention.py  # 阿里 DIN 激活单元代码 (仅展示 Forward Pass)
│   └── sasrec_concept.md        # 阐述如何基于当前架构向自注意力序列演进
├── 06_inference_pipeline/       # 线上推理模拟闭环
│   └── predict.py               # 串联召回与排序，输入 UserID 输出 Top-10 推荐
└── README.md                    # 也就是本文档