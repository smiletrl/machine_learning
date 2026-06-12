# 🚀 机器学习底层推导与工程实战

[English](./README_en.md) | [简体中文](./README.md)

> **撕开机器学习与底层算法的黑盒。**

## 📖 关于这个仓库

这是我从传统服务端开发（Golang/Node.js）跨界 AI 领域的**核心算法与推导实战库**。

面对“理论晦涩难懂，代码全靠调包”的 AI 现状，我决定回归代码本身，为每个核心算法打造 **“分层式”** 学习与实战体系：

1. **手撕底层黑盒 (纯 NumPy)**：脱离任何 AI 框架，手推矩阵求导，用底层代码还原数学逻辑，证明高深公式不过是纸老虎。
2. **对标工业生产 (主流 AI 框架)**：吃透底层后，提供 PyTorch 等现代工业级框架的对照实现，确保在真实生产环境里能打能抗。

不管你是：
- 🎓 **正被晦涩教材折磨，急需一套“说人话”底层推导的学生**
- 💻 **想要跨界进入 AI 赛道，亟需补齐算法底层逻辑的开发工程师**
- 🚀 **厌倦了只做“调包侠”，希望在真实场景中做到“懂底层、能落地”的极客**

欢迎把这里当作你的 AI 底层算法实战沙盒。

*快速体验：纯手写 NumPy 神经网络，实现手写数字识别，5 个 Epoch，总耗时仅需 0.6 秒，准确率直飙 96%+！👉 [点击查看运行效果](01_nn_from_scratch_mnist/README.md#-快速开始-quick-start)*

## 🛠️ 本地开发环境 (Local Setup)

本项目拥抱现代化的 Python 工具链，使用极致快速的 [uv](https://github.com/astral-sh/uv) 作为包和项目管理工具。

```bash
# 1. 克隆仓库并进入目录
git clone git@github.com:smiletrl/machine_learning.git
cd machine_learning

# 2. 一键同步并安装全部依赖 (uv 会自动为你创建虚拟环境 .venv)
uv sync

# 3. 激活当前项目的虚拟环境
source .venv/bin/activate
```

## 🗂️ 项目索引 (Projects)

| 编号 | 项目名称 | 核心技术点 | 状态 | 视频教程 |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [从零手撕神经网络 (MNIST 手写数字识别)](./01_nn_from_scratch_mnist) | MNIST 识别实战, NumPy 手推反向传播 | 🟢 已完成 | [即将发布]() |
| 02 | [经典机器学习核心算法纯手写 (规划中)](#) | 决策树, SVM, K-Means, 降维理论 | 🟡 规划中 | - |
| 03 | [凸优化理论与代码实战 (规划中)](#) | 梯度下降, 拉格朗日对偶, 损失曲面 | 🟡 规划中 | - |
| 04 | [大模型 Transformer 底层架构剖析 (规划中)](#) | 自注意力机制, 位置编码, 矩阵分块 | 🟡 规划中 | - |

---
*Follow my journey from traditional engineering to hardcore AI algorithms. Give it a ⭐️ if it inspires you!*
