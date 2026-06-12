[English](./README_en.md) | [简体中文](./README.md)

## Datasets

数据集从 https://huggingface.co/datasets/ylecun/mnist 下载。

执行 `python download_dataset.py`, 下载数据集。数据首次下载后，即被缓存到 `~/.cache/huggingface/datasets`.

同时数据集会被保存到当前目录`./data`下。后续模型训练，可以直接从本地磁盘中拉取，无需通过huggingface。

## 文件说明

建议按照下列顺序浏览使用文件：

| 顺序 | 文件 | 功能说明 | 核心特点 |
| :---: | :--- | :--- | :--- |
| 1 | `download_dataset.py` | 下载 MNIST 数据集 | 自动下载并解压到本地 |
| 2 | `mnist_numpy.py` | 模拟单样本学习 | 使用 NumPy 模拟，适合理解基础原理 |
| 3 | `train_real_mnist.py` | 真实数据全集的单样本学习 | 在完整数据集上训练单样本模型 |
| 4 | `train_batch_mnist.py` | 真实数据集的批量学习 | 支持批量训练 + 测试集验证 |
