[English](./README_en.md) | [简体中文](./README.md)

## Datasets

The dataset is downloaded from https://huggingface.co/datasets/ylecun/mnist.

Run `python download_dataset.py` to download the dataset. After the initial download, the data will be automatically cached in `~/.cache/huggingface/datasets`.

Simultaneously, the dataset will be saved to the local `./data` directory. For subsequent model training, you can load the data directly from your local disk without needing to fetch it through Hugging Face again.

## File Guide

It is highly recommended to browse and execute the files in the following order:

| Order | File | Function | Core Feature |
| :---: | :--- | :--- | :--- |
| 1 | `download_dataset.py` | Download the MNIST dataset | Automatically downloads and caches data locally |
| 2 | `mnist_numpy.py` | Simulated single-sample training | Pure NumPy implementation, perfect for grasping the core mathematical principles |
| 3 | `train_real_mnist.py` | Single-sample training on real data | Trains the model on the complete, real MNIST dataset (item by item) |
| 4 | `train_batch_mnist.py` | Mini-batch training on real data | Supports mini-batch matrix acceleration + test set validation |