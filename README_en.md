# 🚀 Machine Learning: Derivations & Engineering Practice

[English](./README_en.md) | [简体中文](./README.md)

> **Demystifying the black box of Deep Learning and Core Algorithms.**

## 📖 About This Repository

This repository documents my transition from traditional backend development (Golang/Node.js) into AI, serving as a **practical library for core algorithms and low-level derivations**.

Facing the current AI landscape where "theory is obscure, but coding is just calling APIs," I decided to return to the code itself, building a **"Progressive"** practical system for every core algorithm:

1. **Demystifying the Black Box (Pure NumPy):** Ditching all AI frameworks to manually derive matrix calculus and reverse-engineer the math with low-level code, proving that complex formulas are just paper tigers.
2. **Industrial Benchmarking (Modern AI Frameworks):** After mastering the low-level mechanics, I provide equivalent implementations using PyTorch and other industrial-grade frameworks to ensure robust performance in real-world production environments.

Whether you are:
- 🎓 **A student** struggling with obscure textbooks, looking for "plain English" derivations.
- 💻 **A software engineer** looking to pivot into AI and needing to solidify your algorithmic foundation.
- 🚀 **A geek** tired of just "importing libraries", who wants to truly understand the core mechanics and build real-world applications.

Welcome to your AI core algorithm practice sandbox.

*Quick Teaser: A pure NumPy neural network built from scratch for handwritten digit recognition. Training 5 Epochs takes only 0.6 seconds, with accuracy soaring past 96%! 👉 [Click to see it in action](01_nn_from_scratch_mnist/README_en.md#-quick-start)*

## 🛠️ Local Setup

This project embraces modern Python toolchains, utilizing the blazing-fast [uv](https://github.com/astral-sh/uv) for package and project management.

```bash
# 1. Clone the repository and navigate to the directory
git clone git@github.com:smiletrl/machine_learning.git
cd machine_learning

# 2. Sync and install all dependencies (uv will automatically create a .venv virtual environment for you)
uv sync

# 3. Activate the current project's virtual environment
source .venv/bin/activate
```

## 🗂️ Projects Index

| No. | Project Name | Core Technologies | Status | Video Tutorial |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [Neural Network From Scratch (MNIST)](./01_nn_from_scratch_mnist) | MNIST, NumPy Backprop | 🟢 Done | [Coming Soon]() |
| 02 | [Classic Machine Learning Core Algorithms](#) | Decision Trees, SVM, K-Means, PCA | 🟡 Planned | - |
| 03 | [Convex Optimization in Practice](#) | Gradient Descent, Duality, Loss Surfaces | 🟡 Planned | - |
| 04 | [Transformer Architecture Demystified](#) | Self-Attention, Positional Encoding | 🟡 Planned | - |

---
*Follow my journey from traditional engineering to hardcore AI algorithms. Give it a ⭐️ if it inspires you!*
