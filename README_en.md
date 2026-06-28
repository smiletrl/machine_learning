# 🚀 Machine Learning: Derivations & Engineering Practice

[English](./README_en.md) | [简体中文](./README.md)

> **Demystifying the black box of Machine Learning and Core Algorithms.**

## 📖 About This Repository

This repository is a practical sandbox where a Senior Backend Architect explores the low-level computational mechanics of AI through a systems engineering lens.

Facing the current AI landscape where "theory is obscure, and coding is just calling APIs," I decided to follow my instinct as an infrastructure engineer: strip away the frameworks and return to the bare essence of code and mathematics. This project provides a "tiered" learning and validation system for core algorithms:

1. **Demystifying the Black Box (Pure NumPy):** Ditching all AI frameworks to manually derive matrix calculus and reverse-engineer the math with low-level code, proving that complex formulas are just paper tigers.
2. **Industrial Benchmarking (Modern AI Frameworks):** After mastering the low-level mechanics, I provide equivalent implementations using PyTorch and other industrial-grade frameworks to ensure robust performance in real-world production environments.

Whether you are:
- 🎓 **A student** struggling with obscure textbooks, looking for "plain English" derivations.
- 💻 **A software engineer** looking to pivot into AI and needing to solidify your algorithmic foundation.
- 🚀 **A geek** tired of just "importing libraries", who wants to truly understand the core mechanics and build real-world applications.

Welcome to your AI core algorithm practice sandbox.

*Quick Teaser: You can thoroughly master the low-level backpropagation of neural networks via the [Formula Derivation Guide](01_nn_from_scratch_mnist/docs/README_en.md), and get your hands dirty by running the pure NumPy neural network following the [Quick Start Guide](01_nn_from_scratch_mnist/README_en.md#quick-start). Training 5 Epochs takes only 0.6 seconds, with accuracy soaring past 96%!*

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

📺 **Step-by-step mathematical derivations and video tutorials are available on Xiaohongshu (小红书) & Douyin (抖音). Account: @清影Labs**

| No. | Project Name | Core Technologies | Status |
| :--- | :--- | :--- | :--- |
| 01 | [Neural Network From Scratch (MNIST)](./01_nn_from_scratch_mnist) | MNIST, NumPy Backpropagation | 🟢 Done |
| 02 | [PCA Deep Dive: Algebraic Proofs to Industrial SVD](./02_pca_math_to_svd) | SVD Equivalence, Numerical Stability, NumPy Whitebox | 🟢 Done |
| 03 | [Classic ML Algorithms Demystified](#) | Decision Trees, SVM, K-Means Core Logic | 🟡 Planned |
| 04 | [Convex Optimization in Practice](#) | Gradient Descent, Duality, Loss Surfaces | 🟡 Planned |
| 05 | [Transformer Architecture Unveiled](#) | Self-Attention, Positional Encoding, Block Matrix | 🟡 Planned |

---
*Follow my journey bridging high-performance backend engineering with hardcore AI computation. Give it a ⭐️ if it inspires you!*
