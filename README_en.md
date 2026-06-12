# 🚀 AI Core Algorithms & Engineering Practice Lab

[English](./README_en.md) | [简体中文](./README.md)

> **Demystifying the black box of Deep Learning and Core Algorithms.**

## 📖 About This Repository

This repository documents my machine learning practice journey, transitioning from traditional backend development (Golang/Node.js) into the world of AI. 

Facing the current AI landscape where "theory is obscure, but coding is just calling APIs," I decided to tackle this with a radical engineering approach. I built a **"Dual-Track"** practical system for every project here:

1. **From Scratch (Pure NumPy):** Ditching all AI frameworks to manually derive matrix calculus and reverse-engineer the math with low-level code, proving that complex formulas are just paper tigers.
2. **Industry Standard (PyTorch):** After mastering the low-level mechanics, I provide the equivalent implementations using modern, industrial-grade frameworks to ensure robust performance in real-world production environments.

Whether you are:
- 🎓 **A student** struggling with obscure textbooks, looking for "plain English" derivations.
- 💻 **A software engineer** looking to pivot into AI and needing to solidify your algorithmic foundation.
- 🚀 **A geek** tired of just "importing libraries", who wants to truly understand the core mechanics and build real-world applications.

Welcome to your **"Algorithm Arsenal"**.

*Quick Teaser: A pure NumPy neural network built from scratch for handwritten digit recognition. Training 5 Epochs takes only 0.6 seconds, with accuracy soaring past 96%! 👉 [Click to see it in action](01_nn_from_scratch_mnist/README_en.md#quick-start)*

## 🛠️ Local Setup

This project embraces modern Python toolchains, utilizing the blazing-fast [uv](https://github.com/astral-sh/uv) for package and project management.

```bash
# 1. Initialize the local virtual environment
uv init

# 2. Add dependencies (examples used in the project)
uv add torch numpy matplotlib datasets

# 3. Sync all dependencies
uv sync

# 4. Activate the virtual environment
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
