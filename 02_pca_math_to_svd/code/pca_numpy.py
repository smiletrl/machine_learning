import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# 1. 加载与预处理数据
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 去中心化并统一量纲
n_samples, n_features = X_scaled.shape

# 2. 核心数学步骤：奇异值分解 (SVD)
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

# 3. 计算特征值（无偏方差）与方差解释率
eigenvalues = (S ** 2) / (n_samples - 1)
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

# 4. 降维坐标映射 (选取前两个主成分)
V_k = Vt[:2, :].T 
Y_numpy = X_scaled.dot(V_k)

# --- 打印结果与可视化 ---
print("--- 手写 SVD 实现结果 ---")
print(f"前两维特征值 (方差): {eigenvalues[:2]}")
print(f"前两维方差解释率: {explained_variance_ratio[:2]}")

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(Y_numpy[y == i, 0], Y_numpy[y == i, 1], 
                color=color, alpha=.8, lw=2, label=target_name)

plt.title('PCA of Wine Dataset (NumPy SVD Implementation)')
plt.xlabel(f'PC1 (Explained Variance: {explained_variance_ratio[0]:.2%})')
plt.ylabel(f'PC2 (Explained Variance: {explained_variance_ratio[1]:.2%})')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()