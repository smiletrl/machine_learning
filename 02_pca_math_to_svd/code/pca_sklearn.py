import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 加载与预处理数据
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 去中心化并统一量纲

# 2. 工业级调用：执行 PCA 降维
pca = PCA(n_components=2)
Y_sklearn = pca.fit_transform(X_scaled)

# --- 打印结果与可视化 ---
print("--- Scikit-Learn 调包实现结果 ---")
print(f"前两维特征值 (方差): {pca.explained_variance_}")
print(f"前两维方差解释率: {pca.explained_variance_ratio_}")

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(Y_sklearn[y == i, 0], Y_sklearn[y == i, 1], 
                color=color, alpha=.8, lw=2, label=target_name)

plt.title('PCA of Wine Dataset (Scikit-Learn Implementation)')
plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()