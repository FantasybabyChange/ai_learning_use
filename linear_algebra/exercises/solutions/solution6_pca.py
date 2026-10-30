import numpy as np

print("=== 练习6答案：PCA ===\n")

np.random.seed(42)
X = np.random.rand(100, 4)
X[:, 1] = X[:, 0] * 2 + np.random.normal(0, 0.1, 100)
X[:, 2] = X[:, 0] * 0.5 + np.random.normal(0, 0.1, 100)

# 6.1
X_centered = X - X.mean(axis=0)
print(f"6.1 中心化后均值: {X_centered.mean(axis=0).round(6)}\n")

# 6.2
cov_matrix = np.cov(X_centered.T)
print(f"6.2 协方差矩阵shape: {cov_matrix.shape}\n")

# 6.3
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print(f"6.3 特征值(降序): {eigenvalues.round(4)}\n")

# 6.4
k = 2
top_components = eigenvectors[:, :k]
print(f"6.4 前{k}主成分shape: {top_components.shape}\n")

# 6.5
X_pca = X_centered @ top_components
print(f"6.5 降维后shape: {X_pca.shape}\n")

# 6.6
explained_ratio = eigenvalues[:k].sum() / eigenvalues.sum()
print(f"6.6 前{k}主成分解释方差比: {explained_ratio:.4f}\n")
