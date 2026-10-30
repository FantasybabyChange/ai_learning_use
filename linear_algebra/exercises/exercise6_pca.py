import numpy as np

print("=== 练习6：主成分分析 PCA ===\n")

np.random.seed(42)
X = np.random.rand(100, 4)
X[:, 1] = X[:, 0] * 2 + np.random.normal(0, 0.1, 100)
X[:, 2] = X[:, 0] * 0.5 + np.random.normal(0, 0.1, 100)

# ── 练习 6.1：数据中心化 ──────────────────────────────────────
X_centered = None  # TODO

print("练习6.1：数据中心化")
print(
    f"  中心化后均值: {X_centered.mean(axis=0).round(6) if X_centered is not None else None}"
)
print(f"  正确答案: 接近 [0, 0, 0, 0]\n")

# ── 练习 6.2：协方差矩阵 ──────────────────────────────────────
cov_matrix = None  # TODO

print("练习6.2：协方差矩阵")
print(f"  协方差矩阵 shape: {cov_matrix.shape if cov_matrix is not None else None}")
print(f"  正确答案: (4, 4)\n")

# ── 练习 6.3：特征分解 ────────────────────────────────────────
eigenvalues = None  # TODO
eigenvectors = None  # TODO

print("练习6.3：协方差矩阵的特征分解")
print(f"  特征值: {eigenvalues.round(4) if eigenvalues is not None else None}")
print(f"  最大特征值对应第一主成分\n")

# ── 练习 6.4：选择主成分 ──────────────────────────────────────
# TODO: 选取前2个主成分（特征值最大的2个特征向量）
k = 2
top_components = None  # TODO: shape应为(4, 2)

print("练习6.4：选取主成分")
print(
    f"  前{k}个主成分 shape: {top_components.shape if top_components is not None else None}"
)
print(f"  正确答案: (4, 2)\n")

# ── 练习 6.5：降维投影 ────────────────────────────────────────
X_pca = None  # TODO

print("练习6.5：PCA降维")
print(f"  降维后 shape: {X_pca.shape if X_pca is not None else None}")
print(f"  正确答案: (100, 2)\n")

# ── 练习 6.6：方差解释比 ──────────────────────────────────────
if eigenvalues is not None:
    explained_ratio = None  # TODO: 前2个特征值的方差解释比之和
else:
    explained_ratio = None

print("练习6.6：方差解释比")
print(
    f"  前2个主成分解释方差比: {explained_ratio:.4f}"
    if explained_ratio is not None
    else ""
)
print(f"  含义: 降到2维保留了约多少信息\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
