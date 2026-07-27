import numpy as np

print("=== 奇异值分解 (SVD) ===")

# 创建一个矩阵
A = np.array([[1, 2, 3], [4, 5, 6]])
print(f"矩阵 A (形状 {A.shape}):\n{A}")

# 计算SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U (形状 {U.shape}):\n{U}")
print(f"奇异值 S:\n{S}")
print(f"Vt (形状 {Vt.shape}):\n{Vt}")

# 重构矩阵
A_reconstructed = np.dot(U, np.dot(np.diag(S), Vt))
print(f"重构矩阵:\n{A_reconstructed}")
print(f"A 是否等于重构矩阵: {np.allclose(A, A_reconstructed)}")

print("\n=== 图像压缩示例 ===")

# 创建一个简单的图像矩阵
np.random.seed(42)
image = np.random.rand(8, 8)  # 8x8图像

print(f"原始图像矩阵:\n{image}")

# 对图像进行SVD
U, S, Vt = np.linalg.svd(image, full_matrices=False)
print(f"奇异值: {S}")

# 保留不同数量的奇异值进行重构
for k in [1, 3, 5, 8]:
    # 只保留前k个奇异值
    S_k = np.zeros_like(S)
    S_k[:k] = S[:k]
    
    # 重构图像
    image_compressed = np.dot(U, np.dot(np.diag(S_k), Vt))
    print(f"\n保留 {k} 个奇异值重构:")
    print(f"压缩后图像矩阵:\n{image_compressed}")
    
    # 计算重构误差
    error = np.linalg.norm(image - image_compressed)
    print(f"重构误差: {error:.6f}")

print("\n=== SVD在推荐系统中的应用 ===")

# 创建一个用户-物品评分矩阵
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
print(f"用户-物品评分矩阵:\n{ratings}")

# 对评分矩阵进行SVD
U, S, Vt = np.linalg.svd(ratings, full_matrices=False)

# 只保留前2个奇异值
k = 2
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

# 重构评分矩阵
ratings_reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))
print(f"重构评分矩阵 (k={k}):\n{ratings_reconstructed}")

# 预测缺失评分
print(f"\n原始评分矩阵中缺失的位置 (0) 的预测值:")
for i in range(ratings.shape[0]):
    for j in range(ratings.shape[1]):
        if ratings[i, j] == 0:
            print(f"用户 {i+1} 对物品 {j+1} 的预测评分: {ratings_reconstructed[i, j]:.2f}")

print("\n=== SVD在降维中的应用 ===")

# 创建高维数据
np.random.seed(42)
X = np.random.rand(100, 10)  # 100个样本，10个特征

# 对数据矩阵进行SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# 保留不同数量的奇异值
for k in [2, 5, 8]:
    # 降维
    X_reduced = np.dot(U[:, :k], np.diag(S[:k]))
    print(f"降维到 {k} 维后数据形状: {X_reduced.shape}")
    
    # 计算重构误差
    X_reconstructed = np.dot(X_reduced, Vt[:k, :])
    error = np.linalg.norm(X - X_reconstructed)
    print(f"重构误差: {error:.6f}")