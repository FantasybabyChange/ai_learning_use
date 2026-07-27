import numpy as np

print("=== 特征值分解 ===")

# 创建一个对称矩阵
A = np.array([[4, 2], [2, 3]])
print(f"矩阵 A:\n{A}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量 (列向量):\n{eigenvectors}")

# 验证 A * v = λ * v
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    Av_i = np.dot(A, v_i)
    lambda_v_i = lambda_i * v_i
    print(f"验证 A * v{i} = λ{i} * v{i}: {np.allclose(Av_i, lambda_v_i)}")

# 特征分解 A = Q * Λ * Q^T
Q = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = np.dot(Q, np.dot(Lambda, Q.T))
print(f"重构矩阵 A:\n{A_reconstructed}")
print(f"A 是否等于重构矩阵: {np.allclose(A, A_reconstructed)}")

print("\n=== 主成分分析 (PCA) ===")

# 生成一些数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 100个样本，2个特征
# 添加一些相关性
X[:, 1] = X[:, 0] * 2 + np.random.normal(0, 0.1, 100)

# 中心化数据
X_centered = X - X.mean(axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)
print(f"协方差矩阵:\n{cov_matrix}")

# 特征值分解协方差矩阵
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 主成分就是特征向量，特征值表示方差
print(f"各主成分解释的方差比例: {eigenvalues / eigenvalues.sum()}")

# 投影数据到主成分
X_pca = np.dot(X_centered, eigenvectors)
print(f"PCA变换后数据形状: {X_pca.shape}")

print("\n=== 幂迭代法求主特征值 ===")

def power_iteration(A, num_iterations=100, tolerance=1e-10):
    """幂迭代法求矩阵的主特征值和特征向量"""
    n = A.shape[0]
    b_k = np.random.rand(n)
    
    for _ in range(num_iterations):
        # 计算 A * b_k
        b_k1 = np.dot(A, b_k)
        
        # 归一化
        b_k1_norm = np.linalg.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm
        
        # 检查收敛
        if np.linalg.norm(b_k1 - b_k) < tolerance:
            break
        b_k = b_k1
    
    # 计算特征值
    eigenvalue = np.dot(np.dot(b_k.T, A), b_k)
    return eigenvalue, b_k

# 测试幂迭代法
eigenvalue, eigenvector = power_iteration(A)
print(f"幂迭代法求得的主特征值: {eigenvalue}")
print(f"幂迭代法求得的主特征向量: {eigenvector}")