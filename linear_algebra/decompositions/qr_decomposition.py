import numpy as np

print("=== QR分解 ===")

# 创建一个矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"矩阵 A:\n{A}")

# 计算QR分解
Q, R = np.linalg.qr(A)
print(f"正交矩阵 Q:\n{Q}")
print(f"上三角矩阵 R:\n{R}")

# 验证 Q * R = A
A_reconstructed = np.dot(Q, R)
print(f"验证 Q * R = A: {np.allclose(A, A_reconstructed)}")

# 验证 Q 是正交矩阵
print(f"验证 Q^T * Q = I: {np.allclose(np.dot(Q.T, Q), np.eye(3))}")

print("\n=== QR分解在解线性方程组中的应用 ===")

# 解线性方程组 Ax = b
b = np.array([1, 2, 3])
print(f"方程组 Ax = b, 其中 b = {b}")

# 使用QR分解求解
# Ax = b => QRx = b => Rx = Q^T b
x = np.dot(np.linalg.inv(R), np.dot(Q.T, b))
print(f"解 x = {x}")

# 验证解
print(f"验证 Ax = b: {np.allclose(np.dot(A, x), b)}")

print("\n=== Gram-Schmidt正交化 ===")

def gram_schmidt(A):
    """Gram-Schmidt正交化过程"""
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        # 取出第i列
        v = A[:, i].astype(float)
        
        # 减去前面所有向量的投影
        for j in range(i):
            proj = np.dot(Q[:, j], v) / np.dot(Q[:, j], Q[:, j])
            v = v - proj * Q[:, j]
        
        # 归一化
        Q[:, i] = v / np.linalg.norm(v)
    
    return Q

# 测试Gram-Schmidt
A_test = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
Q_gs = gram_schmidt(A_test)
print(f"Gram-Schmidt正交化结果:\n{Q_gs}")
print(f"验证正交性: {np.allclose(np.dot(Q_gs.T, Q_gs), np.eye(3))}")

print("\n=== Householder变换 ===")

def householder(A):
    """Householder QR分解"""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for i in range(n):
        # 提取当前列的下三角部分
        x = R[i:, i]
        
        # 计算Householder向量
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        v = x.copy()
        v[0] -= alpha
        v = v / np.linalg.norm(v)
        
        # 应用Householder变换
        R[i:, i:] -= 2 * np.outer(v, np.dot(v, R[i:, i:]))
        Q[:, i:] -= 2 * np.outer(np.dot(Q[:, i:], v), v)
    
    return Q, R

# 测试Householder
Q_h, R_h = householder(A_test)
print(f"Householder QR分解:")
print(f"Q:\n{Q_h}")
print(f"R:\n{R_h}")
print(f"验证 Q * R = A: {np.allclose(np.dot(Q_h, R_h), A_test)}")

print("\n=== QR分解的应用 ===")
print("1. 解线性方程组")
print("2. 最小二乘问题")
print("3. 特征值计算（QR算法）")
print("4. 正交投影")