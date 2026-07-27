import numpy as np

print("=== Cholesky分解 ===")

# 创建一个对称正定矩阵
A = np.array([[4, 2], [2, 3]])
print(f"对称正定矩阵 A:\n{A}")

# 验证矩阵是对称的
print(f"矩阵是否对称: {np.allclose(A, A.T)}")

# 验证矩阵是正定的（所有特征值大于0）
eigenvalues = np.linalg.eigvals(A)
print(f"特征值: {eigenvalues}")
print(f"矩阵是否正定: {np.all(eigenvalues > 0)}")

# 计算Cholesky分解
L = np.linalg.cholesky(A)
print(f"下三角矩阵 L:\n{L}")

# 验证 L * L^T = A
A_reconstructed = np.dot(L, L.T)
print(f"验证 L * L^T = A: {np.allclose(A, A_reconstructed)}")

print("\n=== Cholesky分解在解线性方程组中的应用 ===")

# 解线性方程组 Ax = b
b = np.array([1, 2])
print(f"方程组 Ax = b, 其中 b = {b}")

# 使用Cholesky分解求解
# Ax = b => LL^T x = b
# 令 y = L^T x, 则 Ly = b
y = np.linalg.solve(L, b)
x = np.linalg.solve(L.T, y)
print(f"解 x = {x}")

# 验证解
print(f"验证 Ax = b: {np.allclose(np.dot(A, x), b)}")

print("\n=== Cholesky分解在多元正态分布中的应用 ===")

# 多元正态分布的概率密度函数
def multivariate_normal(x, mu, sigma):
    """计算多元正态分布的概率密度"""
    n = len(x)
    L = np.linalg.cholesky(sigma)
    
    # 计算 (x - mu)^T * sigma^{-1} * (x - mu)
    z = np.linalg.solve(L, x - mu)
    exponent = -0.5 * np.dot(z, z)
    
    # 计算归一化常数
    normalizer = 1.0 / (np.power(2 * np.pi, n / 2) * np.prod(np.diag(L)))
    
    return normalizer * np.exp(exponent)

# 示例：2D多元正态分布
mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 2]])

# 计算几个点的概率密度
points = [np.array([0, 0]), np.array([1, 1]), np.array([-1, -1])]
for x in points:
    pdf = multivariate_normal(x, mu, sigma)
    print(f"P({x}) = {pdf:.6f}")

print("\n=== Cholesky分解在优化中的应用 ===")

# 在二次规划中，Hessian矩阵通常是对称正定的
def quadratic_function(x, Q, c):
    """二次函数 f(x) = 0.5 * x^T Q x + c^T x"""
    return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x)

def gradient(x, Q, c):
    """二次函数的梯度"""
    return np.dot(Q, x) + c

# 示例二次函数
Q = np.array([[4, 2], [2, 3]])
c = np.array([1, 2])

# 使用Cholesky分解求最小值
# 梯度为0: Qx + c = 0 => x = -Q^{-1}c
L = np.linalg.cholesky(Q)
x_min = -np.linalg.solve(Q, c)

print(f"二次函数最小值点: {x_min}")
print(f"最小值: {quadratic_function(x_min, Q, c):.6f}")

print("\n=== Cholesky分解的数值稳定性 ===")

# 对比Cholesky分解和直接求逆
A_large = np.array([[10, 2, 1], [2, 10, 3], [1, 3, 10]])
print(f"测试矩阵:\n{A_large}")

# Cholesky分解
L = np.linalg.cholesky(A_large)
print(f"Cholesky分解 L:\n{L}")

# 使用Cholesky分解求解
b = np.array([1, 2, 3])
x_cholesky = np.linalg.solve(A_large, b)

# 直接求逆
x_inv = np.dot(np.linalg.inv(A_large), b)

print(f"Cholesky分解解: {x_cholesky}")
print(f"直接求逆解: {x_inv}")
print(f"两种方法结果一致: {np.allclose(x_cholesky, x_inv)}")

print("\n=== Cholesky分解的应用场景 ===")
print("1. 解对称正定线性方程组")
print("2. 多元正态分布采样")
print("3. 优化问题（二次规划）")
print("4. 协方差矩阵的分解")
print("5. 有限元分析")