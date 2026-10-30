import numpy as np

print("=== 练习3答案：特殊矩阵 ===\n")

# 3.1
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
print(f"3.1 Q^T@Q≈I: {np.allclose(Q.T @ Q, np.eye(2))}\n")

# 3.2
D = np.diag([2, 3, 4])
v = np.array([1, 1, 1])
print(f"3.2 D@v = {D @ v}\n")

# 3.3
S = np.array([[2, 1], [1, 3]])
eigenvalues = np.linalg.eigvals(S)
print(f"3.3 特征值: {eigenvalues}, 全实数: {np.all(np.isreal(eigenvalues))}\n")

# 3.4
u = np.array([1, 2, 3])
v = np.array([4, 5])
rank1 = np.outer(u, v)
print(f"3.4 秩一矩阵:\n{rank1}, 秩={np.linalg.matrix_rank(rank1)}\n")

# 3.5
b = np.array([[1], [0]], dtype=float)
P = b @ b.T / (b.T @ b)
test_v = np.array([[3], [4]], dtype=float)
print(f"3.5 P:\n{P}\nP@[3,4]^T:\n{P @ test_v}\n")
