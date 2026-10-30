import numpy as np

print("=== 练习2答案：矩阵运算 ===\n")

# 2.1
I = np.eye(3)
Z = np.zeros((2, 3))
D = np.diag([2, 3, 4])
print(f"2.1 I:\n{I}\nZ:\n{Z}\nD:\n{D}\n")

# 2.2
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"2.2 A+B:\n{A + B}\n")

# 2.3
print(f"2.3 A@B:\n{A @ B}\n")

# 2.4
C = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2.4 C.T:\n{C.T}  shape: {C.T.shape}\n")

# 2.5
A_inv = np.linalg.inv(A)
print(f"2.5 inv(A):\n{A_inv}")
print(f"  A@inv(A)≈I: {np.allclose(A @ A_inv, np.eye(2))}\n")

# 2.6
print(f"2.6 det(A) = {np.linalg.det(A)}\n")

# 2.7
print(f"2.7 tr(A) = {np.trace(A)}\n")

# 2.8
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
N = np.array([[1, 2], [3, 4]])
print(f"2.8 S对称: {np.allclose(S, S.T)}, N对称: {np.allclose(N, N.T)}\n")

# 2.9
print(f"2.9 ||A||_F = {np.linalg.norm(A, 'fro'):.4f}\n")
