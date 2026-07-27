import numpy as np

# 向量基础
print("=== 向量基础 ===")
# 创建向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"向量 v1: {v1}")
print(f"向量 v2: {v2}")

# 向量加法
v_add = v1 + v2
print(f"v1 + v2 = {v_add}")

# 点积
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")

# 向量范数
norm_v1 = np.linalg.norm(v1)
print(f"||v1|| = {norm_v1}")

print("\n=== 矩阵基础 ===")
# 创建矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")

# 矩阵加法
A_add_B = A + B
print(f"A + B:\n{A_add_B}")

# 矩阵乘法
A_mul_B = np.dot(A, B)
print(f"A * B:\n{A_mul_B}")

# 矩阵转置
A_transpose = A.T
print(f"A^T:\n{A_transpose}")

# 矩阵逆
A_inv = np.linalg.inv(A)
print(f"A^-1:\n{A_inv}")

# 验证 A * A^-1 = I
print(f"A * A^-1:\n{np.dot(A, A_inv)}")

# 矩阵的行列式
det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")

# 特殊矩阵
print("\n=== 特殊矩阵 ===")
# 单位矩阵
I = np.eye(2)
print(f"单位矩阵 I:\n{I}")

# 对角矩阵
D = np.diag([1, 2, 3])
print(f"对角矩阵 D:\n{D}")

# 对称矩阵
S = np.array([[1, 2], [2, 1]])
print(f"对称矩阵 S:\n{S}")
print(f"S 是否对称: {np.allclose(S, S.T)}")

# 正交矩阵
Q = np.array([[0, 1], [-1, 0]])
print(f"正交矩阵 Q:\n{Q}")
print(f"Q 是否正交: {np.allclose(np.dot(Q, Q.T), np.eye(2))}")

print("\n=== 张量基础 ===")
# 3D张量（例如，彩色图像）
tensor_3d = np.random.rand(3, 4, 5)  # 3个通道，4x5像素
print(f"3D张量形状: {tensor_3d.shape}")
print(f"张量元素个数: {tensor_3d.size}")

# 张量运算
print(f"张量求和: {tensor_3d.sum()}")
print(f"张量均值: {tensor_3d.mean()}")