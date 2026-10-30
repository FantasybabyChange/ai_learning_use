import numpy as np

print("=== 练习2：矩阵运算 ===\n")

# ── 练习 2.1：矩阵创建 ────────────────────────────────────────
# TODO: 创建一个 3x3 单位矩阵、一个 2x3 全零矩阵、一个对角矩阵 diag([2,3,4])
I = None  # TODO
Z = None  # TODO
D = None  # TODO

print("练习2.1：矩阵创建")
print(f"  单位矩阵 I:\n{I}")
print(f"  全零矩阵 Z:\n{Z}")
print(f"  对角矩阵 D:\n{D}\n")

# ── 练习 2.2：矩阵加法 ────────────────────────────────────────
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result_add = None  # TODO

print("练习2.2：矩阵加法")
print(f"  A + B = \n{result_add}")
print(f"  正确答案:\n[[6 8]\n [10 12]]\n")

# ── 练习 2.3：矩阵乘法 ────────────────────────────────────────
# TODO: 计算 A @ B
result_mul = None  # TODO

print("练习2.3：矩阵乘法")
print(f"  A @ B = \n{result_mul}")
print(f"  正确答案:\n[[19 22]\n [43 50]]\n")

# ── 练习 2.4：转置 ────────────────────────────────────────────
C = np.array([[1, 2, 3], [4, 5, 6]])
C_T = None  # TODO

print("练习2.4：转置")
print(f"  C shape: {C.shape}")
print(f"  C.T = \n{C_T}")
print(f"  C.T shape: {C_T.shape if C_T is not None else None}")
print(f"  正确答案 shape: (3, 2)\n")

# ── 练习 2.5：逆矩阵 ──────────────────────────────────────────
# TODO: 计算 A 的逆矩阵，并验证 A @ inv(A) ≈ I
A_inv = None  # TODO
verify = None  # TODO: A @ A_inv 的结果

print("练习2.5：逆矩阵")
print(f"  inv(A) = \n{A_inv}")
print(
    f"  A @ inv(A) ≈ I: {np.allclose(verify, np.eye(2)) if verify is not None else None}"
)
print(f"  正确答案: True\n")

# ── 练习 2.6：行列式 ──────────────────────────────────────────
det_A = None  # TODO

print("练习2.6：行列式")
print(f"  det(A) = {det_A}")
print(f"  正确答案: -2.0\n")

# ── 练习 2.7：矩阵的迹 ────────────────────────────────────────
trace_A = None  # TODO

print("练习2.7：矩阵的迹")
print(f"  tr(A) = {trace_A}")
print(f"  正确答案: 5\n")

# ── 练习 2.8：对称矩阵判断 ────────────────────────────────────
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
N = np.array([[1, 2], [3, 4]])
is_S_symmetric = None  # TODO
is_N_symmetric = None  # TODO

print("练习2.8：对称矩阵判断")
print(f"  S 是否对称: {is_S_symmetric}  正确答案: True")
print(f"  N 是否对称: {is_N_symmetric}  正确答案: False\n")

# ── 练习 2.9：矩阵范数 ────────────────────────────────────────
norm_fro = None  # TODO: Frobenius范数

print("练习2.9：矩阵范数")
print(f"  ||A||_F = {norm_fro}")
print(f"  正确答案: {np.sqrt(1 + 4 + 9 + 16):.4f}\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
