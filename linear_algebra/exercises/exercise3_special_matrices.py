import numpy as np

print("=== 练习3：特殊矩阵 ===\n")

# ── 练习 3.1：正交矩阵 ────────────────────────────────────────
# TODO: 验证以下矩阵是否为正交矩阵（Q^T @ Q = I）
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
is_orthogonal = None  # TODO: 验证 Q^T @ Q ≈ I

print("练习3.1：正交矩阵")
print(f"  旋转矩阵 Q:\n{Q.round(4)}")
print(f"  Q^T @ Q ≈ I: {is_orthogonal}")
print(f"  正确答案: True\n")

# ── 练习 3.2：对角矩阵运算 ────────────────────────────────────
# TODO: 计算对角矩阵 D=diag([2,3,4]) 与向量 v=[1,1,1] 的乘积
D = np.diag([2, 3, 4])
v = np.array([1, 1, 1])
Dv = None  # TODO

print("练习3.2：对角矩阵")
print(f"  D @ v = {Dv}")
print(f"  正确答案: [2 3 4]\n")

# ── 练习 3.3：对称矩阵的特征值 ────────────────────────────────
# TODO: 计算对称矩阵 S 的特征值，验证它们都是实数
S = np.array([[2, 1], [1, 3]])
eigenvalues = None  # TODO

print("练习3.3：对称矩阵特征值")
print(f"  特征值: {eigenvalues}")
print(
    f"  都是实数: {np.all(np.isreal(eigenvalues)) if eigenvalues is not None else None}"
)
print(f"  正确答案: True\n")

# ── 练习 3.4：秩一矩阵 ────────────────────────────────────────
# TODO: 用外积构造一个秩一矩阵 u @ v^T
u = np.array([1, 2, 3])
v = np.array([4, 5])
rank1 = None  # TODO
rank_val = None  # TODO: 矩阵的秩

print("练习3.4：秩一矩阵")
print(f"  秩一矩阵:\n{rank1}")
print(f"  矩阵的秩: {rank_val}")
print(f"  正确答案: 1\n")

# ── 练习 3.5：投影矩阵 ────────────────────────────────────────
# TODO: 构造向 b=[1,0] 方向的投影矩阵 P = b @ b^T / (b^T @ b)
b = np.array([[1], [0]], dtype=float)
P = None  # TODO

test_v = np.array([[3], [4]], dtype=float)
projected = None  # TODO: P @ test_v

print("练习3.5：投影矩阵")
print(f"  投影矩阵 P:\n{P}")
print(f"  P @ [3,4]^T = \n{projected}")
print(f"  正确答案: [[3],[0]]\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
