import numpy as np

# ============================================================
# 线性方程组示例代码
# 对应 basics/README.md 中的线性方程组章节
# ============================================================

# ── 1. 线性方程组的矩阵表示 Ax = b ─────────────────────────────
print("=== 1. 线性方程组的矩阵表示 ===")
# 方程组：
#   2x + y = 5
#    x + 3y = 7
# 写成矩阵形式 Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])
print(f"系数矩阵 A:\n{A}")
print(f"常数向量 b: {b}")
print("矩阵形式: Ax = b")
print(f"A.shape = {A.shape}  (2个方程, 2个未知数)")

# ── 2. 高斯消元法 ──────────────────────────────────────────────
print("\n=== 2. 高斯消元法 ===")


# 手动实现高斯消元（展示原理）
def gaussian_elimination(A, b):
    """手动高斯消元法求解 Ax = b"""
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    print("增广矩阵 [A|b]:")
    print(M.round(3))

    # 前向消元
    for col in range(n):
        pivot = M[col, col]
        if abs(pivot) < 1e-10:
            raise ValueError("主元为0，需要行交换")
        M[col] = M[col] / pivot
        for row in range(col + 1, n):
            factor = M[row, col]
            M[row] = M[row] - factor * M[col]

    print("\n上三角形式:")
    print(M.round(3))

    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = M[i, -1] - np.dot(M[i, i + 1 : n], x[i + 1 :])

    return x


A_demo = np.array([[2.0, 1.0], [1.0, 3.0]])
b_demo = np.array([5.0, 7.0])
x_manual = gaussian_elimination(A_demo, b_demo)
print(f"\n手动求解结果: x = {x_manual.round(4)}")

# NumPy 求解
x_numpy = np.linalg.solve(A_demo, b_demo)
print(f"np.linalg.solve: x = {x_numpy.round(4)}")
print(f"验证 Ax = b: A @ x = {(A_demo @ x_numpy).round(4)}  (应等于 b = {b_demo})")

# ── 3. 矩阵的秩 ────────────────────────────────────────────────
print("\n=== 3. 矩阵的秩 ===")

# 满秩矩阵
A_full = np.array([[1, 2], [3, 4]])
rank_full = np.linalg.matrix_rank(A_full)
print(f"A_full:\n{A_full}")
print(f"rank(A_full) = {rank_full}  (满秩，有唯一解)")

# 不满秩矩阵（第二行是第一行的倍数）
A_singular = np.array([[1, 2], [2, 4]])
rank_singular = np.linalg.matrix_rank(A_singular)
print(f"\nA_singular:\n{A_singular}")
print(f"rank(A_singular) = {rank_singular}  (不满秩，无解或无穷多解)")

# 秩的意义
print("\n秩的意义:")
print("- rank(A) = 独立方程的个数")
print("- rank(A) = 列空间的维数（列向量张成的空间维度）")
print("- 若 rank(A) = n (未知数个数)，则有唯一解")
print("- 若 rank(A) < n，则有无穷多解或无解")

# ── 4. 线性相关性与线性无关 ─────────────────────────────────────
print("\n=== 4. 线性相关性与线性无关 ===")

# 线性无关
v1 = np.array([1, 0])
v2 = np.array([0, 1])
M_indep = np.column_stack([v1, v2])
rank_indep = np.linalg.matrix_rank(M_indep)
print(f"v1 = {v1}, v2 = {v2}")
print(f"rank([v1, v2]) = {rank_indep}  → 线性无关（不能互相表示）")

# 线性相关
v3 = np.array([2, 4])
v4 = np.array([1, 2])
M_dep = np.column_stack([v3, v4])
rank_dep = np.linalg.matrix_rank(M_dep)
print(f"\nv3 = {v3}, v4 = {v4}")
print(f"rank([v3, v4]) = {rank_dep}  → 线性相关（v3 = 2 × v4）")

print("\n判断方法:")
print("- 向量组线性无关 <=> rank(矩阵) = 向量个数")
print("- 向量组线性相关 <=> rank(矩阵) < 向量个数")
print("- 等价条件: 存在不全为零的系数使 c1*v1 + c2*v2 + ... = 0")

# ── 5. 齐次方程与非齐次方程 ────────────────────────────────────
print("\n=== 5. 齐次方程与非齐次方程 ===")

# 齐次方程 Ax = 0
A_homo = np.array([[1, 2, 3], [2, 4, 6]])
b_zero = np.array([0, 0])
rank_homo = np.linalg.matrix_rank(A_homo)
n_vars = A_homo.shape[1]
print(f"齐次方程 Ax = 0:")
print(f"A:\n{A_homo}")
print(f"rank(A) = {rank_homo}, 未知数个数 n = {n_vars}")
print(f"自由变量个数 = n - rank = {n_vars - rank_homo}")
print("→ 有非零解（无穷多解），解空间维数 = n - rank")

# 非齐次方程 Ax = b
A_nonhomo = np.array([[1, 2], [3, 5]])
b_nonzero = np.array([5, 12])
x_particular = np.linalg.solve(A_nonhomo, b_nonzero)
print(f"\n非齐次方程 Ax = b:")
print(f"A:\n{A_nonhomo}, b: {b_nonzero}")
print(f"特解 x_p = {x_particular}")
print("通解 = 特解 + 齐次方程的通解")

# ── 6. 最小二乘法 ──────────────────────────────────────────────
print("\n=== 6. 最小二乘法（超定方程）===")

# 超定方程组：方程数 > 未知数数，通常无精确解
A_over = np.array([[1, 1], [1, 2], [1, 3]])
b_over = np.array([2, 3, 5])
print(f"超定方程组 A (3×2):\n{A_over}")
print(f"b: {b_over}")
print("方程数(3) > 未知数(2)，一般无精确解")

# 最小二乘解：找 x 使 ||Ax - b||² 最小
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"\n最小二乘解 x = {x_lstsq.round(4)}")
print(f"残差平方和 = {residuals[0]:.4f}")
print(f"验证: A @ x = {(A_over @ x_lstsq).round(4)}  (近似 b = {b_over})")

# 正规方程验证: x = (AᵀA)⁻¹Aᵀb
x_normal_eq = np.linalg.inv(A_over.T @ A_over) @ A_over.T @ b_over
print(f"正规方程解 x = {x_normal_eq.round(4)}  (与lstsq一致)")

# ── 7. AI应用：线性回归与过拟合/欠拟合 ─────────────────────────
print("\n=== 7. AI应用：线性回归的最小二乘解 ===")

# 生成模拟数据
np.random.seed(42)
X_data = np.linspace(0, 10, 20)
y_true = 2 * X_data + 1  # 真实关系: y = 2x + 1
y_noise = y_true + np.random.randn(20) * 2  # 加噪声

# 构造设计矩阵（增加偏置项）
X_design = np.column_stack([np.ones(len(X_data)), X_data])

# 最小二乘拟合
theta, _, _, _ = np.linalg.lstsq(X_design, y_noise, rcond=None)
print(f"真实参数: [截距=1, 斜率=2]")
print(f"拟合参数: [截距={theta[0]:.3f}, 斜率={theta[1]:.3f}]")
print("这就是线性回归! 本质就是解最小二乘问题 min||X*theta - y||^2")

# 过拟合与欠拟合的线性代数视角
print("\n--- 过拟合与欠拟合的线性代数视角 ---")

# 欠拟合：特征太少（模型容量不足）
X_under = np.ones((20, 1))  # 只有偏置项，没有x
theta_under, _, _, _ = np.linalg.lstsq(X_under, y_noise, rcond=None)
y_pred_under = X_under @ theta_under
mse_under = np.mean((y_noise - y_pred_under) ** 2)
print(f"欠拟合（只用常数项）: MSE = {mse_under:.2f}")
print(f"  设计矩阵 rank = {np.linalg.matrix_rank(X_under)}, 特征数 < 真实复杂度")

# 合适拟合
y_pred_fit = X_design @ theta
mse_fit = np.mean((y_noise - y_pred_fit) ** 2)
print(f"合适拟合（一次多项式）: MSE = {mse_fit:.2f}")
print(f"  设计矩阵 rank = {np.linalg.matrix_rank(X_design)}, 匹配真实复杂度")

# 过拟合：特征太多（高阶多项式）
degree = 10
X_overfit = np.column_stack([X_data**d for d in range(degree + 1)])
theta_over, _, _, _ = np.linalg.lstsq(X_overfit, y_noise, rcond=None)
y_pred_over = X_overfit @ theta_over
mse_over_train = np.mean((y_noise - y_pred_over) ** 2)
print(f"过拟合（10次多项式）: 训练MSE = {mse_over_train:.4f}")
print(f"  设计矩阵 rank = {np.linalg.matrix_rank(X_overfit)}, 特征数 >> 真实复杂度")
print(f"  参数绝对值之和 = {np.sum(np.abs(theta_over)):.1f}  (参数爆炸!)")

print("\n线性代数视角总结:")
print("- 欠拟合: rank(X) 太小，列空间无法覆盖 y 的变化模式")
print("- 过拟合: rank(X) 太大，列空间过大，拟合了噪声")
print("- 正则化(Ridge/Lasso): 约束解的大小，相当于限制列空间的有效维度")
