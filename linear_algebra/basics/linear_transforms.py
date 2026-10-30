import numpy as np

# ============================================================
# 线性变换
# 对应 basics/README.md 中的线性变换章节
# ============================================================

# ── 1. 线性变换的定义 ──────────────────────────────────────────
print("=== 1. 线性变换的定义 ===")
# T: V → W 是线性变换 ⟺
#   T(u + v) = T(u) + T(v)    (保持加法)
#   T(c·u) = c·T(u)           (保持数乘)


def is_linear(T, n=3, dim_in=2, dim_out=2, tol=1e-10):
    """数值验证 T 是否为线性变换"""
    for _ in range(n):
        u = np.random.randn(dim_in)
        v = np.random.randn(dim_in)
        c = np.random.randn()
        add_ok = np.allclose(T(u + v), T(u) + T(v), atol=tol)
        scale_ok = np.allclose(T(c * u), c * T(u), atol=tol)
        if not (add_ok and scale_ok):
            return False
    return True


# 线性变换示例：矩阵乘法
A = np.array([[1, 2], [3, 4]], dtype=float)
T_linear = lambda x: A @ x
print(f"T(x) = Ax, A =\n{A}")
print(f"T 是线性变换? {is_linear(T_linear)}")

# 非线性变换示例：加常数偏移
T_nonlinear = lambda x: A @ x + np.array([1, 0])
print(f"\nT(x) = Ax + b (仿射变换)")
print(f"T 是线性变换? {is_linear(T_nonlinear)}")
print("→ 加偏移后不再满足 T(0)=0，不是线性变换（是仿射变换）")

# ── 2. 线性变换的矩阵表示 ─────────────────────────────────────
print("\n=== 2. 线性变换的矩阵表示 ===")
# 任何线性变换 T: R^n → R^m 都可以写成 T(x) = Ax
# A 的第 j 列 = T(e_j)，即标准基向量 e_j 在变换后的像

n, m = 3, 2


# 构造一个自定义线性变换
def my_transform(x):
    return np.array([x[0] + 2 * x[1] - x[2], 3 * x[0] - x[1]])


# 通过作用于标准基来提取矩阵
A_extracted = np.column_stack([my_transform(np.eye(n)[j]) for j in range(n)])
print(f"自定义变换 T 对应的矩阵 A:\n{A_extracted}")

# 验证
x = np.array([1, 2, 3], dtype=float)
print(f"T({x}) = {my_transform(x)}")
print(f"A @ {x} = {A_extracted @ x}")
print(f"两者相等? {np.allclose(my_transform(x), A_extracted @ x)}")

# ── 3. 常见变换 ───────────────────────────────────────────────
print("\n=== 3. 常见变换 ===")

# 3.1 旋转
theta = np.pi / 4  # 45°
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
v = np.array([1, 0], dtype=float)
print(f"--- 旋转 {np.degrees(theta):.0f}° ---")
print(f"旋转矩阵 R:\n{R.round(4)}")
print(f"R @ [1,0] = {(R @ v).round(4)}  (应指向 45° 方向)")
print("几何: 所有向量绕原点逆时针旋转 θ 角，长度不变")

# 3.2 缩放
S = np.diag([2.0, 0.5])
print(f"\n--- 缩放 ---")
print(f"缩放矩阵 S:\n{S}")
print(f"S @ [1,1] = {S @ np.array([1, 1])}  (x拉长2倍, y压缩一半)")
print("几何: 各轴独立缩放，面积变为 det(S)={:.1f} 倍".format(np.linalg.det(S)))

# 3.3 投影（投影到 x 轴）
P = np.array([[1, 0], [0, 0]], dtype=float)
v = np.array([3, 4], dtype=float)
print(f"\n--- 投影到 x 轴 ---")
print(f"投影矩阵 P:\n{P}")
print(f"P @ {v} = {P @ v}  (y 分量被抹掉)")
print("性质: P^2 = P (幂等矩阵)")
print(f"验证 P@P == P? {np.allclose(P @ P, P)}")

# 一般投影到子空间: P = A(A^TA)^{-1}A^T
a = np.array([[1], [1]], dtype=float)  # 投影到 y=x 线
P_general = a @ np.linalg.inv(a.T @ a) @ a.T
print(f"\n投影到 y=x 方向的矩阵:\n{P_general.round(4)}")
print(f"P @ [3,1] = {(P_general @ np.array([3, 1])).round(4)}  (投影到 y=x)")

# 3.4 反射（关于 x 轴的镜像）
F = np.array([[1, 0], [0, -1]], dtype=float)
print(f"\n--- 关于 x 轴反射 ---")
print(f"反射矩阵 F:\n{F}")
print(f"F @ [3,4] = {F @ np.array([3, 4])}  (y 取反)")
print("性质: F^2 = I (对合矩阵)")
print(f"验证 F@F == I? {np.allclose(F @ F, np.eye(2))}")

# ── 4. 变换的复合（矩阵乘法）─────────────────────────────────
print("\n=== 4. 变换的复合 ===")
# 先旋转再缩放 = 缩放矩阵 × 旋转矩阵（注意顺序！右乘先作用）
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S = np.diag([2.0, 1.0])

# 先旋转 R，再缩放 S → 复合变换 = S @ R
composite = S @ R
v = np.array([1, 0], dtype=float)
print(f"先旋转30°, 再x轴拉伸2倍")
print(f"复合矩阵 S@R:\n{composite.round(4)}")
print(f"(S@R) @ [1,0] = {(composite @ v).round(4)}")

step1 = R @ v
step2 = S @ step1
print(f"分步: R@[1,0]={step1.round(4)} → S@result={step2.round(4)}")
print(f"一致? {np.allclose(composite @ v, step2)}")
print("→ 矩阵乘法 = 变换复合；AB 表示先 B 后 A")

# ── 5. 可逆变换与逆矩阵 ───────────────────────────────────────
print("\n=== 5. 可逆变换与逆矩阵 ===")
A = np.array([[1, 2], [3, 5]], dtype=float)
det_A = np.linalg.det(A)
print(f"A =\n{A}")
print(f"det(A) = {det_A:.1f} {'≠ 0 → 可逆' if abs(det_A) > 1e-10 else '= 0 → 不可逆'}")

A_inv = np.linalg.inv(A)
print(f"A^(-1) =\n{A_inv.round(4)}")
print(f"A @ A^(-1) ≈ I? {np.allclose(A @ A_inv, np.eye(2))}")

# 不可逆的例子
B = np.array([[1, 2], [2, 4]], dtype=float)
print(f"\nB =\n{B}")
print(f"rank(B) = {np.linalg.matrix_rank(B)}, det(B) = {np.linalg.det(B):.1f}")
print("秩亏损 → 将二维压成一维 → 信息丢失 → 不可逆")

# AI直觉：可逆变换保留全部信息（如 Normalizing Flow），不可逆变换有损压缩

# ── 6. AI应用 ──────────────────────────────────────────────────
print("\n=== 6. AI应用 ===")

# 6.1 数据预处理中的变换
print("--- 6.1 数据预处理中的变换 ---")
X = np.array([[100, 200, 300], [110, 210, 290], [105, 205, 310]], dtype=float)

mean = X.mean(axis=0)
std = X.std(axis=0, ddof=0)
X_standardized = (X - mean) / std

print(f"原始数据:\n{X}")
print(f"标准化后:\n{X_standardized.round(4)}")
print("标准化 = 平移(减均值) + 缩放(除标准差)")
print("平移不是线性变换，但缩放部分是线性变换 diag(1/std)")

# 6.2 BatchNorm 的线性代数本质
print("\n--- 6.2 BatchNorm 的线性代数本质 ---")
batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
mu = batch.mean(axis=0)
var = batch.var(axis=0)
gamma = np.array([1.0, 1.0])
beta = np.array([0.0, 0.0])

bn_out = gamma * (batch - mu) / np.sqrt(var + 1e-8) + beta
print(f"输入 batch:\n{batch}")
print(f"BatchNorm 输出:\n{bn_out.round(4)}")
print("BatchNorm = 中心化 + 方差归一化 + 仿射变换(gamma, beta)")
print("当 gamma, beta 固定时，BatchNorm 是仿射变换（线性+平移）")
print("训练时 BN 使激活值分布在单位球附近，改善优化地形")

# 6.3 注意力机制作为线性变换
print("\n--- 6.3 Attention 作为线性变换 ---")
seq_len, d = 3, 4
Q = np.random.randn(seq_len, d)
K = np.random.randn(seq_len, d)
V = np.random.randn(seq_len, d)

scores = Q @ K.T / np.sqrt(d)
attn_weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
output = attn_weights @ V

print(f"序列长度={seq_len}, 维度={d}")
print(f"Attention weights (每行和为1):\n{attn_weights.round(3)}")
print(f"输出 shape: {output.shape}")
print("对于固定的 attention weights, output = attn_weights @ V")
print("这是一个以 attn_weights 为矩阵的线性变换: V → output")
print("不同 token 对有不同的权重矩阵 → 上下文相关的动态线性变换")
print("Multi-head = 多个并行的低维线性变换拼接后再做一次线性变换(W_O)")
