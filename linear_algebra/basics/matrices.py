import numpy as np

# ============================================================
# 矩阵基础示例代码
# 对应 basics/README.md 中的矩阵章节
# ============================================================

# ── 1. 矩阵的表示与创建 ──────────────────────────────────────
print("=== 1. 矩阵的表示与创建 ===")

A = np.array([[1, 2, 3], [4, 5, 6]])
print(f"手动创建 2x3 矩阵 A:\n{A}\nshape: {A.shape}")

Z = np.zeros((2, 3))
O = np.ones((2, 3))
I = np.eye(3)
D = np.diag([1, 2, 3])
R = np.random.rand(2, 3)

print(f"\n零矩阵 Z (2x3):\n{Z}")
print(f"全1矩阵 O (2x3):\n{O}")
print(f"单位矩阵 I (3x3):\n{I}")
print(f"对角矩阵 D:\n{D}")
print(f"随机矩阵 R (2x3):\n{R.round(3)}")
print("\n几何直觉：矩阵可以看作一组列向量的排列，也可以看作一个线性变换")
print("AI应用：神经网络的每一层权重就是一个矩阵，输入向量乘以权重矩阵得到输出")

# ── 2. 矩阵加法与数乘 ────────────────────────────────────────
print("\n=== 2. 矩阵加法与数乘 ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

add_result = A + B
scalar_mul = 3 * A

print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A + B:\n{add_result}")
print(f"3 × A:\n{scalar_mul}")
print("\n规则：加法要求两个矩阵形状相同，对应元素相加")
print("数乘将标量乘到每个元素上，相当于均匀缩放整个变换")

# ── 3. 矩阵乘法 ──────────────────────────────────────────────
print("\n=== 3. 矩阵乘法 ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

result_at = A @ B
result_matmul = np.matmul(A, B)
result_dot = np.dot(A, B)

print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A @ B:\n{result_at}")
print(f"np.matmul(A, B):\n{result_matmul}")
print(f"np.dot(A, B):\n{result_dot}")
print("\n三种写法对二维矩阵结果相同，推荐使用 @ 运算符")
print("矩阵乘法不满足交换律：A@B != B@A")

AB = A @ B
BA = B @ A
print(f"\nA @ B:\n{AB}")
print(f"B @ A:\n{BA}")
print(f"A@B == B@A ? {np.allclose(AB, BA)}")

print("\n维度规则：(m×k) @ (k×n) = (m×n)")
C = np.random.rand(2, 3)
D_mat = np.random.rand(3, 4)
E = C @ D_mat
print(f"C shape: {C.shape}, D shape: {D_mat.shape} → C@D shape: {E.shape}")

print("\n几何直觉：矩阵乘法 = 连续施加两个线性变换")
print("AI应用：神经网络前向传播 y = W2 * sigma(W1 * x)，就是多次矩阵乘法嵌套非线性激活")

# ── 4. 转置及其性质 ──────────────────────────────────────────
print("\n=== 4. 转置及其性质 ===")

A = np.array([[1, 2, 3], [4, 5, 6]])
AT = A.T

print(f"A (2x3):\n{A}")
print(f"A.T (3x2):\n{AT}")

print("\n转置的性质：")
print(f"(A.T).T == A ? {np.allclose(AT.T, A)}")

B = np.array([[7, 8], [9, 10], [11, 12]])
AB_T = (A @ B).T
BT_AT = B.T @ A.T
print(f"(A@B).T == B.T@A.T ? {np.allclose(AB_T, BT_AT)}  （反转顺序）")

print("\n几何直觉：转置就是行列互换，相当于把矩阵沿着主对角线翻转")
print("AI应用：在Attention中，K需要转置才能与Q做点积：scores = Q @ K.T")

# ── 5. 逆矩阵 ────────────────────────────────────────────────
print("\n=== 5. 逆矩阵 ===")

A = np.array([[1, 2], [3, 4]], dtype=float)
A_inv = np.linalg.inv(A)

print(f"A:\n{A}")
print(f"A⁻¹:\n{A_inv.round(4)}")
print(f"A @ A⁻¹ ≈ I ?\n{(A @ A_inv).round(4)}")

print("\n可逆条件：方阵且行列式不为零（非奇异矩阵）")
singular = np.array([[1, 2], [2, 4]], dtype=float)
det_singular = np.linalg.det(singular)
print(f"\n奇异矩阵示例:\n{singular}")
print(f"det = {det_singular:.6f}  （接近0，不可逆）")

try:
    np.linalg.inv(singular)
except np.linalg.LinAlgError as e:
    print(f"求逆失败: {e}")

print("\n几何直觉：逆矩阵是原变换的'撤销操作'，A把空间变形，A⁻¹把它变回来")
print("AI应用：求解线性方程组 Ax=b → x=A⁻¹b；实际中用 np.linalg.solve 更高效稳定")

# ── 6. 行列式 ────────────────────────────────────────────────
print("\n=== 6. 行列式 ===")

A = np.array([[1, 2], [3, 4]], dtype=float)
det_A = np.linalg.det(A)

print(f"A:\n{A}")
print(f"det(A) = {det_A:.4f}")

print("\n2x2公式: det([[a,b],[c,d]]) = ad - bc")
manual_det = 1 * 4 - 2 * 3
print(f"手动计算: 1×4 - 2×3 = {manual_det}")

B = np.array([[2, 0], [0, 3]], dtype=float)
print(f"\n对角矩阵 B:\n{B}")
print(f"det(B) = {np.linalg.det(B):.4f}  （等于对角元素之积: 2×3=6）")

print("\n几何直觉：|det(A)| = 变换后面积(体积)的缩放倍数")
print("  det > 0: 保持方向   det < 0: 翻转方向   det = 0: 降维（压扁）")
print("AI应用：Normalizing Flow 中用行列式衡量概率密度的变换比例")

# ── 7. 矩阵的迹 ──────────────────────────────────────────────
print("\n=== 7. 矩阵的迹 ===")

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tr_A = np.trace(A)

print(f"A:\n{A}")
print(f"tr(A) = {tr_A}  （对角线之和: 1+5+9=15）")

print("\n迹的性质：")
B = np.array([[9, 8], [7, 6]])
C = np.array([[1, 2], [3, 4]])
print(
    f"tr(B+C) = {np.trace(B + C)}, tr(B)+tr(C) = {np.trace(B) + np.trace(C)}  （线性）"
)
print(f"tr(BC) = {np.trace(B @ C)}, tr(CB) = {np.trace(C @ B)}  （循环不变性）")

print("\n几何直觉：迹是矩阵所有特征值之和，反映变换在各主轴上的总缩放")
print("AI应用：损失函数中的 Frobenius 范数 ||A||²_F = tr(AᵀA)")

# ── 8. 特殊矩阵 ──────────────────────────────────────────────
print("\n=== 8. 特殊矩阵 ===")

print("--- 单位矩阵 ---")
I = np.eye(3)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
print(f"I:\n{I}")
print(f"A @ I == A ? {np.allclose(A @ I, A)}  （单位矩阵是乘法的'1'）")

print("\n--- 对角矩阵 ---")
D = np.diag([2, 3, 5])
print(f"D:\n{D}")
print(f"D @ A 相当于对A的每一列分别乘以 2, 3, 5")
scaled = D @ A
print(f"D @ A:\n{scaled}")

print("\n--- 对称矩阵 ---")
S = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=float)
print(f"S:\n{S}")
print(f"S == S.T ? {np.allclose(S, S.T)}  （对称矩阵等于自身转置）")
print("协方差矩阵、Gram矩阵都是对称矩阵")

print("\n--- 正交矩阵 ---")
theta = np.pi / 4
Q_ortho = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
print(f"旋转矩阵 Q (θ=45°):\n{Q_ortho.round(4)}")
print(f"Q.T @ Q ≈ I ?\n{(Q_ortho.T @ Q_ortho).round(4)}")
print(f"det(Q) = {np.linalg.det(Q_ortho):.4f}  （正交矩阵行列式为 ±1）")
print("正交矩阵保持向量长度和夹角不变，只做旋转/反射")

print("\nAI应用：BatchNorm后的权重矩阵接近正交时，梯度传播更稳定")

# ── 9. 矩阵范数 ──────────────────────────────────────────────
print("\n=== 9. 矩阵范数 ===")

A = np.array([[1, 2], [3, 4]], dtype=float)

fro_norm = np.linalg.norm(A, "fro")
l1_norm = np.linalg.norm(A, 1)
linf_norm = np.linalg.norm(A, np.inf)
spec_norm = np.linalg.norm(A, 2)

print(f"A:\n{A}")
print(f"Frobenius 范数 ||A||_F = {fro_norm:.4f}  （所有元素平方和再开方）")
print(f"L1 范数 ||A||_1 = {l1_norm:.4f}  （最大列绝对值和）")
print(f"L∞ 范数 ||A||_inf = {linf_norm:.4f}  （最大行绝对值和）")
print(f"谱范数 ||A||_2 = {spec_norm:.4f}  （最大奇异值）")

print("\nFrobenius 范数与迹的关系:")
print(f"||A||_F = sqrt(tr(A.T@A)) = {np.sqrt(np.trace(A.T @ A)):.4f}")

print("\nAI应用：权重正则化常用 Frobenius 范数限制参数大小，防止过拟合")
print("谱范数用于谱归一化(Spectral Normalization)，稳定GAN训练")

# ── 10. AI 应用综合示例 ──────────────────────────────────────
print("\n=== 10. AI 应用综合示例 ===")

print("\n--- 10.1 神经网络权重矩阵 ---")
np.random.seed(42)
input_dim = 4
hidden_dim = 3
output_dim = 2

W1 = np.random.randn(input_dim, hidden_dim) * 0.5
W2 = np.random.randn(hidden_dim, output_dim) * 0.5
x = np.array([1.0, 0.5, -0.3, 0.8])

h = np.tanh(x @ W1)
y = h @ W2

print(f"输入 x shape: {x.shape}")
print(f"W1 shape: {W1.shape}  (输入→隐藏)")
print(f"W2 shape: {W2.shape}  (隐藏→输出)")
print(f"隐藏层 h = tanh(x@W1): {h.round(4)}")
print(f"输出 y = h@W2: {y.round(4)}")
print("矩阵的每一行对应一个输入特征对所有隐藏单元的权重")

print("\n--- 10.2 协方差矩阵 ---")
np.random.seed(0)
data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=1000)
cov_matrix = np.cov(data, rowvar=False)
print(f"数据 shape: {data.shape}")
print(f"样本协方差矩阵:\n{cov_matrix.round(4)}")
print(f"对称性: {np.allclose(cov_matrix, cov_matrix.T)}")
print(f"对角线(方差): [{cov_matrix[0, 0]:.4f}, {cov_matrix[1, 1]:.4f}]")
print(f"非对角线(协方差): {cov_matrix[0, 1]:.4f}")
print("协方差矩阵描述各维度之间的相关性，PCA就是对它做特征分解")

print("\n--- 10.3 Attention 中的 QKV 矩阵 ---")
seq_len = 4
d_model = 8
d_k = d_model

np.random.seed(1)
X = np.random.randn(seq_len, d_model)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_k) * 0.5

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T / np.sqrt(d_k)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


attn_weights = softmax(scores, axis=-1)
output = attn_weights @ V

print(f"输入 X shape: {X.shape}  (seq_len × d_model)")
print(f"W_Q/W_K/W_V shape: {W_Q.shape}  (d_model × d_k)")
print(f"Q/K/V shape: {Q.shape}  (seq_len × d_k)")
print(f"Attention scores shape: {scores.shape}  (seq_len × seq_len)")
print(f"Attention weights (每行之和≈1):\n{attn_weights.round(3)}")
print(f"Output shape: {output.shape}  (seq_len × d_k)")
print("\nQKV 都是矩阵而非向量：批量处理整个序列的所有token")
print("scores[i][j] = token_i 对 token_j 的关注度")
print("最终输出是每个token对所有value的加权平均")

print("\n=== 矩阵基础学习完成 ===")
