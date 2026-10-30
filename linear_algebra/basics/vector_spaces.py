import numpy as np

# ============================================================
# 向量空间与子空间
# 对应 basics/README.md 中的向量空间章节
# ============================================================

# ── 1. 向量空间的定义与性质 ────────────────────────────────────
print("=== 1. 向量空间的定义与性质 ===")
# 向量空间 V 对加法和数乘封闭：
#   若 u, v ∈ V，则 u + v ∈ V
#   若 u ∈ V, c 为标量，则 c·u ∈ V
# R^2 是最直观的向量空间（整个二维平面）
# R^n 是 n 维实向量空间

v1 = np.array([1.0, 2.0])
v2 = np.array([3.0, -1.0])
c = 2.5

print(f"v1 = {v1}, v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}       (仍在 R^2 中)")
print(f"{c} * v1 = {c * v1}   (仍在 R^2 中)")
print("→ R^2 对加法和数乘封闭，构成向量空间")

# ── 2. 子空间：列空间、零空间、行空间 ─────────────────────────
print("\n=== 2. 子空间 ===")
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

rank = np.linalg.matrix_rank(A)
print(f"A =\n{A}")
print(f"rank(A) = {rank}")

# 列空间 C(A)：A 的列向量的所有线性组合
# dim(C(A)) = rank(A)
print(f"\n列空间 C(A) 的维数 = rank(A) = {rank}")
print("列空间是 A 的列向量张成的子空间")

# 零空间 N(A)：满足 Ax = 0 的所有 x
# dim(N(A)) = n - rank(A)，其中 n 是列数
null_dim = A.shape[1] - rank
print(f"零空间 N(A) 的维数 = n - rank = {A.shape[1]} - {rank} = {null_dim}")

# 用 SVD 求零空间基向量
U, s, Vt = np.linalg.svd(A)
tol = 1e-10
null_basis = Vt[np.abs(s) < tol].T if np.any(np.abs(s) < tol) else Vt[rank:].T
if null_basis.size > 0:
    print(f"零空间基向量:\n{null_basis.round(4)}")
    # 验证 A @ null_basis ≈ 0
    residual = A @ null_basis
    print(f"验证 A @ null_basis ≈ 0: max|residual| = {np.max(np.abs(residual)):.2e}")
else:
    print("该矩阵零空间仅为零向量（满列秩时）")

# 行空间 C(A^T)：A 的行向量的所有线性组合
# dim(C(A^T)) = rank(A)
print(f"\n行空间 C(A^T) 的维数 = rank(A) = {rank}")

# ── 3. 线性组合与生成空间（Span）─────────────────────────────
print("\n=== 3. 线性组合与生成空间 ===")
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])

# v3 = 1*v1 + 1*v2，所以 v3 在 span(v1, v2) 中
combo = 1 * v1 + 1 * v2
print(f"v1 = {v1}, v2 = {v2}, v3 = {v3}")
print(f"1*v1 + 1*v2 = {combo} == v3? {np.allclose(combo, v3)}")
print("→ v3 属于 span(v1, v2)")

# span(v1, v2) 只生成 xy 平面，无法生成 z 方向
target = np.array([0, 0, 1])
M = np.column_stack([v1, v2])
try:
    coeffs, residuals, _, _ = np.linalg.lstsq(M, target, rcond=None)
    recon = M @ coeffs
    print(f"\n尝试用 span(v1,v2) 表示 {target}:")
    print(f"最佳逼近 = {recon}, 残差范数 = {np.linalg.norm(target - recon):.4f}")
    print("→ z 轴向量不在 span(v1, v2) 中")
except Exception as e:
    print(e)

# ── 4. 线性无关与线性相关 ─────────────────────────────────────
print("\n=== 4. 线性无关与线性相关 ===")


def check_independence(vectors, names=None):
    """判断一组向量是否线性无关"""
    M = np.column_stack(vectors)
    r = np.linalg.matrix_rank(M)
    n = len(vectors)
    independent = r == n
    label = "线性无关" if independent else "线性相关"
    print(f"  向量个数={n}, rank={r} → {label}")
    return independent


print("例1: 标准基")
check_independence([np.array([1, 0]), np.array([0, 1])], ["e1", "e2"])

print("例2: 含冗余向量")
check_independence([np.array([1, 0]), np.array([0, 1]), np.array([1, 1])])

print("例3: 三维空间中的三个无关向量")
check_independence([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])

print("例4: 共线向量")
check_independence([np.array([1, 2, 3]), np.array([2, 4, 6])])

# AI直觉：特征矩阵的列如果线性相关，说明存在冗余特征
print("\nAI直觉: 特征选择中，线性相关的特征列可以删除而不丢失信息")

# ── 5. 基与维数 ───────────────────────────────────────────────
print("\n=== 5. 基与维数 ===")
# 基 = 线性无关 + 张成整个空间
# 维数 = 基向量的个数 = rank

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)  # 注意改成满秩
r = np.linalg.matrix_rank(A)
print(f"A =\n{A}")
print(f"rank(A) = {r} → A 的列空间维数为 {r}")

# 用 QR 分解提取列空间的正交基
Q, R = np.linalg.qr(A)
print(f"\nQR 分解得到的正交基 Q (前 {r} 列):\n{Q[:, :r].round(4)}")
print("Q 的列构成 C(A) 的一组标准正交基")

# ── 6. 坐标变换 ───────────────────────────────────────────────
print("\n=== 6. 坐标变换 ===")
# 同一向量在不同基下的坐标不同
# 标准基 e1=[1,0], e2=[0,1]
# 新基 b1=[1,1], b2=[1,-1]

B = np.array([[1, 1], [1, -1]], dtype=float)  # 新基作为列

x_standard = np.array([3, 1], dtype=float)  # 标准基下的坐标
x_new = np.linalg.solve(B, x_standard)  # 新基下的坐标

print(f"标准基下坐标: {x_standard}")
print(f"新基 B =\n{B}")
print(f"新基下坐标: {x_new.round(4)}")
print(f"验证: B @ x_new = {(B @ x_new).round(4)} == {x_standard}")

# AI直觉：PCA 就是把数据变换到主成分基下，使坐标（特征值）按重要性排列
print("\nAI直觉: PCA 本质上是找到一组新基，让数据在新基下的坐标方差最大")

# ── 7. AI应用：特征空间、降维、嵌入空间 ──────────────────────
print("\n=== 7. AI应用 ===")

# 7.1 特征空间
print("--- 7.1 特征空间 ---")
# 假设 4 个样本，每个有 3 个特征
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 5]], dtype=float)
print(f"数据矩阵 X shape: {X.shape}")
print(f"rank(X) = {np.linalg.matrix_rank(X)}")
print("每一行是一个样本在 R^3 特征空间中的点")
print("列空间维数 = 数据实际占据的子空间维数")

# 7.2 降维的本质
print("\n--- 7.2 降维的本质 (PCA) ---")
X_centered = X - X.mean(axis=0)
cov = X_centered.T @ X_centered / (X.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

k = 2
W = eigenvectors[:, :k]
X_reduced = X_centered @ W

print(f"原始维度: {X.shape[1]}, 降到: {k}")
print(f"前 {k} 个主成分解释方差比: {(eigenvalues[:k] / eigenvalues.sum()).round(4)}")
print(f"降维后数据 shape: {X_reduced.shape}")
print("降维 = 将数据投影到 rank-k 子空间上")

# 7.3 嵌入空间
print("\n--- 7.3 嵌入空间 ---")
embeddings = {
    "king": np.array([0.8, 0.3, -0.1, 0.5]),
    "queen": np.array([0.7, 0.4, 0.2, 0.5]),
    "man": np.array([0.6, 0.1, -0.3, 0.2]),
    "woman": np.array([0.5, 0.2, 0.0, 0.2]),
}

result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
sims = {}
for name, vec in embeddings.items():
    cos = np.dot(result, vec) / (np.linalg.norm(result) * np.linalg.norm(vec))
    sims[name] = cos

best = max(sims, key=sims.get)
print(f"'king' - 'man' + 'woman' = {result.round(3)}")
print(f"与各词余弦相似度: {', '.join(f'{k}={v:.3f}' for k, v in sims.items())}")
print(f"最接近的词: '{best}' → 嵌入空间中向量运算保留语义关系")
