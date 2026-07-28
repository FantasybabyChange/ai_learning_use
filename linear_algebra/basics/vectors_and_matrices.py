import numpy as np

# ============================================================
# 向量基础示例代码
# 对应 basics/README.md 中的每一个概念
# ============================================================

# ── 1. 向量的表示：行向量 vs 列向量 ──────────────────────────
print("=== 1. 向量的表示 ===")
v_row = np.array([1, 2, 3])             # 行向量，shape: (3,)
v_col = np.array([[1], [2], [3]])       # 列向量，shape: (3, 1)
print(f"行向量 v_row: {v_row}  shape: {v_row.shape}")
print(f"列向量 v_col:\n{v_col}  shape: {v_col.shape}")

# ── 2. 向量加法 ──────────────────────────────────────────────
print("\n=== 2. 向量加法 ===")
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v_add = v1 + v2
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v_add}")   # [5, 7, 9]

# ── 3. 标量乘法（数乘）───────────────────────────────────────
print("\n=== 3. 标量乘法 ===")
v = np.array([1, 2, 3])
print(f"原始向量 v = {v}")
print(f"2 × v = {2 * v}")        # 拉长：[2, 4, 6]
print(f"0.5 × v = {0.5 * v}")    # 缩短：[0.5, 1.0, 1.5]
print(f"-1 × v = {-1 * v}")      # 反向：[-1, -2, -3]

# ── 4. 点积（内积）───────────────────────────────────────────
print("\n=== 4. 点积（内积）===")
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot1 = np.dot(v1, v2)            # 方式一
dot2 = v1 @ v2                   # 方式二（推荐）
dot3 = (v1 * v2).sum()           # 方式三（逐元素乘再求和）
print(f"v1 · v2 = {dot1}  （三种写法结果相同：{dot2}, {dot3}）")
# 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

# 点积与夹角
a = np.array([1, 0])
b = np.array([0, 1])
theta = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"\na = {a}, b = {b}")
print(f"a · b = {np.dot(a, b)}  夹角 θ = {np.degrees(theta):.1f}°（垂直时点积=0）")

# ── 5. 叉积（外积）───────────────────────────────────────────
print("\n=== 5. 叉积（外积）===")
a = np.array([1, 0, 0])   # x 轴方向
b = np.array([0, 1, 0])   # y 轴方向
cross = np.cross(a, b)     # 结果垂直于 a 和 b
print(f"a = {a}  (x轴)")
print(f"b = {b}  (y轴)")
print(f"a × b = {cross}  (z轴，符合右手定则)")

# ── 6. 向量范数（长度）───────────────────────────────────────
print("\n=== 6. 向量范数 ===")
v = np.array([3, -4, 0])
l1  = np.linalg.norm(v, ord=1)          # L1：各分量绝对值之和
l2  = np.linalg.norm(v)                 # L2：欧氏长度（默认）
inf = np.linalg.norm(v, ord=np.inf)     # 无穷范数：最大绝对值
print(f"v = {v}")
print(f"L1  范数 ||v||_1 = {l1}")        # |3| + |-4| + |0| = 7
print(f"L2  范数 ||v||_2 = {l2}")        # sqrt(9+16+0) = 5.0
print(f"Inf 范数 ||v||_inf = {inf}")     # max(3, 4, 0) = 4.0

# ── 7. 单位向量与归一化 ───────────────────────────────────────
print("\n=== 7. 单位向量与归一化 ===")
v = np.array([3.0, 4.0])
v_unit = v / np.linalg.norm(v)
print(f"原始向量 v = {v}")
print(f"归一化后 v_unit = {v_unit}")                              # [0.6, 0.8]
print(f"归一化后范数 = {np.linalg.norm(v_unit):.1f}（应为 1.0）")

# ── 8. 余弦相似度 ─────────────────────────────────────────────
print("\n=== 8. 余弦相似度 ===")
def cosine_similarity(a, b):
    """衡量两个向量方向的相似程度，结果范围 [-1, 1]"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 0.0])   # 向右
b = np.array([1.0, 0.0])   # 同方向
c = np.array([0.0, 1.0])   # 垂直
d = np.array([-1.0, 0.0])  # 反方向

print(f"同方向 cos_sim = {cosine_similarity(a, b):.2f}  （θ=0°，完全相似）")
print(f"垂  直 cos_sim = {cosine_similarity(a, c):.2f}  （θ=90°，无关）")
print(f"反方向 cos_sim = {cosine_similarity(a, d):.2f}  （θ=180°，完全相反）")

# ── 9. 向量投影 ───────────────────────────────────────────────
print("\n=== 9. 向量投影 ===")
a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])   # x 轴方向
scalar_proj = np.dot(a, b) / np.linalg.norm(b)          # 标量投影（长度）
vector_proj = (np.dot(a, b) / np.dot(b, b)) * b         # 向量投影
print(f"a = {a},  b = {b}")
print(f"a 在 b 方向上的标量投影 = {scalar_proj}")        # 3.0
print(f"a 在 b 方向上的向量投影 = {vector_proj}")        # [3. 0.]

# ── 10. 向量正交 ──────────────────────────────────────────────
print("\n=== 10. 向量正交 ===")
a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1, 1])
print(f"a = {a},  b = {b}  →  a·b = {np.dot(a,b)}  正交: {np.isclose(np.dot(a,b), 0)}")
print(f"a = {a},  c = {c}  →  a·c = {np.dot(a,c)}  正交: {np.isclose(np.dot(a,c), 0)}")

# ── 11. AI 应用：词向量相似度 ─────────────────────────────────
print("\n=== 11. AI 应用：词向量相似度 ===")
# 模拟简化的词向量（实际中是几百到几千维）
word_vectors = {
    "猫":  np.array([0.9, 0.8, 0.1, 0.0]),
    "狗":  np.array([0.8, 0.9, 0.1, 0.1]),
    "汽车": np.array([0.0, 0.1, 0.9, 0.8]),
}
print("词语之间的余弦相似度：")
for w1 in word_vectors:
    for w2 in word_vectors:
        if w1 < w2:
            sim = cosine_similarity(word_vectors[w1], word_vectors[w2])
            print(f"  cos_sim('{w1}', '{w2}') = {sim:.3f}")
# 猫 和 狗 相似度高，猫/狗 和 汽车 相似度低

# ── 12. AI 应用：Attention 点积计算 ──────────────────────────
print("\n=== 12. AI 应用：Attention 点积 ===")
# 简化版 Self-Attention 的相似度分数计算
# Q × Kᵀ / √d
Q = np.array([[1.0, 0.0],   # token1 的 Query 向量
              [0.0, 1.0]])   # token2 的 Query 向量
K = np.array([[1.0, 0.1],   # token1 的 Key 向量
              [0.1, 1.0]])   # token2 的 Key 向量
d = Q.shape[1]              # 向量维度

scores = Q @ K.T / np.sqrt(d)   # Attention 分数矩阵
print(f"Q:\n{Q}")
print(f"K:\n{K}")
print(f"Attention 分数 (Q x K.T / sqrt(d)):\n{scores.round(3)}")
print("→ scores[i][j] 越大，token_i 对 token_j 越关注")

# ============================================================
# 矩阵基础（后续学习）
# ============================================================
print("\n\n=== 矩阵基础 ===")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"A + B:\n{A + B}")
print(f"A × B:\n{A @ B}")
print(f"A.T:\n{A.T}")
print(f"inv(A):\n{np.linalg.inv(A).round(3)}")
print(f"det(A) = {np.linalg.det(A):.1f}")

print("\n=== 特殊矩阵 ===")
print(f"单位矩阵 I:\n{np.eye(3)}")
print(f"对角矩阵 D:\n{np.diag([1, 2, 3])}")

print("\n=== 张量基础 ===")
tensor_3d = np.random.rand(3, 4, 5)
print(f"3D张量（模拟彩色图像）shape: {tensor_3d.shape}  元素总数: {tensor_3d.size}")
