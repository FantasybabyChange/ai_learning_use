import numpy as np

print("=== 练习1 答案：向量运算 ===\n")

# ── 1.1：点积 ─────────────────────────────────────────────────
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)          # 1×4 + 2×5 + 3×6 = 32
print("1.1 点积")
print(f"  v1·v2 = {dot_product}\n")

# ── 1.2：L2 范数 ──────────────────────────────────────────────
v = np.array([3, 4])
norm_l2 = np.linalg.norm(v)           # √(9+16) = 5.0
print("1.2 L2 范数")
print(f"  ||[3,4]||_2 = {norm_l2}\n")

# ── 1.3：L1 范数 ──────────────────────────────────────────────
v = np.array([3, -4, 0])
norm_l1 = np.linalg.norm(v, ord=1)    # |3|+|-4|+|0| = 7.0
print("1.3 L1 范数")
print(f"  ||[3,-4,0]||_1 = {norm_l1}\n")

# ── 1.4：无穷范数 ─────────────────────────────────────────────
v = np.array([3, -4, 0])
norm_inf = np.linalg.norm(v, ord=np.inf)  # max(3,4,0) = 4.0
print("1.4 无穷范数")
print(f"  ||[3,-4,0]||_inf = {norm_inf}\n")

# ── 1.5：标量乘法 ─────────────────────────────────────────────
v = np.array([1, 2, 3])
v_times3  = 3 * v                     # [3, 6, 9]
v_reverse = -1 * v                    # [-1, -2, -3]
print("1.5 标量乘法")
print(f"  3×v = {v_times3}")
print(f"  -1×v = {v_reverse}\n")

# ── 1.6：归一化 ───────────────────────────────────────────────
v = np.array([3.0, 4.0])
v_normalized = v / np.linalg.norm(v)  # [0.6, 0.8]
print("1.6 归一化")
print(f"  v_unit = {v_normalized}  范数验证: {np.linalg.norm(v_normalized):.1f}\n")

# ── 1.7：余弦相似度 ───────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 0.0])
sim_same    = cosine_similarity(a, np.array([ 2.0, 0.0]))   # 1.0
sim_perp    = cosine_similarity(a, np.array([ 0.0, 1.0]))   # 0.0
sim_reverse = cosine_similarity(a, np.array([-1.0, 0.0]))   # -1.0
print("1.7 余弦相似度")
print(f"  同方向: {sim_same:.1f}")
print(f"  垂  直: {sim_perp:.1f}")
print(f"  反方向: {sim_reverse:.1f}\n")

# ── 1.8：向量投影 ─────────────────────────────────────────────
a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])
projection = (np.dot(a, b) / np.dot(b, b)) * b              # [3. 0.]
print("1.8 向量投影")
print(f"  proj = {projection}\n")

# ── 1.9：叉积 ─────────────────────────────────────────────────
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross_product = np.cross(a, b)        # [0, 0, 1]
print("1.9 叉积")
print(f"  a×b = {cross_product}\n")

# ── 1.10：正交判断 ────────────────────────────────────────────
a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1, 1])
is_ab_orthogonal = np.isclose(np.dot(a, b), 0)  # True
is_ac_orthogonal = np.isclose(np.dot(a, c), 0)  # False
print("1.10 向量正交")
print(f"  a⊥b: {is_ab_orthogonal}")
print(f"  a⊥c: {is_ac_orthogonal}\n")

print("=" * 50)
print("所有练习完成!")
