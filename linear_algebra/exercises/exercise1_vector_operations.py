import numpy as np

print("=== 练习1：向量运算 ===\n")

# ── 练习 1.1：点积 ────────────────────────────────────────────
# TODO: 创建向量 v1=[1,2,3] 和 v2=[4,5,6]，计算它们的点积
v1 = None  # TODO
v2 = None  # TODO
dot_product = None  # TODO

print("练习1.1：点积")
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")
print(f"  点积 = {dot_product}")
print(f"  正确答案: 32\n")

# ── 练习 1.2：L2 范数 ─────────────────────────────────────────
# TODO: 计算向量 v=[3,4] 的 L2 范数
v = np.array([3, 4])
norm_l2 = None  # TODO

print("练习1.2：L2 范数")
print(f"  v = {v}")
print(f"  L2 范数 = {norm_l2}")
print(f"  正确答案: 5.0\n")

# ── 练习 1.3：L1 范数 ─────────────────────────────────────────
# TODO: 计算向量 v=[3,-4,0] 的 L1 范数（各分量绝对值之和）
v = np.array([3, -4, 0])
norm_l1 = None  # TODO

print("练习1.3：L1 范数")
print(f"  v = {v}")
print(f"  L1 范数 = {norm_l1}")
print(f"  正确答案: 7.0\n")

# ── 练习 1.4：无穷范数 ───────────────────────────────────────
# TODO: 计算向量 v=[3,-4,0] 的无穷范数（最大绝对值）
v = np.array([3, -4, 0])
norm_inf = None  # TODO

print("练习1.4：无穷范数")
print(f"  v = {v}")
print(f"  无穷范数 = {norm_inf}")
print(f"  正确答案: 4.0\n")

# ── 练习 1.5：标量乘法 ───────────────────────────────────────
# TODO: 将向量 v=[1,2,3] 分别乘以 3、-1，观察结果
v = np.array([1, 2, 3])
v_times3 = None   # TODO：乘以 3
v_reverse = None  # TODO：乘以 -1（反向）

print("练习1.5：标量乘法")
print(f"  v = {v}")
print(f"  3 × v = {v_times3}")
print(f"  正确答案: [3 6 9]")
print(f"  -1 × v = {v_reverse}")
print(f"  正确答案: [-1 -2 -3]\n")

# ── 练习 1.6：归一化 ─────────────────────────────────────────
# TODO: 将向量 v=[3,4] 归一化为单位向量
v = np.array([3.0, 4.0])
v_normalized = None  # TODO

print("练习1.6：向量归一化")
print(f"  原始向量: {v}")
print(f"  归一化后: {v_normalized}")
print(f"  正确答案: [0.6 0.8]\n")

# ── 练习 1.7：余弦相似度 ─────────────────────────────────────
# TODO: 计算以下三对向量的余弦相似度
# 公式：cos_sim(a, b) = a·b / (||a|| × ||b||)
a = np.array([1.0, 0.0])
b_same    = np.array([2.0, 0.0])   # 同方向
b_perp    = np.array([0.0, 1.0])   # 垂直
b_reverse = np.array([-1.0, 0.0])  # 反方向

sim_same    = None  # TODO
sim_perp    = None  # TODO
sim_reverse = None  # TODO

print("练习1.7：余弦相似度")
print(f"  同方向: cos_sim = {sim_same}   正确答案: 1.0")
print(f"  垂  直: cos_sim = {sim_perp}   正确答案: 0.0")
print(f"  反方向: cos_sim = {sim_reverse}  正确答案: -1.0\n")

# ── 练习 1.8：向量投影 ───────────────────────────────────────
# TODO: 计算向量 a=[3,4] 在向量 b=[1,0] 上的向量投影
a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])
projection = None  # TODO

print("练习1.8：向量投影")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a 在 b 上的投影 = {projection}")
print(f"  正确答案: [3. 0.]\n")

# ── 练习 1.9：叉积 ───────────────────────────────────────────
# TODO: 计算向量 a=[1,0,0] 和 b=[0,1,0] 的叉积
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross_product = None  # TODO

print("练习1.9：叉积")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  叉积 = {cross_product}")
print(f"  正确答案: [0 0 1]\n")

# ── 练习 1.10：判断向量是否正交 ──────────────────────────────
# TODO: 分别判断以下两对向量是否正交（点积为0即正交）
a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1, 1])
is_ab_orthogonal = None  # TODO: a 和 b 是否正交
is_ac_orthogonal = None  # TODO: a 和 c 是否正交

print("练习1.10：向量正交")
print(f"  a={a}, b={b}  正交: {is_ab_orthogonal}  正确答案: True")
print(f"  a={a}, c={c}  正交: {is_ac_orthogonal}  正确答案: False\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
