import numpy as np

print("=== 练习1答案：向量运算 ===")

# 练习1.1：创建两个向量并计算它们的点积
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)

print(f"练习1.1:")
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")
print(f"  点积 = {dot_product}")
print()

# 练习1.2：计算向量的范数
v = np.array([3, 4])
norm_v = np.linalg.norm(v)

print(f"练习1.2:")
print(f"  v = {v}")
print(f"  L2范数 = {norm_v}")
print()

# 练习1.3：向量归一化
v = np.array([3, 4])
v_normalized = v / np.linalg.norm(v)

print(f"练习1.3:")
print(f"  原始向量: {v}")
print(f"  归一化后: {v_normalized}")
print()

# 练习1.4：向量投影
a = np.array([3, 4])
b = np.array([1, 0])
projection = (np.dot(a, b) / np.dot(b, b)) * b

print(f"练习1.4:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a在b上的投影 = {projection}")
print()

# 练习1.5：向量叉积
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross_product = np.cross(a, b)

print(f"练习1.5:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  叉积 = {cross_product}")
print()

# 练习1.6：判断向量是否正交
a = np.array([1, 0])
b = np.array([0, 1])
is_orthogonal = np.isclose(np.dot(a, b), 0)

print(f"练习1.6:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  是否正交: {is_orthogonal}")
print()

print("=== 所有练习完成 ===")