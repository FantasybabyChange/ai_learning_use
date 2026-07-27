import numpy as np

print("=== 练习1：向量运算 ===")

# 练习1.1：创建两个向量并计算它们的点积
# TODO: 创建向量 v1 = [1, 2, 3] 和 v2 = [4, 5, 6]，然后计算它们的点积
v1 = None  # TODO: 填写代码
v2 = None  # TODO: 填写代码
dot_product = None  # TODO: 填写代码

print(f"练习1.1:")
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")
print(f"  点积 = {dot_product}")
print(f"  正确答案: 32")
print()

# 练习1.2：计算向量的范数
# TODO: 计算向量 v = [3, 4] 的L2范数
v = np.array([3, 4])
norm_v = None  # TODO: 填写代码

print(f"练习1.2:")
print(f"  v = {v}")
print(f"  L2范数 = {norm_v}")
print(f"  正确答案: 5.0")
print()

# 练习1.3：向量归一化
# TODO: 将向量 v = [3, 4] 归一化为单位向量
v = np.array([3, 4])
v_normalized = None  # TODO: 填写代码

print(f"练习1.3:")
print(f"  原始向量: {v}")
print(f"  归一化后: {v_normalized}")
print(f"  正确答案: [0.6, 0.8]")
print()

# 练习1.4：向量投影
# TODO: 计算向量 a 在向量 b 上的投影
a = np.array([3, 4])
b = np.array([1, 0])
projection = None  # TODO: 填写代码

print(f"练习1.4:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a在b上的投影 = {projection}")
print(f"  正确答案: [3, 0]")
print()

# 练习1.5：向量叉积
# TODO: 计算向量 a 和 b 的叉积（在3D空间中）
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross_product = None  # TODO: 填写代码

print(f"练习1.5:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  叉积 = {cross_product}")
print(f"  正确答案: [0, 0, 1]")
print()

# 练习1.6：判断向量是否正交
# TODO: 判断向量 a 和 b 是否正交（点积为0）
a = np.array([1, 0])
b = np.array([0, 1])
is_orthogonal = None  # TODO: 填写代码

print(f"练习1.6:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  是否正交: {is_orthogonal}")
print(f"  正确答案: True")
print()

print("=== 完成练习后，运行此脚本检查答案 ===")
print("如果答案不正确，请修改代码直到正确为止。")