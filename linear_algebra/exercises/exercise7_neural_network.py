import numpy as np

print("=== 练习7：神经网络中的线性代数 ===\n")

np.random.seed(42)

# ── 练习 7.1：线性层前向传播 ──────────────────────────────────
# y = x @ W + b
x = np.random.randn(3, 4)
W = np.random.randn(4, 5)
b = np.random.randn(5)
y = None  # TODO

print("练习7.1：线性层")
print(f"  输入 shape: {x.shape}")
print(f"  权重 shape: {W.shape}")
print(f"  输出 shape: {y.shape if y is not None else None}")
print(f"  正确答案: (3, 5)\n")

# ── 练习 7.2：ReLU 激活 ──────────────────────────────────────
z = np.array([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=float)
a = None  # TODO

print("练习7.2：ReLU激活")
print(f"  z = \n{z}")
print(f"  ReLU(z) = \n{a}")
print(f"  正确答案: 负数变0，正数不变\n")

# ── 练习 7.3：Softmax ─────────────────────────────────────────
logits = np.array([[1, 2, 3]], dtype=float)
probs = None  # TODO: softmax(logits)

print("练习7.3：Softmax")
print(f"  logits: {logits}")
print(f"  probs: {probs}")
print(f"  概率之和: {probs.sum() if probs is not None else None}")
print(f"  正确答案: 和为1.0\n")

# ── 练习 7.4：反向传播梯度 ────────────────────────────────────
# dy/dW = x^T @ dy
x_single = np.random.randn(3, 4)
dy = np.random.randn(3, 5)
dW = None  # TODO

print("练习7.4：反向传播")
print(f"  dW shape: {dW.shape if dW is not None else None}")
print(f"  正确答案: (4, 5)，与W同shape\n")

# ── 练习 7.5：两层网络 ────────────────────────────────────────
# TODO: 实现 x -> Linear -> ReLU -> Linear -> output
W1 = np.random.randn(4, 6) * 0.1
b1 = np.zeros(6)
W2 = np.random.randn(6, 2) * 0.1
b2 = np.zeros(2)

h = None  # TODO: 隐藏层（含ReLU）
output = None  # TODO: 输出层

print("练习7.5：两层网络前向传播")
print(f"  隐藏层 shape: {h.shape if h is not None else None}")
print(f"  输出 shape: {output.shape if output is not None else None}")
print(f"  正确答案: (3,6), (3,2)\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
