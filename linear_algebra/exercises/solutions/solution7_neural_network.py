import numpy as np

print("=== 练习7答案：神经网络 ===\n")

np.random.seed(42)

# 7.1
x = np.random.randn(3, 4)
W = np.random.randn(4, 5)
b = np.random.randn(5)
y = x @ W + b
print(f"7.1 输出shape: {y.shape}\n")

# 7.2
z = np.array([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=float)
a = np.maximum(0, z)
print(f"7.2 ReLU(z):\n{a}\n")

# 7.3
logits = np.array([[1, 2, 3]], dtype=float)
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / exp_logits.sum()
print(f"7.3 probs: {probs}, 和: {probs.sum():.4f}\n")

# 7.4
dy = np.random.randn(3, 5)
dW = x.T @ dy
print(f"7.4 dW shape: {dW.shape}\n")

# 7.5
W1 = np.random.randn(4, 6) * 0.1
b1 = np.zeros(6)
W2 = np.random.randn(6, 2) * 0.1
b2 = np.zeros(2)
h = np.maximum(0, x @ W1 + b1)
output = h @ W2 + b2
print(f"7.5 h:{h.shape}, output:{output.shape}\n")
