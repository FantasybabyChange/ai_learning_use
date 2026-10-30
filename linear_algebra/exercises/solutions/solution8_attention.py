import numpy as np

print("=== 练习8答案：注意力机制 ===\n")

np.random.seed(42)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


Q = np.random.randn(4, 8)
K = np.random.randn(4, 8)
V = np.random.randn(4, 8)
d_k = Q.shape[-1]

# 8.1
scores = Q @ K.T / np.sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights @ V
print(
    f"8.1 scores:{scores.shape}, weights:{attention_weights.shape}, output:{output.shape}\n"
)

# 8.2
row_sums = attention_weights.sum(axis=-1)
print(f"8.2 每行权重之和: {row_sums.round(4)}\n")

# 8.3
seq_len = 4
mask = np.tril(np.ones((seq_len, seq_len)))
print(f"8.3 因果掩码:\n{mask}\n")

# 8.4
num_heads = 2
d_model = 8
d_head = d_model // num_heads
X = np.random.randn(2, 6, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
Q_proj = X @ W_q
Q_heads = Q_proj.reshape(2, 6, num_heads, d_head).transpose(0, 2, 1, 3)
print(f"8.4 Q_heads shape: {Q_heads.shape}\n")

# 8.5
X_small = np.random.randn(3, 4)
self_scores = X_small @ X_small.T
print(f"8.5 对称: {np.allclose(self_scores, self_scores.T)}\n")
