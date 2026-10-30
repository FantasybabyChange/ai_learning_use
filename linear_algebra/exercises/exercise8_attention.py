import numpy as np

print("=== 练习8：注意力机制 ===\n")

np.random.seed(42)

# ── 练习 8.1：缩放点积注意力 ──────────────────────────────────
Q = np.random.randn(4, 8)
K = np.random.randn(4, 8)
V = np.random.randn(4, 8)
d_k = Q.shape[-1]

scores = None  # TODO: Q @ K^T / sqrt(d_k)
attention_weights = None  # TODO: softmax(scores)
output = None  # TODO: attention_weights @ V


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


print("练习8.1：缩放点积注意力")
print(f"  scores shape: {scores.shape if scores is not None else None}")
print(
    f"  weights shape: {attention_weights.shape if attention_weights is not None else None}"
)
print(f"  output shape: {output.shape if output is not None else None}")
print(f"  正确答案: (4,4), (4,4), (4,8)\n")

# ── 练习 8.2：注意力权重归一化 ────────────────────────────────
if attention_weights is not None:
    row_sums = None  # TODO: 每行之和
else:
    row_sums = None

print("练习8.2：注意力权重验证")
print(f"  每行权重之和: {row_sums.round(4) if row_sums is not None else None}")
print(f"  正确答案: 全为1.0\n")

# ── 练习 8.3：因果掩码 ────────────────────────────────────────
seq_len = 4
mask = None  # TODO: 下三角掩码矩阵

print("练习8.3：因果掩码")
print(f"  掩码矩阵:\n{mask}")
print(f"  含义: token_i只能看到token_j (j<=i)\n")

# ── 练习 8.4：多头注意力分割 ──────────────────────────────────
num_heads = 2
d_model = 8
d_head = d_model // num_heads

X = np.random.randn(2, 6, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
Q_proj = X @ W_q

# TODO: 将 Q_proj reshape 为 (batch, seq_len, num_heads, d_head) 再 transpose
Q_heads = None  # TODO

print("练习8.4：多头分割")
print(f"  Q_heads shape: {Q_heads.shape if Q_heads is not None else None}")
print(f"  正确答案: (2, 2, 6, 4)\n")

# ── 练习 8.5：自注意力的对称性 ────────────────────────────────
# TODO: 当 Q=K=V=X 时，scores矩阵有什么性质？
X_small = np.random.randn(3, 4)
self_scores = None  # TODO: X @ X^T

print("练习8.5：自注意力分数")
print(f"  self_scores:\n{self_scores.round(3) if self_scores is not None else None}")
print(
    f"  是否对称: {np.allclose(self_scores, self_scores.T) if self_scores is not None else None}"
)
print(f"  正确答案: True（因为 X@X^T 总是对称的）\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
