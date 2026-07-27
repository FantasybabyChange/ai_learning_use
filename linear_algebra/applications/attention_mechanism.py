import numpy as np

print("=== 注意力机制中的矩阵运算 ===")

# 缩放点积注意力（Scaled Dot-Product Attention）
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: 查询矩阵 (batch_size, seq_len_q, d_k)
    K: 键矩阵 (batch_size, seq_len_k, d_k)
    V: 值矩阵 (batch_size, seq_len_k, d_v)
    mask: 掩码矩阵
    """
    d_k = Q.shape[-1]
    
    # 计算注意力分数
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # 应用掩码（如果有）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # 应用softmax得到注意力权重
    attention_weights = softmax(scores, axis=-1)
    
    # 计算输出
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 创建示例数据
batch_size = 2
seq_len = 4
d_model = 8

# 输入序列
X = np.random.randn(batch_size, seq_len, d_model)
print(f"输入序列 X (形状 {X.shape}):\n{X[0, :2, :2]}...")  # 只显示部分

# 线性投影矩阵
W_q = np.random.randn(d_model, d_model) * 0.01
W_k = np.random.randn(d_model, d_model) * 0.01
W_v = np.random.randn(d_model, d_model) * 0.01

# 生成Q, K, V
Q = np.matmul(X, W_q)
K = np.matmul(X, W_k)
V = np.matmul(X, W_v)

print(f"\n查询矩阵 Q (形状 {Q.shape}):\n{Q[0, :2, :2]}...")
print(f"键矩阵 K (形状 {K.shape}):\n{K[0, :2, :2]}...")
print(f"值矩阵 V (形状 {V.shape}):\n{V[0, :2, :2]}...")

# 计算注意力
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\n注意力输出 (形状 {output.shape}):\n{output[0, :2, :2]}...")
print(f"注意力权重 (形状 {attention_weights.shape}):\n{attention_weights[0]}")

print("\n=== 多头注意力机制 ===")

# 多头注意力参数
num_heads = 2
d_k = d_model // num_heads

# 为每个头创建投影矩阵
W_q_heads = np.random.randn(d_model, d_model) * 0.01
W_k_heads = np.random.randn(d_model, d_model) * 0.01
W_v_heads = np.random.randn(d_model, d_model) * 0.01
W_o = np.random.randn(d_model, d_model) * 0.01

# 线性投影
Q_proj = np.matmul(X, W_q_heads)
K_proj = np.matmul(X, W_k_heads)
V_proj = np.matmul(X, W_v_heads)

print(f"投影后的 Q (形状 {Q_proj.shape}):\n{Q_proj[0, :2, :2]}...")

# 分割成多个头
batch_size, seq_len, _ = Q_proj.shape
Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

print(f"分割后的 Q 头 (形状 {Q_heads.shape}):\n{Q_heads[0, 0, :2, :2]}...")

# 对每个头计算注意力
attention_outputs = []
attention_weights_all = []
for i in range(num_heads):
    out, weights = scaled_dot_product_attention(
        Q_heads[:, i, :, :], 
        K_heads[:, i, :, :], 
        V_heads[:, i, :, :]
    )
    attention_outputs.append(out)
    attention_weights_all.append(weights)

# 合并所有头的输出
multi_head_output = np.concatenate(attention_outputs, axis=-1)
print(f"多头注意力输出 (形状 {multi_head_output.shape}):\n{multi_head_output[0, :2, :2]}...")

# 最终线性投影
final_output = np.matmul(multi_head_output, W_o)
print(f"最终输出 (形状 {final_output.shape}):\n{final_output[0, :2, :2]}...")

print("\n=== 自注意力 vs 交叉注意力 ===")
print("自注意力: Q, K, V 来自同一个序列")
print("交叉注意力: Q 来自一个序列，K, V 来自另一个序列")

# 交叉注意力示例
encoder_output = np.random.randn(batch_size, seq_len, d_model)  # 编码器输出
decoder_input = np.random.randn(batch_size, seq_len, d_model)  # 解码器输入

# Q来自解码器，K和V来自编码器
Q_cross = np.matmul(decoder_input, W_q)
K_cross = np.matmul(encoder_output, W_k)
V_cross = np.matmul(encoder_output, W_v)

output_cross, weights_cross = scaled_dot_product_attention(Q_cross, K_cross, V_cross)
print(f"交叉注意力输出 (形状 {output_cross.shape}):\n{output_cross[0, :2, :2]}...")

print("\n=== 注意力机制的计算复杂度 ===")
print(f"标准注意力: O(n^2 * d)")
print(f"  n: 序列长度, d: 特征维度")
print(f"当序列很长时，计算复杂度很高，这也是Transformer需要优化的地方")