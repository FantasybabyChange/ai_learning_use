import numpy as np

print("=== 神经网络前向传播中的矩阵运算 ===")

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 简单的两层神经网络
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        """
        前向传播
        X: 输入矩阵 (batch_size, input_size)
        """
        # 第一层：线性变换 + 激活函数
        self.z1 = np.dot(X, self.W1) + self.b1  # 线性变换
        self.a1 = relu(self.z1)  # 激活函数
        
        # 第二层：线性变换 + softmax
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 线性变换
        self.a2 = softmax(self.z2)  # 输出层激活
        
        return self.a2

# 创建神经网络
input_size = 4
hidden_size = 5
output_size = 3
nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# 创建一些输入数据
batch_size = 10
X = np.random.randn(batch_size, input_size)
print(f"输入数据 X (形状 {X.shape}):\n{X[:2]}...")  # 只显示前2个样本

# 前向传播
output = nn.forward(X)
print(f"输出数据 (形状 {output.shape}):\n{output[:2]}...")

# 验证输出是概率分布
print(f"输出概率和 (每个样本): {np.sum(output, axis=1)[:2]}")

print("\n=== 矩阵维度变化 ===")
print(f"输入层 -> 隐藏层: {X.shape} -> {nn.a1.shape}")
print(f"隐藏层 -> 输出层: {nn.a1.shape} -> {nn.a2.shape}")

print("\n=== 矩阵运算细节 ===")
print(f"第一层线性变换: z1 = X * W1 + b1")
print(f"  X形状: {X.shape}, W1形状: {nn.W1.shape}, b1形状: {nn.b1.shape}")
print(f"  z1形状: {nn.z1.shape}")

print(f"第二层线性变换: z2 = a1 * W2 + b2")
print(f"  a1形状: {nn.a1.shape}, W2形状: {nn.W2.shape}, b2形状: {nn.b2.shape}")
print(f"  z2形状: {nn.z2.shape}")

print("\n=== 批量处理的优势 ===")
print(f"单样本处理: 需要 {input_size * hidden_size + hidden_size * output_size} 次乘法")
print(f"批量处理: 只需 {batch_size * (input_size * hidden_size + hidden_size * output_size)} 次乘法")
print(f"效率提升: {batch_size} 倍（通过并行计算）")

print("\n=== 参数数量 ===")
total_params = nn.W1.size + nn.b1.size + nn.W2.size + nn.b2.size
print(f"总参数数量: {total_params}")
print(f"  W1: {nn.W1.size} 参数")
print(f"  b1: {nn.b1.size} 参数")
print(f"  W2: {nn.W2.size} 参数")
print(f"  b2: {nn.b2.size} 参数")