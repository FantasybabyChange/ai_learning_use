import numpy as np

print("=== 优化算法中的向量运算 ===")

print("\n=== 1. 梯度下降的向量形式 ===")


def gradient_descent(grad_fn, x0, lr=0.01, epochs=100, tol=1e-8):
    """梯度下降的向量实现"""
    x = x0.copy()
    history = [x.copy()]

    for i in range(epochs):
        g = grad_fn(x)
        x = x - lr * g
        history.append(x.copy())

        if np.linalg.norm(g) < tol:
            print(f"在第 {i + 1} 步收敛")
            break

    return x, history


def quadratic_grad(x):
    """f(x) = x1^2 + 2*x2^2 的梯度"""
    return np.array([2 * x[0], 4 * x[1]])


x0 = np.array([5.0, 5.0])
x_opt, history = gradient_descent(quadratic_grad, x0, lr=0.1, epochs=200)
print(f"初始点: {x0}")
print(f"最优解: {x_opt}")
print(f"最优值 f(x*) = {x_opt[0] ** 2 + 2 * x_opt[1] ** 2:.6f}")
print("梯度下降的核心公式: x_{k+1} = x_k - lr * grad(f)")
print("这就是向量的减法和数乘！")

print("\n=== 2. 随机梯度下降 SGD ===")


def sgd(grad_fn, x0, lr=0.01, epochs=100, batch_size=32, n_samples=1000):
    """随机梯度下降"""
    x = x0.copy()

    for epoch in range(epochs):
        indices = np.random.randint(0, n_samples, batch_size)
        g = grad_fn(x, indices)
        x = x - lr * g

        if np.linalg.norm(g) < 1e-6:
            break

    return x


def linear_regression_grad(w, indices, X, y):
    """线性回归的mini-batch梯度"""
    X_batch = X[indices]
    y_batch = y[indices]
    residual = X_batch @ w - y_batch
    return (2 / len(indices)) * (X_batch.T @ residual)


np.random.seed(42)
n, d = 1000, 5
X = np.random.randn(n, d)
w_true = np.array([1, 2, 3, 4, 5], dtype=float)
y = X @ w_true + np.random.randn(n) * 0.1

w0 = np.zeros(d)
w_sgd = sgd(
    lambda w, idx: linear_regression_grad(w, idx, X, y), w0, lr=0.01, epochs=500
)
print(f"真实权重: {w_true}")
print(f"SGD恢复权重: {w_sgd.round(3)}")
print("SGD的关键：每次只用一小批数据估算梯度（向量运算）")

print("\n=== 3. Momentum 动量法 ===")


def momentum_descent(grad_fn, x0, lr=0.01, beta=0.9, epochs=100):
    """带momentum的梯度下降"""
    x = x0.copy()
    v = np.zeros_like(x)

    for i in range(epochs):
        g = grad_fn(x)
        v = beta * v + g
        x = x - lr * v

        if np.linalg.norm(g) < 1e-8:
            break

    return x


x_opt_mom = momentum_descent(quadratic_grad, x0, lr=0.1, beta=0.9)
print(f"Momentum最优解: {x_opt_mom}")
print("动量公式: v = beta * v + grad; x = x - lr * v")
print("v 是梯度的指数加权移动平均（向量累加）")

print("\n=== 4. Adam 优化器 ===")


def adam(grad_fn, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, epochs=100):
    """Adam优化器"""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, epochs + 1):
        g = grad_fn(x)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

        if np.linalg.norm(g) < 1e-8:
            break

    return x


x_opt_adam = adam(quadratic_grad, x0, lr=0.1)
print(f"Adam最优解: {x_opt_adam}")
print("Adam = Momentum + RMSProp")
print("m: 梯度的一阶矩估计（均值向量）")
print("v: 梯度的二阶矩估计（方差向量）")
print("所有运算都是逐元素的向量运算")

print("\n=== 5. 反向传播中的链式法则与矩阵运算 ===")


def forward_backward_example():
    """简单的两层网络前向+反向传播"""
    np.random.seed(42)
    x = np.array([[1.0, 2.0]])
    y = np.array([[1.0]])

    W1 = np.random.randn(2, 3) * 0.1
    W2 = np.random.randn(3, 1) * 0.1

    z1 = x @ W1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2
    loss = 0.5 * (z2 - y) ** 2

    dz2 = z2 - y
    dW2 = a1.T @ dz2
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = x.T @ dz1

    print(f"损失: {loss.item():.6f}")
    print(f"dW1 shape: {dW1.shape}, dW2 shape: {dW2.shape}")
    print(f"dW1:\n{dW1}")
    print(f"dW2:\n{dW2}")
    print("反向传播 = 一系列矩阵乘法和转置运算")
    print("链式法则在线性代数中就是矩阵乘法的链式组合")


forward_backward_example()

print("\n=== 6. 学习率与条件数的关系 ===")

A = np.array([[1, 0], [0, 100]])
cond = np.linalg.cond(A)
print(f"矩阵 A:\n{A}")
print(f"条件数 cond(A) = {cond}")
print(f"最大学习率上界 ≈ 2 / (lambda_max + lambda_min) = {2 / (1 + 100):.4f}")
print("条件数大 → 等高线椭圆很扁 → 梯度下降震荡 → 需要小学习率")
print("这就是为什么预处理（如BatchNorm）能加速训练")

print("\n=== 总结：优化中的线性代数 ===")
print("1. 梯度 = 向量（标量函数对向量参数的导数）")
print("2. 参数更新 = 向量减法 x = x - lr * grad")
print("3. Hessian = 矩阵（二阶导数），决定优化地形")
print("4. 条件数 = 矩阵性质，决定收敛速度")
print("5. 反向传播 = 矩阵乘法链")
