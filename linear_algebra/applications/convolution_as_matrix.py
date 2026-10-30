import numpy as np

print("=== 卷积运算的矩阵表示 ===")


def conv2d(image, kernel):
    """手动实现2D卷积"""
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i : i + kh, j : j + kw] * kernel)

    return output


image = np.array(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float
)

kernel = np.array([[1, 0], [0, -1]], dtype=float)

result = conv2d(image, kernel)
print(f"输入图像 (4x4):\n{image}")
print(f"卷积核 (2x2):\n{kernel}")
print(f"卷积结果 (3x3):\n{result}")

print("\n=== 将卷积转化为矩阵乘法 ===")


def im2col(image, kh, kw):
    """将图像patches展开为矩阵的列"""
    h, w = image.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    cols = np.zeros((kh * kw, out_h * out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i : i + kh, j : j + kw]
            cols[:, i * out_w + j] = patch.flatten()

    return cols, out_h, out_w


cols, out_h, out_w = im2col(image, 2, 2)
kernel_flat = kernel.flatten()

result_matmul = (kernel_flat @ cols).reshape(out_h, out_w)
print(f"im2col 展开后的矩阵 (4x9):\n{cols}")
print(f"展平的卷积核: {kernel_flat}")
print(f"矩阵乘法结果:\n{result_matmul}")
print(f"与手动卷积一致: {np.allclose(result, result_matmul)}")

print("\n=== 为什么CNN要用im2col？ ===")
print("1. 将卷积转化为矩阵乘法，可以利用高度优化的BLAS库")
print("2. GPU擅长大规模矩阵乘法，不擅长循环")
print("3. PyTorch/TensorFlow底层都用这种方式加速卷积")
print("4. 代价是需要额外内存存储展开后的矩阵")

print("\n=== 多通道卷积 ===")

image_rgb = np.random.rand(3, 6, 6)
kernel_3ch = np.random.rand(3, 3, 3)

print(f"RGB图像 shape: {image_rgb.shape} (channels, height, width)")
print(f"3通道卷积核 shape: {kernel_3ch.shape}")

output_single = 0
for c in range(3):
    cols_c, oh, ow = im2col(image_rgb[c], 3, 3)
    output_single += kernel_3ch[c].flatten() @ cols_c

output_single = output_single.reshape(oh, ow)
print(f"单输出通道卷积结果 shape: {output_single.shape}")
print("每个输出通道 = 所有输入通道的卷积结果之和")

print("\n=== 卷积与全连接的线性代数关系 ===")
print("卷积本质上是特殊的全连接层：")
print("- 权重共享：同一个卷积核在所有位置复用")
print("- 局部连接：每个输出只依赖局部patch")
print("- 从矩阵角度看：卷积 = 稀疏Toeplitz矩阵 x 输入向量")
print("- 这解释了为什么CNN参数远少于全连接网络")

print("\n=== AI应用：理解CNN的线性代数本质 ===")
print("1. 卷积层 = 线性变换（im2col后就是矩阵乘法）+ 非线性激活")
print("2. BatchNorm = 对特征图做仿射变换（也是线性操作）")
print("3. Pooling = 特殊的线性下采样")
print("4. 整个CNN的前向传播 = 一系列矩阵运算的组合")
