import numpy as np

print("=== 练习4答案：特征值和特征向量 ===\n")

A = np.array([[4, 2], [2, 3]])

# 4.1
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"4.1 特征值: {eigenvalues}\n特征向量:\n{eigenvectors}\n")

# 4.2
Av = A @ eigenvectors[:, 0]
lambda_v = eigenvalues[0] * eigenvectors[:, 0]
print(f"4.2 Av={Av}, λv={lambda_v}, ≈:{np.allclose(Av, lambda_v)}\n")

# 4.3
Lambda = np.diag(eigenvalues)
A_reconstructed = eigenvectors @ Lambda @ eigenvectors.T
print(f"4.3 重构≈A: {np.allclose(A, A_reconstructed)}\n")

# 4.4
print(
    f"4.4 tr(A)={np.trace(A)}, sum(λ)={eigenvalues.sum()}, 相等:{np.isclose(np.trace(A), eigenvalues.sum())}\n"
)

# 4.5
print(
    f"4.5 det(A)={np.linalg.det(A):.4f}, prod(λ)={eigenvalues.prod():.4f}, 相等:{np.isclose(np.linalg.det(A), eigenvalues.prod())}\n"
)
