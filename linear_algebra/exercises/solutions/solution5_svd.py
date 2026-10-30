import numpy as np

print("=== 练习5答案：SVD ===\n")

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

# 5.1
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"5.1 U:{U.shape}, S:{S.round(3)}, Vt:{Vt.shape}\n")

# 5.2
A_reconstructed = U @ np.diag(S) @ Vt
print(f"5.2 重构≈A: {np.allclose(A, A_reconstructed)}\n")

# 5.3
k = 1
A_rank1 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"5.3 秩1近似:\n{A_rank1.round(3)}\n误差:{np.linalg.norm(A - A_rank1):.4f}\n")

# 5.4
energy_ratio = (S[0] ** 2) / np.sum(S**2)
print(f"5.4 前1个奇异值能量占比: {energy_ratio:.4f}\n")

# 5.5
rank_svd = np.sum(S > 1e-10)
print(f"5.5 矩阵的秩: {rank_svd}\n")
