import numpy as np

print("=== 练习9答案：推荐系统SVD ===\n")

ratings = np.array(
    [
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ],
    dtype=float,
)

# 9.1
U, S, Vt = np.linalg.svd(ratings, full_matrices=False)
print(f"9.1 U:{U.shape}, S:{S.round(3)}, Vt:{Vt.shape}\n")

# 9.2
k = 2
ratings_pred = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"9.2 预测矩阵:\n{ratings_pred.round(2)}\n")

# 9.3
for i in range(ratings.shape[0]):
    for j in range(ratings.shape[1]):
        if ratings[i, j] == 0:
            print(f"  用户{i + 1} 物品{j + 1}: {ratings_pred[i, j]:.2f}")
print()

# 9.4
for k_test in range(1, min(ratings.shape) + 1):
    pred = U[:, :k_test] @ np.diag(S[:k_test]) @ Vt[:k_test, :]
    error = np.linalg.norm(ratings[ratings > 0] - pred[ratings > 0])
    print(f"  k={k_test}: 非零位置误差={error:.4f}")
print()

# 9.5
user_vectors = U[:, :2] * S[:2]
norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
sim_01 = np.dot(user_vectors[0], user_vectors[1]) / (norms[0] * norms[1])
print(f"9.5 用户0与用户1相似度: {sim_01.item():.4f}\n")
