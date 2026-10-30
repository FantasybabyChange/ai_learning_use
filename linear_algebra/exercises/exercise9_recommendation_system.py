import numpy as np

print("=== 练习9：推荐系统中的SVD应用 ===\n")

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

# ── 练习 9.1：SVD分解评分矩阵 ─────────────────────────────────
U = None  # TODO
S = None  # TODO
Vt = None  # TODO

print("练习9.1：SVD分解")
print(f"  U shape: {U.shape if U is not None else None}")
print(f"  S: {S.round(3) if S is not None else None}")
print(f"  Vt shape: {Vt.shape if Vt is not None else None}\n")

# ── 练习 9.2：低秩近似预测缺失值 ──────────────────────────────
k = 2
ratings_pred = None  # TODO: 用前k个奇异值重构

print("练习9.2：低秩近似 (k=2)")
print(
    f"  预测评分矩阵:\n{ratings_pred.round(2) if ratings_pred is not None else None}\n"
)

# ── 练习 9.3：填补零值 ────────────────────────────────────────
# TODO: 找出原始矩阵中为0的位置，输出预测值
if ratings_pred is not None:
    predictions = []
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[1]):
            if ratings[i, j] == 0:
                predictions.append((i + 1, j + 1, ratings_pred[i, j]))
else:
    predictions = None

print("练习9.3：缺失值预测")
if predictions:
    for user, item, score in predictions:
        print(f"  用户{user} 对 物品{item} 预测评分: {score:.2f}")
print()

# ── 练习 9.4：选择最优k ───────────────────────────────────────
# TODO: 尝试 k=1,2,3,4，观察重构误差变化
errors = []
if U is not None and S is not None and Vt is not None:
    for k_test in range(1, min(ratings.shape) + 1):
        error = None  # TODO: 计算Frobenius范数误差
        errors.append(error)

print("练习9.4：不同k的重构误差")
for i, err in enumerate(errors, 1):
    print(f"  k={i}: 误差={err:.4f}" if err is not None else f"  k={i}: TODO")
print(f"  规律: k越大误差越小，但过大的k会过拟合\n")

# ── 练习 9.5：用户相似度 ──────────────────────────────────────
# TODO: 用U矩阵的行向量计算用户间的余弦相似度
if U is not None and S is not None:
    user_vectors = U[:, :2] * S[:2]  # 加权用户向量
    sim_01 = None  # TODO: 用户0和用户1的余弦相似度
else:
    sim_01 = None

print("练习9.5：用户相似度")
print(f"  用户0与用户1相似度: {sim_01:.4f}" if sim_01 is not None else "  TODO")
print(f"  含义: 相似度高的用户有相似的偏好\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
