import numpy as np

print("=== 练习5：奇异值分解 ===\n")

# ── 练习 5.1：SVD 分解 ───────────────────────────────────────
A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
U = None  # TODO
S = None  # TODO
Vt = None  # TODO

print("练习5.1：SVD分解")
print(f"  U shape: {U.shape if U is not None else None}")
print(f"  S: {S}")
print(f"  Vt shape: {Vt.shape if Vt is not None else None}\n")

# ── 练习 5.2：SVD 重构 ───────────────────────────────────────
A_reconstructed = None  # TODO

print("练习5.2：SVD重构")
print(
    f"  A ≈ 重构: {np.allclose(A, A_reconstructed) if A_reconstructed is not None else None}"
)
print(f"  正确答案: True\n")

# ── 练习 5.3：低秩近似 ────────────────────────────────────────
# TODO: 只保留前1个奇异值，重构矩阵
k = 1
A_rank1 = None  # TODO

print("练习5.3：低秩近似 (k=1)")
print(f"  秩1近似:\n{A_rank1}")
error = np.linalg.norm(A - A_rank1) if A_rank1 is not None else None
print(f"  重构误差: {error}\n")

# ── 练习 5.4：能量比 ──────────────────────────────────────────
# TODO: 计算前k个奇异值的能量占比 sum(s[:k]^2) / sum(s^2)
if S is not None:
    energy_ratio = None  # TODO
else:
    energy_ratio = None

print("练习5.4：能量比")
print(f"  前1个奇异值能量占比: {energy_ratio}")
print(f"  含义: 用1个分量能保留多少信息\n")

# ── 练习 5.5：SVD与秩 ─────────────────────────────────────────
# TODO: 通过非零奇异值个数判断矩阵的秩
rank_svd = None  # TODO

print("练习5.5：SVD求秩")
print(f"  矩阵的秩: {rank_svd}")
print(f"  正确答案: 2\n")

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
