import numpy as np

print("=== 练习4：特征值和特征向量 ===\n")

# ── 练习 4.1：求特征值和特征向量 ──────────────────────────────
A = np.array([[4, 2], [2, 3]])
eigenvalues = None  # TODO
eigenvectors = None  # TODO

print("练习4.1：特征分解")
print(f"  特征值: {eigenvalues}")
print(f"  特征向量:\n{eigenvectors}")
print(f"  正确答案: 特征值 [5, 2]\n")

# ── 练习 4.2：验证 Av = λv ───────────────────────────────────
# TODO: 验证第一个特征对满足 Av = λv
if eigenvalues is not None and eigenvectors is not None:
    Av = None  # TODO
    lambda_v = None  # TODO
    is_valid = None  # TODO
else:
    Av = lambda_v = is_valid = None

print("练习4.2：验证特征方程")
print(f"  Av = {Av}")
print(f"  λv = {lambda_v}")
print(f"  Av ≈ λv: {is_valid}")
print(f"  正确答案: True\n")

# ── 练习 4.3：特征分解重构 ────────────────────────────────────
# TODO: 用 A = Q @ Lambda @ Q^T 重构矩阵
if eigenvalues is not None and eigenvectors is not None:
    Lambda = None  # TODO
    A_reconstructed = None  # TODO
else:
    Lambda = A_reconstructed = None

print("练习4.3：特征分解重构")
print(f"  重构矩阵:\n{A_reconstructed}")
print(
    f"  A ≈ 重构: {np.allclose(A, A_reconstructed) if A_reconstructed is not None else None}"
)
print(f"  正确答案: True\n")

# ── 练习 4.4：迹与特征值的关系 ────────────────────────────────
# TODO: 验证 tr(A) = sum(eigenvalues)
trace_sum = None  # TODO
eigenvalue_sum = None  # TODO

print("练习4.4：迹与特征值")
print(f"  tr(A) = {trace_sum}")
print(f"  sum(λ) = {eigenvalue_sum}")
print(
    f"  相等: {np.isclose(trace_sum, eigenvalue_sum) if trace_sum is not None else None}\n"
)

# ── 练习 4.5：行列式与特征值的关系 ────────────────────────────
det_product = None  # TODO
eigenvalue_prod = None  # TODO

print("练习4.5：行列式与特征值")
print(f"  det(A) = {det_product}")
print(f"  prod(λ) = {eigenvalue_prod}")
print(
    f"  相等: {np.isclose(det_product, eigenvalue_prod) if det_product is not None else None}\n"
)

print("=" * 50)
print("完成所有 TODO 后运行此脚本，对照正确答案检查。")
