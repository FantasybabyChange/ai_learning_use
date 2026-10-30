# 线性代数公式速查表

## 向量运算

| 运算 | 公式 | NumPy |
|------|------|-------|
| 加法 | a + b = [a1+b1, a2+b2, ...] | `a + b` |
| 数乘 | k*a = [k*a1, k*a2, ...] | `k * a` |
| 点积 | a·b = sum(ai*bi) = \|\|a\|\|\|\|b\|\|cos(θ) | `a @ b` / `np.dot(a,b)` |
| 叉积 | a×b (仅3D，结果垂直于a,b) | `np.cross(a,b)` |
| L1范数 | \|\|v\|\|₁ = sum(\|vi\|) | `np.linalg.norm(v, ord=1)` |
| L2范数 | \|\|v\|\|₂ = sqrt(sum(vi²)) | `np.linalg.norm(v)` |
| Inf范数 | \|\|v\|\|∞ = max(\|vi\|) | `np.linalg.norm(v, ord=np.inf)` |
| 归一化 | v̂ = v / \|\|v\|\| | `v / np.linalg.norm(v)` |
| 余弦相似度 | cos(θ) = a·b / (\|\|a\|\|\|\|b\|\|) | `np.dot(a,b) / (norm(a)*norm(b))` |
| 投影 | proj_b(a) = (a·b/b·b) * b | `(np.dot(a,b)/np.dot(b,b))*b` |
| 正交判定 | a⊥b ⟺ a·b = 0 | `np.isclose(np.dot(a,b), 0)` |

## 矩阵运算

| 运算 | 公式 | NumPy |
|------|------|-------|
| 加法 | A + B (对应元素相加) | `A + B` |
| 数乘 | k*A | `k * A` |
| 乘法 | (AB)[i,j] = sum(A[i,k]*B[k,j]) | `A @ B` / `np.matmul(A,B)` |
| 转置 | (A^T)[i,j] = A[j,i] | `A.T` |
| 逆 | A * A^(-1) = I | `np.linalg.inv(A)` |
| 行列式 | det(A) | `np.linalg.det(A)` |
| 迹 | tr(A) = sum(A[i,i]) | `np.trace(A)` |
| Frobenius范数 | \|\|A\|\|_F = sqrt(sum(aij²)) | `np.linalg.norm(A, 'fro')` |
| 秩 | rank(A) | `np.linalg.matrix_rank(A)` |
| 条件数 | cond(A) = σ_max / σ_min | `np.linalg.cond(A)` |

## 特殊矩阵

| 矩阵 | 性质 | NumPy |
|------|------|-------|
| 单位矩阵 | I*A = A*I = A | `np.eye(n)` |
| 对角矩阵 | D[i,j]=0 (i≠j) | `np.diag([d1,d2,...])` |
| 对称矩阵 | A = A^T | 判断: `np.allclose(A, A.T)` |
| 正交矩阵 | Q^T*Q = I, Q^(-1) = Q^T | 判断: `np.allclose(Q.T@Q, I)` |
| 正定矩阵 | x^T*A*x > 0 (∀x≠0) | 判断: 所有特征值>0 |

## 线性方程组

| 问题 | 方法 | NumPy |
|------|------|-------|
| Ax=b (方阵) | 直接求解 | `np.linalg.solve(A, b)` |
| Ax=b (超定) | 最小二乘 | `np.linalg.lstsq(A, b, rcond=None)` |
| 齐次 Ax=0 | 非零解 ⟺ rank(A) < n | `np.linalg.matrix_rank(A)` |

## 矩阵分解

| 分解 | 形式 | 条件 | NumPy | AI用途 |
|------|------|------|-------|--------|
| 特征分解 | A = QΛQ^T | 对称方阵 | `np.linalg.eig(A)` | PCA、谱聚类 |
| SVD | A = UΣV^T | 任意矩阵 | `np.linalg.svd(A)` | 降维、推荐、压缩 |
| QR | A = QR | 任意矩阵 | `np.linalg.qr(A)` | 最小二乘、稳定求解 |
| Cholesky | A = LL^T | 对称正定 | `np.linalg.cholesky(A)` | 高斯采样、GP |

## 特征值与PCA

| 概念 | 公式 | 说明 |
|------|------|------|
| 特征方程 | Av = λv | v方向不变，λ为缩放倍数 |
| 迹-特征值 | tr(A) = sum(λi) | 迹等于特征值之和 |
| 行列式-特征值 | det(A) = prod(λi) | 行列式等于特征值之积 |
| PCA步骤 | 中心化→协方差→特征分解→投影 | 主成分=最大特征值的特征向量 |
| 方差解释比 | λk / sum(λi) | 第k个主成分保留的信息比例 |

## Attention机制

| 公式 | 说明 |
|------|------|
| scores = Q @ K^T / sqrt(d_k) | 缩放点积注意力分数 |
| weights = softmax(scores) | 归一化注意力权重（每行和=1） |
| output = weights @ V | 加权求和得到输出 |
| MultiHead: reshape→分头→各自attention→concat→W_o | 多头注意力流程 |

## 优化算法

| 算法 | 更新公式 | 关键向量运算 |
|------|---------|-------------|
| GD | x = x - lr * grad | 向量减法+数乘 |
| Momentum | v = β*v + grad; x = x - lr*v | 向量累加 |
| Adam | m=β1*m+(1-β1)*g; v=β2*v+(1-β2)*g² | 逐元素向量运算 |
| 反向传播 | dW = x^T @ dy | 矩阵乘法链 |

## 常用NumPy函数速查

```python
# 创建
np.array(), np.zeros(), np.ones(), np.eye(), np.diag()
np.random.randn(), np.random.rand()

# 形状
a.shape, a.reshape(), a.flatten(), a.T
np.concatenate(), np.stack(), np.hstack(), np.vstack()

# 运算
np.dot(), np.matmul(), @, np.outer()
np.sum(), np.mean(), np.max(), np.min()
np.sqrt(), np.exp(), np.log()

# 线性代数
np.linalg.norm(), np.linalg.det(), np.linalg.inv()
np.linalg.eig(), np.linalg.svd(), np.linalg.qr()
np.linalg.cholesky(), np.linalg.solve(), np.linalg.lstsq()
np.linalg.matrix_rank(), np.linalg.cond()
```
