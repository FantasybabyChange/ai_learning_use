# 大模型底层原理学习路径

> **目标**：快速理解大模型（LLM）的底层原理，以 Transformer 为核心终点，数学够用即可，不求深、求理解。
>
> **策略**：以大模型为终点倒推所需知识，遇到不懂的数学回头补，边学边用代码实践。

---

## 总览

```
数学基础（够用）
├── 线性代数
├── 概率论
└── 微积分
        ↓
深度学习入门
        ↓
Transformer & 注意力机制  ⭐ 核心
        ↓
GPT 工作原理
        ↓
大模型应用（RAG / Fine-tuning / Agent）
```

---

## 阶段一：数学基础「够用就行」

> ⏱️ 预计时间：3 周  
> 💡 三个模块并行学，不求精通，求看懂公式、理解直觉

### 1.1 线性代数（1 周）

**目标**：能看懂 Attention 公式里的矩阵运算

| 概念 | 在大模型中的用途 |
|------|----------------|
| 向量、矩阵 | 表示词语、句子、参数 |
| 矩阵乘法 | Q/K/V 的核心计算 |
| 点积（内积） | 计算两个向量的相似度 |
| 矩阵转置 | 公式里的 Wᵀ |
| 维度 / Shape | 理解张量在网络中的流动 |

**学习资源**
- 🎬 [3Blue1Brown 线性代数本质（B站中文字幕）](https://www.bilibili.com/video/BV1ys411472E)
  - 共 15 集，每集约 10 分钟，1 周刷完
- 📁 项目代码：`linear_algebra/basics/vectors_and_matrices.py`

**学完标准**：能理解 `Q × Kᵀ / √d` 这个公式在做什么

---

### 1.2 概率论（3 天）

**目标**：理解模型输出是概率分布，而不是确定答案

| 概念 | 在大模型中的用途 |
|------|----------------|
| 概率分布 | 模型对每个 token 输出一个概率 |
| Softmax | 把原始分数转成概率（大模型核心操作） |
| 交叉熵 Loss | 衡量模型预测有多"错" |
| KL 散度 | RLHF 对齐训练中用到 |

**学习资源**
- 🎬 [StatQuest - Probability（YouTube）](https://www.youtube.com/@statquest)
- 📖 只看概率分布、最大似然、交叉熵三个章节即可

**学完标准**：能解释为什么语言模型训练用交叉熵作为损失函数

---

### 1.3 微积分（3 天）

**目标**：理解模型训练的本质是梯度下降

| 概念 | 在大模型中的用途 |
|------|----------------|
| 导数 / 偏导数 | 参数朝哪个方向更新 |
| 梯度下降 | 训练的核心循环 |
| 链式法则 | 反向传播的数学基础 |
| Adam 优化器 | GPT 系列默认使用的优化器 |

**学习资源**
- 🎬 [3Blue1Brown 微积分本质（B站）](https://www.bilibili.com/video/BV1qW411N7FU)
  - 只看前 6 集即可
- 🎬 [3Blue1Brown 神经网络反向传播直觉](https://www.bilibili.com/video/BV16x411V7Qg)

**学完标准**：能用一句话解释"反向传播在做什么"

---

## 阶段二：深度学习入门

> ⏱️ 预计时间：2 周  
> 💡 不要从头推公式，先跑起来看效果，理解整体流程

### 核心概念

```
输入数据 → 神经网络 → 输出预测 → 和正确答案比较（Loss）→ 调整参数 → 循环
```

| 概念 | 说明 |
|------|------|
| 前向传播 | 数据从输入到输出的计算过程 |
| 反向传播 | 从 Loss 反推每个参数的梯度 |
| 激活函数 | ReLU / GELU / SiLU，给网络加入非线性 |
| Dropout | 随机丢弃神经元，防止过拟合 |
| LayerNorm | 稳定训练，Transformer 每层都用 |
| 残差连接 | `输出 = F(x) + x`，解决梯度消失 |

**学习资源**
- 📖 [《动手学深度学习》李沐 - zh.d2l.ai](https://zh.d2l.ai)
  - 免费、中文、有代码、有配套视频
  - 重点阅读：第 2 章（预备知识）、第 3 章（线性神经网络）、第 4 章（多层感知机）
- 📁 项目代码：`linear_algebra/applications/neural_network_forward.py`

**学完标准**：能用 PyTorch 写一个简单的两层神经网络并训练

---

## 阶段三：Transformer & 注意力机制 ⭐

> ⏱️ 预计时间：3 周  
> 💡 这是整个学习路径的核心，GPT / Claude / Gemini 全基于此

### 3.1 词向量（Embedding）

```
文字 → 数字向量
"猫" → [0.2, -0.5, 0.8, 0.1, ...]

含义相近的词，向量距离更近：
"猫" 和 "狗" 的向量距离 < "猫" 和 "汽车" 的向量距离
```

### 3.2 自注意力机制（Self-Attention）

```
核心公式：
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V

直觉理解：
Q（Query）= 我想查询什么
K（Key）  = 我能提供什么标签
V（Value）= 我实际的内容

每个词通过计算与其他词的相似度，决定"我该关注谁"，
然后加权汇聚其他词的信息，得到新的表示。
```

### 3.3 多头注意力（Multi-Head Attention）

```
= 多个注意力头并行运行
= 从多个角度同时理解句子
= 最后拼接所有头的结果
```

### 3.4 完整 Transformer 结构

```
Transformer Block（重复 N 层）
├── Multi-Head Self-Attention
│   └── 残差连接 + LayerNorm
├── Feed-Forward Network（FFN）
│   └── 残差连接 + LayerNorm
└── （Decoder 额外有 Cross-Attention 层）
```

### 学习资源（按顺序）

| 顺序 | 资源 | 类型 | 说明 |
|------|------|------|------|
| 1 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图文 | 全网最好的图解，必看 ⭐ |
| 2 | [3Blue1Brown 大模型可视化系列（B站）](https://www.bilibili.com/video/BV1TZ421j7Ke) | 视频 | 极度直观，强烈推荐 ⭐ |
| 3 | [李沐精读 Attention Is All You Need](https://www.bilibili.com/video/BV1pu411o7BE) | 视频 | 中文逐段讲解原始论文 |
| 4 | [《动手学深度学习》第 10 章 - 注意力机制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html) | 图文+代码 | 有完整 PyTorch 实现 |

**学完标准**：能手绘 Transformer 结构图，并解释每个模块的作用

**实践任务**：重写 `linear_algebra/applications/attention_mechanism.py`，加上详细注释

---

## 阶段四：GPT 工作原理

> ⏱️ 预计时间：1 周  
> 💡 理解从"预训练"到"能对话"的完整过程

### 4.1 预训练

```
任务：给模型海量文本，让它不断预测"下一个词"
      "今天天气___" → 预测 "很好"

结果：模型学会了语言规律、世界知识、推理能力
规模：GPT-3 用了 3000 亿个 token 的文本训练
```

### 4.2 指令微调（SFT）

```
任务：用"问题-答案"对继续训练，让模型学会"听指令"
      之前：续写文本
      之后：回答问题、执行指令
```

### 4.3 RLHF（基于人类反馈的强化学习）

```
目标：让模型回答更符合人类期望（有帮助、无害、诚实）

流程：
1. 让模型对同一问题生成多个答案
2. 人类给这些答案打分排序
3. 用排序数据训练"奖励模型"（RM）
4. 用 RL 让语言模型最大化奖励模型的分数
```

### 4.4 推理过程（Inference）

```
输入 tokens
    ↓
经过 N 层 Transformer
    ↓
输出词表上每个 token 的概率分布
    ↓
采样（Temperature / Top-p 控制随机性）
    ↓
得到下一个 token
    ↓
循环，直到生成结束符
```

### 学习资源

- 🎬 [李沐 GPT 论文精读](https://www.bilibili.com/video/BV1AF411b7xQ)
- 🎬 [李沐 InstructGPT 论文精读（RLHF）](https://www.bilibili.com/video/BV1hd4y187CR)
- 📄 原始论文：[Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

---

## 阶段五：大模型应用（对接 Spring AI 项目）

> ⏱️ 持续学习  
> 💡 学完前四阶段，回头看 `spring_ai_learning` 里的文档，会豁然开朗

| 技术 | 原理 | 项目对应 |
|------|------|---------|
| RAG（检索增强生成） | Embedding 相似度检索 + 注入 Prompt | `spring_ai_learning/docs/core_concepts.MD` |
| Tool Calling（工具调用） | 模型输出结构化指令，应用执行后返回结果 | `spring_ai_learning/docs/core_concepts.MD` |
| Fine-Tuning（微调） | LoRA / QLoRA，低成本适配特定任务 | — |
| Prompt Engineering | 系统提示词、Few-shot、Chain-of-Thought | — |
| 结构化输出 | 强制模型输出 JSON 等格式 | `spring_ai_learning/docs/core_concepts.MD` |

---

## 学习计划时间表

| 周次 | 内容 | 每日时间 | 完成标志 |
|------|------|---------|---------|
| 第 1 周 | 3B1B 线性代数本质（全 15 集）| 1.5 小时 | - [ ] 完成 |
| 第 2 周 | 概率论基础 + 微积分基础 | 1.5 小时 | - [ ] 完成 |
| 第 3 周 | 《动手学深度学习》第 2-4 章，跑通代码 | 2 小时 | - [ ] 完成 |
| 第 4 周 | The Illustrated Transformer + 3B1B 大模型系列 | 2 小时 | - [ ] 完成 |
| 第 5 周 | 李沐精读 Attention Is All You Need + 重写 attention 代码 | 2 小时 | - [ ] 完成 |
| 第 6 周 | GPT 原理 + 李沐 GPT/InstructGPT 精读 | 2 小时 | - [ ] 完成 |
| 第 7-8 周 | 大模型应用，对接 Spring AI 项目实践 | 2 小时 | - [ ] 完成 |

---

## 关键心态

> **不要等数学学完再看大模型**
>
> ✅ 正确方式：看大模型 → 遇到不懂的数学 → 回头查那个知识点 → 继续看大模型
>
> ❌ 错误方式：线性代数学完 → 概率论学完 → 微积分学完 → 才敢看 Transformer

---

## 里程碑检查点

- [ ] 能解释向量点积的几何意义
- [ ] 能用一句话解释 softmax 在做什么
- [ ] 能解释梯度下降的直觉
- [ ] 能手绘 Self-Attention 的计算流程
- [ ] 能解释 Q、K、V 分别代表什么
- [ ] 能解释 GPT 预训练的任务是什么
- [ ] 能解释 RAG 和 Fine-Tuning 的区别
- [ ] 重写 `attention_mechanism.py` 并加上完整注释 ⭐
