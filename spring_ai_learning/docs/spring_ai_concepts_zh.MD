# **AI Concepts（Spring AI 概念）中文翻译**

[Spring AI 核心概念文档](https://docs.spring.io/spring-ai/reference/concepts.html)

本章节介绍 Spring AI 使用到的一些核心概念。建议你通读一遍，以便理解 Spring AI 是如何实现这些想法的。

## 关键术语（Key Terms）

- LLM：Large Language Model（大语言模型）
- RAG：Retrieval Augmented Generation（检索增强生成）
- Embeddings：嵌入 / 向量表示
- Tokens：令牌 / 标记
- Prompt Engineering：提示词工程
- Fine Tuning：微调
- Tool Calling：工具调用
- Structured Output：结构化输出
- Context Window：上下文窗口
- Pre-trained：预训练特性

## Models（模型）

AI 模型是用于处理并生成信息的算法，它们常常会模仿人类的认知功能。通过从大规模数据集中学习规律（patterns）与洞察（insights），模型可以进行预测、生成文本、生成图片或输出其他形式的结果，从而为各行各业提供支撑。

模型的类型很多，且每一种都更适合特定的使用场景。虽然 ChatGPT 及其生成式 AI 能力因“文本输入-文本输出”的交互方式而广受关注，但现实中还有许多模型与公司，提供多种输入与输出形式。ChatGPT 之前，人们对“文本生成图片”的模型（例如 Midjourney、Stable Diffusion）也早已有浓厚兴趣。

Spring AI 目前支持把输入与输出表示为语言（Language）、图片（Image）与音频（Audio）等不同模态的模型。下表按输入与输出类型对不同模型做了归类（并且表格的最后一行是文本输入、输出数字，对应文本嵌入 embedding text，用于表达语义的向量/数据结构。）

| 模型类型 (Model Type) | 输入 (Input) | 输出 (Output) |
|---|---|---|
| **LLM（大语言模型）** | 语言/代码 (Language/Code) | 语言/代码 (Language/Code) |
| **图片生成模型** | 语言/图片 (Language/Images) | 图片/视频 (Images/Videos) |
| **文本语音模型（Text-to-Speech）** | 音频/代码 (Audio/Code) | 语言/代码 (Language/Code) |
| **嵌入模型（Embedding Models）** | 文本/图片/音频/视频 (Text/Images/Audio/Video) | 向量/嵌入 (Vectors/Embeddings) |

Spring AI 也提供嵌入能力，以支持更高级的用例。

GPT 这类模型之所以特别，是因为它们具有预训练特性（pre-trained nature）。这一点体现在 GPT 里的 “P”：Chat Generative Pre-trained Transformer（聊天式生成式预训练 Transformer）。预训练能力使得 AI 更像是通用的开发者工具，而不需要你具备深入的机器学习或模型训练背景。

## Prompts（提示词）

提示词（prompts）是用于生成“语言类输出”的基础输入，它会引导 AI 模型产生特定结果。对于熟悉 ChatGPT 的人来说，提示词可能只是你在对话框里输入的那段文本，并发送给 API。

但提示词的内涵远不止“简单字符串”。在许多 AI 模型中，提示词包含多个文本输入，并且每个文本输入都被赋予不同的角色（role）。

例如，存在 `system` 角色：它会告诉模型如何表现，并设置交互上下文；也存在 `user` 角色：它通常就是用户输入的内容。

编写高效提示词既是“艺术”也是“科学”。ChatGPT 面向人类对话进行设计，这与用 SQL 这样的方式去“问一个问题”差异很大。你需要以“与另一个人交谈”的方式来和 AI 模型沟通。

正因为这种交互风格的重要性，“提示词工程（Prompt Engineering）”逐渐成为一门独立的学科。围绕如何改进提示词有效性，已经积累了一系列技术手段。投入时间精心设计提示词，往往能显著提升最终输出质量。

此外，“分享提示词”也成为了一种社区实践，并且围绕提示词效果的研究也持续进行。比如有论文指出，某些最有效的提示词以类似 “Take a deep breath and work on this step by step.” 这样的句子开头会更有效。这在一定程度上能说明语言在提示词中为什么如此重要。我们目前仍没有完全理解如何把这项技术的历史迭代（例如 ChatGPT 3.5）用到极致，更不用说那些正在开发的新版本了。

### Prompt Templates（提示词模板）

要创建高效提示词，关键在于建立请求的上下文（context），并把请求中的可变部分替换成与用户输入相关的值。

这一过程通常使用传统的文本模板引擎来完成提示词的创建与管理。Spring AI 使用开源库 [StringTemplate](https://www.stringtemplate.org/) 来实现模板渲染。

例如，一个简单的提示词模板可以是：

```text
Tell me a {adjective} joke about {content}.
```

在 Spring AI 中，提示词模板可以类比为 Spring MVC 架构里的 “View”。会把一个模型对象（model object，通常是 `java.util.Map`）传入模板，用于填充模板中的占位符。渲染后的字符串（rendered string）就会成为传给 AI 模型的最终提示词内容。

提示词发往模型时的数据格式差异很大：起初提示词可能只是简单字符串；后来发展出包含多条消息（multiple messages）的结构，而且每条消息里的每个字符串都代表模型所使用的一个明确角色（role）。

## Embeddings（嵌入）

嵌入（embeddings）是对文本、图片或视频的数值化表示，它会捕捉输入之间的关系。

嵌入的工作方式是：把文本、图片与视频转换成浮点数数组（vectors）。这些向量用于表达语义内容。向量数组的长度称为向量的维度（dimensionality）。

当应用通过计算两个向量表示之间的“数值距离（distance）”来判断相似性时，它就能够找出用于生成这些嵌入向量的对象之间的相似程度。

对于探索 AI 的 Java 开发者来说，没有必要理解这些向量表示背后的复杂数学理论或具体实现细节。理解嵌入在 AI 系统中的作用与功能已经足够，尤其是在你把 AI 能力集成到自己的应用中时。

嵌入在实际场景中尤其与检索增强生成（RAG, Retrieval Augmented Generation）相关。它可以把数据表示为语义空间（semantic space 中的点），类似于欧几里得几何里的二维平面（2-D），只不过语义空间维度更高。

这意味着：在欧几里得几何中，平面上的点会因为坐标不同而“近”或“远”；而在语义空间中，点与点之间的距离会反映它们语义含义的相似程度。关于相似主题的句子会在这个多维语义空间中彼此更靠近，效果类似于在图表中相邻点会更近。也正是这种“邻近性”，让它能用于文本分类、语义搜索，甚至产品推荐：AI 可以根据这些向量点的“位置”来识别并聚合相关概念。

你可以把这个语义空间简单理解为一个向量空间（vector）。

## Tokens（令牌/标记）

Tokens 是 AI 模型工作的“构建积木”。在输入侧，模型会把词语转换为 token；在输出侧，模型再把 token 转回词语。

在英语语境下，一个 token 大致对应 75% 的一个单词。作为参考，莎士比亚的完整作品约 900,000 个单词，折算下来约为 1.2 million tokens。

更重要的是：Tokens = Money（token 就是钱）。对于托管型 AI 模型，你的计费通常由使用的 token 数量决定，而且输入与输出都会计入 token 统计。

此外，模型还有 token 限制，它会限制一次 API 调用中可处理的文本量。这个阈值常被称为 “context window（上下文窗口）”。超出该限制的文本将不会被模型处理。

例如：ChatGPT3 的 token 限制是 4K；GPT4 提供不同档位（8K、16K、32K）；Anthropic 的 Claude 模型有 100K 上下文限制；Meta 的近期研究产生了一个支持 1M token 上下文限制的模型。

要把莎士比亚的全部作品输入到 GPT4 的上下文窗口里，你需要设计软件工程策略来把数据切分（chop up），并在不超出上下文窗口限制的前提下分段呈现。Spring AI 项目可以帮助你完成这些工作。

## Structured Output（结构化输出）

AI 模型的输出传统上会以 `java.lang.String` 的形式返回——即使你在提示词里要求“以 JSON 格式返回”。它可能是“看起来像 JSON 的正确文本”，但它并不是一个真正可直接使用的 JSON 数据结构，只是字符串而已。同时，要求“以 JSON 输出”作为提示的一部分也并不完全精确。

正因为这种细节，出现了一个专门的领域：通过设计提示词来引导模型生成你想要的输出格式；然后再把得到的字符串转换成应用可用的数据结构，以完成集成。

[Structured output conversion](https://docs.spring.io/spring-ai/reference/api/structured-output-converter.html#_structuredoutputconverter) 使用的是精心设计的 prompts，且通常需要与模型进行多轮交互，才能得到期望的格式。

## Bringing Your Data & APIs to the AI Model（把你的数据与 API 接入到 AI 模型）

如果模型并未在你的数据上训练完成，你该如何让 AI 模型“使用你的信息”？

注意：GPT 3.5/4.0 的训练数据集只延伸到 2021 年 9 月。因此当你问需要 2021 年 9 月之后才有的知识时，模型会说它不知道答案。一个冷知识是：这个数据集大约有 650GB。

要把你的数据纳入 AI 模型，有三种主要技术路线：

Fine Tuning（微调）：这是传统机器学习技术路线，通过调整模型并改变其内部权重来“让模型适配你的数据”。但它对机器学习专家来说仍然是一个挑战，且由于模型规模巨大，对像 GPT 这类模型也非常资源密集。另外，也并非所有模型都提供微调选项。

Prompt Stuffing（把数据塞进提示词）：更实用的替代方案是把你的数据嵌入到提供给模型的提示词中。由于模型有 token 限制，你需要用一系列策略把“相关数据”塞进上下文窗口。这个做法也常被口语称为 “stuffing the prompt（把提示词塞满）”。Spring AI 库提供基于该方案的实现能力，也就是常说的 Retrieval Augmented Generation（RAG）。

Tool Calling（工具调用）：该技术允许注册工具（你自定义的服务），把大语言模型与外部系统的 API 连接起来。Spring AI 能显著简化你实现 [tool calling](https://docs.spring.io/spring-ai/reference/api/tools.html) 所需要的代码。

### Retrieval Augmented Generation（检索增强生成，RAG）

为了解决“如何把相关数据放入提示词以获得准确回答”的问题，出现了一种叫做 Retrieval Augmented Generation（RAG）的技术。

RAG 的思路是一种“批处理式（batch processing）”编程模型：读取你文档中的非结构化数据，对其进行转换，然后写入向量数据库（vector database）。从更高层次看，它本质上是一个 ETL（Extract, Transform and Load）流水线。向量数据库用于 RAG 中的“检索（retrieval）”部分。

在把非结构化数据加载到向量数据库时，最重要的转换之一是把原始文档切分成更小的片段（chunks）。切分通常包含两个关键步骤：

1. 在尽量保留内容语义边界的情况下切分。例如，如果文档包含段落与表格，就应避免在段落或表格中间截断；对代码来说，应避免在方法实现的中间位置切断。
2. 再把这些片段进一步切分，使其大小只占 AI 模型 token 限制的一小部分（small percentage）。

RAG 的下一阶段是处理用户输入。当用户提问需要由 AI 模型回答时，会把问题与所有“相似”的文档片段一起放入将发送给 AI 模型的提示词中。这就是为什么要使用向量数据库：向量数据库非常擅长查找相似内容。

[ETL Pipeline](https://docs.spring.io/spring-ai/reference/api/etl-pipeline.html) 提供了更多关于如何编排“从数据源提取数据并存储到结构化向量存储”的信息，以确保在把数据交给 AI 模型检索时它处于最佳格式。

[ChatClient - RAG](https://docs.spring.io/spring-ai/reference/api/chatclient.html#_retrieval_augmented_generation) 说明如何在应用中使用 `QuestionAnswerAdvisor` 来启用 RAG 能力。

### Tool Calling（工具调用）

大型语言模型（LLMs）在训练结束后就会“冻结”知识，不再具备实时访问或修改外部数据的能力，因此可能出现知识过时的问题。

[Tool Calling](https://docs.spring.io/spring-ai/reference/api/tools.html) 机制用于弥补这些不足。它允许你注册自己的服务作为“工具（tools）”，把大模型连接到外部系统的 API；这些系统可以为 LLM 提供实时数据，并代为执行数据处理动作。

Spring AI 极大简化了你支持工具调用所需的代码。它会帮你处理“工具调用的对话流程（tool invocation conversation）”。你可以把工具作为一个 `@Tool` 注解的方法来提供，并通过 prompt 的选项把它暴露给模型；你还可以在同一个提示词中定义并引用多个工具。

当你希望让某个工具对模型可用时，会把它的定义放入聊天请求（chat request）中。每个工具定义包括：工具名称（name）、工具描述（description）以及输入参数的 schema（结构/格式定义）。

当模型决定调用工具时，它会在响应中返回工具名称与根据 schema 构造出的输入参数。

应用端负责根据工具名称识别并使用提供的输入参数执行该工具。

工具调用结果会被应用处理后，再发送回模型。

模型随后会结合工具调用结果生成最终回答，并把该结果作为额外上下文。

你可以阅读 [Tool Calling](https://docs.spring.io/spring-ai/reference/api/tools.html) 文档以了解如何在不同的 AI 模型上使用此能力。

## Evaluating AI responses（评估 AI 回答）

有效评估 AI 系统在用户请求下产生的输出，对于确保最终应用的准确性与可用性非常重要。一些新兴技术可以让你直接利用预训练模型本身来完成评估。

评估过程涉及判断生成的回答是否符合用户意图（intent）以及问题上下文（context）。常用指标包括相关性（relevance）、连贯性（coherence）以及事实正确性（factual correctness），用于衡量回答质量。

一种做法是：把“用户请求”和“AI 模型的回答”同时展示给另一个模型，再追问该回答是否与提供的数据一致。

此外，利用向量数据库中存储的信息作为补充数据，也可以增强评估效果，帮助判断回答的相关性。

Spring AI 提供了 `Evaluator` API，目前可用于接入一些基础评估策略。更多信息请参阅 [Evaluation Testing](https://docs.spring.io/spring-ai/reference/api/testing.html) 文档。

