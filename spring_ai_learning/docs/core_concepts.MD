# **Ai concepts**
[SpringAi核心概念文档](https://docs.spring.io/spring-ai/reference/concepts.html)
以下是Ai的核心概念,以及Spring的实现
## 关键名词(Key Terms)
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
- vector's dimensionality: 向量维度
## Models
模型是设计用于处理和产生信息的算法.  通过模仿人类认知功能. 通过从大数据中学习规律(patterns)和深层关联(insights), 这些模型可以用做预测,文字,图片生成,或者其他输出,从而对各行各业提供大量支撑.  
   模型有许多不同的类型: 

| 模型类型 (Model Type)　　　　　　　　　　　| 输入 (Input)　　　　　　　　　　　　　　　　　| 输出 (Output)　　　　　　　　　|
| --------------------------------------------| -----------------------------------------------| --------------------------------|
| **LLM** (大语言模型)　　　　　　　　　　　 | 语言/代码 (Language/Code)　　　　　　　　　　 | 语言/代码 (Language/Code)　　　|
| **图片生成模型** (Image Generation Models) | 语言/图片 (Language/Images)　　　　　　　　　 | 图片/视频 (Images/Videos)　　　|
| **文本语音模型** (Text-to-Speech Models)　 | 音频/代码 (Audio/Code)　　　　　　　　　　　　| 语言/代码 (Language/Code)　　　|
| **嵌入模型** (Embedding Models)　　　　　　| 文本/图片/音频/视频 (Text/Images/Audio/Video) | 向量/嵌入 (Vectors/Embeddings) |
## Prompts(提示词)
&emsp;&emsp;提示词是用来指引AI模型产生特定的输出.参考ChatGPT, 提示词通过在输入框中输入文本,然后发送给API. 当然,提示词包含的更广,不仅是简单的字符串.  
&emsp;&emsp;ChatGtp的API在一个prompt中有多个文本输入,每个文本输入被分配一个角色. ```例如，存在 system 角色：它会告诉模型如何表现，并设置交互上下&文；也存在 user 角色：它通常就是用户输入的内容。```  
&emsp;&emsp;制作有效的提示词是一门艺术和科学.ChatGpt专门为人类对话所设计.这和用类似sql去问一个问题完全不同. 和AI更像与其他人类交流.
&emsp;&emsp; 这种交互风格的重要性催生了"**Prompt Engineering**"这个科目. 这是一个蓬勃增长的技术集合 用来提高prompts的效率.在制作prompt投资时间可以直接的提升输出结果的质量.  
&emsp;&emsp; 分享关键词已经变成了一个社会实践, 在这个主题上有很多活跃的学术调研. 一个创建有效提示词的反直觉例子(对比SQL), 一个[研究论文](https://arxiv.org/abs/2205.11916)发现最有效的提示词之一是开头:"深呼吸一步一步的完成这项工作". 这告诉你为什么语言如此重要. 我们不能完全理解怎么去通过之前的版本来创建最有效的提示词, 比如ChatGPT3.5, 那就跟不用说新版本了》
### Prompt Templates
 创建一个高效的提示词, 首先是**建立上下文**,并且酱请求容特定部分替换成用户输入.  
 对于提示词的创建和管理这个过程使用传统的文本模板引擎.Spring AI 使用 OSS(开源)库 [StringTemplate](https://www.stringtemplate.org/)来实现.
 ```text
    Tell me a {adjective} joke about {content}.
 ```
 在Spring AI中,提示词模板如同 SpringMVC中的"View".  一个模型对象,比如Map集合,用来把占位符填充到模版内,而渲染者"render"字符变成了给AI模型供应提示词的内容.  
 &emsp;&emsp; 将提示词的特定数据格式发送给模型有相当大的差异. 初始的开始 比如从简单的字符串, 提示词演变成了包含多个消息, 每个字符串在每个消息代表了对于模型的不同的角色.


 ## Embeddings
&emsp;&emsp;嵌入是使用数值来表示文字,图片或者视频. 他用来捕捉输入的数据之间的关联性.  
&emsp;&emsp;工作原理: 通过把文本图片视频等转换到浮点型数字的数组重(向量). 向量用来捕捉文本图片视频等的含义. 内嵌数组的长度叫做**向量的维度**.
通过计算向量代表的两个文本数值之间的距离, 一个应用可以判断两个嵌入向量对象的**相似度**.

![embedding-ai](pic/spring-ai-embeddings.jpg)

&emsp;&emsp;对于探索AI的java码农来说,你包管复杂的数学理论和向量表示的具体实现,当你集成到你自己的应用程序中时候,基本理解他们在AI系统中的作用和功能(虎丘调用就行了).  
&emsp;&emsp; Embedding在实践应用中特别的关键. 比如 **检索增强模式**(Retrieval Augmented Generation),他们可以在**语义空间**(Semantic space)中把数据表示成点,有点类似于一个欧几里得几何的2D空间,但在更高维度中.这也意味着在欧几里得平面上的点可以基于他们的坐标变得远或者近,在语义空间中,点之间的的**临近程度**代表**含义的相似度**.&emsp; 句子中相似的主题在多维度空间中是坐标相近的.就像图表中的点靠近彼此. &emsp;这种相近性在文本分类,语义搜素,甚至产品推荐都有帮助,它允许AI基于他们的**位置**辨别和分组相关的概念在这个扩展的语义的土地中.  
你可以把语义空间理解为向量.
 ## Tokens
 &emsp;&emsp; Tokens 是AI模型工作的基础单元. 输入, 模型把单词转换成tokens.输出,他们把token还原成单词.  
 &emsp;&emsp; 在英语中,一个token 大概相当于一个单词的%75。比如,莎士比亚的完全工作,总共大约900w个单词,翻译成大约1200万个token.
 &emsp;&emsp;
 ```
   如同ChatGPT这样的大语言模型,他生产token而不是单词.尽管token经常变成完整的单词,理解其中的区别对理解LLM的配置如何影响他们的输出,为什么会出现通用对抗触发器(universal adversarial trigger)这种奇怪的现象,以及AI文本检测(AI-text detector)的工作等是很重要的.
 ```
 &emsp;&emsp; 也许更重要的是 tokens=money. 在AI模型托管的情况下,你使用token的数量决定你的费用. 输入输出都会被总体计入token count(令牌数量).
 &emsp;&emsp;当然,模型是受限于token的限制,在每个单一api的调用中文本处理的数量是受限的.而这个阈值通常称为"上下文窗口".当超过这个限制后模型不处理任何文本.


