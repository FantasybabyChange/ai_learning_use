# **Ai concepts**
[SpringAi核心概念文档](https://docs.spring.io/spring-ai/reference/concepts.html)
以下是Ai的核心概念,以及Spring的实现
## Models
模型是设计用于处理和产生信息的算法.  通过模仿人类认知功能. 通过从大数据中学习规律(patterns)和深层关联(insights), 这些模型可以用做预测,文字,图片生成,或者其他输出,从而对各行各业提供大量支撑.  
   模型有许多不同的类型:
   |模型|输入|输出|
   |--|--|--|
   |LLM  large language Model 大语言模型|语言/代码|语言/代码|
   |图片生成模型|语言/图片|图片/视频|
   |文本 语音模型 |音频/代码|语言/代码|
   |嵌入模型(embedding model) |文本/图片/音频/视频|向量/嵌入|
   
## Prompts(提示词)
提示词是用来指引AI模型产生特定的输出.参考ChatGPT, 提示词通过在输入框中输入文本,然后发送给API. 当然,提示词包含的更广,不仅是简单的字符串.
### Prompt Templates
 创建一个高效的提示词, 首先是**建立上下文**,并且酱请求容特定部分替换成用户输入.  
 对于提示词的创建和管理这个过程使用传统的文本模板引擎.Spring AI 使用 OSS(开源)库 [StringTemplate](https://www.stringtemplate.org/)来实现.
 ```text
    Tell me a {adjective} joke about {content}.
 ```
 在Spring AI中,提示词模板如同 SpringMVC中的"View".  一个模型对象,比如Map集合,用来填充模板的占位符.

 ## Embeddings
嵌入是使用数值来表示文字,图片或者视频. 他用来捕捉输入的数据之间的关联性.  
工作原理: 通过把文本图片视频等转换到浮点型数字的数组重(向量). 向量用来捕捉文本图片视频等的含义. 内嵌数组的长度叫做**向量的维度**.
通过计算向量代表的两个文本数值之间的距离, 一个应用可以判断两个嵌入向量对象的**相似度**.

![embedding-ai](pic/spring-ai-embeddings.jpg)