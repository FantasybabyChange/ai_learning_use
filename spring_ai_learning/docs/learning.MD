# **OverViews**
本项目使用SpringAi来关联各个Ai项目的使用
[SpringAi文档](https://docs.spring.io/spring-ai/reference/index.html)

常见Ai模型:
* Anthropic  Claude 3
* OpenAI GPT
* Microsoft  Azure AI
* Amazon AWS AI (Bedrock, SageMaker)
* Google Gemini
* Ollama

**SpringAi 支持的特性**
* **Chat Completion** 聊天补全
* **Embedding**  将文本/数据转换为数值向量，用于语义搜索、聚类等
* **Text to Image**
* **Audio Transcription**
* **Text to Speech**
* **Moderation** 内容审核  

**Structured Output** 结构化输出  
    是 AI 系统生成的一种规范化、可解析的数据格式（如 JSON、XML、表格等），而非非结构化的纯文本。  
**支持的向量数据库**  
  向量数据库:是专门用于存储、检索高维向量数据（如AI模型生成的嵌入向量）的数据库.
  流行的数据库包括: Apache Cassandra, Azure Cosmos DB, Azure Vector Search, Chroma, Elasticsearch, GemFire, MariaDB, Milvus, MongoDB Atlas, Neo4j, OpenSearch, Oracle, PostgreSQL/PGVector, PineCone, Qdrant, Redis, SAP Hana, Typesense and Weaviate.
**Api跨供应商**,包括新的类似sql的筛查API  
**工具和函数调用** 允许请求客户端工具和函数的执行, 从而根据需要获取必要的实时信息采取行动.  
**可观测** 提供AI相关操作中的insights.  
**为数据引擎整合了**[ETL框架](https://docs.spring.io/spring-ai/reference/api/etl-pipeline.html)文档  
**AI 模型评估** 提供的工具用来帮助评估产生的内容和防止Ai模型产生的幻觉回答(hallucinated response)  
**Spring boot 自动配置**  
**ChatClient Api** 通过fluent api 来关联AI聊天模型,惯用的比如webClient和restClientApi.  
**advisors api** 封装了可以重复使用的AI模式,通过LLMs来传输数据,并且提供跨模型和用例的可移植性.  
支持**聊天会话记忆**和**检索增强生成**(RAG-Retrieval Augmented Generation)

[下面是Ai的一些概念](./core_concepts.MD)


