# RAG 文档问答助手

基于 RAG（检索增强生成）架构的本地 PDF 文档智能问答系统。上传 PDF 文档后，系统自动解析、向量化，支持多轮对话式提问，并附上参考来源片段。

## 功能特性

* PDF 文档自动解析与文本抽取
* 文档内容自动分块与向量化存储（FAISS）
* 基于通义千问大模型的智能问答
* 检索结果附带来源片段引用，答案可追溯
* 多用户会话隔离，支持最多 10 个并发会话
* 会话自动超时清理（默认 1 小时），后台守护线程定期回收资源
* 基于 Gradio 的 Web 交互界面，开箱即用

## 技术栈

Python 3.11 + Gradio + LangChain + DashScope（通义千问 + 文本嵌入）+ FAISS + PyPDF

## 系统架构

    用户浏览器 (Gradio Web UI)
           │
           ▼
       main.py (入口，配置校验 + 启动清理线程)
           │
           ├── config.py (全局配置，dataclass)
           │
           ├── session_manager.py (多会话管理，线程安全)
           │      │
           │      └── rag_engine.py (RAGSession，每用户一个)
           │             ├── PyPDFLoader (PDF 解析)
           │             ├── RecursiveCharacterTextSplitter (文本分块)
           │             ├── DashscopeEmbeddings (文本嵌入)
           │             │      └── dashscope.TextEmbedding API
           │             ├── FAISS (本地向量存储)
           │             ├── ChatTongyi (大模型推理)
           │             │      └── dashscope.Generation API
           │             └── ConversationalRetrievalChain (检索增强链)
           │
           └── ui.py (Gradio 界面)

## 环境要求

* Python 3.10+
* 阿里云 DashScope API Key（[获取地址](https://dashscope.console.aliyun.com/)）

## 快速开始

### 1. 克隆项目

    git clone <repository-url>
    cd RAG文档问答助手（新）

### 2. 安装依赖

    pip install -r requirements.txt

### 3. 配置 API Key

在项目根目录创建 `.env` 文件，填入你的 DashScope API Key：

    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

可选配置项（不设置则使用默认值）：

    # LLM 模型，默认 qwen-plus
    LLM_MODEL=qwen-plus
    
    # 文档分块大小，默认 500
    CHUNK_SIZE=500
    
    # 检索返回片段数，默认 3
    RETRIEVAL_K=3
    
    # 会话超时秒数，默认 3600
    SESSION_TIMEOUT=3600
    
    # 最大并发会话数，默认 10
    MAX_SESSIONS=10

### 4. 启动服务

    python main.py

服务默认在 `http://127.0.0.1:7862` 启动，浏览器打开即可使用。

### 5. 使用流程

1. 在左侧上传 PDF 文档，点击「解析并加载文档」
2. 等待状态提示「文档加载完成」
3. 在右侧聊天框输入问题，系统将基于文档内容回答
4. 答案下方会显示参考来源片段

## 项目结构

    ├── main.py              # 入口：配置校验、启动清理线程、启动 Gradio 服务
    ├── config.py            # 全局配置（dataclass），集中管理所有参数
    ├── rag_engine.py        # RAG 核心引擎：PDF 加载、向量存储、检索链
    ├── embeddings.py        # DashScope 文本嵌入适配器
    ├── llm.py               # 通义千问 LLM 适配器
    ├── session_manager.py   # 多会话管理器（线程安全 + 超时清理）
    ├── ui.py                # Gradio Web 界面
    ├── requirements.txt     # Python 依赖清单
    ├── .env                 # 环境变量配置（需自行创建）
    └── .gitignore           # Git 忽略规则

## 配置说明

所有可配置项集中定义在 `config.py` 中：

| 配置项 | 默认值 | 说明  |
| --- | --- | --- |
| `embedding_model` | text-embedding-v1 | 文本嵌入模型 |
| `llm_model` | qwen-plus | 大语言模型 |
| `chunk_size` | 500 | 文档分块大小 |
| `chunk_overlap` | 50  | 分块重叠字符数 |
| `retrieval_k` | 3   | 检索返回 Top-K 片段 |
| `session_timeout` | 3600 | 会话超时（秒） |
| `max_sessions` | 10  | 最大并发会话数 |
| `cleanup_interval` | 300 | 后台清理间隔（秒） |
| `server_port` | 7862 | Gradio 服务端口 |

## 注意事项

* API Key 通过环境变量注入，禁止在代码中硬编码；`.env` 文件已加入 `.gitignore`，不会被提交到仓库
* 异常信息不会泄露到前端，均返回通用错误提示
* 每个浏览器标签页拥有独立会话，互不干扰
* 文档重新上传后会自动重建向量存储，覆盖旧数据
* 当前仅支持 PDF 格式文档

## License

MIT
