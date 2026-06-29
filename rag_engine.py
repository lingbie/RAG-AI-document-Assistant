# rag_engine.py
# [FIXED] 将会话状态封装为类实例，替代全局可变状态
# [FIXED] QA Chain 在文档加载后创建一次，后续对话复用
# [FIXED] 异常信息不再泄露给前端
import logging
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config
from embeddings import DashscopeEmbeddings
from llm import ChatTongyi

logger = logging.getLogger(__name__)

# [FIXED] 通用错误消息，不暴露内部细节
GENERIC_ERROR_MSG = "处理时遇到问题，请稍后重试。如持续出现，请联系管理员。"


class RAGSession:
    """单个文档会话的 RAG 状态管理器

    [FIXED] 将全局变量封装为实例属性，解决线程安全和可测试性问题
    """

    def __init__(self) -> None:
        self.vector_store: Optional[FAISS] = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.chain: Optional[ConversationalRetrievalChain] = None

        # 引擎组件延迟初始化
        self.llm = ChatTongyi()
        self.embeddings = DashscopeEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load_document(self, file_path: str) -> str:
        """加载 PDF 文档并构建向量存储和 QA Chain

        [FIXED] 添加文件路径有效性检查
        [FIXED] 一次性构建 chain，后续对话复用
        """
        # [FIXED] 校验文件是否存在
        import os

        if not os.path.isfile(file_path):
            logger.warning("文件不存在: %s", file_path)
            return "文件不存在，请检查文件路径。"

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except Exception:
            logger.exception("PDF 加载失败: %s", file_path)
            return "PDF 文件加载失败，请确认文件未损坏且为有效的 PDF 格式。"

        if not documents:
            return "PDF 文件中未检测到可读文本，请确认文件包含文字内容。"

        texts = self.text_splitter.split_documents(documents)

        try:
            # [FIXED] 使用新的嵌入适配器构建向量存储
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
        except Exception:
            logger.exception("向量存储构建失败")
            return GENERIC_ERROR_MSG

        # [FIXED] 构建 QA Chain（仅一次），后续 chat 复用
        try:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": config.retrieval_k}
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
            )
        except Exception:
            logger.exception("QA Chain 构建失败")
            self.vector_store = None
            return GENERIC_ERROR_MSG

        self.memory.clear()
        logger.info("文档加载成功: %s, 文本块数: %d", file_path, len(texts))
        return f"成功加载文档并提取了 {len(texts)} 个文本块，现在可以提问了！"

    def chat(self, question: str) -> str:
        """执行一轮问答

        [FIXED] 复用已构建的 chain，不再每轮重建
        [FIXED] 异常信息不泄露给前端
        """
        if self.chain is None:
            return "请先加载文档！"

        # [FIXED] 空问题校验
        if not question or not question.strip():
            return "请输入有效的问题。"

        try:
            result = self.chain.invoke({"question": question})
        except Exception:
            logger.exception("问答调用失败")
            return GENERIC_ERROR_MSG

        answer = result.get("answer", "未找到相关答案")
        sources = result.get("source_documents", [])

        if sources:
            source_parts = ["\n\n**参考来源:**"]
            for i, doc in enumerate(sources):
                content = doc.page_content
                preview = content[:100] if len(content) > 100 else content
                source_parts.append(f"片段{i + 1}：...{preview}...")
            return answer + "\n".join(source_parts)

        return answer

    def reset(self) -> str:
        """重置会话"""
        self.vector_store = None
        self.chain = None
        self.memory.clear()
        return "已重置，请重新上传文档。"
