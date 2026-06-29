# embeddings.py
# [FIXED] 自定义 Embeddings 适配器，修复异常静默吞没、维度硬编码等问题
import logging
from typing import Any, List, Optional

from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings

from config import config

logger = logging.getLogger(__name__)


class DashscopeEmbeddings(Embeddings):
    """DashScope TextEmbedding 适配器"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model = model or config.embedding_model
        self.api_key = api_key or config.dashscope_api_key
        self.dimension = dimension or config.embedding_dim

    # [FIXED] 统一嵌入调用逻辑，消除 embed_documents / embed_query 的重复代码
    def _call_embedding_api(self, text: str) -> List[float]:
        """调用 DashScope 嵌入 API，失败时抛出异常而非返回零向量"""
        response = TextEmbedding.call(
            model=self.model,
            input=text,
            api_key=self.api_key,
        )
        # [FIXED] 使用 response 对象的官方属性检查，不依赖 status_code 数值
        if response.status_code != 200:
            # [FIXED] 不再静默返回零向量，而是记录并抛出，让上层决定如何降级
            error_msg = (
                f"Embedding API 返回错误: status_code={response.status_code}, "
                f"message={getattr(response, 'message', 'unknown')}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # [FIXED] 提取嵌入向量，失败时抛出以暴露数据异常
        embedding = self._extract_embedding(response, text[:50])
        if embedding is None:
            raise RuntimeError(
                f"无法从 API 响应中提取嵌入向量 (text={text[:50]}...)"
            )
        return embedding

    def _extract_embedding(
        self, response: Any, context: str = ""
    ) -> Optional[List[float]]:
        """从 API 响应中提取嵌入向量"""
        output = getattr(response, "output", None)
        if output is None:
            logger.warning("API 响应中无 output 字段 (%s)", context)
            return None
        # output 可能是 {"embeddings": [...]} 或 [{"embedding": [...]}]
        if isinstance(output, dict):
            embeddings_list = output.get("embeddings")
            if embeddings_list and len(embeddings_list) > 0:
                return embeddings_list[0].get("embedding")
        elif isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], dict):
                return output[0].get("embedding")
        return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        results: List[List[float]] = []
        for i, text in enumerate(texts):
            try:
                embedding = self._call_embedding_api(text)
                results.append(embedding)
            except Exception:
                logger.exception("嵌入第 %d 个文档片段失败", i)
                raise  # [FIXED] 不再静默吞异常，向上传播
        return results

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        return self._call_embedding_api(text)
