# config.py
# [FIXED] 集中管理所有配置，消除散落各处的魔法数字
# [FIXED] API Key 从环境变量读取，不再硬编码
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """RAG 系统配置"""

    # [FIXED] API Key 只能通过环境变量或传入方式设置，禁止硬编码
    dashscope_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY")
    )

    # Embedding 模型配置
    embedding_model: str = "text-embedding-v1"
    embedding_dim: int = 1024  # text-embedding-v1 输出维度

    # LLM 模型配置
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # 文档分割配置
    chunk_size: int = 500
    chunk_overlap: int = 50

    # 检索配置
    retrieval_k: int = 3

    # 会话管理配置
    session_timeout: int = 3600  # 会话超时时间（秒），默认1小时
    max_sessions: int = 10  # 最大并发会话数
    cleanup_interval: int = 300  # 清理线程执行间隔（秒），默认5分钟

    # Gradio 服务配置
    server_name: str = "127.0.0.1"
    server_port: int = 7862
    share: bool = False

    def validate(self) -> None:
        """启动前校验必须的配置"""
        if not self.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置。请在环境变量中设置，或创建 .env 文件。\n"
                "示例: DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx"
            )


# 全局配置实例
config = RAGConfig()
