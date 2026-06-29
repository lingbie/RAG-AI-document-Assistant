# session_manager.py
# 多会话管理器，确保用户间的会话隔离
import logging
import threading
import time
import uuid
from typing import Dict, Optional

from rag_engine import RAGSession

logger = logging.getLogger(__name__)


class SessionManager:
    """管理多用户的 RAGSession 实例，确保会话隔离

    功能：
    - 为每个用户创建独立的 RAGSession
    - 超时自动清理
    - 并发访问保护
    - 达到上限时拒绝新会话
    """

    def __init__(
        self,
        timeout_seconds: int = 3600,
        max_sessions: int = 10,
    ):
        """
        Args:
            timeout_seconds: 会话超时时间（秒），默认1小时
            max_sessions: 最大并发会话数，默认10
        """
        self._sessions: Dict[str, RAGSession] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._timeout = timeout_seconds
        self._max_sessions = max_sessions

    def create_session(self) -> str:
        """创建新会话，返回 session_id

        Raises:
            RuntimeError: 当达到最大并发会话数时
        """
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                logger.warning(
                    "达到最大并发会话数 %d，拒绝创建新会话",
                    self._max_sessions,
                )
                raise RuntimeError("系统繁忙，请稍后再试")

            session_id = str(uuid.uuid4())
            self._sessions[session_id] = RAGSession()
            self._last_access[session_id] = time.time()
            logger.info(
                "创建新会话: %s，当前活跃会话数: %d",
                session_id[:8],
                len(self._sessions),
            )
            return session_id

    def get_session(self, session_id: str) -> Optional[RAGSession]:
        """获取指定会话，不存在则返回 None"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                self._last_access[session_id] = time.time()
            return session

    def remove_session(self, session_id: str) -> None:
        """移除指定会话"""
        with self._lock:
            removed = self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)
            if removed:
                logger.info(
                    "移除会话: %s，当前活跃会话数: %d",
                    session_id[:8],
                    len(self._sessions),
                )

    def cleanup_expired(self) -> int:
        """清理超时会话，返回清理数量"""
        now = time.time()
        expired = []
        with self._lock:
            for sid, last_time in self._last_access.items():
                if now - last_time > self._timeout:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
                del self._last_access[sid]
            if expired:
                logger.info(
                    "清理了 %d 个超时会话，当前活跃会话数: %d",
                    len(expired),
                    len(self._sessions),
                )
        return len(expired)

    @property
    def active_count(self) -> int:
        """当前活跃会话数"""
        with self._lock:
            return len(self._sessions)
