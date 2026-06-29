# llm.py
# [FIXED] 自定义 LLM 适配器，补充 _acall 异步支持，修复字段声明方式
import logging
from typing import Any, List, Mapping, Optional

from dashscope import Generation
from langchain_core.language_models import LLM

from config import config

logger = logging.getLogger(__name__)


class ChatTongyi(LLM):
    """DashScope 通义千问 LLM 适配器"""

    # [FIXED] Pydantic 字段声明方式，支持默认值且可被实例化覆盖
    model: str = config.llm_model
    temperature: float = config.llm_temperature
    max_tokens: int = config.llm_max_tokens
    api_key: str = ""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model or config.llm_model,
            temperature=(
                temperature if temperature is not None else config.llm_temperature
            ),
            max_tokens=max_tokens or config.llm_max_tokens,
            api_key=api_key or config.dashscope_api_key or "",
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "tongyi"

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """同步调用"""
        response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )
        if response.status_code == 200:
            output = response.output
            if isinstance(output, dict):
                return output.get("text", "")
            return str(output)
        logger.error(
            "LLM API 返回错误: status_code=%s, message=%s",
            response.status_code,
            getattr(response, "message", "unknown"),
        )
        # [FIXED] 不再返回中性错误消息误导用户，改为抛出明确异常
        raise RuntimeError(
            f"LLM 调用失败 (status_code={response.status_code})"
        )

    # [FIXED] 补充 _acall 异步实现，确保在异步链中可用
    async def _acall(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """异步调用（DashScope SDK 暂不支持原生 async，使用同步调用）"""
        return self._call(prompt, stop=stop, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
