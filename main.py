# main.py
# [FIXED] 入口文件，统一启动逻辑
# [FIXED] 启动前校验配置，避免 API Key 缺失时静默失败
# [FIXED] 启动后台清理线程，自动清理超时会话
import logging
import sys
import threading
import time

# [FIXED] 优先从 .env 文件加载环境变量，再使用系统环境变量
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv 可选依赖

from config import config
from session_manager import SessionManager
from ui import create_ui

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def start_cleanup_thread(manager: SessionManager, interval: int) -> None:
    """启动后台清理线程，定期清理超时会话"""

    def cleanup_loop() -> None:
        while True:
            time.sleep(interval)
            count = manager.cleanup_expired()
            if count > 0:
                logger.info(
                    "会话清理完成，清理: %d，当前活跃: %d",
                    count,
                    manager.active_count,
                )

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info("会话清理线程已启动，间隔: %d 秒", interval)


def main() -> None:
    # [FIXED] 启动前强制校验 API Key
    try:
        config.validate()
    except ValueError as e:
        logger.error("配置校验失败: %s", e)
        print(f"\n❌ 启动失败: {e}\n", file=sys.stderr)
        sys.exit(1)

    # 启动会话清理线程
    from ui import session_manager

    start_cleanup_thread(session_manager, config.cleanup_interval)

    logger.info("正在启动 RAG 文档问答助手...")
    logger.info(
        "会话配置 - 最大并发: %d，超时时间: %d 秒",
        config.max_sessions,
        config.session_timeout,
    )
    demo = create_ui()
    demo.launch(
        share=config.share,
        server_name=config.server_name,
        server_port=config.server_port,
        # [FIXED] 建议生产环境启用身份验证，取消下面一行的注释并设置凭据：
        # auth=("admin", "your_password"),
    )


if __name__ == "__main__":
    main()
