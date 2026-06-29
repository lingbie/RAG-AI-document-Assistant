# ui.py
# [FIXED] UI 层与业务逻辑分离
# [FIXED] 添加身份验证支持
# [FIXED] 支持多用户并发，会话隔离
import gradio as gr

from config import config
from session_manager import SessionManager

# 全局会话管理器（进程级单例）
session_manager = SessionManager(
    timeout_seconds=config.session_timeout,
    max_sessions=config.max_sessions,
)


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    def init_session() -> str:
        """初始化用户会话，生成唯一 session_id"""
        try:
            return session_manager.create_session()
        except RuntimeError as e:
            return ""

    def process_pdf(file_obj, session_id: str) -> str:
        if not session_id:
            return "会话无效，请刷新页面重试。"
        session = session_manager.get_session(session_id)
        if not session:
            return "会话已过期，请刷新页面重试。"
        if file_obj is None:
            return "请先选择一个 PDF 文件！"
        if isinstance(file_obj, dict):
            file_path = file_obj.get("path") or file_obj.get("name", "")
        elif isinstance(file_obj, str):
            file_path = file_obj
        else:
            file_path = getattr(file_obj, "name", "")
        return session.load_document(file_path)

    def chat_with_bot(question: str, history, session_id: str) -> str:
        if not session_id:
            return "会话无效，请刷新页面重试。"
        session = session_manager.get_session(session_id)
        if not session:
            return "会话已过期，请刷新页面重试。"
        return session.chat(question)

    def clear_all(session_id: str) -> tuple:
        if not session_id:
            return "会话无效", None
        session = session_manager.get_session(session_id)
        if not session:
            return "会话已过期", None
        msg = session.reset()
        return msg, None

    with gr.Blocks(title="RAG 本地文档智能问答系统") as demo:
        gr.Markdown("# 📚 RAG 本地文档智能问答系统")
        gr.Markdown("上传 PDF 文档，然后进行智能问答")

        # 隐藏的 session_id 状态
        session_state = gr.State(value="")

        # 页面加载时初始化会话
        demo.load(fn=init_session, outputs=[session_state])

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="上传 PDF 文档", file_types=[".pdf"])
                upload_btn = gr.Button("解析并加载文档", variant="primary")
                status_output = gr.Textbox(label="处理状态", interactive=False)

            with gr.Column(scale=2):
                chatbot = gr.ChatInterface(
                    fn=chat_with_bot,
                    additional_inputs=[session_state],
                    title="问答对话框",
                    description="请先在左侧上传 PDF 文档，然后在此输入您的问题。",
                )

        upload_btn.click(
            fn=process_pdf, inputs=[pdf_input, session_state], outputs=[status_output]
        )

        gr.Button("重置系统").click(
            fn=clear_all, inputs=[session_state], outputs=[status_output, chatbot]
        )

    return demo
