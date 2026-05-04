"""邮件通知子模块。

主要导出：
    - EmailSender   ：负责通过 SMTP 推送 HTML 邮件
    - render_signal_html / render_signal_subject
                    ：根据交易信号渲染邮件主题与正文
"""
from app.notification.email_sender import (
    EmailSender,
    render_signal_html,
    render_signal_subject,
)

__all__ = [
    "EmailSender",
    "render_signal_html",
    "render_signal_subject",
]
