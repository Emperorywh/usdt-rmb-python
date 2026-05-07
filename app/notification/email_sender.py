"""邮件通知发送器（基于 Resend HTTP API + HTML 模板）。

设计目标
========
* 当 LLM 给出明确方向（``bias`` ∈ {long, short}）且本轮为真实 LLM 调用
  （``from_cache=False``）时，向 ``notification_emails`` 表中所有
  ``enabled=TRUE`` 的邮箱推送一封 HTML 提醒。
* 观望（neutral）以及节流缓存命中的判断不发邮件，避免噪声。
* 保持发送链路与信号生成主路径解耦：
    * 整个发送过程在 asyncio 后台任务里跑，失败只打日志，不会反向阻塞
      ``SignalService.generate``；
    * Resend SDK 的 ``Emails.send`` 是同步阻塞 HTTP 调用，通过
      ``asyncio.to_thread`` 丢到默认 executor 上执行，不会拖慢事件循环。

实现要点
========
* 不再依赖 ``smtplib``：直接用 Resend 官方 Python SDK（``resend`` 包）走
  HTTPS REST，规避国内云主机出口 25/465/587 端口被封禁、STARTTLS 抖动
  等典型 SMTP 问题，且天然支持模板/退信回执等更高级特性。
* 模块级 ``resend.api_key`` 在 EmailSender 构造时统一设置一次；后续每个
  请求复用同一份配置。
* HTML 模板使用 inline CSS（很多邮件客户端会 strip ``<style>``），
  并提供纯文本备选（``text`` 字段）以兼容粗暴的纯文本客户端。
* 价位 / 仓位等数值统一格式化为 4 位小数 / 百分比，避免 Decimal 直拼。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from html import escape
from typing import Any, Dict, List, Optional, Sequence

import resend

from app.config import Settings
from app.logging_config import get_logger
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


def _format_resend_error(exc: BaseException) -> str:
    """
    把 Resend SDK / requests 抛出的异常压成一行可读的错误描述
    --------------------------------------------------------------
    参数：
        exc : 任意异常对象
    返回：
        形如 "ResendError(403): The mail.imywh.com domain is not verified"
        这样的单行字符串；普通异常退化为 "ClassName: str(exc)"。
    说明：
        Resend Python SDK 的异常对象通常带有 ``message`` / ``error_type`` /
        ``code`` / ``status_code`` 等属性，但同一异常类在不同版本里字段名
        略有差异，这里用 getattr 做兼容性提取，避免出现 "ApiError()" 这种
        看不出原因的日志。
    """
    cls_name = type(exc).__name__
    parts: list[str] = []
    status_code = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
    )
    if status_code is not None:
        parts.append(f"status={status_code}")
    error_type = getattr(exc, "error_type", None) or getattr(exc, "name", None)
    if error_type:
        parts.append(f"type={error_type}")
    message = (
        getattr(exc, "message", None)
        or getattr(exc, "detail", None)
        or str(exc)
    )
    if message:
        parts.append(f"msg={message}")
    suffix = ", ".join(p for p in parts if p)
    return f"{cls_name}({suffix})" if suffix else cls_name


# ----------------------------------------------------------------------
# 字面量映射 / 工具函数
# ----------------------------------------------------------------------
# 方向偏置 → 中文 + 主题色（HTML 内联样式用）
_BIAS_LABEL: Dict[str, str] = {
    "long": "做多",
    "short": "做空",
    "neutral": "观望",
}
_BIAS_COLOR: Dict[str, str] = {
    "long": "#16a34a",
    "short": "#dc2626",
    "neutral": "#64748b",
}
_BIAS_BG: Dict[str, str] = {
    "long": "#ecfdf5",
    "short": "#fef2f2",
    "neutral": "#f1f5f9",
}


def _confidence_label(conf: Optional[float]) -> str:
    """
    把 [0, 1] 置信度映射为中文档位
    --------------------------------------------------------------
    参数：
        conf : 置信度（None 时返回 '未知'）
    返回：
        '高' (≥0.7) / '中' (≥0.4) / '低' / '未知'
    """
    if conf is None:
        return "未知"
    if conf >= 0.7:
        return "高"
    if conf >= 0.4:
        return "中"
    return "低"


def _fmt_price(v: Any, digits: int = 4) -> str:
    """
    把价格 / 数值安全转成保留指定小数的字符串
    --------------------------------------------------------------
    参数：
        v      : 任意值
        digits : 小数位数
    返回：
        格式化后的字符串；None / 不可转时返回 '-'。
    """
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v: Any, digits: int = 2) -> str:
    """
    把 [0, 1] 区间的百分比小数格式化为带 % 的字符串
    --------------------------------------------------------------
    参数：
        v      : float（如 0.0875 表示 8.75%）
        digits : 小数位数
    返回：
        格式化字符串；None / 不可转时返回 '-'。
    """
    if v is None:
        return "-"
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(v)


def _now_cn_str(dt: Optional[datetime] = None) -> str:
    """
    渲染一个对中文用户友好的时间戳（UTC+8）
    --------------------------------------------------------------
    参数：
        dt : 任意 datetime；None 表示取当前 UTC
    返回：
        如 '2026-05-04 22:01:35 (UTC+8)' 的字符串
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    cn = dt.astimezone(tz=None)
    try:
        from datetime import timedelta, timezone as _tz

        cn = dt.astimezone(_tz(timedelta(hours=8)))
    except Exception:  # noqa: BLE001
        pass
    return cn.strftime("%Y-%m-%d %H:%M:%S") + " (UTC+8)"


# ----------------------------------------------------------------------
# 主题 / HTML 渲染
# ----------------------------------------------------------------------
def render_signal_subject(symbol: str, signal: TradingSignal) -> str:
    """
    生成邮件主题（中文 + emoji）
    --------------------------------------------------------------
    参数：
        symbol : 合约代码
        signal : TradingSignal
    返回：
        如 '【ETH 量化提醒】ETH-USDT-SWAP 做多信号 · 置信度高（85%）'
    """
    bias = signal.bias
    bias_label = _BIAS_LABEL.get(bias, bias)
    conf_label = _confidence_label(signal.confidence)
    conf_pct = _fmt_pct(signal.confidence, digits=0)
    return f"【ETH 量化提醒】{symbol} {bias_label}信号 · 置信度{conf_label}({conf_pct})"


def render_signal_html(
    *,
    symbol: str,
    signal: TradingSignal,
    rule_score: Optional[float] = None,
    factors: Optional[Dict[str, Any]] = None,
    signal_id: Optional[int] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """
    把交易信号渲染成 HTML 邮件正文
    --------------------------------------------------------------
    参数：
        symbol       : 合约代码
        signal       : 完整 TradingSignal（含交易计划）
        rule_score   : 规则引擎打分，可空
        factors      : 因子聚合 dict（用于摘要 regime / current_price）
        signal_id    : signals.id（可空，用于在邮件页脚展示）
        generated_at : 信号生成时间（默认 = now UTC）
    返回：
        完整的 HTML 字符串
    说明：
        - 使用 inline CSS：很多邮件客户端（Outlook / 网易邮箱）会 strip 掉
          ``<style>`` 块里的样式；
        - 不引入外链图片 / 资源，避免被某些邮件客户端拦截或显示破图；
        - 颜色按多空方向自动切换，主体卡片含核心交易计划要点。
    """
    bias = signal.bias
    bias_label = _BIAS_LABEL.get(bias, bias)
    bias_color = _BIAS_COLOR.get(bias, "#0f172a")
    bias_bg = _BIAS_BG.get(bias, "#f1f5f9")

    confidence_pct = _fmt_pct(signal.confidence, digits=0)
    confidence_label = _confidence_label(signal.confidence)

    factors = factors or {}
    regime = factors.get("regime") if isinstance(factors, dict) else None
    liquidity = factors.get("liquidity") if isinstance(factors, dict) else {}
    current_price = (
        liquidity.get("current_price") if isinstance(liquidity, dict) else None
    )

    # 交易计划
    entry_zone = signal.entry_zone
    entry_low = entry_zone[0] if entry_zone else None
    entry_high = entry_zone[1] if entry_zone else None
    take_profit = list(signal.take_profit or [])
    tp_rows_html = "".join(
        f"""
        <tr>
            <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#475569;">止盈 {idx}</td>
            <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;color:#0f172a;">
                {_fmt_price(tp)}
            </td>
        </tr>
        """
        for idx, tp in enumerate(take_profit, start=1)
    )

    # 多周期方向
    tf_order = ["5m", "15m", "1h", "4h", "1d"]
    tfa = signal.timeframe_alignment or {}
    tf_cells = []
    for tf in tf_order:
        b = tfa.get(tf) or "-"
        b_label = _BIAS_LABEL.get(b, b)
        b_color = _BIAS_COLOR.get(b, "#64748b")
        tf_cells.append(
            f"""
            <td style="padding:8px 4px;border:1px solid #e2e8f0;text-align:center;
                       font-size:13px;color:{b_color};font-weight:600;">
                <div style="color:#94a3b8;font-weight:400;font-size:11px;">{tf}</div>
                {escape(b_label)}
            </td>
            """
        )
    tf_row_html = "".join(tf_cells)

    # 失效条件
    invalidation_items = "".join(
        f"<li style=\"margin:4px 0;color:#475569;line-height:1.6;\">{escape(str(item))}</li>"
        for item in (signal.invalidation_conditions or [])
    )
    if not invalidation_items:
        invalidation_items = (
            "<li style=\"color:#94a3b8;\">（LLM 未返回量化失效条件）</li>"
        )

    generated_at_str = _now_cn_str(generated_at)

    reason_html = escape(signal.reason or "").replace("\n", "<br>")
    risk_html = escape(signal.risk or "").replace("\n", "<br>")
    suggestion_html = escape(signal.suggestion or "").replace("\n", "<br>")

    rule_score_str = (
        f"{float(rule_score):+.4f}" if rule_score is not None else "-"
    )
    regime_str = str(regime) if regime else "-"
    current_price_str = _fmt_price(current_price)
    rr_str = _fmt_price(signal.risk_reward_ratio, digits=2)
    pos_str = _fmt_pct(signal.position_size_pct, digits=2)
    signal_id_str = str(signal_id) if signal_id is not None else "-"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ETH 量化交易提醒</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',Helvetica,Arial,sans-serif;color:#0f172a;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background:#f1f5f9;padding:24px 0;">
    <tr><td align="center">
      <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="640" style="max-width:640px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(15,23,42,0.06);">
        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);padding:24px 28px;color:#f8fafc;">
            <div style="font-size:13px;letter-spacing:1px;color:#94a3b8;">ETH 量化交易系统 · LLM 信号提醒</div>
            <div style="font-size:22px;font-weight:700;margin-top:6px;">
              {escape(symbol)} <span style="color:{bias_color};">{escape(bias_label)}</span>
            </div>
            <div style="font-size:13px;color:#cbd5e1;margin-top:4px;">
              生成时间：{escape(generated_at_str)}
            </div>
          </td>
        </tr>

        <!-- 核心摘要卡片 -->
        <tr>
          <td style="padding:24px 28px 8px 28px;">
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-collapse:separate;border-spacing:0;">
              <tr>
                <td style="background:{bias_bg};padding:18px 20px;border-radius:10px;border:1px solid {bias_color}33;">
                  <div style="font-size:12px;color:#64748b;letter-spacing:0.5px;">方向 · 置信度</div>
                  <div style="font-size:24px;font-weight:700;color:{bias_color};margin-top:4px;">
                    {escape(bias_label)} · {confidence_pct}
                  </div>
                  <div style="font-size:13px;color:#475569;margin-top:6px;">
                    置信度档位：<strong>{escape(confidence_label)}</strong>
                    &nbsp;·&nbsp; 当前价：<strong>{current_price_str}</strong>
                    &nbsp;·&nbsp; 市场状态：<strong>{escape(regime_str)}</strong>
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- 交易计划 -->
        <tr>
          <td style="padding:16px 28px 8px 28px;">
            <div style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:10px;">📋 结构化交易计划</div>
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-collapse:collapse;font-size:13px;">
              <tr>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#475569;width:30%;">入场区间</td>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;color:#0f172a;">
                  {_fmt_price(entry_low)} ~ {_fmt_price(entry_high)}
                </td>
              </tr>
              <tr>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#475569;">止损</td>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;color:#dc2626;">
                  {_fmt_price(signal.stop_loss)}
                </td>
              </tr>
              {tp_rows_html}
              <tr>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#475569;">盈亏比 (RR)</td>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;color:#0f172a;">{rr_str}</td>
              </tr>
              <tr>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#475569;">建议仓位</td>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;color:#0f172a;">{pos_str}</td>
              </tr>
              <tr>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#475569;">规则引擎打分</td>
                <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:600;color:#0f172a;">{rule_score_str}</td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- 多周期方向 -->
        <tr>
          <td style="padding:16px 28px 8px 28px;">
            <div style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:10px;">🧭 多周期方向投票</div>
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-collapse:collapse;">
              <tr>{tf_row_html}</tr>
            </table>
          </td>
        </tr>

        <!-- 判断依据 / 风险 -->
        <tr>
          <td style="padding:16px 28px 8px 28px;">
            <div style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:10px;">🧠 判断依据</div>
            <div style="background:#f8fafc;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13px;color:#334155;line-height:1.7;">
              {reason_html or '（LLM 未返回 reason）'}
            </div>
          </td>
        </tr>
        <tr>
          <td style="padding:8px 28px;">
            <div style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:10px;">⚠️ 主要风险</div>
            <div style="background:#fffbeb;border-left:3px solid #f59e0b;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13px;color:#78350f;line-height:1.7;">
              {risk_html or '（LLM 未返回 risk）'}
            </div>
          </td>
        </tr>

        <!-- 失效条件 -->
        <tr>
          <td style="padding:8px 28px;">
            <div style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:10px;">❌ 量化失效条件</div>
            <ul style="margin:0;padding:0 0 0 18px;font-size:13px;">
              {invalidation_items}
            </ul>
          </td>
        </tr>

        <!-- 操作建议 -->
        <tr>
          <td style="padding:8px 28px 24px 28px;">
            <div style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:10px;">💡 操作建议</div>
            <div style="background:#eff6ff;border-left:3px solid #2563eb;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13px;color:#1e3a8a;line-height:1.7;">
              {suggestion_html or '（LLM 未返回 suggestion）'}
            </div>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f8fafc;padding:14px 28px;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:12px;line-height:1.6;">
            信号 ID：{escape(signal_id_str)} &nbsp;|&nbsp; 来源：DeepSeek + 规则引擎多周期共振<br>
            本邮件由系统自动发送，仅供参考，不构成交易指令。请结合自身风险承受能力独立决策。
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def render_signal_text(
    *,
    symbol: str,
    signal: TradingSignal,
    rule_score: Optional[float] = None,
    signal_id: Optional[int] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """
    渲染纯文本版本的邮件正文（兼容粗暴的纯文本邮件客户端）
    --------------------------------------------------------------
    参数：见 render_signal_html
    返回：
        多行纯文本字符串。
    """
    bias_label = _BIAS_LABEL.get(signal.bias, signal.bias)
    conf_label = _confidence_label(signal.confidence)
    conf_pct = _fmt_pct(signal.confidence, digits=0)
    rr_str = _fmt_price(signal.risk_reward_ratio, digits=2)
    pos_str = _fmt_pct(signal.position_size_pct, digits=2)
    entry_low = signal.entry_zone[0] if signal.entry_zone else None
    entry_high = signal.entry_zone[1] if signal.entry_zone else None
    rule_score_str = (
        f"{float(rule_score):+.4f}" if rule_score is not None else "-"
    )

    tp_lines = "\n".join(
        f"  - 止盈{i}：{_fmt_price(tp)}"
        for i, tp in enumerate(signal.take_profit or [], start=1)
    )
    inval_lines = "\n".join(
        f"  - {item}" for item in (signal.invalidation_conditions or [])
    ) or "  -（无）"

    return (
        f"【ETH 量化交易提醒】{symbol} {bias_label}信号\n"
        f"生成时间：{_now_cn_str(generated_at)}\n"
        f"信号 ID：{signal_id if signal_id is not None else '-'}\n"
        f"\n"
        f"== 核心判断 ==\n"
        f"方向：{bias_label}\n"
        f"置信度：{conf_pct}（{conf_label}）\n"
        f"规则引擎打分：{rule_score_str}\n"
        f"\n"
        f"== 交易计划 ==\n"
        f"入场区间：{_fmt_price(entry_low)} ~ {_fmt_price(entry_high)}\n"
        f"止损：{_fmt_price(signal.stop_loss)}\n"
        f"{tp_lines}\n"
        f"盈亏比：{rr_str}\n"
        f"建议仓位：{pos_str}\n"
        f"\n"
        f"== 判断依据 ==\n{signal.reason or '-'}\n"
        f"\n"
        f"== 主要风险 ==\n{signal.risk or '-'}\n"
        f"\n"
        f"== 量化失效条件 ==\n{inval_lines}\n"
        f"\n"
        f"== 操作建议 ==\n{signal.suggestion or '-'}\n"
        f"\n"
        f"本邮件由系统自动发送，仅供参考，不构成交易指令。\n"
    )


# ----------------------------------------------------------------------
# EmailSender
# ----------------------------------------------------------------------
class EmailSender:
    """
    Resend 邮件发送器
    --------------------------------------------------------------
    职责：
        - 加载 Resend 配置（API Key / 发件人地址）；
        - 在事件循环里以非阻塞方式发送 HTML 邮件
          （``resend.Emails.send`` 是同步阻塞 HTTP 调用，
            通过 ``asyncio.to_thread`` 丢到默认 executor 上跑）；
        - 不抛异常：发送失败仅打日志，不影响信号主路径。
    使用：
        sender = EmailSender(settings)
        await sender.send_signal_alert(
            recipients=[...], symbol=..., signal=..., factors=..., ...
        )
    """

    def __init__(self, settings: Settings):
        """
        构造邮件发送器并把 Resend API Key 写入 SDK 模块全局变量
        --------------------------------------------------------------
        参数：
            settings : 全局 Settings；本类只读取
                       ``resend_api_key`` / ``resend_from`` /
                       ``enable_email_notification`` 三个字段。
        说明：
            - Resend SDK 把 ``resend.api_key`` 设计为模块级变量，
              所有调用都会读它。这里在 EmailSender 实例化时统一赋值，
              后续 ``Emails.send`` 直接复用，避免每次发送都重新设置；
            - 若 API Key 留空，仅记录一次 INFO 日志后整个发送链路降级
              为 no-op（``enabled`` 返回 False）。
        """
        self.settings = settings
        api_key = (getattr(settings, "resend_api_key", "") or "").strip()
        if api_key:
            resend.api_key = api_key

    @property
    def enabled(self) -> bool:
        """
        是否启用邮件通知
        --------------------------------------------------------------
        判定依据：
            - 主开关 ``enable_email_notification`` 为 True；
            - ``resend_api_key`` 已配置；
            - ``resend_from`` 已配置（Resend 强制要求 from 字段）。
        三者满足才视为启用，否则整个通知链路降级为 no-op。
        """
        return (
            bool(getattr(self.settings, "enable_email_notification", False))
            and bool((getattr(self.settings, "resend_api_key", "") or "").strip())
            and bool((getattr(self.settings, "resend_from", "") or "").strip())
        )

    def _send_one_blocking(
        self,
        *,
        recipient: str,
        subject: str,
        html_body: str,
        text_body: str,
    ) -> Dict[str, Any]:
        """
        同步发送一封邮件（被 to_thread 丢到 executor 跑）
        --------------------------------------------------------------
        参数：
            recipient : 收件邮箱
            subject   : 邮件主题
            html_body : HTML 正文
            text_body : 纯文本备选正文（多数邮件客户端不会显示，仅作降级）
        返回：
            Resend SDK 返回的 dict（含 id 字段，可用于排错 / 退信查询）。
        异常：
            resend.exceptions.* / requests.RequestException 等任意异常会
            原样抛出，由调用方记日志。
        说明：
            - Resend 的 ``from`` 字段支持两种格式：
                  "onboarding@resend.dev"
                  "ETH 量化 <onboarding@resend.dev>"
              直接把 settings.resend_from 透传即可。
            - SDK 的同步接口会内部调用 ``requests``，不需要自己再起会话；
              当前发送频率 ≤ 每 15 分钟一次，不需要长连接复用。
        """
        from_addr = (self.settings.resend_from or "").strip()
        params: resend.Emails.SendParams = {
            "from": from_addr,
            "to": [recipient],
            "subject": subject,
            "html": html_body,
            "text": text_body,
        }
        try:
            return resend.Emails.send(params)
        except Exception as exc:  # noqa: BLE001
            err_desc = _format_resend_error(exc)
            logger.warning(
                "Resend HTTP API 调用失败：from=%s -> to=%s 错误=%s",
                from_addr,
                recipient,
                err_desc,
                exc_info=True,
            )
            raise RuntimeError(f"Resend API 调用失败：{err_desc}") from exc

    async def send_signal_alert(
        self,
        *,
        recipients: Sequence[str],
        symbol: str,
        signal: TradingSignal,
        rule_score: Optional[float] = None,
        factors: Optional[Dict[str, Any]] = None,
        signal_id: Optional[int] = None,
        generated_at: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        给若干收件人发送一封"明确方向"的交易信号 HTML 邮件
        --------------------------------------------------------------
        参数：
            recipients   ：收件人邮箱列表（已过滤 enabled=TRUE）
            symbol       ：合约代码
            signal       ：完整 TradingSignal
            rule_score   ：规则引擎打分，可空
            factors      ：因子聚合 dict（用于摘要 regime / current_price）
            signal_id    ：signals.id（可空）
            generated_at ：信号生成时间，默认 now UTC
        返回：
            {"sent": N, "failed": M, "skipped": K} 三元统计字典
        说明：
            - 仅当 ``signal.bias`` ∈ {long, short} 且 ``self.enabled`` 时才真正发送；
              否则统计为 skipped，并打 INFO 日志。
            - 任意一封失败只记日志，不打断剩余收件人；与信号生成主路径解耦。
        """
        result: Dict[str, int] = {"sent": 0, "failed": 0, "skipped": 0}

        if not self.enabled:
            logger.info(
                "邮件通知未启用（enable_email_notification 或 RESEND_API_KEY/RESEND_FROM 缺失），跳过发送",
            )
            result["skipped"] = len(list(recipients))
            return result

        if signal.bias not in ("long", "short"):
            logger.debug(
                "信号方向为 %s，按设计不发邮件提醒",
                signal.bias,
            )
            result["skipped"] = len(list(recipients))
            return result

        clean_recipients: List[str] = [
            (r or "").strip() for r in recipients if (r or "").strip()
        ]
        if not clean_recipients:
            logger.info("无可发送的收件人（notification_emails 表空 / 全部 disabled）")
            return result

        subject = render_signal_subject(symbol=symbol, signal=signal)
        html_body = render_signal_html(
            symbol=symbol,
            signal=signal,
            rule_score=rule_score,
            factors=factors,
            signal_id=signal_id,
            generated_at=generated_at,
        )
        text_body = render_signal_text(
            symbol=symbol,
            signal=signal,
            rule_score=rule_score,
            signal_id=signal_id,
            generated_at=generated_at,
        )

        from_addr = (self.settings.resend_from or "").strip()
        for recipient in clean_recipients:
            try:
                resp = await asyncio.to_thread(
                    self._send_one_blocking,
                    recipient=recipient,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                )
                # Resend 同步 SDK 返回的 dict 至少包含 id 字段；这里把它一并打日志
                # 便于在 Resend 控制台按 id 反查投递状态 / 退信明细。
                msg_id = (resp or {}).get("id") if isinstance(resp, dict) else None
                logger.info(
                    "邮件已发送：%s -> %s（symbol=%s，bias=%s，resend_id=%s）",
                    from_addr,
                    recipient,
                    symbol,
                    signal.bias,
                    msg_id,
                )
                result["sent"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "邮件发送失败 -> %s（symbol=%s，bias=%s，from=%s）：%s",
                    recipient,
                    symbol,
                    signal.bias,
                    from_addr,
                    _format_resend_error(exc),
                    exc_info=True,
                )
                result["failed"] += 1

        return result

    async def send_test_email(self, recipient: str) -> Dict[str, Any]:
        """
        发送一封测试邮件（管理员校验 Resend 配置用）
        --------------------------------------------------------------
        参数：
            recipient ：测试收件邮箱
        返回：
            Resend SDK 返回的 dict（含 id 字段）。
        异常：
            发送失败时原样抛出，由路由层捕获并返回 500 / 400。
        """
        if not self.enabled:
            raise RuntimeError(
                "邮件通知未启用（enable_email_notification 或 RESEND_API_KEY/RESEND_FROM 缺失）"
            )
        subject = "【ETH 量化交易系统】Resend 配置测试邮件"
        html_body = (
            "<div style=\"font-family:'PingFang SC','Microsoft YaHei',Arial;color:#0f172a;\">"
            "<h2 style=\"color:#16a34a;\">Resend 配置测试成功 ✅</h2>"
            "<p>如果你看到这封邮件，说明 ETH 量化交易系统的 Resend 通道已经可以正常发件。</p>"
            f"<p>发送时间：{_now_cn_str()}</p>"
            "<p style=\"color:#94a3b8;font-size:12px;\">本邮件由系统自动发送，无需回复。</p>"
            "</div>"
        )
        text_body = (
            "ETH 量化交易系统：Resend 配置测试邮件\n"
            f"发送时间：{_now_cn_str()}\n"
        )
        return await asyncio.to_thread(
            self._send_one_blocking,
            recipient=recipient,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )
