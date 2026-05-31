"""公共工具函数。

提供全局复用的安全转换、格式化等基础工具。
"""
from __future__ import annotations

import math
from typing import Optional


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """统一的 float 安全转换。

    处理 None / str / NaN / Inf 等所有边界情况。

    Args:
        value: 任意输入值
        default: 转换失败时返回的默认值 (默认 None)

    Returns:
        float 或 default
    """
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result
