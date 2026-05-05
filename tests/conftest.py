"""pytest 全局配置：把项目根目录加入 sys.path
---------------------------------------------------------
本项目把所有源代码放在 ``app/`` 包下，但仓库根目录没有 setup.py / pyproject.toml，
直接在仓库根跑 ``pytest`` 时，``app`` 这个顶级包不会自动被解析进 sys.path。
此处显式把仓库根插入到 sys.path[0]，让 ``from app.xxx import ...`` 在
所有 test_*.py 里都能直接生效，避免在每个测试文件里重复 sys.path hack。
"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/ 与 app/ 是平级目录，仓库根 = tests 父目录。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
