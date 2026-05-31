"""Backward-compatible re-export.

The original ``Repositories`` class has been split into per-table repo modules
under ``app.data_storage.repos/``.  This file re-exports ``Repositories``
so that existing ``from app.data_storage.repositories import Repositories``
imports continue to work without any changes.
"""
from __future__ import annotations

from app.data_storage.repos import Repositories  # noqa: F401

__all__ = ["Repositories"]
