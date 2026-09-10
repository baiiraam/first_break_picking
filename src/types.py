# file location: src/types.py

"""
Shared type definitions for the project.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ✅ Use loguru.Logger type (exists in type stubs)
    from loguru import Logger as LoggerType
else:
    # ✅ At runtime, use Any to avoid import issues
    from typing import Any

    LoggerType = Any

__all__ = ["LoggerType"]
