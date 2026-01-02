"""Validators for different shader languages, C externals, and Max help patchers.

This module provides lazy imports for validators that may have heavy dependencies
(like clangd, pylsp_jsonrpc). The maxhelp subpackage is always available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Always export the maxhelp subpackage
from max_linter.validators import maxhelp

if TYPE_CHECKING:
    from max_linter.validators.c_semantic import CSemanticValidator
    from max_linter.validators.clangd import ClangdValidator
    from max_linter.validators.glsl import GLSLValidator
    from max_linter.validators.maxhelp_validator import MaxhelpValidator


def __getattr__(name: str) -> type:
    """Lazy import validators to avoid loading heavy dependencies."""
    if name == "CSemanticValidator":
        from max_linter.validators.c_semantic import CSemanticValidator

        return CSemanticValidator
    if name == "ClangdValidator":
        from max_linter.validators.clangd import ClangdValidator

        return ClangdValidator
    if name == "GLSLValidator":
        from max_linter.validators.glsl import GLSLValidator

        return GLSLValidator
    if name == "MaxhelpValidator":
        try:
            from max_linter.validators.maxhelp_validator import MaxhelpValidator

            return MaxhelpValidator
        except ImportError as err:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from err
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CSemanticValidator",
    "ClangdValidator",
    "GLSLValidator",
    "MaxhelpValidator",
    "maxhelp",
]
