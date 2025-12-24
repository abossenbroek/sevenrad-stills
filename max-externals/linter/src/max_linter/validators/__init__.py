"""Validators for different shader languages."""

from max_linter.validators.clangd import ClangdValidator
from max_linter.validators.glsl import GLSLValidator

__all__ = ["GLSLValidator", "ClangdValidator"]
