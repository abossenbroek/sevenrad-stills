"""Validators for different shader languages and C externals."""

from max_linter.validators.c_semantic import CSemanticValidator
from max_linter.validators.clangd import ClangdValidator
from max_linter.validators.glsl import GLSLValidator

# MaxhelpValidator is imported from the standalone file, not the package
# This will be updated once the package migration is complete
try:
    from max_linter.validators.maxhelp_validator import MaxhelpValidator
except ImportError:
    # Fallback: import from old location during migration
    MaxhelpValidator = None  # type: ignore[misc, assignment]

__all__ = [
    "CSemanticValidator",
    "ClangdValidator",
    "GLSLValidator",
    "MaxhelpValidator",
]
