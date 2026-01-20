"""
Embedded code validation for TouchDesigner .text files.

This module provides validators for GLSL shaders and Python scripts
embedded in TouchDesigner project files.
"""

from td_linter.embedded.constants import (
    DETECTION_THRESHOLD,
    EXECUTE_CALLBACKS,
    EXPRESSION_GLOBALS,
    GLSL_PATTERNS,
    PYTHON_DAT_TYPES,
    PYTHON_PATTERNS,
    TD_BUILTINS,
)
from td_linter.embedded.expression_validator import (
    EXPRESSION_MODE,
    STRING_EXPRESSION_MODE,
    Expression,
    ExpressionValidator,
)
from td_linter.embedded.glsl_validator import (
    GLSLError,
    GLSLValidator,
    ShaderType,
)
from td_linter.embedded.language_detector import (
    DetectionContext,
    DetectionResult,
    Language,
    LanguageDetector,
)
from td_linter.embedded.python_validator import (
    PythonValidationResult,
    PythonValidator,
    UndefinedName,
)

__all__ = [
    "DETECTION_THRESHOLD",
    "EXECUTE_CALLBACKS",
    "EXPRESSION_GLOBALS",
    "EXPRESSION_MODE",
    "GLSL_PATTERNS",
    "PYTHON_DAT_TYPES",
    "PYTHON_PATTERNS",
    "STRING_EXPRESSION_MODE",
    "TD_BUILTINS",
    "DetectionContext",
    "DetectionResult",
    "Expression",
    "ExpressionValidator",
    "GLSLError",
    "GLSLValidator",
    "Language",
    "LanguageDetector",
    "PythonValidationResult",
    "PythonValidator",
    "ShaderType",
    "UndefinedName",
]
