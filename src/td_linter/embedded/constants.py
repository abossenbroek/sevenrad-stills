"""
Shared constants for embedded code validation.

This module contains patterns and builtins used by the language detector
and validators.
"""

import re
from typing import Pattern

# GLSL detection patterns with weights
# Higher weight = stronger indicator of GLSL
GLSL_PATTERNS: dict[Pattern[str], int] = {
    # Standard GLSL types
    re.compile(r"\bvec[234]\b"): 3,
    re.compile(r"\bmat[234]\b"): 3,
    re.compile(r"\bivec[234]\b"): 3,
    re.compile(r"\buvec[234]\b"): 3,
    re.compile(r"\bbvec[234]\b"): 3,
    # Qualifiers
    re.compile(r"\buniform\b"): 2,
    re.compile(r"\bvarying\b"): 2,
    re.compile(r"\battribute\b"): 2,
    re.compile(r"\bin\b\s+\w+"): 1,  # in qualifier (avoid matching Python 'in')
    re.compile(r"\bout\b\s+\w+"): 1,  # out qualifier
    # Samplers
    re.compile(r"\bsampler2D\b"): 3,
    re.compile(r"\bsampler3D\b"): 3,
    re.compile(r"\bsamplerCube\b"): 3,
    re.compile(r"\bsampler2DArray\b"): 3,
    # Built-in variables
    re.compile(r"\bgl_FragColor\b"): 3,
    re.compile(r"\bgl_FragCoord\b"): 3,
    re.compile(r"\bgl_Position\b"): 3,
    re.compile(r"\bfragColor\b"): 2,
    # Layout
    re.compile(r"\blayout\s*\("): 2,
    # Main function (GLSL style)
    re.compile(r"\bvoid\s+main\s*\(\s*\)"): 2,
    # TouchDesigner-specific GLSL
    re.compile(r"\bTDOutputSwizzle\b"): 4,
    re.compile(r"\bsTD2DInputs\b"): 4,
    re.compile(r"\bsTD3DInputs\b"): 4,
    re.compile(r"\bTDImageStoreOutput\b"): 4,
    re.compile(r"\buTDOutputInfo\b"): 3,
    re.compile(r"\buTDPass\b"): 3,
    re.compile(r"\bTDAlphaOfOutput\b"): 3,
}

# Python detection patterns with weights
# Higher weight = stronger indicator of Python
PYTHON_PATTERNS: dict[Pattern[str], int] = {
    # Function/class definitions
    re.compile(r"\bdef\s+\w+\s*\("): 3,
    re.compile(r"\bclass\s+\w+\s*[:\(]"): 3,
    # Imports
    re.compile(r"\bimport\s+\w+"): 3,
    re.compile(r"\bfrom\s+\w+\s+import\b"): 3,
    # Control flow (Python-specific keywords)
    re.compile(r"\belif\b"): 2,  # elif is Python-only
    re.compile(r"\bexcept\b"): 2,
    re.compile(r"\braise\b"): 2,
    re.compile(r"\byield\b"): 2,
    re.compile(r"\bawait\b"): 2,
    re.compile(r"\basync\s+def\b"): 3,
    # Python built-ins
    re.compile(r"\bprint\s*\("): 1,
    re.compile(r"\blen\s*\("): 1,
    re.compile(r"\brange\s*\("): 1,
    # Decorators
    re.compile(r"^@\w+", re.MULTILINE): 2,
    # TouchDesigner-specific Python
    re.compile(r"\bop\s*\("): 4,
    re.compile(r"\bme\."): 4,
    re.compile(r"\bparent\(\)"): 3,
    re.compile(r"\bproject\."): 3,
    re.compile(r"\babsTime\b"): 4,
    re.compile(r"\bpar\."): 3,
    re.compile(r"\bmod\."): 2,
    re.compile(r"\bext\."): 2,
    # TD callbacks
    re.compile(r"\bdef\s+on[A-Z]\w+\s*\("): 3,  # onCook, onPulse, etc.
}

# DAT types that indicate Python content
PYTHON_DAT_TYPES: set[str] = {
    "execute",
    "script",
    "text",  # when contains callback patterns
    "panel",
    "table",  # callbacks
}

# TouchDesigner Python builtins (comprehensive list)
# These should NOT be flagged as undefined
TD_BUILTINS: set[str] = {
    # Core operator access
    "op",
    "ops",
    "me",
    "parent",
    "iop",
    "ipar",
    # Extension and module access
    "mod",
    "ext",
    # Parameter access
    "par",
    "pars",
    # Storage
    "storage",
    "fetch",
    "store",
    # Project/root
    "project",
    "root",
    "ui",
    # Time
    "absTime",
    # System
    "app",
    "sysinfo",
    "monitors",
    # Common functions
    "run",
    "cook",
    "debug",
    "passive",
    "var",
    "vardict",
    # TD modules
    "td",
    "tdu",
    "TDF",
    "TDJSON",
    "TDStoreTools",
    "TDFunctions",
}

# Expression-specific globals (subset for ast.parse(mode='eval'))
# Available in parameter expressions like mode 49/17
EXPRESSION_GLOBALS: set[str] = {
    # Core access
    "me",
    "op",
    "ops",
    "parent",
    # Time
    "absTime",
    # Project
    "project",
    # Utilities
    "tdu",
    "math",
    # Python built-ins commonly used in expressions
    "int",
    "float",
    "str",
    "bool",
    "abs",
    "min",
    "max",
    "pow",
    "round",
    "floor",
    "ceil",
    # Math functions (from math module, often available)
    "sin",
    "cos",
    "tan",
    "sqrt",
    "exp",
    "log",
    "log10",
    "pi",
}

# Standard Execute DAT callbacks
EXECUTE_CALLBACKS: set[str] = {
    "onStart",
    "onCreate",
    "onExit",
    "onFrameStart",
    "onFrameEnd",
    "onPlayStateChange",
    "onDeviceChange",
    "onProjectPreSave",
    "onProjectPostSave",
}

# Threshold for language detection scoring
# If |glsl_score - python_score| < threshold, return UNKNOWN
DETECTION_THRESHOLD: int = 2
