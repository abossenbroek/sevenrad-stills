"""Built-in lint rules for td-linter."""

from td_linter.rules.builtin.connection import NoDanglingInputsRule, NoInvalidCyclesRule
from td_linter.rules.builtin.glsl import (
    GLSLNoVersionRule,
    GLSLSyntaxRule,
    GLSLTDOutputRule,
)
from td_linter.rules.builtin.performance import (
    CookEveryFrameRule,
    DeepNestingRule,
    ExcessiveInputsRule,
    HeavyTextureChainsRule,
    UnoptimizedFeedbackRule,
)
from td_linter.rules.builtin.python_rules import (
    PythonSyntaxRule,
    PythonUndefinedNameRule,
    TDExecuteDatCallbacksRule,
)
from td_linter.rules.builtin.reference import (
    ValidOperatorReferencesRule,
    ValidPathReferencesRule,
)
from td_linter.rules.builtin.syntax import (
    TocCompletenessRule,
    ValidNFileSyntaxRule,
    ValidParmFileSyntaxRule,
)
from td_linter.rules.builtin.type_rules import TypeCompatibilityRule

__all__ = [
    "CookEveryFrameRule",
    "DeepNestingRule",
    "ExcessiveInputsRule",
    "GLSLNoVersionRule",
    "GLSLSyntaxRule",
    "GLSLTDOutputRule",
    "HeavyTextureChainsRule",
    "NoDanglingInputsRule",
    "NoInvalidCyclesRule",
    "PythonSyntaxRule",
    "PythonUndefinedNameRule",
    "TDExecuteDatCallbacksRule",
    "TocCompletenessRule",
    "TypeCompatibilityRule",
    "UnoptimizedFeedbackRule",
    "ValidNFileSyntaxRule",
    "ValidOperatorReferencesRule",
    "ValidParmFileSyntaxRule",
    "ValidPathReferencesRule",
]


def get_all_builtin_rules() -> list[type]:
    """Return all built-in rule classes."""
    return [
        # Syntax (S)
        ValidNFileSyntaxRule,
        ValidParmFileSyntaxRule,
        TocCompletenessRule,
        # Connection (C)
        NoInvalidCyclesRule,
        NoDanglingInputsRule,
        # Type (T)
        TypeCompatibilityRule,
        # Reference (R)
        ValidOperatorReferencesRule,
        ValidPathReferencesRule,
        # GLSL (G)
        GLSLSyntaxRule,
        GLSLNoVersionRule,
        GLSLTDOutputRule,
        # Python (P)
        PythonSyntaxRule,
        PythonUndefinedNameRule,
        TDExecuteDatCallbacksRule,
        # Performance (F)
        DeepNestingRule,
        ExcessiveInputsRule,
        HeavyTextureChainsRule,
        UnoptimizedFeedbackRule,
        CookEveryFrameRule,
    ]
