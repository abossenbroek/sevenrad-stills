"""GLSL rules (G): Shader code validation."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, OptionValue, Violation


class GLSLSyntaxRule(LintRule):
    """
    G001: Validate GLSL shader syntax.

    Checks that GLSL code in .text files has valid syntax using
    the glslangValidator tool.

    Note: This rule operates on embedded code validation results
    stored in graph metadata from the linting pipeline.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "G001"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "glsl-syntax"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate GLSL shader syntax"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for GLSL syntax errors."""
        # Check for GLSL errors stored in graph metadata
        glsl_errors = graph.graph.get("glsl_errors", [])
        for error in glsl_errors:
            yield Violation(
                rule=self.rule_id,
                message=f"GLSL syntax error: {error.get('message', 'unknown error')}",
                path=error.get("path", "unknown"),
                severity=self.severity,
                source_file=error.get("source_file"),
                line=error.get("line"),
                context={"glsl_error": error},
            )


class GLSLNoVersionRule(LintRule):
    """
    G002: Detect missing #version directive in GLSL shaders.

    GLSL shaders should include a #version directive to ensure
    consistent behavior across different graphics drivers.

    Note: TouchDesigner typically injects version directives, so
    this rule may produce false positives.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "G002"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "glsl-no-version"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect missing #version directive in GLSL"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for missing #version directives."""
        # Check for version warnings stored in graph metadata
        version_warnings = graph.graph.get("glsl_version_warnings", [])
        for warning in version_warnings:
            yield Violation(
                rule=self.rule_id,
                message="GLSL shader missing #version directive",
                path=warning.get("path", "unknown"),
                severity=self.severity,
                source_file=warning.get("source_file"),
                line=1,
                context={"shader_type": warning.get("shader_type")},
            )


class GLSLTDOutputRule(LintRule):
    """
    G003: Check for TouchDesigner output requirements.

    Validates that fragment shaders properly use TouchDesigner's
    output mechanism (TDOutputSwizzle, fragColor, etc.).

    Note: Placeholder for future implementation.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "G003"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "glsl-td-output"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Check for proper TD shader output usage"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:  # noqa: ARG002
        """Check for TD output requirements."""
        # Placeholder: Full implementation requires analyzing shader code
        # for proper output usage
        return iter(())
