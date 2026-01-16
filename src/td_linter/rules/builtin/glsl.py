"""GLSL rules (G): Shader code validation."""

from pathlib import Path
from typing import Iterator

import networkx as nx

from td_linter.rules.base import Fix, LintRule, OptionValue, Replacement, Violation


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
    G002: Detect #version directive in GLSL shaders for TouchDesigner.

    TouchDesigner automatically injects version directives, so user-supplied
    #version lines will conflict and should be removed.

    This rule is fixable - it can automatically remove #version directives.
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
        return "Detect #version directive that conflicts with TouchDesigner"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    @property
    def fixable(self) -> bool:
        """Return whether this rule can auto-fix violations."""
        return True

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for #version directives that should be removed."""
        # Check for version issues stored in graph metadata
        version_issues = graph.graph.get("glsl_version_issues", [])
        for issue in version_issues:
            source_file = issue.get("source_file")
            line_num = issue.get("line", 1)

            # Create fix to remove the version line
            fix = None
            if source_file:
                fix = Fix(
                    description=f"Remove #version directive from {Path(source_file).name}",
                    replacements=[
                        Replacement(
                            file_path=Path(source_file),
                            start_line=line_num,
                            end_line=line_num,
                            new_text="",  # Delete the line
                        )
                    ],
                )

            yield Violation(
                rule=self.rule_id,
                message="GLSL shader contains #version directive (conflicts with TD)",
                path=issue.get("path", "unknown"),
                severity=self.severity,
                source_file=source_file,
                line=line_num,
                context={"shader_type": issue.get("shader_type")},
                fix=fix,
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
