"""Main linter orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from td_linter.embedded import (
    ExpressionValidator,
    GLSLValidator,
    Language,
    LanguageDetector,
    PythonValidator,
)
from td_linter.graph.builder import NetworkGraphBuilder
from td_linter.parsers import NFileParser, ParmFileParser, TocParser
from td_linter.rules.base import LintRule, Violation
from td_linter.rules.builtin import (
    NoDanglingInputsRule,
    NoInvalidCyclesRule,
    TypeCompatibilityRule,
    ValidOperatorReferencesRule,
)

if TYPE_CHECKING:
    import networkx as nx

    from td_linter.rules.loader import LintConfig


def get_all_rules() -> list[LintRule]:
    """Return all available lint rules (legacy compatibility)."""
    return [
        NoInvalidCyclesRule(),
        NoDanglingInputsRule(),
        ValidOperatorReferencesRule(),
        TypeCompatibilityRule(),
    ]


def _validate_toc_entries(
    toe_dir: Path,
    toc_path: Path,
    toc_entries: list[str],
) -> list[Violation]:
    """Validate that all TOC entries exist in the project."""
    violations: list[Violation] = []
    for entry in toc_entries:
        if entry.startswith("."):
            continue  # Skip special entries
        entry_path = toe_dir / entry
        if not entry_path.exists():
            violations.append(
                Violation(
                    rule="S003",  # toc-completeness
                    message=f"TOC entry does not exist: {entry}",
                    path=entry,
                    severity="error",
                    source_file=toc_path,
                )
            )
    return violations


def _run_graph_rules(
    graph: "nx.DiGraph",
    rules: list[LintRule],
    config: "LintConfig | None",
) -> list[Violation]:
    """Run graph-based rules and collect violations."""
    violations: list[Violation] = []
    graph_rule_categories = {"C", "T", "R"}
    for rule in rules:
        if rule.category_code in graph_rule_categories:
            for violation in rule.check(graph):
                if config:
                    violation.severity = config.get_rule_severity(rule.rule_id)
                violations.append(violation)
    return violations


def _collect_embedded_violations(
    toe_dir: Path,
    rules: list[LintRule],
    config: "LintConfig | None",
) -> list[Violation]:
    """Collect and filter embedded code violations."""
    violations: list[Violation] = []
    embedded_violations = list(_validate_embedded_code(toe_dir))
    enabled_rule_ids = {r.rule_id for r in rules}
    for v in embedded_violations:
        if v.rule in enabled_rule_ids or v.rule.startswith(("G", "P")):
            if config:
                v.severity = config.get_rule_severity(v.rule)
            violations.append(v)
    return violations


def run_lint(
    toe_dir: Path,
    validate_expressions: bool = False,
    validate_embedded: bool = True,
    rules: list[LintRule] | None = None,
    config: "LintConfig | None" = None,
) -> list[Violation]:
    """
    Run all lint rules on a .toe.dir project.

    Args:
        toe_dir: Path to the .toe.dir project directory.
        validate_expressions: Whether to validate expressions in .parm files.
        validate_embedded: Whether to validate embedded GLSL/Python code.
        rules: Optional list of rule instances to run. If None, uses default rules.
        config: Optional lint configuration for severity overrides.

    Returns:
        List of violations found.

    """
    violations: list[Violation] = []

    # Use provided rules or fall back to legacy defaults
    if rules is None:
        rules = get_all_rules()

    # Initialize parsers
    n_parser = NFileParser()
    parm_parser = ParmFileParser()
    toc_parser = TocParser()

    # Parse TOC and validate manifest
    toc_path = toe_dir.parent / f"{toe_dir.name.replace('.toe.dir', '.toe.toc')}"
    if toc_path.exists():
        parsed_toc = toc_parser.parse(toc_path)
        violations.extend(_validate_toc_entries(toe_dir, toc_path, parsed_toc.entries))

    # Build the network graph
    builder = NetworkGraphBuilder(n_parser, parm_parser)
    try:
        graph = builder.build(toe_dir)
    except Exception as e:
        violations.append(
            Violation(
                rule="S001",  # valid-n-file-syntax
                message=f"Failed to build network graph: {e}",
                path=str(toe_dir),
                severity="error",
            )
        )
        return violations

    # Run graph-based rules (C, T, R categories)
    violations.extend(_run_graph_rules(graph, rules, config))

    # Phase 3: Embedded code validation
    if validate_embedded:
        violations.extend(_collect_embedded_violations(toe_dir, rules, config))

    # Expression validation (opt-in)
    if validate_expressions:
        for v in _validate_expressions(toe_dir, parm_parser):
            if config:
                v.severity = config.get_rule_severity(v.rule)
            violations.append(v)

    return violations


def _validate_glsl_file(
    content: str,
    text_file: Path,
    lines_stripped: int,
    glsl_validator: GLSLValidator,
) -> Iterator[Violation]:
    """Validate a GLSL file and yield violations."""
    for violation in glsl_validator.validate(content, source_file=text_file):
        if violation.line is not None:
            violation.line += lines_stripped
        violation.rule = "G001"  # glsl-syntax
        yield violation


def _validate_python_file(
    content: str,
    text_file: Path,
    lines_stripped: int,
    python_validator: PythonValidator,
) -> Iterator[Violation]:
    """Validate a Python file and yield violations."""
    for violation in python_validator.validate(content, source_file=text_file):
        if violation.line is not None:
            violation.line += lines_stripped
        # Use rule ID based on violation type
        violation.rule = "P001" if "syntax" in violation.rule.lower() else "P002"
        yield violation


def _validate_embedded_code(toe_dir: Path) -> Iterator[Violation]:
    """
    Validate embedded GLSL and Python code in .text files.

    Args:
        toe_dir: Path to the .toe.dir project directory.

    Yields:
        Violations found in embedded code.

    """
    lang_detector = LanguageDetector()
    glsl_validator = GLSLValidator()
    python_validator = PythonValidator()

    for text_file in toe_dir.rglob("*.text"):
        try:
            content = text_file.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        stripped_content, lines_stripped = lang_detector.strip_text_header(content)
        if not stripped_content.strip():
            continue

        result = lang_detector.detect(content)
        if result.language == Language.GLSL:
            yield from _validate_glsl_file(
                stripped_content, text_file, lines_stripped, glsl_validator
            )
        elif result.language == Language.PYTHON:
            yield from _validate_python_file(
                stripped_content, text_file, lines_stripped, python_validator
            )


def _validate_expressions(
    toe_dir: Path,
    parm_parser: ParmFileParser,
) -> Iterator[Violation]:
    """
    Validate Python expressions in .parm files.

    Args:
        toe_dir: Path to the .toe.dir project directory.
        parm_parser: Parser for .parm files.

    Yields:
        Violations found in expressions.

    """
    expr_validator = ExpressionValidator()

    for parm_file in toe_dir.rglob("*.parm"):
        try:
            parsed = parm_parser.parse(parm_file)
        except Exception:
            continue

        if parsed.has_errors:
            continue

        yield from expr_validator.validate_parm_file(parsed)
