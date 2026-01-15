"""Main linter orchestration."""

from pathlib import Path
from typing import Iterator

from td_linter.graph.builder import NetworkGraphBuilder
from td_linter.parsers import NFileParser, ParmFileParser, TocParser
from td_linter.rules.base import LintRule, Violation
from td_linter.rules.no_dangling_inputs import NoDanglingInputsRule
from td_linter.rules.no_invalid_cycles import NoInvalidCyclesRule
from td_linter.rules.type_compatibility import TypeCompatibilityRule
from td_linter.rules.valid_references import ValidReferencesRule


def get_all_rules() -> list[LintRule]:
    """Return all available lint rules."""
    return [
        NoInvalidCyclesRule(),
        NoDanglingInputsRule(),
        ValidReferencesRule(),
        TypeCompatibilityRule(),
    ]


def run_lint(toe_dir: Path) -> list[Violation]:
    """Run all lint rules on a .toe.dir project."""
    violations: list[Violation] = []

    # Initialize parsers
    n_parser = NFileParser()
    parm_parser = ParmFileParser()
    toc_parser = TocParser()

    # Parse TOC and validate manifest
    toc_path = toe_dir.parent / f"{toe_dir.name.replace('.toe.dir', '.toe.toc')}"
    if not toc_path.exists():
        # Try looking for .toc inside the .toe.dir
        toc_entries: list[str] = []
    else:
        parsed_toc = toc_parser.parse(toc_path)
        toc_entries = parsed_toc.entries

        # Validate TOC entries exist
        for entry in toc_entries:
            if entry.startswith("."):
                continue  # Skip special entries
            entry_path = toe_dir / entry
            if not entry_path.exists():
                violations.append(
                    Violation(
                        rule="toc-entry-missing",
                        message=f"TOC entry does not exist: {entry}",
                        path=entry,
                        severity="error",
                        source_file=toc_path,
                    )
                )

    # Build the network graph
    builder = NetworkGraphBuilder(n_parser, parm_parser)
    try:
        graph = builder.build(toe_dir)
    except Exception as e:
        violations.append(
            Violation(
                rule="parse-error",
                message=f"Failed to build network graph: {e}",
                path=str(toe_dir),
                severity="error",
            )
        )
        return violations

    # Run all rules
    rules = get_all_rules()
    for rule in rules:
        for violation in rule.check(graph):
            violations.append(violation)

    return violations
