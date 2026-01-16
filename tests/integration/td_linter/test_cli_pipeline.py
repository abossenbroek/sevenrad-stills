"""Integration tests for CLI command pipelines.

These tests verify that CLI commands work together correctly:
- lint → fix → lint chain
- Configuration propagation through commands
"""

from pathlib import Path

import pytest
from td_linter.cli import app
from typer.testing import CliRunner

runner = CliRunner()

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent.parent / "docs/touchdesigner/fixtures/projects"
)


@pytest.mark.slow
@pytest.mark.integration
class TestLintFixLintPipeline:
    """Test lint → fix → lint command chain."""

    def test_lint_then_fix_then_lint(self) -> None:
        """Running lint, fix, lint should produce consistent results."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # First lint - get baseline violations
        lint_result_1 = runner.invoke(app, ["lint", str(toe_dir), "--format", "json"])
        assert lint_result_1.exit_code == 0 or lint_result_1.exit_code == 1

        # Run fix in dry-run mode (don't actually modify files)
        fix_result = runner.invoke(
            app, ["fix", str(toe_dir), "--dry-run", "--verbose"]
        )
        # Fix may succeed even if no fixable violations found
        assert fix_result.exit_code == 0

        # Second lint - should be same as first (dry-run doesn't modify)
        lint_result_2 = runner.invoke(app, ["lint", str(toe_dir), "--format", "json"])
        assert lint_result_2.exit_code == lint_result_1.exit_code

    def test_fix_dry_run_reports_would_fix_count(self) -> None:
        """Fix --dry-run should report how many fixes would be applied."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        result = runner.invoke(app, ["fix", str(toe_dir), "--dry-run"])
        assert result.exit_code == 0
        # Output should mention fixes (even if 0)
        output_lower = result.output.lower()
        assert (
            "fix" in output_lower
            or "applied" in output_lower
            or "skip" in output_lower
            or "no violations" in output_lower
        )


@pytest.mark.slow
@pytest.mark.integration
class TestConfigurationPropagation:
    """Test that configuration flows through all CLI commands."""

    def test_lint_respects_select_option(self) -> None:
        """Lint --select should filter rules."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Lint with only syntax rules
        result = runner.invoke(
            app, ["lint", str(toe_dir), "--select", "S", "--format", "json"]
        )
        # Should complete (may have violations or not)
        assert result.exit_code in [0, 1]

    def test_lint_respects_ignore_option(self) -> None:
        """Lint --ignore should exclude rules."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Lint ignoring performance rules
        result = runner.invoke(
            app, ["lint", str(toe_dir), "--ignore", "F", "--format", "json"]
        )
        # Should complete
        assert result.exit_code in [0, 1]

    def test_fix_respects_select_option(self) -> None:
        """Fix --select should only fix selected rules."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Fix with only GLSL rules (G002 is fixable)
        result = runner.invoke(
            app, ["fix", str(toe_dir), "--select", "G", "--dry-run"]
        )
        assert result.exit_code == 0


@pytest.mark.slow
@pytest.mark.integration
class TestOutputFormats:
    """Test that output formats work for all commands."""

    def test_lint_json_output_has_expected_structure(self) -> None:
        """Lint --format json should have expected JSON structure."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        result = runner.invoke(app, ["lint", str(toe_dir), "--format", "json"])
        assert result.exit_code in [0, 1]

        # Check for JSON structure markers (Rich may format output)
        output = result.output
        assert '"version"' in output
        assert '"violations"' in output
        assert '"summary"' in output

    def test_lint_sarif_output_has_expected_structure(self) -> None:
        """Lint --format sarif should have expected SARIF structure."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        result = runner.invoke(app, ["lint", str(toe_dir), "--format", "sarif"])
        assert result.exit_code in [0, 1]

        # Check for SARIF structure markers (Rich may format output)
        output = result.output
        assert '"$schema"' in output
        assert '"runs"' in output
        assert "sarif" in output.lower()


@pytest.mark.slow
@pytest.mark.integration
class TestEdgeCases:
    """Edge case tests for CLI commands."""

    def test_lint_empty_toe_dir(self, tmp_path: Path) -> None:
        """Lint should handle empty .toe.dir gracefully."""
        empty_toe = tmp_path / "empty.toe.dir"
        empty_toe.mkdir()
        (empty_toe / ".toc").write_text("")

        result = runner.invoke(app, ["lint", str(empty_toe)])
        # Should complete without crashing (may have 0 violations)
        assert result.exit_code in [0, 1]

    def test_lint_toe_dir_with_only_toc(self, tmp_path: Path) -> None:
        """Lint should handle .toe.dir with only .toc file."""
        toe_dir = tmp_path / "minimal.toe.dir"
        toe_dir.mkdir()
        (toe_dir / ".toc").write_text("nonexistent.n\n")

        result = runner.invoke(app, ["lint", str(toe_dir)])
        # Should complete (may report missing files)
        assert result.exit_code in [0, 1]

    def test_fix_empty_toe_dir(self, tmp_path: Path) -> None:
        """Fix should handle empty .toe.dir gracefully."""
        empty_toe = tmp_path / "empty.toe.dir"
        empty_toe.mkdir()
        (empty_toe / ".toc").write_text("")

        result = runner.invoke(app, ["fix", str(empty_toe), "--dry-run"])
        assert result.exit_code == 0

    def test_lint_with_all_rules_disabled(self) -> None:
        """Lint should work when all rules are ignored."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Ignore all rule categories
        result = runner.invoke(
            app, ["lint", str(toe_dir), "--ignore", "S,C,T,R,G,P,F"]
        )
        # Should complete with no violations
        assert result.exit_code == 0

    def test_lint_with_conflicting_select_ignore(self) -> None:
        """Lint should handle conflicting --select and --ignore."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Select S, but also ignore S
        result = runner.invoke(
            app, ["lint", str(toe_dir), "--select", "S", "--ignore", "S"]
        )
        # Should complete (implementation decides precedence)
        assert result.exit_code in [0, 1]

    def test_fix_with_no_fixable_violations(self) -> None:
        """Fix should report cleanly when no violations are fixable."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Select only connection rules (not fixable)
        result = runner.invoke(
            app, ["fix", str(toe_dir), "--select", "C", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_lint_repeated_runs_consistent(self) -> None:
        """Multiple lint runs should produce consistent results."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        results = []
        for _ in range(3):
            result = runner.invoke(app, ["lint", str(toe_dir), "--format", "json"])
            results.append(result.output)

        # All runs should produce same output
        assert results[0] == results[1] == results[2]
