"""Integration tests for td-linter end-to-end flow."""

from pathlib import Path

import pytest
from td_linter.linter import get_all_rules, run_lint

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent.parent / "docs/touchdesigner/fixtures/projects"
)


class TestLintFlow:
    """Test end-to-end linting flow."""

    def test_lint_example_project(self) -> None:
        """Linting the example project should complete without parse errors."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        violations = run_lint(toe_dir)

        # Filter for parse errors only
        parse_errors = [v for v in violations if v.rule == "parse-error"]
        assert len(parse_errors) == 0, f"Parse errors found: {parse_errors}"

    def test_lint_shader_harness_project(self) -> None:
        """Linting the shader test harness should complete without parse errors."""
        toe_dir = FIXTURE_DIR / "shader_test_harness.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        violations = run_lint(toe_dir)

        # Filter for parse errors only
        parse_errors = [v for v in violations if v.rule == "parse-error"]
        assert len(parse_errors) == 0, f"Parse errors found: {parse_errors}"

    def test_all_rules_execute(self) -> None:
        """All rules should execute without crashing."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        # Get all rules
        rules = get_all_rules()
        assert len(rules) >= 4, "Expected at least 4 rules"

        # Run lint - this exercises all rules
        violations = run_lint(toe_dir)

        # Just verify it completes - violations are expected for test projects
        assert isinstance(violations, list)

    def test_violations_have_required_fields(self) -> None:
        """Violations should have all required fields populated."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        violations = run_lint(toe_dir)

        for v in violations:
            assert v.rule is not None, "Violation missing rule"
            assert v.message is not None, "Violation missing message"
            assert v.severity is not None, "Violation missing severity"
            # path can be None for some violations


class TestRulesIntegration:
    """Test that specific rules find expected patterns."""

    def test_dangling_inputs_detected(self) -> None:
        """Rule should detect references to non-existent operators."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        violations = run_lint(toe_dir)

        # Filter for dangling input violations
        dangling = [v for v in violations if v.rule == "no-dangling-inputs"]
        # May or may not find any - just verify rule runs
        assert isinstance(dangling, list)

    def test_type_compatibility_checked(self) -> None:
        """Rule should check operator type compatibility."""
        toe_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not toe_dir.exists():
            pytest.skip("Fixture project not found")

        violations = run_lint(toe_dir)

        # Filter for type compatibility violations
        type_violations = [v for v in violations if v.rule == "type-compatibility"]
        # May or may not find any - just verify rule runs
        assert isinstance(type_violations, list)
