"""Tests for SARIFFormatter."""

import json
from pathlib import Path

import pytest

from td_linter.output.sarif import SARIFFormatter
from td_linter.rules.base import Violation


class TestSARIFFormatterBasics:
    """Basic SARIFFormatter tests."""

    def test_name_property(self) -> None:
        """Test that formatter name is 'sarif'."""
        formatter = SARIFFormatter()
        assert formatter.name == "sarif"

    def test_supports_color_false(self) -> None:
        """Test that SARIF formatter does not support color."""
        formatter = SARIFFormatter()
        assert formatter.supports_color is False


class TestSARIFFormatterSchema:
    """Test SARIF 2.1.0 schema compliance."""

    def test_empty_violations_structure(self) -> None:
        """Test empty violations produces valid SARIF structure."""
        formatter = SARIFFormatter()
        output = formatter.format([])
        data = json.loads(output)

        # Check required SARIF fields
        assert "$schema" in data
        assert "sarif-schema-2.1.0" in data["$schema"]
        assert data["version"] == "2.1.0"
        assert "runs" in data
        assert len(data["runs"]) == 1

        run = data["runs"][0]
        assert "tool" in run
        assert "driver" in run["tool"]
        assert run["tool"]["driver"]["name"] == "td-linter"
        assert "results" in run
        assert run["results"] == []

    def test_tool_driver_info(self) -> None:
        """Test tool driver contains correct information."""
        formatter = SARIFFormatter()
        output = formatter.format([])
        data = json.loads(output)

        driver = data["runs"][0]["tool"]["driver"]
        assert driver["name"] == "td-linter"
        assert "version" in driver
        assert "informationUri" in driver


class TestSARIFFormatterViolations:
    """Test violation formatting in SARIF."""

    def test_single_violation_structure(self) -> None:
        """Test single violation produces correct SARIF result."""
        formatter = SARIFFormatter()
        violations = [
            Violation(
                rule="C001",
                message="Test error message",
                path="/project/op1",
                severity="error",
            )
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        results = data["runs"][0]["results"]
        assert len(results) == 1

        result = results[0]
        assert result["ruleId"] == "C001"
        assert result["level"] == "error"
        assert result["message"]["text"] == "Test error message"
        assert "locations" in result

    def test_violation_with_line_number(self) -> None:
        """Test violation with line number includes region."""
        formatter = SARIFFormatter()
        violations = [
            Violation(
                rule="G001",
                message="GLSL error",
                path="/project/shader",
                severity="error",
                source_file=Path("/project/shader.text"),
                line=42,
            )
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        result = data["runs"][0]["results"][0]
        location = result["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == "/project/shader.text"
        assert location["region"]["startLine"] == 42


class TestSARIFFormatterSeverityMapping:
    """Test severity level mapping to SARIF levels."""

    def test_error_maps_to_error(self) -> None:
        """Test error severity maps to SARIF 'error' level."""
        formatter = SARIFFormatter()
        violations = [
            Violation(rule="C001", message="E", path="/p", severity="error")
        ]
        output = formatter.format(violations)
        data = json.loads(output)
        assert data["runs"][0]["results"][0]["level"] == "error"

    def test_warning_maps_to_warning(self) -> None:
        """Test warning severity maps to SARIF 'warning' level."""
        formatter = SARIFFormatter()
        violations = [
            Violation(rule="C002", message="W", path="/p", severity="warning")
        ]
        output = formatter.format(violations)
        data = json.loads(output)
        assert data["runs"][0]["results"][0]["level"] == "warning"

    def test_info_maps_to_note(self) -> None:
        """Test info severity maps to SARIF 'note' level."""
        formatter = SARIFFormatter()
        violations = [
            Violation(rule="P003", message="I", path="/p", severity="info")
        ]
        output = formatter.format(violations)
        data = json.loads(output)
        assert data["runs"][0]["results"][0]["level"] == "note"


class TestSARIFFormatterRules:
    """Test rule definitions in SARIF output."""

    def test_rules_array_populated(self) -> None:
        """Test rules array contains unique rules."""
        formatter = SARIFFormatter()
        violations = [
            Violation(rule="C001", message="E1", path="/p/1", severity="error"),
            Violation(rule="C001", message="E2", path="/p/2", severity="error"),
            Violation(rule="C002", message="W1", path="/p/3", severity="warning"),
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        rules = data["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [r["id"] for r in rules]

        # Should have 2 unique rules, not 3
        assert len(rules) == 2
        assert "C001" in rule_ids
        assert "C002" in rule_ids

    def test_rule_has_short_description(self) -> None:
        """Test each rule has shortDescription."""
        formatter = SARIFFormatter()
        violations = [
            Violation(rule="C001", message="Test", path="/p", severity="error")
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        rule = data["runs"][0]["tool"]["driver"]["rules"][0]
        assert "shortDescription" in rule
        assert "text" in rule["shortDescription"]


class TestSARIFFormatterMultipleViolations:
    """Test multiple violations handling."""

    def test_multiple_violations_from_same_rule(self) -> None:
        """Test multiple violations from same rule are handled."""
        formatter = SARIFFormatter()
        violations = [
            Violation(rule="C001", message="E1", path="/p/1", severity="error"),
            Violation(rule="C001", message="E2", path="/p/2", severity="error"),
            Violation(rule="C001", message="E3", path="/p/3", severity="error"),
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        results = data["runs"][0]["results"]
        assert len(results) == 3

        # All should reference C001
        for result in results:
            assert result["ruleId"] == "C001"

    def test_multiple_files_multiple_rules(self) -> None:
        """Test violations from multiple files and rules."""
        formatter = SARIFFormatter()
        violations = [
            Violation(
                rule="S001",
                message="Syntax error",
                path="/project/file1.n",
                severity="error",
                source_file=Path("/project/file1.n"),
            ),
            Violation(
                rule="G001",
                message="GLSL error",
                path="/project/shader.text",
                severity="error",
                source_file=Path("/project/shader.text"),
            ),
            Violation(
                rule="C002",
                message="Dangling input",
                path="/project/file2.n",
                severity="warning",
                source_file=Path("/project/file2.n"),
            ),
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        results = data["runs"][0]["results"]
        assert len(results) == 3

        # Check different artifact URIs
        uris = [
            r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
            for r in results
        ]
        assert "/project/file1.n" in uris
        assert "/project/shader.text" in uris
        assert "/project/file2.n" in uris
