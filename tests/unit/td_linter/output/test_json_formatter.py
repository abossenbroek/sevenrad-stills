"""Tests for JSONFormatter."""

import json

import pytest

from td_linter.output.json_formatter import JSONFormatter
from td_linter.rules.base import Violation


class TestJSONFormatterBasics:
    """Basic JSONFormatter tests."""

    def test_name_property(self) -> None:
        """Test that formatter name is 'json'."""
        formatter = JSONFormatter()
        assert formatter.name == "json"

    def test_supports_color_false(self) -> None:
        """Test that JSON formatter does not support color."""
        formatter = JSONFormatter()
        assert formatter.supports_color is False


class TestJSONFormatterSchema:
    """Test JSON output schema compliance."""

    def test_empty_violations_schema(self) -> None:
        """Test empty violations produces valid schema."""
        formatter = JSONFormatter()
        output = formatter.format([])
        data = json.loads(output)

        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "violations" in data
        assert data["violations"] == []
        assert "summary" in data
        assert data["summary"]["total"] == 0
        assert data["summary"]["errors"] == 0
        assert data["summary"]["warnings"] == 0
        assert data["summary"]["info"] == 0

    def test_violations_schema(self) -> None:
        """Test violations produce valid schema structure."""
        formatter = JSONFormatter()
        violations = [
            Violation(
                rule="C001",
                message="Test error",
                path="/project/op1",
                severity="error",
                line=10,
            )
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        assert len(data["violations"]) == 1
        v = data["violations"][0]
        assert v["rule"] == "C001"
        assert v["message"] == "Test error"
        assert v["path"] == "/project/op1"
        assert v["severity"] == "error"
        assert v["line"] == 10

    def test_project_path_included(self) -> None:
        """Test that project_path is included when provided."""
        formatter = JSONFormatter()
        output = formatter.format([], project_path="test.toe.dir")
        data = json.loads(output)

        assert "project" in data
        assert data["project"] == "test.toe.dir"


class TestJSONFormatterSummary:
    """Test summary counts in JSON output."""

    def test_summary_counts_correct(self) -> None:
        """Test that summary counts are accurate."""
        formatter = JSONFormatter()
        violations = [
            Violation(rule="C001", message="E1", path="/p/1", severity="error"),
            Violation(rule="C002", message="E2", path="/p/2", severity="error"),
            Violation(rule="C003", message="W1", path="/p/3", severity="warning"),
            Violation(rule="P003", message="I1", path="/p/4", severity="info"),
            Violation(rule="P003", message="I2", path="/p/5", severity="info"),
            Violation(rule="P003", message="I3", path="/p/6", severity="info"),
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        assert data["summary"]["total"] == 6
        assert data["summary"]["errors"] == 2
        assert data["summary"]["warnings"] == 1
        assert data["summary"]["info"] == 3


class TestJSONFormatterViolationFields:
    """Test all violation fields are properly serialized."""

    def test_all_fields_present(self) -> None:
        """Test all violation fields are included in output."""
        formatter = JSONFormatter()
        from pathlib import Path

        violations = [
            Violation(
                rule="G001",
                message="GLSL syntax error",
                path="/project/shader",
                severity="error",
                source_file=Path("/project/shader.text"),
                line=42,
                context={"error_type": "syntax"},
            )
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        v = data["violations"][0]
        assert v["rule"] == "G001"
        assert v["message"] == "GLSL syntax error"
        assert v["path"] == "/project/shader"
        assert v["severity"] == "error"
        assert v["source_file"] == "/project/shader.text"
        assert v["line"] == 42
        assert v["context"] == {"error_type": "syntax"}

    def test_null_fields_when_none(self) -> None:
        """Test that None fields are serialized as null."""
        formatter = JSONFormatter()
        violations = [
            Violation(
                rule="C001",
                message="Test",
                path="/p/1",
                severity="error",
            )
        ]
        output = formatter.format(violations)
        data = json.loads(output)

        v = data["violations"][0]
        assert v["line"] is None
        assert v["source_file"] is None

    def test_special_characters_in_message(self) -> None:
        """Test that special characters in message are properly escaped."""
        formatter = JSONFormatter()
        violations = [
            Violation(
                rule="C001",
                message='Test with "quotes" and \\ backslash',
                path="/p/1",
                severity="error",
            )
        ]
        output = formatter.format(violations)
        # Should be valid JSON
        data = json.loads(output)
        assert '"quotes"' in data["violations"][0]["message"]
        assert "\\" in data["violations"][0]["message"]
