"""Unit tests for TocParser."""

from pathlib import Path

import pytest
from td_linter.parsers.toc_parser import TocParser

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "docs/touchdesigner/fixtures/projects"
)


@pytest.fixture
def parser() -> TocParser:
    """Create a parser instance."""
    return TocParser()


class TestTocParserFixtures:
    """Test TocParser against fixture .toc files."""

    def test_parses_example_toc_file(self, parser: TocParser) -> None:
        """Should parse the example .toc file."""
        toc_file = FIXTURE_DIR / "reference_toe/example.toe.dir/.toc"
        if not toc_file.exists():
            pytest.skip("Fixture file not found")

        result = parser.parse(toc_file)
        assert not result.has_errors
        assert len(result.entries) > 0

    def test_parses_shader_harness_toc_file(self, parser: TocParser) -> None:
        """Should parse the shader test harness .toc file."""
        toc_file = FIXTURE_DIR / "shader_test_harness.toe.dir/.toc"
        if not toc_file.exists():
            pytest.skip("Fixture file not found")

        result = parser.parse(toc_file)
        assert not result.has_errors


class TestTocParserFeatures:
    """Test specific parser features."""

    def test_parses_special_entries(self, parser: TocParser) -> None:
        """Parser should identify special entries (starting with .)."""
        content = """.build
.start
.grps
project1.n
project1.parm
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert ".build" in result.special_entries
        assert ".start" in result.special_entries
        assert ".grps" in result.special_entries
        assert "project1.n" in result.entries
        assert "project1.parm" in result.entries

    def test_parses_nested_paths(self, parser: TocParser) -> None:
        """Parser should handle nested directory paths."""
        content = """project1.n
project1/geo1.n
project1/geo1/box1.n
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.entries) == 3
        assert "project1/geo1/box1.n" in result.entries

    def test_empty_toc_file(self, parser: TocParser) -> None:
        """Parser should handle empty .toc files."""
        content = ""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.entries) == 0
        assert len(result.special_entries) == 0
