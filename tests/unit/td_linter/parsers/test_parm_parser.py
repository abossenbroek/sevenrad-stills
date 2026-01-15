"""Unit tests for ParmFileParser."""

from pathlib import Path

import pytest
from td_linter.parsers.parm_parser import ParmFileParser

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "docs/touchdesigner/fixtures/projects"
)


@pytest.fixture
def parser() -> ParmFileParser:
    """Create a parser instance."""
    return ParmFileParser()


class TestParmFileParserFixtures:
    """Test ParmFileParser against all fixture .parm files."""

    def test_parses_all_fixture_files(self, parser: ParmFileParser) -> None:
        """All .parm fixture files should parse without errors."""
        parm_files = list(FIXTURE_DIR.rglob("*.parm"))
        assert len(parm_files) > 0, "No fixture files found"

        failed = []
        for parm_file in parm_files:
            result = parser.parse(parm_file)
            if result.has_errors:
                failed.append(parm_file.name)

        assert not failed, f"Failed to parse: {failed}"


class TestParmFileParserFeatures:
    """Test specific parser features."""

    def test_parses_empty_parm_file(self, parser: ParmFileParser) -> None:
        """Parser should handle empty parameter files."""
        content = "?\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 0

    def test_parses_simple_parameters(self, parser: ParmFileParser) -> None:
        """Parser should extract basic parameter values."""
        content = """?
diffr 0 0.952
diffg 0 0.5
diffb 0 0.3
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 3
        assert result.parameters[0].name == "diffr"
        assert result.parameters[0].mode == 0

    def test_parses_identifier_values(self, parser: ParmFileParser) -> None:
        """Parser should handle identifier values like 'on', 'hermite'."""
        content = """?
type 0 hermite
blending 0 on
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 2
        assert result.parameters[0].value == "hermite"
        assert result.parameters[1].value == "on"

    def test_parses_relative_path_values(self, parser: ParmFileParser) -> None:
        """Parser should handle relative path values."""
        content = """?
top 0 ./out1
material 0 ./phong1
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 2
        assert result.parameters[0].value == "./out1"
        assert result.parameters[1].value == "./phong1"

    def test_parses_absolute_path_values(self, parser: ParmFileParser) -> None:
        """Parser should handle absolute path values."""
        content = """?
clone 0 /sys/local/time
dat 0 /local/midi/device
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 2
        assert result.parameters[0].value == "/sys/local/time"
        assert result.parameters[1].value == "/local/midi/device"

    def test_parses_expressions_mode_49(self, parser: ParmFileParser) -> None:
        """Parser should extract expressions for mode 49."""
        content = """?
tx 49 6531 absTime.frame*.6
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].name == "tx"
        assert result.parameters[0].mode == 49
        assert result.parameters[0].expression == "absTime.frame*.6"

    def test_parses_string_expressions_mode_17(self, parser: ParmFileParser) -> None:
        """Parser should handle mode 17 string expressions."""
        content = """?
autoexportroot 17 "" me.parent()
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].mode == 17
        assert result.parameters[0].expression == "me.parent()"

    def test_parses_glob_patterns(self, parser: ParmFileParser) -> None:
        """Parser should handle glob patterns as values."""
        content = """?
rownames 0 ?*
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].value == "?*"

    def test_parses_numeric_values(self, parser: ParmFileParser) -> None:
        """Parser should handle various numeric formats."""
        content = """?
w 0 1280
h 0 720
rough 0 0.25
negative 0 -3.5
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 4

    def test_malformed_file_returns_has_errors(self, parser: ParmFileParser) -> None:
        """Malformed content should return has_errors=True."""
        content = "this is not valid .parm content"
        result = parser.parse_string(content)
        assert result.has_errors
