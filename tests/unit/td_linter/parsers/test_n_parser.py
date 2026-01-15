"""Unit tests for NFileParser."""

from pathlib import Path

import pytest
from td_linter.parsers.n_parser import NFileParser

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "docs/touchdesigner/fixtures/projects"
)


@pytest.fixture
def parser() -> NFileParser:
    """Create a parser instance."""
    return NFileParser()


class TestNFileParserFixtures:
    """Test NFileParser against all fixture .n files."""

    def test_parses_all_fixture_files(self, parser: NFileParser) -> None:
        """All .n fixture files should parse without errors."""
        n_files = list(FIXTURE_DIR.rglob("*.n"))
        assert len(n_files) > 0, "No fixture files found"

        failed = []
        for n_file in n_files:
            result = parser.parse(n_file)
            if result.has_errors:
                failed.append(n_file.name)

        assert not failed, f"Failed to parse: {failed}"


class TestNFileParserFeatures:
    """Test specific parser features."""

    def test_parses_negative_tile_coordinates(self, parser: NFileParser) -> None:
        """Parser should handle negative tile coordinates."""
        content = """COMP:base
tile -200 -300 160 130
flags = parlanguage 0
color 0.67 0.67 0.67
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.tile == (-200, -300, 160, 130)

    def test_parses_comment_directive(self, parser: NFileParser) -> None:
        """Parser should extract comment directive."""
        content = """COMP:window
comment "This is a test comment"
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.comment == "This is a test comment"

    def test_parses_standalone_flag_modifiers(self, parser: NFileParser) -> None:
        """Parser should handle flags with on/off modifiers."""
        content = """COMP:geo
tile 450 170 159 130
flags = viewer 1 activate on render on display on parlanguage 0
color 0.67 0.67 0.67
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert "viewer" in result.flags
        assert "activate" in result.flags

    def test_parses_path_refs_in_inputs(self, parser: NFileParser) -> None:
        """Parser should handle path references in inputs."""
        content = """COMP:geo
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0   geo1/out1
1   base1/container1
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 2
        assert result.inputs[0] == (0, "geo1/out1")
        assert result.inputs[1] == (1, "base1/container1")

    def test_parses_v_directive(self, parser: NFileParser) -> None:
        """Parser should extract v (viewport) directive."""
        content = """COMP:container
v -670.261 80.4858 2.14377
tile 200 100 400 244
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.v is not None
        assert abs(result.v[0] - (-670.261)) < 0.001
        assert abs(result.v[1] - 80.4858) < 0.001
        assert abs(result.v[2] - 2.14377) < 0.001

    def test_parses_all_operator_families(self, parser: NFileParser) -> None:
        """Parser should handle all operator family types."""
        families = ["TOP", "CHOP", "SOP", "DAT", "COMP", "MAT", "POP"]
        for family in families:
            content = f"""{family}:test
tile 100 100 100 100
flags = parlanguage 0
end
"""
            result = parser.parse_string(content)
            assert not result.has_errors, f"Failed for family {family}"
            assert result.family == family

    def test_malformed_file_returns_has_errors(self, parser: NFileParser) -> None:
        """Malformed content should return has_errors=True."""
        content = "this is not valid .n content"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_extracts_color_values(self, parser: NFileParser) -> None:
        """Parser should extract color values."""
        content = """TOP:displace
tile 100 100 100 100
flags = parlanguage 0
color 0.67 0.5 0.3
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.color is not None
        assert len(result.color) == 3
        assert abs(result.color[0] - 0.67) < 0.01

    def test_extracts_inputs(self, parser: NFileParser) -> None:
        """Parser should extract input connections."""
        content = """TOP:displace
tile 100 100 100 100
flags = parlanguage 0
inputs
{
0   moviefilein1
1   chopto1
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 2
        assert result.inputs[0] == (0, "moviefilein1")
        assert result.inputs[1] == (1, "chopto1")
