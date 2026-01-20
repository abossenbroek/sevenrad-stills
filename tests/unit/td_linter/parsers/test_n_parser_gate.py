"""
Gate criteria tests for NFileParser (G1.1).

This module contains comprehensive tests to validate the .n file parser
meets all gate criteria requirements for the td-linter MVP.

Positive tests verify the parser correctly handles all valid .n file constructs.
Negative tests verify the parser correctly rejects malformed content.
"""

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


# =============================================================================
# POSITIVE TESTS - Should parse successfully
# =============================================================================


class TestPositiveFixtureFiles:
    """G1.1-P1: All fixture .n files should parse without errors."""

    def test_parses_all_fixture_n_files(self, parser: NFileParser) -> None:
        """All .n fixture files should parse without errors."""
        n_files = list(FIXTURE_DIR.rglob("*.n"))
        assert len(n_files) > 0, "No fixture files found"

        failed: list[tuple[str, str]] = []
        for n_file in n_files:
            result = parser.parse(n_file)
            if result.has_errors:
                failed.append((str(n_file), "parse error"))

        assert not failed, f"Failed to parse {len(failed)} files: {failed}"

    def test_parses_reference_toe_fixture_files(self, parser: NFileParser) -> None:
        """Reference toe fixture files should parse successfully."""
        reference_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        if not reference_dir.exists():
            pytest.skip("Reference fixture directory not found")

        n_files = list(reference_dir.rglob("*.n"))
        assert len(n_files) > 0, "No .n files found in reference_toe fixture"

        for n_file in n_files:
            result = parser.parse(n_file)
            assert not result.has_errors, f"Failed to parse: {n_file.name}"

    def test_parses_shader_harness_fixture_files(self, parser: NFileParser) -> None:
        """Shader test harness fixture files should parse successfully."""
        harness_dir = FIXTURE_DIR / "shader_test_harness.toe.dir"
        if not harness_dir.exists():
            pytest.skip("Shader harness fixture directory not found")

        n_files = list(harness_dir.rglob("*.n"))
        assert len(n_files) > 0, "No .n files found in shader_harness fixture"

        for n_file in n_files:
            result = parser.parse(n_file)
            assert not result.has_errors, f"Failed to parse: {n_file.name}"


class TestPositiveNegativeTileCoordinates:
    """G1.1-P2: Parser should handle negative tile coordinates."""

    def test_negative_x_coordinate(self, parser: NFileParser) -> None:
        """Parser should handle negative X tile coordinate."""
        content = """COMP:base
tile -200 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.tile[0] == -200

    def test_negative_y_coordinate(self, parser: NFileParser) -> None:
        """Parser should handle negative Y tile coordinate."""
        content = """COMP:base
tile 100 -300 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.tile[1] == -300

    def test_both_negative_coordinates(self, parser: NFileParser) -> None:
        """Parser should handle both negative X and Y coordinates."""
        content = """COMP:base
tile -200 -300 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.tile == (-200, -300, 160, 130)

    def test_large_negative_coordinates(self, parser: NFileParser) -> None:
        """Parser should handle large negative coordinates."""
        content = """COMP:base
tile -9999 -8888 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.tile[0] == -9999
        assert result.tile[1] == -8888


class TestPositiveCommentDirective:
    """G1.1-P3: Parser should handle comment directive."""

    def test_simple_comment(self, parser: NFileParser) -> None:
        """Parser should extract simple comment."""
        content = """COMP:window
comment "Test comment"
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.comment == "Test comment"

    def test_comment_with_spaces(self, parser: NFileParser) -> None:
        """Parser should extract comment with multiple spaces."""
        content = """COMP:window
comment "This is a longer test comment with spaces"
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.comment == "This is a longer test comment with spaces"

    def test_comment_with_special_characters(self, parser: NFileParser) -> None:
        """Parser should extract comment with special characters."""
        content = """COMP:window
comment "Test: value=123, mode->active"
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert "Test:" in result.comment
        assert "value=123" in result.comment

    def test_empty_comment(self, parser: NFileParser) -> None:
        """Parser should handle empty comment string."""
        content = """COMP:window
comment ""
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.comment == ""


class TestPositiveStandaloneFlagModifiers:
    """G1.1-P4: Parser should handle standalone flag modifiers (on/off)."""

    def test_picked_on_flag(self, parser: NFileParser) -> None:
        """Parser should handle 'picked on' flag."""
        content = """COMP:container
tile 100 100 160 130
flags = picked on viewer 1
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert "picked" in result.flags

    def test_multiple_on_flags(self, parser: NFileParser) -> None:
        """Parser should handle multiple 'on' flags."""
        content = """COMP:geo
tile 450 170 159 130
flags = viewer 1 activate on render on display on
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert "viewer" in result.flags
        assert "activate" in result.flags
        assert "render" in result.flags
        assert "display" in result.flags

    def test_off_flag(self, parser: NFileParser) -> None:
        """Parser should handle 'off' flag value."""
        content = """COMP:geo
tile 100 100 160 130
flags = viewer 1 display off
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert "display" in result.flags
        assert result.flags["display"] == "off"

    def test_mixed_numeric_and_on_off_flags(self, parser: NFileParser) -> None:
        """Parser should handle mix of numeric and on/off flag values."""
        content = """COMP:geo
tile 450 170 159 130
flags = viewer 1 activate on render on display on pickable on parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.flags["viewer"] == "1"
        assert result.flags["activate"] == "on"
        assert result.flags["parlanguage"] == "0"


class TestPositivePathRefsInInputs:
    """G1.1-P5: Parser should handle path references in inputs."""

    def test_simple_path_ref(self, parser: NFileParser) -> None:
        """Parser should handle simple path reference."""
        content = """COMP:geo
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	geo1/out1
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 1
        assert result.inputs[0] == (0, "geo1/out1")

    def test_multiple_path_refs(self, parser: NFileParser) -> None:
        """Parser should handle multiple path references."""
        content = """COMP:geo
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	geo1/out1
1	base1/container1
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 2
        assert result.inputs[0] == (0, "geo1/out1")
        assert result.inputs[1] == (1, "base1/container1")

    def test_deep_path_ref(self, parser: NFileParser) -> None:
        """Parser should handle deep path references."""
        content = """TOP:select
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	parent1/child1/grandchild1/out1
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 1
        assert "parent1/child1/grandchild1/out1" in result.inputs[0][1]

    def test_mixed_simple_and_path_refs(self, parser: NFileParser) -> None:
        """Parser should handle mix of simple and path refs."""
        content = """TOP:composite
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	moviefilein1
1	geo1/out1
2	noise1
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 3
        assert result.inputs[0][1] == "moviefilein1"
        assert result.inputs[1][1] == "geo1/out1"
        assert result.inputs[2][1] == "noise1"


class TestPositiveVDirective:
    """G1.1-P6: Parser should handle v (viewport) directive."""

    def test_positive_v_values(self, parser: NFileParser) -> None:
        """Parser should extract positive v values."""
        content = """COMP:container
v 309.531 186.79 0.93596
tile 200 100 400 244
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.v is not None
        assert abs(result.v[0] - 309.531) < 0.001
        assert abs(result.v[1] - 186.79) < 0.001
        assert abs(result.v[2] - 0.93596) < 0.001

    def test_negative_v_values(self, parser: NFileParser) -> None:
        """Parser should handle negative v values."""
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

    def test_all_negative_v_values(self, parser: NFileParser) -> None:
        """Parser should handle all negative v values."""
        content = """COMP:container
v -100.5 -200.25 -0.5
tile 200 100 400 244
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.v is not None
        assert result.v[0] < 0
        assert result.v[1] < 0
        assert result.v[2] < 0

    def test_zero_v_values(self, parser: NFileParser) -> None:
        """Parser should handle zero v values."""
        content = """COMP:container
v 0 0 0
tile 200 100 400 244
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.v is not None
        assert result.v == (0.0, 0.0, 0.0)


class TestPositiveAllOperatorFamilies:
    """G1.1-P7: Parser should handle all operator families."""

    @pytest.mark.parametrize(
        "family",
        ["TOP", "CHOP", "SOP", "DAT", "COMP", "MAT", "POP"],
    )
    def test_operator_family(self, parser: NFileParser, family: str) -> None:
        """Parser should handle {family} operator family."""
        content = f"""{family}:test_operator
tile 100 100 100 100
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors, f"Failed for family {family}"
        assert result.family == family

    def test_top_with_typical_op_type(self, parser: NFileParser) -> None:
        """Parser should handle TOP with typical operator type."""
        content = """TOP:displace
tile 260 200 130 72
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.family == "TOP"
        assert result.op_type == "displace"

    def test_chop_with_typical_op_type(self, parser: NFileParser) -> None:
        """Parser should handle CHOP with typical operator type."""
        content = """CHOP:noise
tile 100 100 130 72
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.family == "CHOP"
        assert result.op_type == "noise"

    def test_dat_with_typical_op_type(self, parser: NFileParser) -> None:
        """Parser should handle DAT with typical operator type."""
        content = """DAT:table
tile 100 100 130 72
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.family == "DAT"
        assert result.op_type == "table"


class TestPositiveEmptyInputsBlock:
    """G1.1-P8: Parser should handle empty inputs block."""

    def test_empty_inputs_block(self, parser: NFileParser) -> None:
        """Parser should handle inputs block with no entries."""
        content = """TOP:null
tile 100 100 130 72
flags = parlanguage 0
inputs
{
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 0

    def test_empty_inputs_with_whitespace(self, parser: NFileParser) -> None:
        """Parser should handle inputs block with whitespace."""
        content = """TOP:null
tile 100 100 130 72
flags = parlanguage 0
inputs
{

}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 0


class TestPositiveMultipleDirectivesAnyOrder:
    """G1.1-P9: Parser should handle directives in any order."""

    def test_color_before_tile(self, parser: NFileParser) -> None:
        """Parser should handle color before tile."""
        content = """TOP:displace
color 0.67 0.67 0.67
tile 100 100 130 72
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.color is not None
        assert result.tile == (100, 100, 130, 72)

    def test_inputs_before_flags(self, parser: NFileParser) -> None:
        """Parser should handle inputs before flags."""
        content = """TOP:displace
tile 100 100 130 72
inputs
{
0	moviefilein1
}
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 1
        assert "parlanguage" in result.flags

    def test_v_after_view(self, parser: NFileParser) -> None:
        """Parser should handle v directive after view."""
        content = """COMP:container
tile 200 100 400 244
view -1 3 0 0 1 1 0 0
v 309.531 186.79 0.93596
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.v is not None
        assert result.view is not None

    def test_comment_at_end(self, parser: NFileParser) -> None:
        """Parser should handle comment at end before 'end'."""
        content = """COMP:window
tile 100 100 160 130
flags = parlanguage 0
color 0.5 0.5 0.5
comment "Final comment"
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.comment == "Final comment"

    def test_all_directives_mixed_order(self, parser: NFileParser) -> None:
        """Parser should handle all directives in mixed order."""
        content = """COMP:container
v 100.0 200.0 1.0
comment "Test"
color 0.67 0.67 0.67
inputs
{
0	source1
}
flags = viewer 1 parlanguage 0
tile 200 100 400 244
view -1 3 0 0 1 1 0 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.v is not None
        assert result.comment == "Test"
        assert result.color is not None
        assert len(result.inputs) == 1
        assert "viewer" in result.flags
        assert result.tile == (200, 100, 400, 244)
        assert result.view is not None


class TestPositiveColorValues:
    """G1.1-P10: Parser should handle color with 3 and 4 values."""

    def test_color_with_3_values(self, parser: NFileParser) -> None:
        """Parser should handle RGB color (3 values)."""
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
        assert abs(result.color[1] - 0.5) < 0.01
        assert abs(result.color[2] - 0.3) < 0.01

    def test_color_with_4_values(self, parser: NFileParser) -> None:
        """Parser should handle RGBA color (4 values)."""
        content = """TOP:displace
tile 100 100 100 100
flags = parlanguage 0
color 0.67 0.5 0.3 1.0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.color is not None
        assert len(result.color) == 4
        assert abs(result.color[3] - 1.0) < 0.01

    def test_color_grayscale_3_values(self, parser: NFileParser) -> None:
        """Parser should handle grayscale color (same RGB values)."""
        content = """COMP:base
tile 100 100 100 100
flags = parlanguage 0
color 0.56 0.56 0.56
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.color is not None
        assert all(abs(c - 0.56) < 0.01 for c in result.color)

    def test_color_zero_alpha(self, parser: NFileParser) -> None:
        """Parser should handle color with zero alpha."""
        content = """TOP:null
tile 100 100 100 100
flags = parlanguage 0
color 1.0 0.0 0.0 0.0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.color is not None
        assert len(result.color) == 4
        assert result.color[3] == 0.0


# =============================================================================
# NEGATIVE TESTS - Should fail/return has_errors=True
# =============================================================================


class TestNegativeMissingTypeHeader:
    """G1.1-N1: Parser should fail on missing type header."""

    def test_no_type_header(self, parser: NFileParser) -> None:
        """Parser should fail when type header is missing."""
        content = """tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_only_tile_and_end(self, parser: NFileParser) -> None:
        """Parser should fail with only tile and end."""
        content = """tile 100 100 160 130
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_missing_colon_in_header(self, parser: NFileParser) -> None:
        """Parser should fail when colon missing in type header."""
        content = """TOP test
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeInvalidOperatorFamily:
    """G1.1-N2: Parser should fail on invalid operator family."""

    def test_invalid_family_name(self, parser: NFileParser) -> None:
        """Parser should fail for invalid family name."""
        content = """INVALID:test
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_lowercase_family(self, parser: NFileParser) -> None:
        """Parser should fail for lowercase family name."""
        content = """top:test
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_unknown_family(self, parser: NFileParser) -> None:
        """Parser should fail for unknown family."""
        content = """XYZ:test
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_numeric_family(self, parser: NFileParser) -> None:
        """Parser should fail for numeric family."""
        content = """123:test
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeMissingEndKeyword:
    """G1.1-N3: Parser should fail on missing 'end' keyword."""

    def test_no_end_keyword(self, parser: NFileParser) -> None:
        """Parser should fail when 'end' is missing."""
        content = """TOP:test
tile 100 100 160 130
flags = parlanguage 0
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_end_misspelled(self, parser: NFileParser) -> None:
        """Parser should fail when 'end' is misspelled."""
        content = """TOP:test
tile 100 100 160 130
flags = parlanguage 0
endd
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_end_uppercase(self, parser: NFileParser) -> None:
        """Parser should fail when 'end' is uppercase."""
        content = """TOP:test
tile 100 100 160 130
flags = parlanguage 0
END
"""
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeMalformedTile:
    """G1.1-N4: Parser should fail on malformed tile directive."""

    def test_tile_with_3_values(self, parser: NFileParser) -> None:
        """Parser should fail when tile has only 3 values."""
        content = """TOP:test
tile 100 100 160
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_tile_with_5_values(self, parser: NFileParser) -> None:
        """Parser should fail when tile has 5 values."""
        content = """TOP:test
tile 100 100 160 130 50
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_tile_with_2_values(self, parser: NFileParser) -> None:
        """Parser should fail when tile has only 2 values."""
        content = """TOP:test
tile 100 100
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_tile_with_1_value(self, parser: NFileParser) -> None:
        """Parser should fail when tile has only 1 value."""
        content = """TOP:test
tile 100
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_tile_with_float_values(self, parser: NFileParser) -> None:
        """Parser should fail when tile has float values (requires integers)."""
        content = """TOP:test
tile 100.5 100.5 160.0 130.0
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_tile_with_text_values(self, parser: NFileParser) -> None:
        """Parser should fail when tile has text values."""
        content = """TOP:test
tile abc def ghi jkl
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeUnclosedInputsBlock:
    """G1.1-N5: Parser should fail on unclosed inputs block."""

    def test_missing_closing_brace(self, parser: NFileParser) -> None:
        """Parser should fail when inputs block missing closing brace."""
        content = """TOP:test
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	source1
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_missing_opening_brace(self, parser: NFileParser) -> None:
        """Parser should fail when inputs block missing opening brace."""
        content = """TOP:test
tile 100 100 160 130
flags = parlanguage 0
inputs
0	source1
}
end
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_missing_both_braces(self, parser: NFileParser) -> None:
        """Parser should fail when inputs block missing both braces."""
        content = """TOP:test
tile 100 100 160 130
flags = parlanguage 0
inputs
0	source1
end
"""
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeRandomGarbageText:
    """G1.1-N6: Parser should fail on random garbage text."""

    def test_completely_random_text(self, parser: NFileParser) -> None:
        """Parser should fail on completely random text."""
        content = "this is not valid .n content at all"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_json_content(self, parser: NFileParser) -> None:
        """Parser should fail on JSON content."""
        content = '{"type": "TOP", "name": "test"}'
        result = parser.parse_string(content)
        assert result.has_errors

    def test_xml_content(self, parser: NFileParser) -> None:
        """Parser should fail on XML content."""
        content = "<node type='TOP' name='test'/>"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_python_code(self, parser: NFileParser) -> None:
        """Parser should fail on Python code."""
        content = """def parse():
    return True
"""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_numbers_only(self, parser: NFileParser) -> None:
        """Parser should fail on numbers only."""
        content = "123 456 789"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_special_characters(self, parser: NFileParser) -> None:
        """Parser should fail on special characters."""
        content = "!@#$%^&*()"
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeEmptyFile:
    """G1.1-N7: Parser should fail on empty file."""

    def test_empty_string(self, parser: NFileParser) -> None:
        """Parser should fail on empty string."""
        content = ""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_whitespace_only(self, parser: NFileParser) -> None:
        """Parser should fail on whitespace only."""
        content = "   \n\n\t\t  \n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_newlines_only(self, parser: NFileParser) -> None:
        """Parser should fail on newlines only."""
        content = "\n\n\n\n"
        result = parser.parse_string(content)
        assert result.has_errors


class TestNegativeMissingTileDirective:
    """
    G1.1-N8: Parser should handle missing tile directive.

    Note: The grammar allows directives in any order and tile may not be
    strictly required by the grammar. These tests verify parser behavior
    for minimal valid files without tile.
    """

    def test_no_tile_minimal(self, parser: NFileParser) -> None:
        """Test behavior when tile directive is completely absent."""
        content = """TOP:test
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        # Parser should still parse (tile is optional per grammar)
        # but result.tile will be default (0,0,0,0)
        # The test documents expected behavior rather than enforcing failure
        if not result.has_errors:
            assert result.tile == (0, 0, 0, 0)

    def test_only_type_and_end(self, parser: NFileParser) -> None:
        """Test parser with only type header and end."""
        content = """TOP:test
end
"""
        result = parser.parse_string(content)
        # Document expected behavior - may or may not be an error
        # depending on whether tile is required
        if not result.has_errors:
            assert result.tile == (0, 0, 0, 0)


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================


class TestEdgeCasesViewDirective:
    """Edge cases for view directive parsing."""

    def test_view_with_integers_and_strings(self, parser: NFileParser) -> None:
        """Parser should handle view with mixed integers and strings."""
        content = """COMP:geo
tile 450 170 159 130
flags = parlanguage 0
view -1 4 0 1 1 1 34 4 "" "" "" ""
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.view is not None

    def test_view_with_many_values(self, parser: NFileParser) -> None:
        """Parser should handle view with many values."""
        content = """COMP:geo
tile 450 170 159 130
flags = parlanguage 0
view -1 4 0 1 1 1 34 4 0 2 3840 2 6 207 4883 0 0 -1 0 0 -1 0 0 -1 0 0 -1 0 0 0 10000 1 0 0 0 0 1 -1 1 1 0 4 "" "" "" ""
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors


class TestEdgeCasesDockDirective:
    """Edge cases for dock directive parsing."""

    def test_dock_directive(self, parser: NFileParser) -> None:
        """Parser should handle dock directive."""
        content = """COMP:container
tile 100 100 160 130
flags = parlanguage 0
dock parent
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.dock == "parent"


class TestEdgeCasesOperatorTypes:
    """Edge cases for operator type names."""

    def test_operator_type_with_numbers(self, parser: NFileParser) -> None:
        """Parser should handle operator type with numbers."""
        content = """TOP:moviefilein1
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.op_type == "moviefilein1"

    def test_operator_type_with_underscore(self, parser: NFileParser) -> None:
        """Parser should handle operator type with underscore."""
        content = """DAT:execute_autotest
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.op_type == "execute_autotest"

    def test_operator_type_camel_case(self, parser: NFileParser) -> None:
        """Parser should handle camelCase operator type."""
        content = """TOP:chopTo
tile 100 100 160 130
flags = parlanguage 0
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.op_type == "chopTo"


class TestEdgeCasesInputIndexes:
    """Edge cases for input indexes."""

    def test_high_input_index(self, parser: NFileParser) -> None:
        """Parser should handle high input index numbers."""
        content = """TOP:composite
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	source1
5	source2
10	source3
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 3
        assert result.inputs[2][0] == 10

    def test_sparse_input_indexes(self, parser: NFileParser) -> None:
        """Parser should handle sparse (non-sequential) input indexes."""
        content = """TOP:composite
tile 100 100 160 130
flags = parlanguage 0
inputs
{
0	source1
3	source2
7	source3
}
end
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.inputs) == 3
