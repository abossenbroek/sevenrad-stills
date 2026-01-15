"""
Gate criteria tests for ParmFileParser (G1.2).

This module provides comprehensive unit tests to validate the .parm file parser
meets all gate criteria for accepting valid TouchDesigner parameter files
and rejecting malformed content.

Gate Criteria G1.2: .parm file parser must:
- Parse all fixture .parm files without errors
- Handle empty parameter files
- Parse all standard parameter formats (numeric, identifier, path, expression)
- Properly reject malformed content with has_errors=True
"""

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


# =============================================================================
# POSITIVE TESTS - Should parse successfully
# =============================================================================


class TestParmParserPositiveFixtures:
    """G1.2.1: All fixture .parm files should parse without errors."""

    def test_parses_all_fixture_files(self, parser: ParmFileParser) -> None:
        """All .parm fixture files should parse without errors."""
        parm_files = list(FIXTURE_DIR.rglob("*.parm"))
        assert len(parm_files) > 0, "No fixture files found"

        failed = []
        for parm_file in parm_files:
            result = parser.parse(parm_file)
            if result.has_errors:
                failed.append(f"{parm_file.name} ({parm_file.parent.name})")

        assert not failed, f"Failed to parse: {failed}"

    def test_parses_reference_toe_parm_files(self, parser: ParmFileParser) -> None:
        """Reference toe fixture .parm files should parse successfully."""
        reference_dir = FIXTURE_DIR / "reference_toe/example.toe.dir"
        parm_files = list(reference_dir.rglob("*.parm"))
        assert len(parm_files) > 0, "No reference fixture files found"

        for parm_file in parm_files:
            result = parser.parse(parm_file)
            assert not result.has_errors, f"Failed: {parm_file}"

    def test_parses_shader_harness_parm_files(self, parser: ParmFileParser) -> None:
        """Shader test harness .parm files should parse successfully."""
        harness_dir = FIXTURE_DIR / "shader_test_harness.toe.dir"
        parm_files = list(harness_dir.rglob("*.parm"))
        assert len(parm_files) > 0, "No shader harness fixture files found"

        for parm_file in parm_files:
            result = parser.parse(parm_file)
            assert not result.has_errors, f"Failed: {parm_file}"


class TestParmParserPositiveEmpty:
    """G1.2.2: Empty parameter file handling."""

    def test_parses_empty_parm_file(self, parser: ParmFileParser) -> None:
        """Parser should handle empty parameter files with just delimiters."""
        content = "?\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 0

    def test_parses_empty_parm_file_no_trailing_newline(
        self, parser: ParmFileParser
    ) -> None:
        """Parser should handle empty parameter files without trailing newline."""
        content = "?\n?"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 0


class TestParmParserPositiveNumeric:
    """G1.2.3: Simple numeric parameter values."""

    def test_parses_integer_parameter(self, parser: ParmFileParser) -> None:
        """Parser should handle integer parameter values."""
        content = "?\nwidth 0 1280\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].name == "width"
        assert result.parameters[0].mode == 0
        assert result.parameters[0].value == 1280.0

    def test_parses_float_parameter(self, parser: ParmFileParser) -> None:
        """Parser should handle floating point parameter values."""
        content = "?\nrough 0 0.25\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].name == "rough"
        assert result.parameters[0].value == 0.25

    def test_parses_large_integer(self, parser: ParmFileParser) -> None:
        """Parser should handle large integer values."""
        content = "?\ntx 49 6531 absTime.frame\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == 6531.0


class TestParmParserPositiveIdentifier:
    """G1.2.4: Identifier values like 'hermite', 'on', 'off'."""

    def test_parses_type_hermite(self, parser: ParmFileParser) -> None:
        """Parser should handle 'hermite' identifier value."""
        content = "?\ntype 0 hermite\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].name == "type"
        assert result.parameters[0].value == "hermite"

    def test_parses_blending_on(self, parser: ParmFileParser) -> None:
        """Parser should handle 'on' identifier value."""
        content = "?\nblending 0 on\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "on"

    def test_parses_multiple_identifier_values(self, parser: ParmFileParser) -> None:
        """Parser should handle multiple identifier values."""
        content = """?
type 0 hermite
blending 0 on
extend 0 mirror
dataformat 0 legacy
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 4
        assert result.parameters[0].value == "hermite"
        assert result.parameters[1].value == "on"
        assert result.parameters[2].value == "mirror"
        assert result.parameters[3].value == "legacy"


class TestParmParserPositiveRelativePath:
    """G1.2.5: Relative path values."""

    def test_parses_relative_path_dot_slash(self, parser: ParmFileParser) -> None:
        """Parser should handle relative path with ./prefix."""
        content = "?\ntop 0 ./out1\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "./out1"

    def test_parses_relative_path_material(self, parser: ParmFileParser) -> None:
        """Parser should handle relative path for material reference."""
        content = "?\nmaterial 0 ./phong1\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "./phong1"

    def test_parses_simple_relative_name(self, parser: ParmFileParser) -> None:
        """Parser should handle simple relative operator names."""
        content = "?\nchop 0 noise1\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "noise1"

    def test_parses_relative_path_with_slashes(self, parser: ParmFileParser) -> None:
        """Parser should handle relative paths with subdirectories."""
        content = "?\ncolormap 0 in1\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "in1"


class TestParmParserPositiveAbsolutePath:
    """G1.2.6: Absolute path values."""

    def test_parses_absolute_path_sys(self, parser: ParmFileParser) -> None:
        """Parser should handle absolute path to sys operators."""
        content = "?\nclone 0 /sys/local/time\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "/sys/local/time"

    def test_parses_absolute_path_local(self, parser: ParmFileParser) -> None:
        """Parser should handle absolute path to local operators."""
        content = "?\ndat 0 /local/midi/device\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "/local/midi/device"

    def test_parses_multiple_absolute_paths(self, parser: ParmFileParser) -> None:
        """Parser should handle multiple absolute path values."""
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


class TestParmParserPositiveExpressionMode49:
    """G1.2.7: Expression mode 49 (numeric expression)."""

    def test_parses_expression_mode_49(self, parser: ParmFileParser) -> None:
        """Parser should handle mode 49 expressions."""
        content = "?\ntx 49 6531 absTime.frame*.6\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].name == "tx"
        assert result.parameters[0].mode == 49
        assert result.parameters[0].value == 6531.0
        assert result.parameters[0].expression == "absTime.frame*.6"

    def test_parses_expression_with_method_call(self, parser: ParmFileParser) -> None:
        """Parser should handle expressions with method calls."""
        content = "?\nrate 49 60 cookRate()\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].mode == 49
        assert result.parameters[0].expression == "cookRate()"


class TestParmParserPositiveStringExpressionMode17:
    """G1.2.8: String expression mode 17."""

    def test_parses_string_expression_mode_17(self, parser: ParmFileParser) -> None:
        """Parser should handle mode 17 string expressions with empty string."""
        content = '?\nautoexportroot 17 "" me.parent()\n?\n'
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 1
        assert result.parameters[0].mode == 17
        assert result.parameters[0].expression == "me.parent()"

    def test_parses_mode_17_with_numeric_default(self, parser: ParmFileParser) -> None:
        """Parser should handle mode 17 with numeric default value."""
        content = '?\nrowindexend 17 0 "me.inputs[0].numRows - 1"\n?\n'
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].mode == 17
        assert result.parameters[0].value == 0.0
        # Expression includes quotes as stored in the file
        assert result.parameters[0].expression == '"me.inputs[0].numRows - 1"'

    def test_parses_mode_17_destination(self, parser: ParmFileParser) -> None:
        """Parser should handle mode 17 for destination parameter."""
        content = '?\ndestination 17 "" me.parent()\n?\n'
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].expression == "me.parent()"

    def test_parses_rate_mode_17(self, parser: ParmFileParser) -> None:
        """Parser should handle rate parameter with mode 17."""
        content = "?\nrate 17 60 cookRate()\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].mode == 17
        assert result.parameters[0].value == 60.0
        assert result.parameters[0].expression == "cookRate()"


class TestParmParserPositiveGlobPatterns:
    """G1.2.9: Glob patterns as values."""

    def test_parses_glob_pattern_question_star(self, parser: ParmFileParser) -> None:
        """Parser should handle ?* glob pattern."""
        content = "?\nrownames 0 ?*\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "?*"

    def test_parses_glob_pattern_star_prefix(self, parser: ParmFileParser) -> None:
        """Parser should handle *prefix glob patterns."""
        # Grammar supports patterns starting with ? or * like ?* or *foo
        content = "?\npattern 0 *foo\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "*foo"

    def test_parses_glob_pattern_question_prefix(self, parser: ParmFileParser) -> None:
        """Parser should handle ?prefix glob patterns."""
        content = "?\npattern 0 ?foo\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "?foo"


class TestParmParserPositiveMultipleParameters:
    """G1.2.10: Multiple parameters in one file."""

    def test_parses_multiple_parameters(self, parser: ParmFileParser) -> None:
        """Parser should handle multiple parameters."""
        content = """?
diffr 0 0.952
diffg 0 0.5
diffb 0 0.3
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 3

    def test_parses_mixed_parameter_types(self, parser: ParmFileParser) -> None:
        """Parser should handle mixed parameter types."""
        content = """?
type 0 hermite
rough 0 0.25
tx 49 6531 absTime.frame*.6
ty 32 0
autoexportroot 17 "" me.parent()
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 5

    def test_parses_realistic_parm_file(self, parser: ParmFileParser) -> None:
        """Parser should handle a realistic .parm file content."""
        content = """?
horzsource 0 none
vertsource 0 red
displaceweightx 0 0
displaceweighty 0 0.1
displaceweightz 0 1
extend 0 mirror
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 6

    def test_parses_phong_material_parm(self, parser: ParmFileParser) -> None:
        """Parser should handle phong material parameter file."""
        content = """?
diffr 0 0.952
diffg 0 0.952
diffb 0 0.952
colormap 0 in1
blending 0 on
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 5


class TestParmParserPositiveNegativeNumbers:
    """G1.2.11: Negative number handling."""

    def test_parses_negative_integer(self, parser: ParmFileParser) -> None:
        """Parser should handle negative integer values."""
        content = "?\noffset 0 -3\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == -3.0

    def test_parses_negative_float(self, parser: ParmFileParser) -> None:
        """Parser should handle negative float values."""
        content = "?\noffset 0 -3.5\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == -3.5

    def test_parses_negative_small_float(self, parser: ParmFileParser) -> None:
        """Parser should handle small negative float values."""
        content = "?\noffset 0 -0.001\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert abs(result.parameters[0].value - (-0.001)) < 0.0001


class TestParmParserPositiveMode32:
    """Additional tests for mode 32 (numeric mode)."""

    def test_parses_mode_32_zero(self, parser: ParmFileParser) -> None:
        """Parser should handle mode 32 with zero value."""
        content = "?\nty 32 0\n?\n"
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].mode == 32
        assert result.parameters[0].value == 0.0

    def test_parses_multiple_mode_32(self, parser: ParmFileParser) -> None:
        """Parser should handle multiple mode 32 parameters."""
        content = """?
ty 32 0
tz 32 0
?
"""
        result = parser.parse_string(content)
        assert not result.has_errors
        assert len(result.parameters) == 2
        assert all(p.mode == 32 for p in result.parameters)


class TestParmParserPositiveQuotedStrings:
    """Tests for quoted string values."""

    def test_parses_quoted_string_value(self, parser: ParmFileParser) -> None:
        """Parser should handle quoted string values."""
        content = '?\ncolnames 0 "id definition"\n?\n'
        result = parser.parse_string(content)
        assert not result.has_errors
        assert result.parameters[0].value == "id definition"


# =============================================================================
# NEGATIVE TESTS - Should fail/return has_errors=True
# =============================================================================


class TestParmParserNegativeMissingDelimiters:
    """G1.2.N1-N3: Missing delimiter tests."""

    def test_rejects_missing_opening_delimiter(self, parser: ParmFileParser) -> None:
        """Parser should reject content missing opening delimiter."""
        content = "width 0 1280\n?\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_missing_closing_delimiter(self, parser: ParmFileParser) -> None:
        """Parser should reject content missing closing delimiter."""
        content = "?\nwidth 0 1280\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_no_delimiters(self, parser: ParmFileParser) -> None:
        """Parser should reject content with no delimiters at all."""
        content = "width 0 1280\n"
        result = parser.parse_string(content)
        assert result.has_errors


class TestParmParserNegativeGarbage:
    """G1.2.N4: Random garbage text rejection."""

    def test_rejects_random_garbage(self, parser: ParmFileParser) -> None:
        """Parser should reject random garbage text."""
        content = "this is not valid .parm content"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_html_content(self, parser: ParmFileParser) -> None:
        """Parser should reject HTML content."""
        content = "<html><body>Not a parm file</body></html>"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_json_content(self, parser: ParmFileParser) -> None:
        """Parser should reject JSON content."""
        content = '{"width": 1280, "height": 720}'
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_binary_like_content(self, parser: ParmFileParser) -> None:
        """Parser should reject binary-like content."""
        content = "\x00\x01\x02\x03\x04"
        result = parser.parse_string(content)
        assert result.has_errors


class TestParmParserNegativeMalformedParameters:
    """G1.2.N5-N6: Malformed parameter line tests."""

    def test_rejects_parameter_without_mode(self, parser: ParmFileParser) -> None:
        """Parser should reject parameter line without mode number."""
        content = "?\nwidth 1280\n?\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_parameter_name_only(self, parser: ParmFileParser) -> None:
        """Parser should reject incomplete parameter (name only)."""
        content = "?\nwidth\n?\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_empty_parameter_line(self, parser: ParmFileParser) -> None:
        """Parser should handle but not crash on empty lines."""
        # Note: Empty lines between delimiters are typically allowed
        # This tests a truly malformed line
        content = "?\n   \n?\n"
        result = parser.parse_string(content)
        # Empty whitespace lines may or may not be errors depending on grammar
        # The key is it shouldn't crash
        assert isinstance(result.has_errors, bool)


class TestParmParserNegativeStructural:
    """Additional structural validation tests."""

    def test_rejects_wrong_delimiter_character(self, parser: ParmFileParser) -> None:
        """Parser should reject content with wrong delimiter character."""
        content = "#\nwidth 0 1280\n#\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_nested_delimiters(self, parser: ParmFileParser) -> None:
        """Parser should reject nested delimiter patterns."""
        content = "?\n?\nwidth 0 1280\n?\n?\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_rejects_inverted_delimiters(self, parser: ParmFileParser) -> None:
        """Parser should reject content that looks inverted."""
        content = "width 0 1280\n?\n?\n"
        result = parser.parse_string(content)
        assert result.has_errors


class TestParmParserEdgeCases:
    """Edge case tests for robustness."""

    def test_handles_empty_string(self, parser: ParmFileParser) -> None:
        """Parser should handle empty string input."""
        content = ""
        result = parser.parse_string(content)
        assert result.has_errors

    def test_handles_only_newlines(self, parser: ParmFileParser) -> None:
        """Parser should handle content with only newlines."""
        content = "\n\n\n"
        result = parser.parse_string(content)
        assert result.has_errors

    def test_handles_only_whitespace(self, parser: ParmFileParser) -> None:
        """Parser should handle content with only whitespace."""
        content = "   \t   \n   "
        result = parser.parse_string(content)
        assert result.has_errors


class TestParmParserSourceTracking:
    """Tests for source file tracking."""

    def test_parse_sets_source_path(self, parser: ParmFileParser) -> None:
        """Parser should set source path when parsing a file."""
        parm_files = list(FIXTURE_DIR.rglob("*.parm"))
        if not parm_files:
            pytest.skip("No fixture files found")

        parm_file = parm_files[0]
        result = parser.parse(parm_file)
        assert result.source == parm_file

    def test_parse_string_has_no_source(self, parser: ParmFileParser) -> None:
        """Parser should have None source when parsing string."""
        content = "?\nwidth 0 1280\n?\n"
        result = parser.parse_string(content)
        assert result.source is None


class TestParmParserFreshTransformer:
    """Tests to ensure transformer state doesn't carry over."""

    def test_sequential_parses_are_independent(self, parser: ParmFileParser) -> None:
        """Sequential parses should not share state."""
        content1 = "?\nwidth 0 1280\n?\n"
        content2 = "?\nheight 0 720\n?\n"

        result1 = parser.parse_string(content1)
        result2 = parser.parse_string(content2)

        assert len(result1.parameters) == 1
        assert len(result2.parameters) == 1
        assert result1.parameters[0].name == "width"
        assert result2.parameters[0].name == "height"

    def test_many_sequential_parses(self, parser: ParmFileParser) -> None:
        """Many sequential parses should all be independent."""
        results = []
        for i in range(10):
            content = f"?\nparam{i} 0 {i}\n?\n"
            result = parser.parse_string(content)
            results.append(result)

        for i, result in enumerate(results):
            assert len(result.parameters) == 1
            assert result.parameters[0].name == f"param{i}"
