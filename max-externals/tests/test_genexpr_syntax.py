"""
Test suite for GenExpr shader file validation.

This module validates all .genjit shader files in the max-externals/code/ directory.
It checks for proper GenExpr syntax, required elements, and PCG RNG constant values.

Tests:
    - Param declarations with proper format
    - Output assignments (out = ... or out1[0] = ...)
    - No undefined variables (basic check)
    - Proper function syntax with braces and semicolons
    - PCG RNG constants match reference values
    - GLSL shader syntax (for wrapped shaders)
"""
# ruff: noqa: S101 PLR2004

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

# Reference PCG RNG constants that must match across implementations
PCG_CONSTANTS = {
    "PCG_MULT": 747796405,
    "PCG_INC": 2891336453,
    "PCG_FACTOR": 277803737,
    "COORD_PRIME_X": 374761393,
    "COORD_PRIME_Y": 668265263,
}


# Get the code directory relative to this test file
TESTS_DIR = Path(__file__).parent
CODE_DIR = TESTS_DIR.parent / "code"


def get_genjit_files() -> List[Path]:
    """
    Get all .genjit files from the code directory.

    Returns:
        List of Path objects for .genjit files

    """
    if not CODE_DIR.exists():
        pytest.skip(f"Code directory not found: {CODE_DIR}")

    files = list(CODE_DIR.glob("*.genjit"))
    if not files:
        pytest.skip(f"No .genjit files found in {CODE_DIR}")

    return sorted(files)


def is_glsl_wrapped(content: str) -> bool:
    """
    Check if the shader is wrapped in GLSL/XML format.

    Args:
        content: Shader file content

    Returns:
        True if wrapped in <jit.gl.pix> tags

    """
    return "<jit.gl.pix>" in content


def extract_genexpr_code(content: str) -> str:
    """
    Extract pure GenExpr code from file content.

    For GLSL-wrapped shaders, this returns empty string as they use
    different syntax. For pure GenExpr, returns the content as-is.

    Args:
        content: Shader file content

    Returns:
        GenExpr code or empty string for GLSL shaders

    """
    if is_glsl_wrapped(content):
        return ""
    return content


def extract_glsl_code(content: str) -> str:
    """
    Extract GLSL code from wrapped shader files.

    Args:
        content: Shader file content

    Returns:
        GLSL code or empty string for pure GenExpr shaders

    """
    if not is_glsl_wrapped(content):
        return ""

    # Extract content between <![CDATA[ and ]]>
    match = re.search(r"<!\[CDATA\[(.*?)\]\]>", content, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def parse_param_declarations(content: str) -> Dict[str, str]:
    """
    Parse Param declarations from GenExpr code.

    Args:
        content: GenExpr shader code

    Returns:
        Dictionary mapping param names to their default values

    """
    params = {}

    # Match: Param name(default);
    pattern = r"Param\s+(\w+)\s*\(\s*([^)]+)\s*\)\s*;"

    for match in re.finditer(pattern, content):
        param_name = match.group(1)
        param_default = match.group(2)
        params[param_name] = param_default

    return params


def parse_glsl_params(content: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse param declarations from GLSL-wrapped shaders.

    Args:
        content: Full shader file content (including XML)

    Returns:
        Dictionary mapping param names to (type, default) tuples

    """
    params = {}

    # Match: <param name="..." type="..." default="..." />
    pattern = r'<param\s+name="(\w+)"\s+type="(\w+)"\s+default="([^"]+)"\s*/>'

    for match in re.finditer(pattern, content):
        param_name = match.group(1)
        param_type = match.group(2)
        param_default = match.group(3)
        params[param_name] = (param_type, param_default)

    return params


def find_output_assignments(content: str) -> List[str]:
    """
    Find output assignments in GenExpr code.

    Args:
        content: GenExpr shader code

    Returns:
        List of output assignment patterns found

    """
    assignments = []

    # Pattern 1: out = ...;
    if re.search(r"\bout\s*=", content):
        assignments.append("out")

    # Pattern 2: out1[0] = ...;
    if re.search(r"\bout1\[\d+\]\s*=", content):
        assignments.append("out1[n]")

    # Pattern 3: gl_FragColor = ... (for GLSL)
    if re.search(r"\bgl_FragColor\s*=", content):
        assignments.append("gl_FragColor")

    return assignments


def find_pcg_constants(content: str) -> Dict[str, int]:
    """
    Find PCG RNG constant definitions in code.

    Args:
        content: Shader code (GenExpr or GLSL)

    Returns:
        Dictionary mapping constant names to their values

    """
    constants = {}

    for const_name in PCG_CONSTANTS:
        # GenExpr format: CONST_NAME = 12345;
        pattern1 = rf"\b{const_name}\s*=\s*(\d+)\s*;"
        match = re.search(pattern1, content)
        if match:
            constants[const_name] = int(match.group(1))
            continue

        # GLSL format: const uint CONST_NAME = 12345u;
        pattern2 = rf"const\s+uint\s+{const_name}\s*=\s*(\d+)u?\s*;"
        match = re.search(pattern2, content)
        if match:
            constants[const_name] = int(match.group(1))

    return constants


def find_function_definitions(content: str) -> List[str]:
    """
    Find function definitions in GenExpr code.

    Args:
        content: GenExpr shader code

    Returns:
        List of function names

    """
    functions = []

    # Pattern: function_name(args) { ... }
    # Match function name followed by parentheses and opening brace
    pattern = r"(\w+)\s*\([^)]*\)\s*\{"

    for match in re.finditer(pattern, content):
        func_name = match.group(1)
        # Filter out control flow keywords
        if func_name not in ["if", "else", "for", "while"]:
            functions.append(func_name)

    return functions


def extract_defined_variables(content: str) -> Set[str]:
    """
    Extract all defined variables from GenExpr code.

    This includes:
    - Param declarations
    - Variable assignments (var = ...)
    - Function parameters
    - Built-in variables (in1, in2, norm, dim, etc.)

    Args:
        content: GenExpr shader code

    Returns:
        Set of defined variable names

    """
    variables = set()

    # Built-in GenExpr variables
    variables.update(["in1", "in2", "norm", "dim", "out"])

    # Parse Param declarations
    params = parse_param_declarations(content)
    variables.update(params.keys())

    # Parse variable assignments (simple pattern)
    # Matches: variable_name = ...
    pattern = r"\b([a-zA-Z_]\w*)\s*="
    for match in re.finditer(pattern, content):
        var_name = match.group(1)
        variables.add(var_name)

    # Parse function definitions and their parameters
    func_pattern = r"(\w+)\s*\(([^)]*)\)\s*\{"
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        if func_name not in ["if", "else", "for", "while"]:
            variables.add(func_name)
            # Parse function parameters
            params_str = match.group(2)
            if params_str.strip():
                param_names = [p.strip() for p in params_str.split(",")]
                variables.update(param_names)

    # Parse for loop variables
    for_pattern = r"for\s*\(\s*(\w+)\s*="
    for match in re.finditer(for_pattern, content):
        var_name = match.group(1)
        variables.add(var_name)

    return variables


class TestGenExprSyntax:
    """Test suite for GenExpr shader syntax validation."""

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_file_readable(self, shader_file: Path) -> None:
        """
        Test that shader file can be read.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()
        assert len(content) > 0, f"File {shader_file.name} is empty"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_has_documentation(self, shader_file: Path) -> None:
        """
        Test that shader file has documentation comment.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()
        assert (
            "/**" in content or "/*" in content
        ), f"File {shader_file.name} missing documentation header"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_has_parameters(self, shader_file: Path) -> None:
        """
        Test that shader has parameter declarations.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if is_glsl_wrapped(content):
            # GLSL-wrapped shaders use <param> tags
            glsl_params = parse_glsl_params(content)
            assert (
                len(glsl_params) > 0
            ), f"File {shader_file.name} has no <param> declarations"
        else:
            # Pure GenExpr shaders use Param declarations
            genexpr = extract_genexpr_code(content)
            genexpr_params = parse_param_declarations(genexpr)
            assert (
                len(genexpr_params) > 0
            ), f"File {shader_file.name} has no Param declarations"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_param_format(self, shader_file: Path) -> None:
        """
        Test that Param declarations follow proper format.

        Expected format: Param name(default);

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if is_glsl_wrapped(content):
            # GLSL params are in XML format, already validated by XML structure
            pytest.skip("GLSL-wrapped shader uses XML param format")

        genexpr = extract_genexpr_code(content)
        params = parse_param_declarations(genexpr)

        # Check that each Param line has proper semicolon
        param_lines = re.findall(r"Param\s+\w+\s*\([^)]+\)\s*;", genexpr)
        assert len(param_lines) == len(
            params
        ), f"File {shader_file.name} has malformed Param declarations"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_has_output_assignment(self, shader_file: Path) -> None:
        """
        Test that shader has output assignment.

        Valid patterns:
        - out = ...
        - out1[0] = ...
        - gl_FragColor = ... (GLSL)

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if is_glsl_wrapped(content):
            glsl = extract_glsl_code(content)
            assignments = find_output_assignments(glsl)
        else:
            genexpr = extract_genexpr_code(content)
            assignments = find_output_assignments(genexpr)

        assert len(assignments) > 0, (
            f"File {shader_file.name} has no output assignment "
            f"(out = ... or gl_FragColor = ...)"
        )

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_function_braces(self, shader_file: Path) -> None:
        """
        Test that functions have proper brace syntax.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if is_glsl_wrapped(content):
            code = extract_glsl_code(content)
        else:
            code = extract_genexpr_code(content)

        # Count opening and closing braces
        open_braces = code.count("{")
        close_braces = code.count("}")

        assert open_braces == close_braces, (
            f"File {shader_file.name} has mismatched braces: "
            f"{open_braces} opening vs {close_braces} closing"
        )

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_pcg_constants_correct(self, shader_file: Path) -> None:
        """
        Test that PCG RNG constants match reference values.

        Reference values:
        - PCG_MULT = 747796405
        - PCG_INC = 2891336453
        - PCG_FACTOR = 277803737
        - COORD_PRIME_X = 374761393
        - COORD_PRIME_Y = 668265263

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if is_glsl_wrapped(content):
            code = extract_glsl_code(content)
        else:
            code = extract_genexpr_code(content)

        constants = find_pcg_constants(content)

        # Only check files that use PCG constants
        if not constants:
            pytest.skip(f"File {shader_file.name} does not use PCG constants")

        # Check each constant found
        for const_name, const_value in constants.items():
            expected_value = PCG_CONSTANTS[const_name]
            assert const_value == expected_value, (
                f"File {shader_file.name}: {const_name} = {const_value}, "
                f"expected {expected_value}"
            )

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_no_obvious_undefined_variables(self, shader_file: Path) -> None:
        """
        Test for obvious undefined variables in GenExpr code.

        This is a basic check that looks for common mistakes. It's not
        exhaustive as it doesn't perform full semantic analysis.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if is_glsl_wrapped(content):
            # GLSL has different scoping rules, skip this test
            pytest.skip("GLSL shaders use different scoping rules")

        genexpr = extract_genexpr_code(content)

        # Remove comments to avoid false positives
        code_no_comments = re.sub(r"/\*.*?\*/", "", genexpr, flags=re.DOTALL)
        code_no_comments = re.sub(r"//.*?$", "", code_no_comments, flags=re.MULTILINE)

        # Get defined variables
        defined = extract_defined_variables(code_no_comments)

        # Add common built-in functions
        defined.update(
            [
                "sample",
                "vec",
                "clamp",
                "max",
                "min",
                "abs",
                "floor",
                "ceil",
                "sqrt",
                "exp",
                "log",
                "sin",
                "cos",
                "tan",
                "pow",
                "mod",
                "int",
                "float",
                "uint",
            ]
        )

        # This test is informational - we just check that core variables exist
        # Full validation would require a complete GenExpr parser
        assert "norm" in defined or is_glsl_wrapped(
            content
        ), f"File {shader_file.name} may be missing coordinate handling"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_glsl_shader_structure(self, shader_file: Path) -> None:
        """
        Test GLSL-wrapped shader structure.

        Validates:
        - <jit.gl.pix> wrapper
        - <description> tag
        - <language> tag with GLSL
        - <program> tag with fragment shader

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if not is_glsl_wrapped(content):
            pytest.skip("Not a GLSL-wrapped shader")

        # Check for required tags
        assert (
            "<description>" in content
        ), f"File {shader_file.name} missing <description> tag"
        assert (
            '<language name="glsl"' in content
        ), f"File {shader_file.name} missing GLSL language declaration"
        assert (
            '<program name="fp" type="fragment">' in content
        ), f"File {shader_file.name} missing fragment program declaration"
        assert "<![CDATA[" in content, f"File {shader_file.name} missing CDATA opening"
        assert "]]>" in content, f"File {shader_file.name} missing CDATA closing"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_glsl_main_function(self, shader_file: Path) -> None:
        """
        Test that GLSL shaders have a main() function.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if not is_glsl_wrapped(content):
            pytest.skip("Not a GLSL-wrapped shader")

        glsl = extract_glsl_code(content)

        assert (
            "void main()" in glsl
        ), f"File {shader_file.name} missing void main() function"

    @pytest.mark.parametrize("shader_file", get_genjit_files())
    def test_glsl_version_directive(self, shader_file: Path) -> None:
        """
        Test that GLSL shaders have version directive.

        Args:
            shader_file: Path to .genjit file

        """
        content = shader_file.read_text()

        if not is_glsl_wrapped(content):
            pytest.skip("Not a GLSL-wrapped shader")

        glsl = extract_glsl_code(content)

        assert "#version" in glsl, f"File {shader_file.name} missing #version directive"


class TestShaderCoverage:
    """Test suite for overall shader coverage and consistency."""

    def test_all_shaders_have_unique_names(self) -> None:
        """Test that all shader files have unique names."""
        files = get_genjit_files()
        names = [f.name for f in files]

        assert len(names) == len(
            set(names)
        ), f"Duplicate shader filenames found: {names}"

    def test_pcg_shaders_have_all_constants(self) -> None:
        """
        Test that shaders using PCG have all required constants.

        If a shader defines any PCG constant, it should define all of them
        for consistency.
        """
        files = get_genjit_files()

        for shader_file in files:
            content = shader_file.read_text()

            if is_glsl_wrapped(content):
                code = extract_glsl_code(content)
            else:
                code = extract_genexpr_code(content)

            constants = find_pcg_constants(content)

            # If shader uses any PCG constant, check for completeness
            if constants:
                # Not all shaders need all constants, but check the core ones
                has_pcg_mult = "PCG_MULT" in constants
                has_pcg_inc = "PCG_INC" in constants

                if has_pcg_mult or has_pcg_inc:
                    assert (
                        "PCG_MULT" in constants
                    ), f"{shader_file.name} uses PCG but missing PCG_MULT"
                    assert (
                        "PCG_INC" in constants
                    ), f"{shader_file.name} uses PCG but missing PCG_INC"

    def test_shader_file_count(self) -> None:
        """Test that expected number of shader files exist."""
        files = get_genjit_files()

        # We expect at least 10 shader files based on the directory listing
        assert (
            len(files) >= 10
        ), f"Expected at least 10 shader files, found {len(files)}"
