"""
Gate tests for TDL-034: Parameter Expression Validator.

Tests Python expression validation in .parm files.
"""

import pytest
from td_linter.embedded import Expression, ExpressionValidator


class TestPositiveExpressionSyntax:
    """Syntax validation for expressions."""

    def test_valid_expression_passes(self):
        """Valid expression passes validation."""
        expr = Expression(
            param_name="tx",
            expression="absTime.frame * 0.5",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        syntax_errors = [v for v in violations if v.rule == "expression-syntax-error"]
        assert len(syntax_errors) == 0

    def test_complex_expression_passes(self):
        """Complex valid expression passes."""
        expr = Expression(
            param_name="ty",
            expression="math.sin(absTime.seconds * 2 * 3.14159) * 100",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        syntax_errors = [v for v in violations if v.rule == "expression-syntax-error"]
        assert len(syntax_errors) == 0

    def test_op_lookup_expression_passes(self):
        """Expression with op() lookup passes."""
        expr = Expression(
            param_name="value",
            expression="op('slider1').par.value0",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        syntax_errors = [v for v in violations if v.rule == "expression-syntax-error"]
        assert len(syntax_errors) == 0


class TestNegativeExpressionSyntax:
    """Syntax errors in expressions."""

    def test_syntax_error_caught(self):
        """Syntax error in expression is caught."""
        expr = Expression(
            param_name="tx",
            expression="absTime.frame *",  # Incomplete
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        syntax_errors = [v for v in violations if v.rule == "expression-syntax-error"]
        assert len(syntax_errors) == 1

    def test_statement_not_allowed(self):
        """Statements are not allowed in expressions."""
        expr = Expression(
            param_name="tx",
            expression="x = 5",  # Assignment is a statement
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        syntax_errors = [v for v in violations if v.rule == "expression-syntax-error"]
        assert len(syntax_errors) == 1


class TestExpressionGlobals:
    """Expression-specific globals."""

    def test_me_available(self):
        """Me is available in expressions."""
        expr = Expression(
            param_name="tx",
            expression="me.par.rx.eval",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        undefined = [v for v in violations if v.rule == "expression-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "me" not in names

    def test_absTime_available(self):
        """AbsTime is available in expressions."""
        expr = Expression(
            param_name="frame",
            expression="absTime.frame",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        undefined = [v for v in violations if v.rule == "expression-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "absTime" not in names

    def test_math_functions_available(self):
        """Math functions are available."""
        expr = Expression(
            param_name="value",
            expression="sin(absTime.seconds) + cos(absTime.seconds)",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        undefined = [v for v in violations if v.rule == "expression-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "sin" not in names
        assert "cos" not in names

    def test_undefined_name_warning(self):
        """Undefined names in expressions are warnings."""
        expr = Expression(
            param_name="value",
            expression="custom_var * 2",
            mode=49,
            line=1,
            source_file=None,
        )
        validator = ExpressionValidator()
        violations = list(validator.validate_expression(expr))

        undefined = [v for v in violations if v.rule == "expression-undefined-name"]
        assert len(undefined) == 1
        assert undefined[0].severity == "warning"


class TestExpressionExtraction:
    """Test expression extraction from parm files."""

    def test_extract_mode_49_expression(self):
        """Mode 49 expressions are extracted."""
        from td_linter.parsers.parm_parser import Parameter, ParsedParmFile

        parm_file = ParsedParmFile(
            parameters=[
                Parameter(
                    name="tx", mode=49, value=0, expression="absTime.frame * 0.5"
                ),
                Parameter(
                    name="ty", mode=0, value=5, expression=None
                ),  # Not expression
            ]
        )

        validator = ExpressionValidator()
        expressions = list(validator.extract_expressions(parm_file))

        assert len(expressions) == 1
        assert expressions[0].param_name == "tx"
        assert expressions[0].expression == "absTime.frame * 0.5"

    def test_extract_mode_17_expression(self):
        """Mode 17 (string expression) is extracted."""
        from td_linter.parsers.parm_parser import Parameter, ParsedParmFile

        parm_file = ParsedParmFile(
            parameters=[
                Parameter(
                    name="file",
                    mode=17,
                    value="",
                    expression="project.folder + '/data'",
                ),
            ]
        )

        validator = ExpressionValidator()
        expressions = list(validator.extract_expressions(parm_file))

        assert len(expressions) == 1
        assert expressions[0].param_name == "file"

    def test_skip_constant_parameters(self):
        """Constant parameters (mode 0) are skipped."""
        from td_linter.parsers.parm_parser import Parameter, ParsedParmFile

        parm_file = ParsedParmFile(
            parameters=[
                Parameter(name="tx", mode=0, value=5, expression=None),
                Parameter(name="ty", mode=0, value=10, expression=None),
            ]
        )

        validator = ExpressionValidator()
        expressions = list(validator.extract_expressions(parm_file))

        assert len(expressions) == 0
