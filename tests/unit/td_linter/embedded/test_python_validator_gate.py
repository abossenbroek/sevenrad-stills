"""
Gate tests for TDL-032: Python AST Validator.

Tests Python syntax and undefined name detection with TD builtins.
"""

import pytest
from td_linter.embedded import PythonValidator


class TestPositiveSyntaxValidation:
    """G3.2-P1: Syntax errors should be caught."""

    def test_syntax_error_detected(self):
        """Syntax error in script is caught."""
        content = """
        def onCook():
            x = 5
            y =    # Missing value
        """
        validator = PythonValidator()
        violations = list(validator.validate(content))

        syntax_errors = [v for v in violations if v.rule == "python-syntax-error"]
        assert len(syntax_errors) == 1

    def test_indentation_error_detected(self):
        """Indentation error is caught."""
        content = """
def onCook():
x = 5
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        syntax_errors = [v for v in violations if v.rule == "python-syntax-error"]
        assert len(syntax_errors) == 1

    def test_valid_script_no_syntax_errors(self):
        """Valid script has no syntax errors."""
        content = """
def onCook():
    x = 5
    return x * 2
"""
        validator = PythonValidator()
        violations = list(validator.validate(content, check_undefined=False))

        syntax_errors = [v for v in violations if v.rule == "python-syntax-error"]
        assert len(syntax_errors) == 0


class TestPositiveTDBuiltins:
    """G3.3-P1: TD builtins should not be flagged as undefined."""

    def test_op_not_flagged(self):
        """op() function is not flagged."""
        content = """
target = op('/project1/geo1')
target.par.tx = 5
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "op" not in names

    def test_me_not_flagged(self):
        """Me object is not flagged."""
        content = """
me.par.tx = 10
me.cook(force=True)
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "me" not in names

    def test_absTime_not_flagged(self):
        """AbsTime object is not flagged."""
        content = """
frame = absTime.frame
seconds = absTime.seconds
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "absTime" not in names

    def test_project_not_flagged(self):
        """Project object is not flagged."""
        content = """
name = project.name
folder = project.folder
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "project" not in names

    def test_td_module_not_flagged(self):
        """Td and tdu modules are not flagged."""
        content = """
value = tdu.clamp(x, 0, 1)
color = tdu.Color(1, 0, 0)
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]
        assert "tdu" not in names
        assert "td" not in names


class TestNegativeUndefinedNames:
    """G3.3-N1: Non-TD undefined names should be warnings."""

    def test_custom_undefined_is_warning(self):
        """Custom undefined names are warnings."""
        content = """
result = custom_function(x)
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]

        # custom_function should be flagged
        assert "custom_function" in names

        # All undefined violations should be warnings, not errors
        for v in undefined:
            assert v.severity == "warning"


class TestPositiveCallbackCompleteness:
    """G3-P4: Callback completeness check."""

    def test_missing_callbacks_info(self):
        """Missing callbacks are reported as INFO."""
        content = """
def onStart():
    pass

def onFrameStart(frame):
    pass
"""
        validator = PythonValidator()
        violations = list(
            validator.validate(content, check_undefined=False, check_callbacks=True)
        )

        missing = [v for v in violations if v.rule == "python-missing-callback"]

        # Should have missing callbacks reported
        assert len(missing) > 0

        # All should be INFO severity
        for v in missing:
            assert v.severity == "info"

    def test_no_callbacks_no_warning(self):
        """Script without callbacks doesn't trigger callback check."""
        content = """
def helper():
    return 5

x = helper()
"""
        validator = PythonValidator()
        violations = list(
            validator.validate(content, check_undefined=False, check_callbacks=True)
        )

        missing = [v for v in violations if v.rule == "python-missing-callback"]
        assert len(missing) == 0


class TestDefinedNames:
    """Test that defined names are not flagged."""

    def test_function_args_not_flagged(self):
        """Function arguments are not flagged as undefined."""
        content = """
def process(data, count):
    return data * count
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]

        assert "data" not in names
        assert "count" not in names

    def test_imports_not_flagged(self):
        """Imported names are not flagged."""
        content = """
import math
from os import path

result = math.sin(0)
p = path.join('a', 'b')
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]

        assert "math" not in names
        assert "path" not in names

    def test_for_loop_variable_not_flagged(self):
        """For loop variables are not flagged."""
        content = """
for i in range(10):
    print(i)
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]

        assert "i" not in names

    def test_comprehension_variable_not_flagged(self):
        """Comprehension variables are not flagged."""
        content = """
result = [x * 2 for x in range(10)]
"""
        validator = PythonValidator()
        violations = list(validator.validate(content))

        undefined = [v for v in violations if v.rule == "python-undefined-name"]
        names = [v.context.get("name") for v in undefined]

        assert "x" not in names
