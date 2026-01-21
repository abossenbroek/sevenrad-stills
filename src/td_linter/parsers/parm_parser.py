"""Parser for TouchDesigner .parm (parameter) files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

# Load the grammar from the grammars directory
_GRAMMAR_PATH = Path(__file__).parent.parent / "grammars" / "parm_file.lark"


@dataclass
class Parameter:
    """A single parameter definition."""

    name: str
    mode: int  # 0=constant, 17=string expr, 32=numeric, 49=expression
    value: str | float
    expression: str | None = None


@dataclass
class ParsedParmFile:
    """Parsed .parm file contents."""

    parameters: list[Parameter] = field(default_factory=list)
    has_errors: bool = False
    source: Path | None = None


class ParmFileTransformer(Transformer[Any, ParsedParmFile]):
    """Transform parse tree into ParsedParmFile."""

    def __init__(self) -> None:
        """Initialize transformer."""
        super().__init__()
        self._parameters: list[Parameter] = []

    def param_line(self, items: list[Any]) -> None:
        """Extract a parameter line."""
        if len(items) >= 3:
            name = str(items[0])
            mode = int(items[1])
            value = items[2]
            expression = str(items[3]) if len(items) > 3 else None

            # Convert value to appropriate type
            if isinstance(value, str):
                # Try to convert to float if it looks like a number
                try:
                    value = float(value)
                except ValueError:
                    pass

            self._parameters.append(
                Parameter(name=name, mode=mode, value=value, expression=expression)
            )

    def param_value(self, items: list[Any]) -> Any:
        """Extract parameter value."""
        if items:
            val = items[0]
            # Strip quotes from strings
            if hasattr(val, "type") and val.type == "ESCAPED_STRING":
                return str(val)[1:-1]  # Remove surrounding quotes
            return val
        return ""

    def expression(self, items: list[Any]) -> str:
        """Extract expression text."""
        return str(items[0]).strip() if items else ""

    def start(self, items: list[Any]) -> ParsedParmFile:
        """Build final ParsedParmFile."""
        return ParsedParmFile(parameters=self._parameters)


class ParmFileParser:
    """Parser for TouchDesigner .parm files."""

    def __init__(self) -> None:
        """Initialize the parser with the Lark grammar."""
        grammar = _GRAMMAR_PATH.read_text()
        # Don't use transformer= parameter to avoid state carryover
        self._parser = Lark(grammar, parser="lalr")

    def parse(self, file_path: Path) -> ParsedParmFile:
        """Parse a .parm file into structured data."""
        content = file_path.read_text()
        try:
            tree = self._parser.parse(content)
            # Create fresh transformer for each parse
            transformer = ParmFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedParmFile):
                result.source = file_path
                return result
            return ParsedParmFile(has_errors=True, source=file_path)
        except Exception:
            return ParsedParmFile(has_errors=True, source=file_path)

    def parse_string(self, content: str) -> ParsedParmFile:
        """Parse a .parm file content string into structured data."""
        try:
            tree = self._parser.parse(content)
            # Create fresh transformer for each parse
            transformer = ParmFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedParmFile):
                return result
            return ParsedParmFile(has_errors=True)
        except Exception:
            return ParsedParmFile(has_errors=True)
