"""Parser for TouchDesigner .text (embedded code) files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

# Load the grammar from the grammars directory
_GRAMMAR_PATH = Path(__file__).parent.parent / "grammars" / "text_file.lark"


@dataclass
class ParsedTextFile:
    """Parsed .text file contents."""

    header_byte: int | None = None
    content: str = ""
    has_errors: bool = False
    source: Path | None = None


def _strip_binary_prefix(content: str) -> str:
    """Strip any binary/non-printable prefix from content.

    TouchDesigner .text files sometimes have a binary line after the header
    before the actual text content starts.
    """
    lines = content.split("\n")
    result_lines: list[str] = []
    found_text = False

    for line in lines:
        # Check if line contains printable text (not just binary/special chars)
        # A line is considered text if it has printable ASCII or common code patterns
        is_text_line = False
        if line.strip():
            # Count printable ASCII characters (space through tilde)
            printable_count = sum(1 for c in line if 32 <= ord(c) <= 126)
            # Consider it text if more than half the characters are printable
            # or if it starts with common code patterns
            if printable_count > len(line.strip()) / 2:
                is_text_line = True
            elif line.strip().startswith(("//", "#", "/*", "def ", "class ", "import ")):
                is_text_line = True

        if is_text_line or found_text:
            found_text = True
            result_lines.append(line)
        elif not found_text and not line.strip():
            # Keep empty lines only after we've found text
            pass

    return "\n".join(result_lines)


class TextFileTransformer(Transformer[Any, ParsedTextFile]):
    """Transform parse tree into ParsedTextFile."""

    def __init__(self) -> None:
        """Initialize transformer."""
        super().__init__()
        self._header_byte: int | None = None
        self._content: str = ""

    @v_args(inline=True)
    def start(self, header: Any, content: Any) -> ParsedTextFile:
        """Build final ParsedTextFile."""
        header_byte = int(header)
        raw_content = str(content)
        # Strip any binary prefix from the content
        clean_content = _strip_binary_prefix(raw_content)

        return ParsedTextFile(
            header_byte=header_byte,
            content=clean_content,
        )


class TextFileParser:
    """Parser for TouchDesigner .text files."""

    def __init__(self) -> None:
        """Initialize the parser with the Lark grammar."""
        grammar = _GRAMMAR_PATH.read_text()
        self._parser = Lark(grammar, parser="lalr")

    def parse(self, file_path: Path) -> ParsedTextFile:
        """Parse a .text file into structured data."""
        content = file_path.read_text(errors="replace")
        try:
            tree = self._parser.parse(content)
            transformer = TextFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedTextFile):
                result.source = file_path
                return result
            return ParsedTextFile(has_errors=True, source=file_path)
        except Exception:
            return ParsedTextFile(has_errors=True, source=file_path)

    def parse_string(self, content: str) -> ParsedTextFile:
        """Parse a .text file content string into structured data."""
        try:
            tree = self._parser.parse(content)
            transformer = TextFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedTextFile):
                return result
            return ParsedTextFile(has_errors=True)
        except Exception:
            return ParsedTextFile(has_errors=True)
