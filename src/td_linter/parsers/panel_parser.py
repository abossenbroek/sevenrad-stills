"""Parser for TouchDesigner .panel (panel UI state) files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

# Load the grammar from the grammars directory
_GRAMMAR_PATH = Path(__file__).parent.parent / "grammars" / "panel_file.lark"


@dataclass
class ParsedPanelFile:
    """Parsed .panel file contents."""

    # Header values (3 integers)
    header1: int | None = None
    header2: int | None = None
    header3: int | None = None

    # Named values
    u: float | None = None
    v: float | None = None
    trueu: float | None = None
    truev: float | None = None
    screenw: int | None = None
    screenh: int | None = None

    has_errors: bool = False
    source: Path | None = None


class PanelFileTransformer(Transformer[Any, ParsedPanelFile]):
    """Transform parse tree into ParsedPanelFile."""

    def __init__(self) -> None:
        """Initialize transformer."""
        super().__init__()
        self._headers: list[int] = []
        self._u: float | None = None
        self._v: float | None = None
        self._trueu: float | None = None
        self._truev: float | None = None
        self._screenw: int | None = None
        self._screenh: int | None = None

    @v_args(inline=True)
    def header_line(self, value: Any) -> None:
        """Extract header integer."""
        self._headers.append(int(value))

    @v_args(inline=True)
    def named_entry(self, key: Any, value: Any) -> None:
        """Extract named key-value pair."""
        key_str = str(key)
        if key_str == "u":
            self._u = float(value)
        elif key_str == "v":
            self._v = float(value)
        elif key_str == "trueu":
            self._trueu = float(value)
        elif key_str == "truev":
            self._truev = float(value)
        elif key_str == "screenw":
            self._screenw = int(float(value))
        elif key_str == "screenh":
            self._screenh = int(float(value))

    def start(self, _items: list[Any]) -> ParsedPanelFile:  # noqa: ARG002
        """Build final ParsedPanelFile."""
        return ParsedPanelFile(
            header1=self._headers[0] if len(self._headers) > 0 else None,
            header2=self._headers[1] if len(self._headers) > 1 else None,
            header3=self._headers[2] if len(self._headers) > 2 else None,
            u=self._u,
            v=self._v,
            trueu=self._trueu,
            truev=self._truev,
            screenw=self._screenw,
            screenh=self._screenh,
        )


class PanelFileParser:
    """Parser for TouchDesigner .panel files."""

    def __init__(self) -> None:
        """Initialize the parser with the Lark grammar."""
        grammar = _GRAMMAR_PATH.read_text()
        self._parser = Lark(grammar, parser="lalr")

    def parse(self, file_path: Path) -> ParsedPanelFile:
        """Parse a .panel file into structured data."""
        content = file_path.read_text()
        try:
            tree = self._parser.parse(content)
            transformer = PanelFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedPanelFile):
                result.source = file_path
                return result
            return ParsedPanelFile(has_errors=True, source=file_path)
        except Exception:
            return ParsedPanelFile(has_errors=True, source=file_path)

    def parse_string(self, content: str) -> ParsedPanelFile:
        """Parse a .panel file content string into structured data."""
        try:
            tree = self._parser.parse(content)
            transformer = PanelFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedPanelFile):
                return result
            return ParsedPanelFile(has_errors=True)
        except Exception:
            return ParsedPanelFile(has_errors=True)
