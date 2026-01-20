"""Parser for TouchDesigner .build (version/build metadata) files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

# Load the grammar from the grammars directory
_GRAMMAR_PATH = Path(__file__).parent.parent / "grammars" / "build_file.lark"


@dataclass
class ParsedBuildFile:
    """Parsed .build file contents."""

    version: str | None = None
    build: str | None = None
    time: str | None = None
    osname: str | None = None
    osversion: str | None = None
    has_errors: bool = False
    source: Path | None = None


class BuildFileTransformer(Transformer[Any, ParsedBuildFile]):
    """Transform parse tree into ParsedBuildFile."""

    def __init__(self) -> None:
        """Initialize transformer."""
        super().__init__()
        self._version: str | None = None
        self._build: str | None = None
        self._time: str | None = None
        self._osname: str | None = None
        self._osversion: str | None = None

    @v_args(inline=True)
    def entry(self, key: Any, value: Any) -> None:
        """Extract an entry (key-value pair)."""
        key_str = str(key).strip()
        value_str = str(value).strip()

        if key_str == "version":
            self._version = value_str
        elif key_str == "build":
            self._build = value_str
        elif key_str == "time":
            self._time = value_str
        elif key_str == "osname":
            self._osname = value_str
        elif key_str == "osversion":
            self._osversion = value_str

    def start(self, _items: list[Any]) -> ParsedBuildFile:  # noqa: ARG002
        """Build final ParsedBuildFile."""
        return ParsedBuildFile(
            version=self._version,
            build=self._build,
            time=self._time,
            osname=self._osname,
            osversion=self._osversion,
        )


class BuildFileParser:
    """Parser for TouchDesigner .build files."""

    def __init__(self) -> None:
        """Initialize the parser with the Lark grammar."""
        grammar = _GRAMMAR_PATH.read_text()
        self._parser = Lark(grammar, parser="lalr")

    def parse(self, file_path: Path) -> ParsedBuildFile:
        """Parse a .build file into structured data."""
        content = file_path.read_text()
        try:
            tree = self._parser.parse(content)
            transformer = BuildFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedBuildFile):
                result.source = file_path
                return result
            return ParsedBuildFile(has_errors=True, source=file_path)
        except Exception:
            return ParsedBuildFile(has_errors=True, source=file_path)

    def parse_string(self, content: str) -> ParsedBuildFile:
        """Parse a .build file content string into structured data."""
        try:
            tree = self._parser.parse(content)
            transformer = BuildFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedBuildFile):
                return result
            return ParsedBuildFile(has_errors=True)
        except Exception:
            return ParsedBuildFile(has_errors=True)
