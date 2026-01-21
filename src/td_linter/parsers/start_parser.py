"""Parser for TouchDesigner .start (runtime config) files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

# Load the grammar from the grammars directory
_GRAMMAR_PATH = Path(__file__).parent.parent / "grammars" / "start_file.lark"


@dataclass
class ClockSettings:
    """Clock configuration settings."""

    f: int | None = None  # Frame flag
    s: int | None = None  # Start flag
    o: int | None = None  # Offset flag
    w: int | None = None  # Wait flag


@dataclass
class ParsedStartFile:
    """Parsed .start file contents."""

    cookrate: int | None = None
    clock: ClockSettings | None = None
    realtime: bool | None = None
    viewers: bool | None = None
    resetaudioondevicechange: bool | None = None
    comments: list[str] = field(default_factory=list)
    extra_settings: dict[str, list[Any]] = field(default_factory=dict)
    has_errors: bool = False
    source: Path | None = None


class StartFileTransformer(Transformer[Any, ParsedStartFile]):
    """Transform parse tree into ParsedStartFile."""

    def __init__(self) -> None:
        """Initialize transformer."""
        super().__init__()
        self._cookrate: int | None = None
        self._clock: ClockSettings | None = None
        self._realtime: bool | None = None
        self._viewers: bool | None = None
        self._resetaudioondevicechange: bool | None = None
        self._comments: list[str] = []
        self._extra_settings: dict[str, list[Any]] = {}

    def setting_value(self, items: list[Any]) -> Any:
        """Extract setting value."""
        if items:
            val = items[0]
            val_str = str(val)
            # Handle on/off
            if val_str == "on":
                return True
            if val_str == "off":
                return False
            # Handle flag values like "-f 1"
            if val_str.startswith("-") and len(val_str) > 2:
                parts = val_str.split()
                if len(parts) == 2:
                    return (parts[0][1], int(parts[1]))  # ('f', 1)
            # Handle numbers
            try:
                return int(val_str)
            except ValueError:
                try:
                    return float(val_str)
                except ValueError:
                    return val_str
        return None

    def setting(self, items: list[Any]) -> None:
        """Process a setting line."""
        if not items:
            return

        name = str(items[0])
        values = items[1:]

        if name == "cookrate" and values:
            self._cookrate = values[0] if isinstance(values[0], int) else None
        elif name == "clock":
            clock = ClockSettings()
            for val in values:
                if isinstance(val, tuple) and len(val) == 2:
                    flag, num = val
                    if flag == "f":
                        clock.f = num
                    elif flag == "s":
                        clock.s = num
                    elif flag == "o":
                        clock.o = num
                    elif flag == "w":
                        clock.w = num
            self._clock = clock
        elif name == "realtime":
            self._realtime = values[0] if values and isinstance(values[0], bool) else None
        elif name == "viewers":
            self._viewers = values[0] if values and isinstance(values[0], bool) else None
        elif name == "resetaudioondevicechange":
            self._resetaudioondevicechange = (
                values[0] if values and isinstance(values[0], bool) else None
            )
        else:
            self._extra_settings[name] = values

    @v_args(inline=True)
    def comment(self, text: Any) -> None:
        """Extract comment text."""
        self._comments.append(str(text))

    def line(self, items: list[Any]) -> None:  # noqa: ARG002
        """Process a line (already handled by setting/comment)."""
        pass

    def start(self, _items: list[Any]) -> ParsedStartFile:  # noqa: ARG002
        """Build final ParsedStartFile."""
        return ParsedStartFile(
            cookrate=self._cookrate,
            clock=self._clock,
            realtime=self._realtime,
            viewers=self._viewers,
            resetaudioondevicechange=self._resetaudioondevicechange,
            comments=self._comments,
            extra_settings=self._extra_settings,
        )


class StartFileParser:
    """Parser for TouchDesigner .start files."""

    def __init__(self) -> None:
        """Initialize the parser with the Lark grammar."""
        grammar = _GRAMMAR_PATH.read_text()
        self._parser = Lark(grammar, parser="lalr")

    def parse(self, file_path: Path) -> ParsedStartFile:
        """Parse a .start file into structured data."""
        content = file_path.read_text()
        try:
            tree = self._parser.parse(content)
            transformer = StartFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedStartFile):
                result.source = file_path
                return result
            return ParsedStartFile(has_errors=True, source=file_path)
        except Exception:
            return ParsedStartFile(has_errors=True, source=file_path)

    def parse_string(self, content: str) -> ParsedStartFile:
        """Parse a .start file content string into structured data."""
        try:
            tree = self._parser.parse(content)
            transformer = StartFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedStartFile):
                return result
            return ParsedStartFile(has_errors=True)
        except Exception:
            return ParsedStartFile(has_errors=True)
