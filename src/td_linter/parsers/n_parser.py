"""Parser for TouchDesigner .n (node definition) files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

# Load the grammar from the grammars directory
_GRAMMAR_PATH = Path(__file__).parent.parent / "grammars" / "n_file.lark"


@dataclass
class ParsedNFile:
    """Parsed .n file contents."""

    family: str
    op_type: str
    tile: tuple[int, int, int, int]
    flags: dict[str, str] = field(default_factory=dict)
    inputs: list[tuple[int, str]] = field(default_factory=list)
    color: tuple[float, ...] | None = None
    dock: str | None = None
    view: list[Any] | None = None
    has_errors: bool = False
    source: Path | None = None


class NFileTransformer(Transformer[Any, ParsedNFile]):
    """Transform parse tree into ParsedNFile."""

    def __init__(self) -> None:
        """Initialize transformer."""
        super().__init__()
        self._family: str = ""
        self._op_type: str = ""
        self._tile: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._flags: dict[str, str] = {}
        self._inputs: list[tuple[int, str]] = []
        self._color: tuple[float, ...] | None = None
        self._dock: str | None = None
        self._view: list[Any] | None = None

    @v_args(inline=True)
    def type_header(self, family: Any, op_type: Any) -> None:
        """Extract type header."""
        self._family = str(family)
        self._op_type = str(op_type)

    @v_args(inline=True)
    def tile_directive(self, x: Any, y: Any, w: Any, h: Any) -> None:
        """Extract tile position."""
        self._tile = (int(x), int(y), int(w), int(h))

    def flags_directive(self, items: list[Any]) -> None:
        """Extract flags."""
        # items are flag_pair results
        for item in items:
            if isinstance(item, tuple):
                self._flags[item[0]] = item[1]

    @v_args(inline=True)
    def flag_pair(self, name: Any, value: Any) -> tuple[str, str]:
        """Extract a single flag pair."""
        return (str(name), str(value))

    @v_args(inline=True)
    def flag_value(self, value: Any) -> str:
        """Extract flag value."""
        return str(value)

    def inputs_block(self, items: list[Any]) -> None:
        """Extract inputs."""
        for item in items:
            if isinstance(item, tuple):
                self._inputs.append(item)

    @v_args(inline=True)
    def input_entry(self, index: Any, ref: Any) -> tuple[int, str]:
        """Extract a single input entry."""
        return (int(index), str(ref))

    def color_directive(self, items: list[Any]) -> None:
        """Extract color values."""
        self._color = tuple(float(x) for x in items)

    @v_args(inline=True)
    def dock_directive(self, name: Any) -> None:
        """Extract dock reference."""
        self._dock = str(name)

    def view_directive(self, items: list[Any]) -> None:
        """Extract view values."""
        self._view = [
            str(x) if hasattr(x, "type") and x.type == "ESCAPED_STRING" else x
            for x in items
        ]

    def view_value(self, items: list[Any]) -> Any:
        """Extract single view value."""
        return items[0] if items else None

    def body(self, items: list[Any]) -> None:
        """Process body items (inputs, color, dock, view)."""
        # Body items are already processed by their specific handlers
        pass

    def start(self, items: list[Any]) -> ParsedNFile:
        """Build final ParsedNFile."""
        return ParsedNFile(
            family=self._family,
            op_type=self._op_type,
            tile=self._tile,
            flags=self._flags,
            inputs=self._inputs,
            color=self._color,
            dock=self._dock,
            view=self._view,
        )


class NFileParser:
    """Parser for TouchDesigner .n files."""

    def __init__(self) -> None:
        """Initialize the parser with the Lark grammar."""
        grammar = _GRAMMAR_PATH.read_text()
        # Don't use transformer= parameter to avoid state carryover
        self._parser = Lark(grammar, parser="lalr")

    def parse(self, file_path: Path) -> ParsedNFile:
        """Parse a .n file into structured data."""
        content = file_path.read_text()
        try:
            tree = self._parser.parse(content)
            # Create fresh transformer for each parse
            transformer = NFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedNFile):
                result.source = file_path
                return result
            # If transformer didn't produce ParsedNFile, something went wrong
            return ParsedNFile(
                family="UNKNOWN",
                op_type="unknown",
                tile=(0, 0, 0, 0),
                has_errors=True,
                source=file_path,
            )
        except Exception:
            return ParsedNFile(
                family="UNKNOWN",
                op_type="unknown",
                tile=(0, 0, 0, 0),
                has_errors=True,
                source=file_path,
            )

    def parse_string(self, content: str) -> ParsedNFile:
        """Parse a .n file content string into structured data."""
        try:
            tree = self._parser.parse(content)
            # Create fresh transformer for each parse
            transformer = NFileTransformer()
            result = transformer.transform(tree)
            if isinstance(result, ParsedNFile):
                return result
            return ParsedNFile(
                family="UNKNOWN",
                op_type="unknown",
                tile=(0, 0, 0, 0),
                has_errors=True,
            )
        except Exception:
            return ParsedNFile(
                family="UNKNOWN",
                op_type="unknown",
                tile=(0, 0, 0, 0),
                has_errors=True,
            )
