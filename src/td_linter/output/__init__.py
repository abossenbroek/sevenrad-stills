"""Output formatters for td-linter.

This module provides formatters for different output formats:
- text: Human-readable colored terminal output
- json: Machine-readable JSON with stable schema
- sarif: SARIF 2.1.0 for GitHub Code Scanning
"""

from td_linter.output.base import OutputFormatter
from td_linter.output.json_formatter import JSONFormatter
from td_linter.output.sarif import SARIFFormatter
from td_linter.output.text import TextFormatter

__all__ = [
    "OutputFormatter",
    "TextFormatter",
    "JSONFormatter",
    "SARIFFormatter",
    "get_formatter",
]

FORMATTERS: dict[str, type[OutputFormatter]] = {
    "text": TextFormatter,
    "json": JSONFormatter,
    "sarif": SARIFFormatter,
}


def get_formatter(name: str, **kwargs: object) -> OutputFormatter:
    """Get a formatter by name.

    Args:
        name: Formatter name ('text', 'json', 'sarif')
        **kwargs: Formatter-specific options (e.g., no_color for text)

    Returns:
        OutputFormatter instance

    Raises:
        ValueError: If formatter name is unknown
    """
    if name not in FORMATTERS:
        valid = ", ".join(FORMATTERS.keys())
        msg = f"Unknown formatter: {name}. Valid options: {valid}"
        raise ValueError(msg)

    formatter_class = FORMATTERS[name]

    # Only pass kwargs that the formatter accepts
    if name == "text" and "no_color" in kwargs:
        return formatter_class(no_color=bool(kwargs["no_color"]))  # type: ignore[call-arg]

    return formatter_class()
