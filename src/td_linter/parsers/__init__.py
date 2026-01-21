"""Parsers for TouchDesigner file formats."""

from td_linter.parsers.build_parser import BuildFileParser, ParsedBuildFile
from td_linter.parsers.n_parser import NFileParser, ParsedNFile
from td_linter.parsers.panel_parser import PanelFileParser, ParsedPanelFile
from td_linter.parsers.parm_parser import ParmFileParser, ParsedParmFile
from td_linter.parsers.start_parser import (
    ClockSettings,
    ParsedStartFile,
    StartFileParser,
)
from td_linter.parsers.text_parser import ParsedTextFile, TextFileParser
from td_linter.parsers.toc_parser import ParsedToc, TocParser

__all__ = [
    "BuildFileParser",
    "ClockSettings",
    "NFileParser",
    "PanelFileParser",
    "ParmFileParser",
    "ParsedBuildFile",
    "ParsedNFile",
    "ParsedPanelFile",
    "ParsedParmFile",
    "ParsedStartFile",
    "ParsedTextFile",
    "ParsedToc",
    "StartFileParser",
    "TextFileParser",
    "TocParser",
]
