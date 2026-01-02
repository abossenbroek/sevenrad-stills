"""Maxhelp linter orchestrator combining all validator mixins.

This module provides the MaxhelpLinter class that orchestrates all
validation rules by inheriting from the validator mixins.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from max_linter.lint_error import LintError
from max_linter.lint_graph import LintGraph
from max_linter.types import Severity

from .connection import ConnectionValidatorMixin
from .context import ContextValidatorMixin
from .dead_code import DeadCodeValidatorMixin
from .dial import DialValidatorMixin
from .feedback import FeedbackValidatorMixin
from .flow import FlowValidatorMixin
from .genexpr import GenExprValidatorMixin
from .metadata import MetadataValidatorMixin
from .param_ui import ParamUIValidatorMixin
from .signal_flow import SignalFlowValidatorMixin
from .trigger import TriggerValidatorMixin
from .ui_overlap import OverlapValidatorMixin

if TYPE_CHECKING:
    from max_linter.genexpr import GenExprValidator

# Try to import GenExprValidator
try:
    from max_linter.genexpr import GenExprValidator as _GenExprValidator

    GENEXPR_AVAILABLE = True
except ImportError:
    GENEXPR_AVAILABLE = False
    _GenExprValidator = None  # type: ignore[assignment, misc]


class MaxhelpLinter(
    ContextValidatorMixin,
    ConnectionValidatorMixin,
    SignalFlowValidatorMixin,
    OverlapValidatorMixin,
    GenExprValidatorMixin,  # Provides _find_genjit_file, _extract_gen_shader
    ParamUIValidatorMixin,  # Provides _find_param_messages
    DialValidatorMixin,  # Depends on GenExpr and ParamUI mixins
    FlowValidatorMixin,
    DeadCodeValidatorMixin,
    FeedbackValidatorMixin,
    TriggerValidatorMixin,
    MetadataValidatorMixin,
):
    """Validates .maxhelp files using graph-based analysis.

    This class combines all validator mixins to provide comprehensive
    validation of Max/MSP help patchers.

    Attributes:
        strict: If True, treat warnings as errors.
        verbose: If True, show info messages.
        errors: List of error-level lint messages.
        warnings: List of warning-level lint messages.
        filepath: Path to the current file being validated.
        data: Parsed JSON data from the maxhelp file.
        graph: NetworkX directed graph of patcher connections.
        boxes: Dict mapping box_id to box properties.
        lint_graph: LintGraph instance for advanced analysis.
        genexpr_validator: GenExprValidator for shader validation.
        c_external_params: Parameter metadata for C externals.
    """

    def __init__(self, strict: bool = False, verbose: bool = False) -> None:
        """Initialize the linter.

        Args:
            strict: Treat warnings as errors.
            verbose: Show info messages.
        """
        self.strict = strict
        self.verbose = verbose
        self.errors: list[LintError] = []
        self.warnings: list[LintError] = []
        self.filepath: Path | None = None
        self.data: dict[str, Any] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.boxes: dict[str, dict[str, Any]] = {}
        self.lint_graph: LintGraph = LintGraph(
            graph=nx.DiGraph(),
            boxes={},
            type_map={},
            cycles=[],
            orphans=set(),
            dead_branches=set(),
        )
        # Initialize GenExprValidator if available
        self.genexpr_validator: GenExprValidator | None = (
            _GenExprValidator() if GENEXPR_AVAILABLE else None
        )
        # Load C external parameter bounds metadata
        self.c_external_params = self._load_c_external_params()

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Add an error."""
        self.errors.append(LintError(Severity.ERROR, rule, message, object_id))

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Add a warning."""
        self.warnings.append(LintError(Severity.WARNING, rule, message, object_id))

    def info(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Add an info message (only shown in verbose mode)."""
        if self.verbose:
            self.warnings.append(LintError(Severity.INFO, rule, message, object_id))

    def _load_c_external_params(self) -> dict[str, dict[str, Any]]:
        """Load C external parameter bounds from metadata file.

        The metadata file (c_external_params.json) provides parameter bounds
        for C externals like sr.maskgen and sr.tilegen that don't have .genjit
        files to parse.

        Returns:
            Dict mapping external name -> param name -> {min, max, default, type}
            Returns empty dict if file not found or invalid.
        """
        # Look for metadata file in tools directory
        tools_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "tools"
        metadata_path = tools_dir / "c_external_params.json"
        if not metadata_path.exists():
            return {}
        try:
            data: dict[str, dict[str, Any]] = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            return data
        except (json.JSONDecodeError, OSError):
            return {}

    def _build_graph(self) -> nx.DiGraph:
        """Build directed graph where nodes are (box_id, port_type, port_num) tuples.

        This allows tracking signal flow through the patcher.
        """
        graph = nx.DiGraph()

        # Add nodes for each box's inlets and outlets
        for box_wrapper in self.data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            box_id = box.get("id", "")

            # Store box reference in node attributes
            for i in range(box.get("numoutlets", 0)):
                graph.add_node((box_id, "out", i), box=box)
            for i in range(box.get("numinlets", 0)):
                graph.add_node((box_id, "in", i), box=box)

            # Also add a "box" node for easier lookup
            graph.add_node((box_id, "box"), box=box)

        # Add edges from patchlines with type information
        for line in self.data.get("patcher", {}).get("lines", []):
            patchline = line.get("patchline", {})
            src = patchline.get("source", [])
            dst = patchline.get("destination", [])

            if len(src) >= 2 and len(dst) >= 2:
                src_box_id = src[0]
                src_outlet = src[1]
                dst_box_id = dst[0]
                dst_inlet = dst[1]

                # Get source and destination boxes
                src_box = self.boxes.get(src_box_id, {})
                dst_box = self.boxes.get(dst_box_id, {})

                # Determine outlet and inlet types
                outlet_type = self._get_outlet_type(src_box, src_outlet)
                inlet_type = self._get_expected_inlet_type(dst_box, dst_inlet)

                src_node = (src_box_id, "out", src_outlet)
                dst_node = (dst_box_id, "in", dst_inlet)

                # Store type info in edge attributes
                graph.add_edge(
                    src_node,
                    dst_node,
                    outlet_type=outlet_type,
                    inlet_type=inlet_type,
                )

                # Also connect outlet to inlet at box level for path finding
                graph.add_edge((src_box_id, "box"), (dst_box_id, "box"))

        return graph

    def _get_box_text(self, box_id: str) -> str:
        """Get the text content of a box."""
        box = self.boxes.get(box_id, {})
        return str(box.get("text", ""))

    def _find_boxes_by_type(self, type_pattern: str) -> list[str]:
        """Find all boxes whose text starts with the given pattern."""
        result = []
        for box_id, box in self.boxes.items():
            text = box.get("text", "")
            if text.startswith(type_pattern):
                result.append(box_id)
        return result

    def _find_boxes_by_maxclass(self, maxclass: str) -> list[str]:
        """Find all boxes with the given maxclass."""
        result = []
        for box_id, box in self.boxes.items():
            if box.get("maxclass") == maxclass:
                result.append(box_id)
        return result

    def _extract_context_name(self, text: str) -> str | None:
        """Extract context name from object text like 'jit.world sr_ctx @visible 0'."""
        parts = text.split()
        if len(parts) >= 2:
            # Context name is usually the second word (after object name)
            candidate = parts[1]
            # Skip if it's an attribute (starts with @)
            if not candidate.startswith("@"):
                return candidate
        return None

    def _extract_drawto(self, text: str) -> str | None:
        """Extract @drawto value from object text."""
        match = re.search(r"@drawto\s+(\S+)", text)
        if match:
            return match.group(1)
        return None

    def _get_expected_context_name(self) -> str:
        """Derive expected context name from filename."""
        if self.filepath is None:
            return "sr_unknown_ctx"
        stem = self.filepath.stem  # e.g., "sr.bandswap"
        # Convert dots to underscores and add _ctx suffix
        name = stem.replace(".", "_") + "_ctx"
        return name

    def validate_file(self, filepath: Path) -> bool:
        """Validate a single .maxhelp file.

        Args:
            filepath: Path to the .maxhelp file.

        Returns:
            True if valid, False otherwise.
        """
        self.errors.clear()
        self.warnings.clear()
        self.filepath = filepath

        if not filepath.exists():
            self.error("file", f"File does not exist: {filepath}")
            return False

        if filepath.suffix != ".maxhelp":
            self.error(
                "file", f"File must have .maxhelp extension, got: {filepath.suffix}"
            )
            return False

        # Read and parse JSON
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            self.error("file", f"Failed to read file: {e}")
            return False

        try:
            self.data = json.loads(content)
        except json.JSONDecodeError as e:
            self.error("json", f"Invalid JSON syntax: {e.msg} at line {e.lineno}")
            return False

        # Build box lookup and graph
        self.boxes = {}
        for box_wrapper in self.data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            box_id = box.get("id", "")
            if box_id:
                self.boxes[box_id] = box

        self.graph = self._build_graph()

        # Build LintGraph once for all validation rules
        self.lint_graph = LintGraph.build(self.data, self)

        # Run all validations
        valid = True
        valid &= self._validate_structure()
        valid &= self._validate_context_naming()
        valid &= self._validate_context_initialization()
        valid &= self._validate_context_rules()
        valid &= self._validate_init_order()
        valid &= self._validate_signal_flow()
        valid &= self._validate_inlet_connections()
        valid &= self._validate_connection_types()
        valid &= self._validate_strict_types()
        valid &= self._validate_display_sink_sources()
        valid &= self._validate_metadata()
        valid &= self._validate_parameter_ui()
        valid &= self._validate_dial_param_ranges()
        valid &= self._validate_c_external_dial_ranges()
        valid &= self._validate_dial_decimals()
        valid &= self._validate_dial_float_output()
        valid &= self._validate_no_overlaps()
        valid &= self._validate_param_initialization()
        valid &= self._validate_dial_initialization()
        valid &= self._validate_known_objects()
        valid &= self._validate_flow_rules()
        valid &= self._validate_dead_code()
        valid &= self._validate_feedback_loops()
        valid &= self._validate_trigger_order()

        return valid

    def _validate_structure(self) -> bool:
        """Validate basic JSON structure."""
        valid = True

        if "patcher" not in self.data:
            self.error("structure", "Missing 'patcher' key")
            return False

        patcher = self.data["patcher"]

        if "boxes" not in patcher:
            self.error("structure", "Missing 'boxes' key in patcher")
            valid = False

        if "lines" not in patcher:
            self.warning(
                "structure", "Missing 'lines' key - patcher has no connections"
            )

        return valid

    def print_results(self, filepath: Path) -> None:
        """Print validation results."""
        if self.errors or self.warnings:
            print(f"\n{filepath}:")
            for err in self.errors:
                print(err)
            for warn in self.warnings:
                print(warn)
        elif self.verbose:
            print(f"\n{filepath}: OK")

    def has_errors(self) -> bool:
        """Check if there are any errors (or warnings in strict mode)."""
        return bool(self.errors or (self.strict and self.warnings))
