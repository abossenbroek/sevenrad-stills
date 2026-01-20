"""Build NetworkX graph from parsed .toe.dir structure."""

from pathlib import Path
from typing import Iterator

import networkx as nx

from td_linter.graph.model import Connection, OperatorNode, TilePosition
from td_linter.graph.types import OperatorFamily
from td_linter.parsers.n_parser import NFileParser, ParsedNFile
from td_linter.parsers.parm_parser import ParmFileParser


class NetworkGraphBuilder:
    """Builds NetworkX graph from .toe.dir structure."""

    def __init__(self, n_parser: NFileParser, parm_parser: ParmFileParser) -> None:
        """Initialize with parsers."""
        self._n_parser = n_parser
        self._parm_parser = parm_parser

    def build(self, toe_dir: Path) -> nx.DiGraph:
        """Build network graph from .toe.dir directory."""
        graph: nx.DiGraph = nx.DiGraph()
        graph.graph["toe_dir_path"] = str(toe_dir)

        # Phase 1: Create all nodes from .n files
        n_files = list(self._iter_n_files(toe_dir))

        for n_file in n_files:
            parsed = self._n_parser.parse(n_file)
            if parsed.has_errors:
                continue

            node = self._create_node(parsed, n_file, toe_dir)
            graph.add_node(
                node.path,
                operator=node,
                family=node.family.value,
                op_type=node.op_type,
                source_file=str(n_file),
            )

        # Phase 2: Create edges from inputs
        for n_file in n_files:
            parsed = self._n_parser.parse(n_file)
            if parsed.has_errors or not parsed.inputs:
                continue

            node_path = self._file_to_path(n_file, toe_dir)
            parent_dir = self._get_parent_path(node_path)

            for input_idx, ref_name in parsed.inputs:
                source_path = self._resolve_reference(ref_name, parent_dir, graph)

                if source_path not in graph.nodes:
                    # Create placeholder for missing reference
                    missing_path = f"MISSING:{source_path}"
                    graph.add_node(
                        missing_path,
                        family="MISSING",
                        op_type="unknown",
                        missing=True,
                    )
                    source_path = missing_path

                graph.add_edge(
                    source_path,
                    node_path,
                    input_index=input_idx,
                    missing=source_path.startswith("MISSING:"),
                )

        return graph

    def _iter_n_files(self, toe_dir: Path) -> Iterator[Path]:
        """Iterate all .n files in the project."""
        return toe_dir.rglob("*.n")

    def _create_node(
        self, parsed: ParsedNFile, n_file: Path, toe_dir: Path
    ) -> OperatorNode:
        """Create an OperatorNode from parsed data."""
        node_path = self._file_to_path(n_file, toe_dir)
        name = n_file.stem

        return OperatorNode(
            name=name,
            family=OperatorFamily.from_string(parsed.family),
            op_type=parsed.op_type,
            path=node_path,
            tile=TilePosition(*parsed.tile),
            source_file=n_file,
            flags=parsed.flags,
            inputs=parsed.inputs,
            color=parsed.color,
        )

    def _file_to_path(self, n_file: Path, toe_dir: Path) -> str:
        """Convert file path to operator path."""
        relative = n_file.relative_to(toe_dir)
        # Remove .n extension and convert to path string
        return str(relative.with_suffix(""))

    def _get_parent_path(self, node_path: str) -> str:
        """Get the parent container path for a node."""
        if "/" in node_path:
            return "/".join(node_path.split("/")[:-1])
        return ""

    def _resolve_reference(self, ref: str, parent_path: str, graph: nx.DiGraph) -> str:
        """
        Resolve an operator reference to a full path.

        References can be:
        - Bare name: "noise1" -> sibling in same container
        - Relative: "./noise1" -> same container
        - Absolute: "/project1/noise1" -> from root
        """
        if ref.startswith("/"):
            # Absolute path
            return ref.lstrip("/")
        elif ref.startswith("./"):
            # Explicit relative
            ref = ref[2:]

        # Sibling reference - look in same container
        if parent_path:
            candidate = f"{parent_path}/{ref}"
        else:
            candidate = ref

        # Check if this path exists in the graph
        if candidate in graph.nodes:
            return candidate

        # Try without parent path (root level)
        if ref in graph.nodes:
            return ref

        return candidate  # Return as-is, will be marked as missing
