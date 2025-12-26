"""Extractor for patcher structure from .maxhelp files."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class MaxObject:
    """Object (box) extracted from a .maxhelp file.

    Attributes:
        id: Object identifier (e.g., "obj-1")
        maxclass: Max class name (e.g., "newobj", "jit.pwindow")
        text: Object text (for newobj) containing class name and args
        numinlets: Number of inlets
        numoutlets: Number of outlets
        outlettype: Type of each outlet (e.g., ["jit_gl_texture", ""])
        attributes: Parsed attributes from text (e.g., {"output_texture": "1"})
    """

    id: str
    maxclass: str
    text: str | None = None
    numinlets: int = 0
    numoutlets: int = 0
    outlettype: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        """Get the Max class name from text or maxclass.

        For newobj boxes, parses the first word from text.
        """
        if self.maxclass != "newobj" or not self.text:
            return self.maxclass
        parts = self.text.split()
        return parts[0] if parts else self.maxclass

    def has_attribute(self, name: str, value: str | None = None) -> bool:
        """Check if object has an attribute with optional value match.

        Args:
            name: Attribute name (without @)
            value: Optional value to match

        Returns:
            True if attribute exists (and matches value if specified)
        """
        if name not in self.attributes:
            return False
        if value is None:
            return True
        return self.attributes[name] == value


@dataclass
class MaxConnection:
    """Connection (patchline) between objects.

    Attributes:
        source_id: Source object ID
        source_outlet: Source outlet index (0-based)
        dest_id: Destination object ID
        dest_inlet: Destination inlet index (0-based)
    """

    source_id: str
    source_outlet: int
    dest_id: str
    dest_inlet: int


@dataclass
class MaxPatcher:
    """Parsed .maxhelp patcher structure.

    Attributes:
        filepath: Path to the .maxhelp file
        objects: Dict of object ID -> MaxObject
        connections: List of connections
        description: Patcher description
        tags: Patcher tags (comma-separated)
    """

    filepath: Path
    objects: dict[str, MaxObject] = field(default_factory=dict)
    connections: list[MaxConnection] = field(default_factory=list)
    description: str = ""
    tags: str = ""

    def find_objects_by_class(self, class_name: str) -> list[MaxObject]:
        """Find all objects with a given class name.

        Args:
            class_name: Class name to search for (e.g., "jit.movie")

        Returns:
            List of matching objects
        """
        return [obj for obj in self.objects.values() if obj.class_name == class_name]

    def find_objects_by_class_prefix(self, prefix: str) -> list[MaxObject]:
        """Find all objects whose class name starts with prefix.

        Args:
            prefix: Class name prefix (e.g., "sr.")

        Returns:
            List of matching objects
        """
        return [
            obj for obj in self.objects.values() if obj.class_name.startswith(prefix)
        ]

    def has_utility_tag(self) -> bool:
        """Check if patcher has 'utility' in tags."""
        return "utility" in self.tags.lower()


class MaxhelpExtractor:
    """Extracts patcher structure from .maxhelp files.

    .maxhelp files are JSON patcher files that contain the help
    documentation and examples for Max objects. This extractor
    parses the structure and builds a connection graph for analysis.
    """

    def __init__(self) -> None:
        """Initialize extractor."""
        pass

    def extract(self, filepath: Path) -> MaxPatcher | None:
        """Extract patcher structure from a .maxhelp file.

        Args:
            filepath: Path to the .maxhelp file

        Returns:
            MaxPatcher object or None if parsing fails
        """
        try:
            content = filepath.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            return None
        except OSError as e:
            logger.error(f"Cannot read {filepath}: {e}")
            return None

        patcher_data = data.get("patcher", {})

        patcher = MaxPatcher(
            filepath=filepath,
            description=patcher_data.get("description", ""),
            tags=patcher_data.get("tags", ""),
        )

        # Extract objects
        for box_wrapper in patcher_data.get("boxes", []):
            box = box_wrapper.get("box", {})
            obj = self._parse_box(box)
            if obj:
                patcher.objects[obj.id] = obj

        # Extract connections
        for line_wrapper in patcher_data.get("lines", []):
            patchline = line_wrapper.get("patchline", {})
            conn = self._parse_connection(patchline)
            if conn:
                patcher.connections.append(conn)

        return patcher

    def _parse_box(self, box: dict[str, Any]) -> MaxObject | None:
        """Parse a box dictionary into a MaxObject.

        Args:
            box: Box data from JSON

        Returns:
            MaxObject or None if missing required fields
        """
        obj_id = box.get("id")
        if not obj_id:
            return None

        maxclass = box.get("maxclass", "")
        text = box.get("text")
        attributes = self._parse_text_attributes(text) if text else {}

        return MaxObject(
            id=obj_id,
            maxclass=maxclass,
            text=text,
            numinlets=box.get("numinlets", 0),
            numoutlets=box.get("numoutlets", 0),
            outlettype=box.get("outlettype", []),
            attributes=attributes,
        )

    def _parse_text_attributes(self, text: str) -> dict[str, str]:
        """Parse @attribute value pairs from object text.

        Args:
            text: Object text (e.g., "jit.movie @loop 1 @output_texture 1")

        Returns:
            Dict of attribute name -> value
        """
        attributes: dict[str, str] = {}

        # Pattern: @name value (value is next token after @name)
        pattern = r"@(\w+)\s+(\S+)"
        for match in re.finditer(pattern, text):
            name = match.group(1)
            value = match.group(2)
            attributes[name] = value

        return attributes

    def _parse_connection(self, patchline: dict[str, Any]) -> MaxConnection | None:
        """Parse a patchline dictionary into a MaxConnection.

        Args:
            patchline: Patchline data from JSON

        Returns:
            MaxConnection or None if missing required fields
        """
        source = patchline.get("source", [])
        dest = patchline.get("destination", [])

        if len(source) < 2 or len(dest) < 2:
            return None

        return MaxConnection(
            source_id=source[0],
            source_outlet=source[1],
            dest_id=dest[0],
            dest_inlet=dest[1],
        )

    def build_connection_graph(self, patcher: MaxPatcher) -> nx.DiGraph:
        """Build a networkx directed graph from patcher connections.

        Nodes are object IDs, edges are connections with outlet/inlet metadata.

        Args:
            patcher: Parsed MaxPatcher

        Returns:
            networkx DiGraph
        """
        graph = nx.DiGraph()

        # Add nodes with object data
        for obj_id, obj in patcher.objects.items():
            graph.add_node(obj_id, obj=obj)

        # Add edges with connection data
        for conn in patcher.connections:
            if conn.source_id in patcher.objects and conn.dest_id in patcher.objects:
                graph.add_edge(
                    conn.source_id,
                    conn.dest_id,
                    outlet=conn.source_outlet,
                    inlet=conn.dest_inlet,
                )

        return graph

    def get_objects_upstream(
        self, graph: nx.DiGraph, obj_id: str
    ) -> list[tuple[str, MaxObject]]:
        """Get all objects that connect to a given object (directly or indirectly).

        Args:
            graph: Connection graph from build_connection_graph
            obj_id: Object ID to find upstream objects for

        Returns:
            List of (object_id, MaxObject) tuples
        """
        result = []
        try:
            ancestors = nx.ancestors(graph, obj_id)
            for ancestor_id in ancestors:
                obj = graph.nodes[ancestor_id].get("obj")
                if obj:
                    result.append((ancestor_id, obj))
        except nx.NetworkXError:
            pass
        return result

    def get_objects_downstream(
        self, graph: nx.DiGraph, obj_id: str
    ) -> list[tuple[str, MaxObject]]:
        """Get all objects that receive from a given object (directly or indirectly).

        Args:
            graph: Connection graph from build_connection_graph
            obj_id: Object ID to find downstream objects for

        Returns:
            List of (object_id, MaxObject) tuples
        """
        result = []
        try:
            descendants = nx.descendants(graph, obj_id)
            for desc_id in descendants:
                obj = graph.nodes[desc_id].get("obj")
                if obj:
                    result.append((desc_id, obj))
        except nx.NetworkXError:
            pass
        return result

    def has_path(self, graph: nx.DiGraph, source_id: str, dest_id: str) -> bool:
        """Check if there is a path from source to destination.

        Args:
            graph: Connection graph
            source_id: Source object ID
            dest_id: Destination object ID

        Returns:
            True if path exists
        """
        try:
            result: bool = nx.has_path(graph, source_id, dest_id)
            return result
        except nx.NetworkXError:
            return False

    def find_direct_connections(
        self, patcher: MaxPatcher, from_id: str
    ) -> list[tuple[str, MaxObject, int]]:
        """Find objects directly connected from a given object.

        Args:
            patcher: MaxPatcher instance
            from_id: Source object ID

        Returns:
            List of (dest_id, dest_obj, inlet) tuples
        """
        result = []
        for conn in patcher.connections:
            if conn.source_id == from_id:
                dest_obj = patcher.objects.get(conn.dest_id)
                if dest_obj:
                    result.append((conn.dest_id, dest_obj, conn.dest_inlet))
        return result
