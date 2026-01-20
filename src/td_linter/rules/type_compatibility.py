"""Rule: Check type compatibility between connected operators."""

from typing import Iterator

import networkx as nx

from td_linter.graph.types import COMPATIBLE_CONNECTIONS, OperatorFamily
from td_linter.rules.base import LintRule, Violation


class TypeCompatibilityRule(LintRule):
    """
    Checks that connected operators have compatible types.

    TouchDesigner operators can only connect to operators of compatible
    families:
    - TOP -> TOP, MAT
    - CHOP -> CHOP
    - SOP -> SOP, MAT
    - DAT -> DAT
    - COMP -> COMP
    - MAT -> MAT

    Cross-family connections require converter operators (chopto, sopto, etc.)
    """

    @property
    def id(self) -> str:
        """Return rule identifier."""
        return "type-compatibility"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Check operator connections are type-compatible"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for type-incompatible connections."""
        for source, target, data in graph.edges(data=True):
            # Skip missing references (handled by other rules)
            if source.startswith("MISSING:"):
                continue

            source_data = graph.nodes.get(source, {})
            target_data = graph.nodes.get(target, {})

            source_family_str = source_data.get("family", "UNKNOWN")
            target_family_str = target_data.get("family", "UNKNOWN")

            # Skip if we can't determine families
            if source_family_str == "UNKNOWN" or target_family_str == "UNKNOWN":
                continue

            source_family = OperatorFamily.from_string(source_family_str)
            target_family = OperatorFamily.from_string(target_family_str)

            # Check compatibility
            if not self._is_compatible(source_family, target_family, target_data):
                yield Violation(
                    rule=self.id,
                    message=(
                        f"Type mismatch: {source_family.value} operator '{source}' "
                        f"connected to {target_family.value} operator '{target}'"
                    ),
                    path=target,
                    severity="error",
                    context={
                        "source_family": source_family.value,
                        "target_family": target_family.value,
                    },
                )

    def _is_compatible(
        self,
        source_family: OperatorFamily,
        target_family: OperatorFamily,
        target_data: dict[str, object],
    ) -> bool:
        """Check if source can connect to target."""
        # Get compatible targets for source family
        compatible = COMPATIBLE_CONNECTIONS.get(source_family, set())

        if target_family in compatible:
            return True

        # Check if target is a converter operator
        target_op_type = str(target_data.get("op_type", ""))
        converter_types = {"chopto", "sopto", "tochop", "datto", "topto"}

        if target_op_type.lower() in converter_types:
            return True

        return False
