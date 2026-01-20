"""JSON output formatter with stable schema."""

import json
from typing import TYPE_CHECKING

from td_linter.output.base import OutputFormatter

if TYPE_CHECKING:
    from td_linter.rules.base import Violation

# JSON output schema version
OUTPUT_SCHEMA_VERSION = "1.0.0"


class JSONFormatter(OutputFormatter):
    """Machine-readable JSON output with stable schema.

    Produces output conforming to the td-linter-output.schema.json schema.
    """

    @property
    def name(self) -> str:
        """Return the formatter name."""
        return "json"

    def format(
        self,
        violations: list["Violation"],
        project_path: str | None = None,
    ) -> str:
        """Format violations as JSON.

        Args:
            violations: List of violations to format
            project_path: Optional project path (included in output)

        Returns:
            JSON string with schema version, violations array, and summary
        """
        counts = {"error": 0, "warning": 0, "info": 0}
        violation_dicts = []

        for v in violations:
            sev = v.severity
            counts[sev] = counts.get(sev, 0) + 1
            violation_dicts.append({
                "rule": v.rule,
                "message": v.message,
                "path": v.path,
                "severity": v.severity,
                "line": v.line,
                "source_file": str(v.source_file) if v.source_file else None,
                "context": v.context if v.context else None,
            })

        output: dict[str, object] = {
            "version": OUTPUT_SCHEMA_VERSION,
            "violations": violation_dicts,
            "summary": {
                "total": len(violations),
                "errors": counts["error"],
                "warnings": counts["warning"],
                "info": counts["info"],
            },
        }

        if project_path:
            output["project"] = project_path

        return json.dumps(output, indent=2)
