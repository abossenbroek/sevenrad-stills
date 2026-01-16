"""SARIF 2.1.0 output formatter for GitHub Code Scanning integration."""

import json
from typing import TYPE_CHECKING

from td_linter import __version__
from td_linter.output.base import OutputFormatter

if TYPE_CHECKING:
    from td_linter.rules.base import Violation


SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/"
    "sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
)

# Map internal severity to SARIF levels
SEVERITY_TO_SARIF = {
    "error": "error",
    "warning": "warning",
    "info": "note",
}


class SARIFFormatter(OutputFormatter):
    """SARIF 2.1.0 output for GitHub Code Scanning integration.

    Produces output compatible with GitHub's upload-sarif action and
    Code Scanning API.
    """

    @property
    def name(self) -> str:
        """Return the formatter name."""
        return "sarif"

    def format(
        self,
        violations: list["Violation"],
        project_path: str | None = None,
    ) -> str:
        """Format violations as SARIF 2.1.0.

        Args:
            violations: List of violations to format
            project_path: Optional project path for artifact URIs

        Returns:
            SARIF JSON string compatible with GitHub Code Scanning
        """
        # Build unique rules list
        rules_seen: dict[str, dict[str, object]] = {}
        results: list[dict[str, object]] = []

        for v in violations:
            # Add rule definition if not seen
            if v.rule not in rules_seen:
                rules_seen[v.rule] = {
                    "id": v.rule,
                    "shortDescription": {"text": v.rule},
                }

            # Build location
            artifact_uri = str(v.source_file) if v.source_file else v.path
            physical_location: dict[str, object] = {
                "artifactLocation": {"uri": artifact_uri},
            }
            if v.line:
                physical_location["region"] = {"startLine": v.line}

            location = {"physicalLocation": physical_location}

            results.append({
                "ruleId": v.rule,
                "level": SEVERITY_TO_SARIF.get(v.severity, "warning"),
                "message": {"text": v.message},
                "locations": [location],
            })

        sarif_output: dict[str, object] = {
            "$schema": SARIF_SCHEMA,
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "td-linter",
                            "version": __version__,
                            "informationUri": "https://github.com/sevenrad/td-linter",
                            "rules": list(rules_seen.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

        return json.dumps(sarif_output, indent=2)
