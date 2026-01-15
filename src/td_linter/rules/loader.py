"""Configuration loader for td-linter rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError, validate

# Package paths
_PACKAGE_DIR = Path(__file__).parent.parent
_SCHEMA_PATH = _PACKAGE_DIR / "schemas" / "td-linter-rules.schema.json"
_PRESETS_DIR = Path(__file__).parent / "presets"

# Available preset names
PRESET_NAMES = frozenset({"recommended", "strict", "minimal", "pedantic"})

# Category code to name mapping
CATEGORY_CODES = {
    "S": "syntax",
    "C": "connection",
    "T": "type",
    "R": "reference",
    "G": "glsl",
    "P": "python",
    "F": "performance",
}


@dataclass
class RuleConfig:
    """Configuration for a single rule."""

    enabled: bool = True
    severity: str = "error"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LintConfig:
    """Complete linting configuration."""

    version: str = "1.0.0"
    rules: dict[str, RuleConfig] = field(default_factory=dict)

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a rule is enabled."""
        if rule_id in self.rules:
            return self.rules[rule_id].enabled
        return True  # Default to enabled

    def get_rule_severity(self, rule_id: str) -> str:
        """Get the severity for a rule."""
        if rule_id in self.rules:
            return self.rules[rule_id].severity
        return "error"  # Default severity

    def get_rule_options(self, rule_id: str) -> dict[str, Any]:
        """Get options for a rule."""
        if rule_id in self.rules:
            return self.rules[rule_id].options
        return {}


class ConfigLoader:
    """Loads and merges linter configuration from YAML files."""

    _schema: dict[str, object]

    def __init__(self) -> None:
        """Initialize the config loader."""
        self._schema = self._load_schema()

    def _load_schema(self) -> dict[str, object]:
        """Load the JSON schema for validation."""
        with open(_SCHEMA_PATH) as f:
            result: dict[str, object] = json.load(f)
            return result

    def load(self, config_path: Path | None = None) -> LintConfig:
        """
        Load configuration from file or use defaults.

        Config discovery order:
        1. Explicit config_path if provided
        2. td-linter.yaml in current directory
        3. .td-linter.yaml in current directory
        4. Default (recommended preset)
        """
        if config_path is not None:
            return self._load_from_file(config_path)

        # Try default config file locations
        cwd = Path.cwd()
        for name in ("td-linter.yaml", ".td-linter.yaml"):
            candidate = cwd / name
            if candidate.exists():
                return self._load_from_file(candidate)

        # Fall back to recommended preset
        return self._load_preset("recommended")

    def _load_from_file(self, path: Path) -> LintConfig:
        """Load and validate configuration from a YAML file."""
        with open(path) as f:
            raw_config = yaml.safe_load(f) or {}

        # Validate against schema
        self._validate(raw_config)

        return self._process_config(raw_config)

    def _validate(self, config: dict[str, Any]) -> None:
        """Validate configuration against JSON schema."""
        try:
            validate(config, self._schema)
        except ValidationError as e:
            msg = f"Invalid configuration: {e.message}"
            raise ConfigValidationError(msg) from e

    def _process_config(self, raw_config: dict[str, Any]) -> LintConfig:
        """Process raw config dict into LintConfig, resolving presets."""
        # Start with empty config
        config = LintConfig(version=raw_config.get("version", "1.0.0"))

        # Resolve extends (presets)
        extends = raw_config.get("extends")
        if extends:
            preset_names = [extends] if isinstance(extends, str) else extends
            for preset_name in preset_names:
                preset_config = self._load_preset(preset_name)
                self._merge_configs(config, preset_config)

        # Apply rule overrides from raw config
        raw_rules = raw_config.get("rules", {})
        for rule_id, rule_data in raw_rules.items():
            if rule_id not in config.rules:
                config.rules[rule_id] = RuleConfig()
            self._apply_rule_override(config.rules[rule_id], rule_data)

        # Apply select/ignore filters
        select = raw_config.get("select")
        ignore = raw_config.get("ignore")

        if select is not None:
            self._apply_select(config, select)
        if ignore is not None:
            self._apply_ignore(config, ignore)

        return config

    def _load_preset(self, name: str) -> LintConfig:
        """Load a preset configuration."""
        if name not in PRESET_NAMES:
            msg = f"Unknown preset: {name}"
            raise ConfigValidationError(msg)

        preset_path = _PRESETS_DIR / f"{name}.yaml"
        with open(preset_path) as f:
            raw_config = yaml.safe_load(f) or {}

        config = LintConfig(version=raw_config.get("version", "1.0.0"))

        raw_rules = raw_config.get("rules", {})
        for rule_id, rule_data in raw_rules.items():
            config.rules[rule_id] = RuleConfig(
                enabled=rule_data.get("enabled", True),
                severity=rule_data.get("severity", "error"),
                options=rule_data.get("options", {}),
            )

        return config

    def _merge_configs(self, target: LintConfig, source: LintConfig) -> None:
        """Merge source config into target (source values override target)."""
        for rule_id, rule_config in source.rules.items():
            if rule_id not in target.rules:
                target.rules[rule_id] = RuleConfig()
            target.rules[rule_id].enabled = rule_config.enabled
            target.rules[rule_id].severity = rule_config.severity
            target.rules[rule_id].options.update(rule_config.options)

    def _apply_rule_override(
        self, rule_config: RuleConfig, override: dict[str, Any]
    ) -> None:
        """Apply override values to a rule config."""
        if "enabled" in override:
            rule_config.enabled = override["enabled"]
        if "severity" in override:
            rule_config.severity = override["severity"]
        if "options" in override:
            rule_config.options.update(override["options"])

    def _apply_select(self, config: LintConfig, select: list[str]) -> None:
        """Enable only rules matching select patterns."""
        selected_rules = self._expand_patterns(select)

        # Disable all rules not in selection
        for rule_id in config.rules:
            if rule_id not in selected_rules:
                config.rules[rule_id].enabled = False

    def _apply_ignore(self, config: LintConfig, ignore: list[str]) -> None:
        """Disable rules matching ignore patterns."""
        ignored_rules = self._expand_patterns(ignore)

        for rule_id in ignored_rules:
            if rule_id in config.rules:
                config.rules[rule_id].enabled = False

    def _expand_patterns(self, patterns: list[str]) -> set[str]:
        """Expand category codes and rule IDs into a set of rule IDs."""
        result: set[str] = set()

        for pattern in patterns:
            if len(pattern) == 1 and pattern in CATEGORY_CODES:
                # Category code - expand to all rules in category
                # (e.g., "S" -> S001, S002, S003)
                for rule_id in self._get_rules_by_category(pattern):
                    result.add(rule_id)
            else:
                # Specific rule ID
                result.add(pattern)

        return result

    def _get_rules_by_category(self, category_code: str) -> list[str]:
        """Get all rule IDs for a category code."""
        # Return known rule IDs for the category
        # This is based on the planned rule set
        category_rules = {
            "S": ["S001", "S002", "S003"],
            "C": ["C001", "C002"],
            "T": ["T001"],
            "R": ["R001", "R002"],
            "G": ["G001", "G002", "G003"],
            "P": ["P001", "P002", "P003"],
            "F": ["F001", "F002"],
        }
        return category_rules.get(category_code, [])


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""


def load_config(config_path: Path | None = None) -> LintConfig:
    """
    Load linter configuration from file or use defaults.

    Args:
        config_path: Optional path to config file. If None, uses discovery.

    Returns:
        LintConfig with resolved settings.

    """
    loader = ConfigLoader()
    return loader.load(config_path)
