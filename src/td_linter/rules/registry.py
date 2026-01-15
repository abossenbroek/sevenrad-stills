"""Rule registry for td-linter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from td_linter.rules.base import CATEGORIES, LintRule
from td_linter.rules.builtin import get_all_builtin_rules
from td_linter.rules.loader import LintConfig, load_config

if TYPE_CHECKING:
    from pathlib import Path


class RuleRegistry:
    """
    Registry for managing lint rules.

    Handles rule registration, instantiation, and filtering based on
    configuration.
    """

    def __init__(self, config: LintConfig | None = None) -> None:
        """
        Initialize the registry with optional configuration.

        Args:
            config: Lint configuration. If None, uses recommended preset.

        """
        self._config = config or load_config(None)
        self._rule_classes = get_all_builtin_rules()
        self._rules: dict[str, LintRule] = {}

        # Instantiate all rules with their configuration
        self._instantiate_rules()

    def _instantiate_rules(self) -> None:
        """Instantiate all registered rule classes with config options."""
        for rule_class in self._rule_classes:
            # Create a temporary instance to get the rule_id
            temp_instance = rule_class()
            rule_id = temp_instance.rule_id

            # Get options from config
            options = self._config.get_rule_options(rule_id)

            # Create the configured instance
            rule = rule_class(options=options)
            self._rules[rule_id] = rule

    @classmethod
    def from_config_file(cls, config_path: Path | None = None) -> "RuleRegistry":
        """
        Create a registry from a config file path.

        Args:
            config_path: Path to config file, or None for auto-discovery.

        Returns:
            Configured RuleRegistry instance.

        """
        config = load_config(config_path)
        return cls(config)

    @property
    def config(self) -> LintConfig:
        """Return the current configuration."""
        return self._config

    def all(self) -> list[LintRule]:
        """Return all registered rules."""
        return list(self._rules.values())

    def enabled(self) -> list[LintRule]:
        """Return only enabled rules based on configuration."""
        return [
            rule
            for rule in self._rules.values()
            if self._config.is_rule_enabled(rule.rule_id)
        ]

    def by_category(self, category_code: str) -> list[LintRule]:
        """
        Return rules in a specific category.

        Args:
            category_code: Single-letter category code (S, C, T, R, G, P, F).

        Returns:
            List of rules in the category.

        """
        return [
            rule for rule in self._rules.values() if rule.category_code == category_code
        ]

    def get(self, rule_id: str) -> LintRule | None:
        """
        Get a specific rule by ID.

        Args:
            rule_id: Rule ID (e.g., "C001").

        Returns:
            The rule instance, or None if not found.

        """
        return self._rules.get(rule_id)

    def select(self, patterns: list[str]) -> list[LintRule]:
        """
        Select rules matching patterns.

        Args:
            patterns: List of category codes or rule IDs.

        Returns:
            List of matching rules.

        """
        selected: list[LintRule] = []

        for pattern in patterns:
            if len(pattern) == 1 and pattern in CATEGORIES:
                # Category code
                selected.extend(self.by_category(pattern))
            else:
                # Specific rule ID
                rule = self.get(pattern)
                if rule and rule not in selected:
                    selected.append(rule)

        return selected

    def filter_enabled(self, rules: list[LintRule]) -> list[LintRule]:
        """
        Filter a list of rules to only include enabled ones.

        Args:
            rules: List of rules to filter.

        Returns:
            Filtered list with only enabled rules.

        """
        return [rule for rule in rules if self._config.is_rule_enabled(rule.rule_id)]

    def get_rule_info(self, rule_id: str) -> dict[str, object] | None:
        """
        Get detailed info about a rule.

        Args:
            rule_id: Rule ID (e.g., "C001").

        Returns:
            Dict with rule information, or None if not found.

        """
        rule = self.get(rule_id)
        if not rule:
            return None

        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "category": rule.category,
            "category_code": rule.category_code,
            "severity": self._config.get_rule_severity(rule_id),
            "enabled": self._config.is_rule_enabled(rule_id),
            "options": self._config.get_rule_options(rule_id),
        }

    def list_all_rules(self) -> list[dict[str, object]]:
        """
        List all rules with their configuration status.

        Returns:
            List of dicts with rule information.

        """
        return [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "category": rule.category,
                "category_code": rule.category_code,
                "default_severity": rule.severity,
                "config_severity": self._config.get_rule_severity(rule.rule_id),
                "enabled": self._config.is_rule_enabled(rule.rule_id),
            }
            for rule in sorted(self._rules.values(), key=lambda r: r.rule_id)
        ]

    def get_categories(self) -> dict[str, str]:
        """Return available category codes and names."""
        return dict(CATEGORIES)


def get_registry(config_path: Path | None = None) -> RuleRegistry:
    """
    Get a configured rule registry.

    Args:
        config_path: Optional path to config file.

    Returns:
        Configured RuleRegistry instance.

    """
    return RuleRegistry.from_config_file(config_path)
