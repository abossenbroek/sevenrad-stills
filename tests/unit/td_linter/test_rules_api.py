"""
Unit tests for rule registry API (TDL-043).

These tests verify that the RuleRegistry correctly:
- Returns all registered rules
- Filters rules by enabled status
- Filters rules by category
- Provides rule information
"""

from pathlib import Path

import pytest
import yaml
from td_linter.rules.loader import LintConfig, RuleConfig
from td_linter.rules.registry import RuleRegistry, get_registry


class TestRuleRegistryBasics:
    """Basic registry functionality tests."""

    def test_registry_loads_all_rules(self) -> None:
        """Registry should load all 19 builtin rules."""
        registry = RuleRegistry()
        rules = registry.all()

        assert len(rules) == 19

    def test_registry_has_all_categories(self) -> None:
        """Registry should have rules in all categories."""
        registry = RuleRegistry()
        categories = registry.get_categories()

        assert "S" in categories  # syntax
        assert "C" in categories  # connection
        assert "T" in categories  # type
        assert "R" in categories  # reference
        assert "G" in categories  # glsl
        assert "P" in categories  # python
        assert "F" in categories  # performance


class TestRuleFiltering:
    """Tests for rule filtering methods."""

    def test_by_category_returns_correct_rules(self) -> None:
        """by_category should return rules in the specified category."""
        registry = RuleRegistry()

        syntax_rules = registry.by_category("S")
        assert len(syntax_rules) == 3
        assert all(r.category_code == "S" for r in syntax_rules)

        connection_rules = registry.by_category("C")
        assert len(connection_rules) == 2
        assert all(r.category_code == "C" for r in connection_rules)

    def test_select_by_category(self) -> None:
        """Select should expand category codes."""
        registry = RuleRegistry()

        selected = registry.select(["S", "C"])
        assert len(selected) == 5  # 3 syntax + 2 connection

    def test_select_by_rule_id(self) -> None:
        """Select should accept specific rule IDs."""
        registry = RuleRegistry()

        selected = registry.select(["C001", "G001"])
        assert len(selected) == 2
        rule_ids = [r.rule_id for r in selected]
        assert "C001" in rule_ids
        assert "G001" in rule_ids

    def test_select_mixed(self) -> None:
        """Select should handle mixed categories and IDs."""
        registry = RuleRegistry()

        selected = registry.select(["S", "C001"])
        assert len(selected) == 4  # 3 syntax + 1 specific


class TestEnabledRules:
    """Tests for enabled/disabled rule handling."""

    def test_enabled_with_default_config(self) -> None:
        """enabled() should return rules based on recommended preset."""
        registry = RuleRegistry()
        enabled = registry.enabled()

        # Core rules should be enabled in recommended preset
        enabled_ids = [r.rule_id for r in enabled]
        assert "S001" in enabled_ids
        assert "C001" in enabled_ids
        assert "T001" in enabled_ids

    def test_enabled_with_custom_config(self) -> None:
        """enabled() should respect custom config."""
        config = LintConfig(
            rules={
                "C001": RuleConfig(enabled=False),
                "C002": RuleConfig(enabled=True),
            }
        )
        registry = RuleRegistry(config)
        enabled = registry.enabled()

        enabled_ids = [r.rule_id for r in enabled]
        assert "C001" not in enabled_ids
        assert "C002" in enabled_ids


class TestRuleInfo:
    """Tests for rule information retrieval."""

    def test_get_rule_by_id(self) -> None:
        """get() should return rule by ID."""
        registry = RuleRegistry()

        rule = registry.get("C001")
        assert rule is not None
        assert rule.rule_id == "C001"
        assert rule.name == "no-invalid-cycles"

    def test_get_nonexistent_rule(self) -> None:
        """get() should return None for unknown rules."""
        registry = RuleRegistry()

        rule = registry.get("X999")
        assert rule is None

    def test_get_rule_info(self) -> None:
        """get_rule_info() should return detailed rule info."""
        registry = RuleRegistry()

        info = registry.get_rule_info("C001")
        assert info is not None
        assert info["rule_id"] == "C001"
        assert info["name"] == "no-invalid-cycles"
        assert info["category"] == "connection"
        assert info["category_code"] == "C"
        assert "enabled" in info
        assert "severity" in info

    def test_list_all_rules(self) -> None:
        """list_all_rules() should return info for all rules."""
        registry = RuleRegistry()

        all_info = registry.list_all_rules()
        assert len(all_info) == 19

        # Should be sorted by rule_id
        ids = [info["rule_id"] for info in all_info]
        assert ids == sorted(ids)


class TestConfigIntegration:
    """Tests for config file integration."""

    def test_from_config_file(self, tmp_path: Path) -> None:
        """from_config_file should load config and create registry."""
        config_content = {
            "version": "1.0.0",
            "extends": "pedantic",
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)

        # Pedantic enables all rules
        assert len(registry.enabled()) == 19

    def test_options_passed_to_rules(self) -> None:
        """Rule options from config should be passed to rule instances."""
        config = LintConfig(
            rules={
                "F001": RuleConfig(options={"max_depth": 25}),
            }
        )
        registry = RuleRegistry(config)

        rule = registry.get("F001")
        assert rule is not None
        assert rule.get_option("max_depth") == 25


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_get_registry(self) -> None:
        """get_registry() should return a configured registry."""
        registry = get_registry()

        assert isinstance(registry, RuleRegistry)
        assert len(registry.all()) == 19
