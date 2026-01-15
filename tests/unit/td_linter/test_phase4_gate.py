"""
Phase 4 Gate Tests: Rule System.

Gate Criteria:
- G4.1: Rules load from YAML (custom config overrides defaults)
- G4.2: Rules can be disabled (enabled: false silences rule)
- G4.3: Presets work (extends: pedantic enables all rules)
"""

from pathlib import Path

import yaml
from td_linter.rules.loader import load_config
from td_linter.rules.registry import RuleRegistry, get_registry


class TestG41RulesLoadFromYAML:
    """G4.1: Rules load from YAML - custom config overrides defaults."""

    def test_custom_config_overrides_severity(self, tmp_path: Path) -> None:
        """Custom config should override default severity."""
        config_content = {
            "version": "1.0.0",
            "extends": "recommended",
            "rules": {
                "C001": {"severity": "warning"},  # Override from error to warning
            },
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)

        # Verify override applied
        assert config.get_rule_severity("C001") == "warning"
        # Verify other rules unchanged
        assert config.get_rule_severity("S001") == "error"

    def test_custom_config_adds_rule_options(self, tmp_path: Path) -> None:
        """Custom config should set rule-specific options."""
        config_content = {
            "version": "1.0.0",
            "extends": "recommended",
            "rules": {
                "F001": {
                    "enabled": True,
                    "options": {"max_depth": 25},
                },
            },
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)
        rule = registry.get("F001")

        assert rule is not None
        assert rule.get_option("max_depth") == 25

    def test_registry_uses_config_file(self, tmp_path: Path) -> None:
        """Registry should use provided config file."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",  # All rules enabled
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = get_registry(config_file)

        # Strict preset enables all rules
        assert len(registry.enabled()) == 16


class TestG42RulesCanBeDisabled:
    """G4.2: Rules can be disabled - enabled: false silences rule."""

    def test_disabled_rule_not_in_enabled_list(self, tmp_path: Path) -> None:
        """Disabled rules should not appear in enabled() list."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "rules": {
                "C001": {"enabled": False},
            },
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)
        enabled = registry.enabled()
        enabled_ids = [r.rule_id for r in enabled]

        # C001 should be disabled
        assert "C001" not in enabled_ids
        # Other rules should still be enabled
        assert "C002" in enabled_ids
        assert "S001" in enabled_ids

    def test_ignore_pattern_disables_rules(self, tmp_path: Path) -> None:
        """Ignore patterns should disable matching rules."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "ignore": ["F"],  # Ignore all performance rules
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)

        # F rules should be disabled
        assert config.is_rule_enabled("F001") is False
        assert config.is_rule_enabled("F002") is False
        # Other rules should be enabled
        assert config.is_rule_enabled("S001") is True

    def test_ignore_specific_rule(self, tmp_path: Path) -> None:
        """Specific rule IDs in ignore should be disabled."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "ignore": ["C002", "P003"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)

        # Ignored rules should be disabled
        assert config.is_rule_enabled("C002") is False
        assert config.is_rule_enabled("P003") is False
        # Other rules in same category should be enabled
        assert config.is_rule_enabled("C001") is True
        assert config.is_rule_enabled("P001") is True


class TestG43PresetsWork:
    """G4.3: Presets work - extends: pedantic enables all rules."""

    def test_pedantic_enables_all_rules(self, tmp_path: Path) -> None:
        """Pedantic preset should enable all 16 rules."""
        config_content = {
            "version": "1.0.0",
            "extends": "pedantic",
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)
        enabled = registry.enabled()

        # All 16 rules should be enabled
        assert len(enabled) == 16

    def test_pedantic_upgrades_severities(self, tmp_path: Path) -> None:
        """Pedantic preset should upgrade severities to error."""
        config_content = {
            "version": "1.0.0",
            "extends": "pedantic",
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)

        # C002 should be upgraded from warning to error in pedantic
        assert config.get_rule_severity("C002") == "error"
        # G002 should be upgraded from warning to error
        assert config.get_rule_severity("G002") == "error"

    def test_pedantic_has_stricter_thresholds(self, tmp_path: Path) -> None:
        """Pedantic preset should have stricter performance thresholds."""
        config_content = {
            "version": "1.0.0",
            "extends": "pedantic",
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)
        f001 = registry.get("F001")

        assert f001 is not None
        # Pedantic has max_depth of 5 (stricter than default 10)
        assert f001.get_option("max_depth") == 5

    def test_minimal_disables_most_rules(self, tmp_path: Path) -> None:
        """Minimal preset should only enable critical rules."""
        config_content = {
            "version": "1.0.0",
            "extends": "minimal",
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)
        enabled = registry.enabled()
        enabled_ids = [r.rule_id for r in enabled]

        # Only syntax and critical connection rules enabled
        assert "S001" in enabled_ids
        assert "S002" in enabled_ids
        assert "S003" in enabled_ids
        assert "C001" in enabled_ids
        # Non-critical rules disabled
        assert "T001" not in enabled_ids
        assert "G001" not in enabled_ids
        assert "F001" not in enabled_ids

    def test_multiple_presets_merge_correctly(self, tmp_path: Path) -> None:
        """Multiple presets should merge in order."""
        config_content = {
            "version": "1.0.0",
            "extends": ["minimal", "recommended"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)

        # recommended (later) should override minimal
        # G001 disabled in minimal but enabled in recommended
        assert config.is_rule_enabled("G001") is True


class TestSelectIgnoreIntegration:
    """Integration tests for select/ignore patterns."""

    def test_select_only_categories(self, tmp_path: Path) -> None:
        """Select should enable only specified categories."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "select": ["S", "C"],  # Only syntax and connection
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        registry = RuleRegistry.from_config_file(config_file)
        enabled = registry.enabled()
        enabled_ids = [r.rule_id for r in enabled]

        # S and C rules enabled
        assert "S001" in enabled_ids
        assert "C001" in enabled_ids
        # Other categories disabled
        assert "G001" not in enabled_ids
        assert "P001" not in enabled_ids
        assert "T001" not in enabled_ids

    def test_select_and_ignore_combine(self, tmp_path: Path) -> None:
        """Select and ignore should combine correctly."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "select": ["S", "C", "T"],  # Enable these
            "ignore": ["C002"],  # But not this one
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)

        # C001 enabled (in select)
        assert config.is_rule_enabled("C001") is True
        # C002 disabled (in ignore)
        assert config.is_rule_enabled("C002") is False
        # T001 enabled (in select)
        assert config.is_rule_enabled("T001") is True
        # G001 disabled (not in select)
        assert config.is_rule_enabled("G001") is False
