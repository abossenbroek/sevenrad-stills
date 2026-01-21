"""
Unit tests for configuration schema validation (TDL-040).

These tests verify that the JSON schema correctly validates and rejects
td-linter configuration files.
"""

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

# Load schema once for all tests
SCHEMA_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "td_linter"
    / "schemas"
    / "td-linter-rules.schema.json"
)


@pytest.fixture
def schema() -> dict:
    """Load the JSON schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


class TestValidConfigs:
    """Tests for configurations that should be valid."""

    def test_minimal_config(self, schema: dict) -> None:
        """Empty config (using defaults) should be valid."""
        config: dict = {}
        validate(config, schema)

    def test_version_only(self, schema: dict) -> None:
        """Config with just version should be valid."""
        config = {"version": "1.0.0"}
        validate(config, schema)

    def test_extends_single_preset(self, schema: dict) -> None:
        """Config extending a single preset should be valid."""
        config = {"version": "1.0.0", "extends": "recommended"}
        validate(config, schema)

    def test_extends_multiple_presets(self, schema: dict) -> None:
        """Config extending multiple presets should be valid."""
        config = {"version": "1.0.0", "extends": ["minimal", "recommended"]}
        validate(config, schema)

    def test_extends_pedantic(self, schema: dict) -> None:
        """Config extending pedantic preset should be valid."""
        config = {"version": "1.0.0", "extends": "pedantic"}
        validate(config, schema)

    def test_select_categories(self, schema: dict) -> None:
        """Config with category selection should be valid."""
        config = {"version": "1.0.0", "select": ["S", "C", "T"]}
        validate(config, schema)

    def test_select_specific_rules(self, schema: dict) -> None:
        """Config selecting specific rule IDs should be valid."""
        config = {"version": "1.0.0", "select": ["S001", "C001", "G001"]}
        validate(config, schema)

    def test_select_mixed(self, schema: dict) -> None:
        """Config with mixed category and rule selection should be valid."""
        config = {"version": "1.0.0", "select": ["S", "C001", "G"]}
        validate(config, schema)

    def test_ignore_categories(self, schema: dict) -> None:
        """Config with category ignores should be valid."""
        config = {"version": "1.0.0", "ignore": ["F", "P"]}
        validate(config, schema)

    def test_ignore_specific_rules(self, schema: dict) -> None:
        """Config ignoring specific rules should be valid."""
        config = {"version": "1.0.0", "ignore": ["F001", "F002"]}
        validate(config, schema)

    def test_rules_override_enabled(self, schema: dict) -> None:
        """Config with rule enabled override should be valid."""
        config = {"version": "1.0.0", "rules": {"C001": {"enabled": False}}}
        validate(config, schema)

    def test_rules_override_severity(self, schema: dict) -> None:
        """Config with severity override should be valid."""
        config = {"version": "1.0.0", "rules": {"C002": {"severity": "error"}}}
        validate(config, schema)

    def test_rules_override_options(self, schema: dict) -> None:
        """Config with rule options should be valid."""
        config = {
            "version": "1.0.0",
            "rules": {"F001": {"enabled": True, "options": {"max_depth": 15}}},
        }
        validate(config, schema)

    def test_full_config(self, schema: dict) -> None:
        """Full configuration with all features should be valid."""
        config = {
            "version": "1.0.0",
            "extends": "recommended",
            "select": ["S", "C", "T", "R", "G", "P"],
            "ignore": ["F001", "F002"],
            "rules": {
                "C001": {"severity": "error"},
                "F001": {"enabled": True, "options": {"max_depth": 15}},
                "P002": {"enabled": False},
            },
        }
        validate(config, schema)


class TestInvalidConfigs:
    """Tests for configurations that should be rejected."""

    def test_invalid_version_format(self, schema: dict) -> None:
        """Version not matching semver should be rejected."""
        config = {"version": "1.0"}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_version_type(self, schema: dict) -> None:
        """Non-string version should be rejected."""
        config = {"version": 100}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_preset_name(self, schema: dict) -> None:
        """Unknown preset name should be rejected."""
        config = {"extends": "unknown_preset"}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_preset_in_list(self, schema: dict) -> None:
        """Unknown preset in list should be rejected."""
        config = {"extends": ["recommended", "unknown"]}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_select_pattern(self, schema: dict) -> None:
        """Invalid category/rule pattern should be rejected."""
        config = {"select": ["X001"]}  # X is not a valid category
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_ignore_pattern(self, schema: dict) -> None:
        """Invalid ignore pattern should be rejected."""
        config = {"ignore": ["invalid-rule"]}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_severity_value(self, schema: dict) -> None:
        """Invalid severity value should be rejected."""
        config = {"rules": {"C001": {"severity": "critical"}}}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_invalid_enabled_type(self, schema: dict) -> None:
        """Non-boolean enabled should be rejected."""
        config = {"rules": {"C001": {"enabled": "yes"}}}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_unknown_rule_property(self, schema: dict) -> None:
        """Unknown property in rule config should be rejected."""
        config = {"rules": {"C001": {"unknown_prop": True}}}
        with pytest.raises(ValidationError):
            validate(config, schema)

    def test_unknown_top_level_property(self, schema: dict) -> None:
        """Unknown top-level property should be rejected."""
        config = {"unknown_property": "value"}
        with pytest.raises(ValidationError):
            validate(config, schema)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_select_list(self, schema: dict) -> None:
        """Empty select list should be valid."""
        config = {"select": []}
        validate(config, schema)

    def test_empty_ignore_list(self, schema: dict) -> None:
        """Empty ignore list should be valid."""
        config = {"ignore": []}
        validate(config, schema)

    def test_empty_rules_object(self, schema: dict) -> None:
        """Empty rules object should be valid."""
        config = {"rules": {}}
        validate(config, schema)

    def test_all_valid_categories(self, schema: dict) -> None:
        """All valid category codes should be accepted."""
        config = {"select": ["S", "C", "T", "R", "G", "P", "F"]}
        validate(config, schema)

    def test_all_valid_severities(self, schema: dict) -> None:
        """All valid severity values should be accepted."""
        config = {
            "rules": {
                "S001": {"severity": "error"},
                "S002": {"severity": "warning"},
                "S003": {"severity": "info"},
            }
        }
        validate(config, schema)
