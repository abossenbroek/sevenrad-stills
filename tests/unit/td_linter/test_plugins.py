"""Unit tests for plugin loading functionality."""

from pathlib import Path
from typing import Iterator

import pytest

from td_linter.plugins import PluginLoadError, PluginLoader, load_plugins
from td_linter.rules.base import LintRule, Violation


# Sample plugin rule for testing
class SamplePluginRule(LintRule):
    """Sample plugin rule for testing."""

    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "sample-plugin-rule"

    @property
    def description(self) -> str:
        return "Sample plugin rule for testing"

    def check(self, graph: object) -> Iterator[Violation]:
        return iter(())


class TestPluginLoader:
    """Tests for PluginLoader class."""

    def test_loader_initialization(self) -> None:
        """Should initialize with empty rules and errors."""
        loader = PluginLoader()
        assert loader.rules == []
        assert loader.errors == []

    def test_load_from_path_valid_plugin(self, tmp_path: Path) -> None:
        """Should load rules from a valid Python file."""
        plugin_file = tmp_path / "my_rules.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class MyCustomRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "my-custom-rule"

    @property
    def description(self) -> str:
        return "My custom rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        assert len(rules) == 1
        assert rules[0].__name__ == "MyCustomRule"
        assert len(loader.rules) == 1

    def test_load_from_path_multiple_rules(self, tmp_path: Path) -> None:
        """Should load multiple rules from one file."""
        plugin_file = tmp_path / "rules.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class RuleA(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "rule-a"

    @property
    def description(self) -> str:
        return "Rule A"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())

class RuleB(LintRule):
    @property
    def rule_id(self) -> str:
        return "X002"

    @property
    def name(self) -> str:
        return "rule-b"

    @property
    def description(self) -> str:
        return "Rule B"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        assert len(rules) == 2
        rule_names = {r.__name__ for r in rules}
        assert rule_names == {"RuleA", "RuleB"}

    def test_load_from_path_file_not_found(self, tmp_path: Path) -> None:
        """Should raise PluginLoadError for missing file."""
        loader = PluginLoader()
        nonexistent = tmp_path / "nonexistent.py"

        with pytest.raises(PluginLoadError, match="not found"):
            loader.load_from_path(nonexistent)

    def test_load_from_path_not_python_file(self, tmp_path: Path) -> None:
        """Should raise PluginLoadError for non-.py file."""
        loader = PluginLoader()
        txt_file = tmp_path / "rules.txt"
        txt_file.write_text("not python")

        with pytest.raises(PluginLoadError, match=".py file"):
            loader.load_from_path(txt_file)

    def test_load_from_path_syntax_error(self, tmp_path: Path) -> None:
        """Should raise PluginLoadError for file with syntax error."""
        plugin_file = tmp_path / "bad.py"
        plugin_file.write_text("this is not valid python !!!")

        loader = PluginLoader()

        with pytest.raises(PluginLoadError):
            loader.load_from_path(plugin_file)

    def test_load_from_path_import_error(self, tmp_path: Path) -> None:
        """Should raise PluginLoadError for file with import error."""
        plugin_file = tmp_path / "bad_import.py"
        plugin_file.write_text("from nonexistent_module import something")

        loader = PluginLoader()

        with pytest.raises(PluginLoadError):
            loader.load_from_path(plugin_file)

    def test_load_from_path_ignores_non_rules(self, tmp_path: Path) -> None:
        """Should ignore non-LintRule classes."""
        plugin_file = tmp_path / "mixed.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class Helper:
    pass

def utility_function():
    pass

class MyRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "my-rule"

    @property
    def description(self) -> str:
        return "My rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        assert len(rules) == 1
        assert rules[0].__name__ == "MyRule"

    def test_load_from_config_with_path(self, tmp_path: Path) -> None:
        """Should load rules from config with path spec."""
        plugin_file = tmp_path / "rules.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class ConfigRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "config-rule"

    @property
    def description(self) -> str:
        return "Config rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        loader = PluginLoader()
        config = [{"path": str(plugin_file)}]
        rules = loader.load_from_config(config)

        assert len(rules) == 1
        assert rules[0].__name__ == "ConfigRule"

    def test_load_from_config_invalid_spec(self) -> None:
        """Should record error for invalid config spec."""
        loader = PluginLoader()
        config = [{"invalid": "spec"}]
        rules = loader.load_from_config(config)

        assert len(rules) == 0
        assert len(loader.errors) == 1

    def test_load_from_config_missing_file(self, tmp_path: Path) -> None:
        """Should record error for missing file in config."""
        loader = PluginLoader()
        config = [{"path": str(tmp_path / "missing.py")}]
        rules = loader.load_from_config(config)

        assert len(rules) == 0
        assert len(loader.errors) == 1


class TestLoadPluginsFunction:
    """Tests for load_plugins function."""

    def test_load_plugins_empty(self) -> None:
        """Should return empty list with no config."""
        rules, errors = load_plugins(plugins_config=None, load_entry_points=False)
        assert rules == []
        assert errors == []

    def test_load_plugins_from_config(self, tmp_path: Path) -> None:
        """Should load plugins from config."""
        plugin_file = tmp_path / "rules.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class LoadedRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "loaded-rule"

    @property
    def description(self) -> str:
        return "Loaded rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        config = [{"path": str(plugin_file)}]
        rules, errors = load_plugins(plugins_config=config, load_entry_points=False)

        assert len(rules) == 1
        assert len(errors) == 0


class TestPluginRuleIntegration:
    """Integration tests for plugin rules with the registry."""

    def test_plugin_rule_instance(self, tmp_path: Path) -> None:
        """Should be able to instantiate a loaded plugin rule."""
        plugin_file = tmp_path / "rules.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class InstantiableRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "instantiable-rule"

    @property
    def description(self) -> str:
        return "Can be instantiated"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        # Should be able to instantiate the rule
        rule_instance = rules[0]()
        assert rule_instance.rule_id == "X001"
        assert rule_instance.name == "instantiable-rule"

    def test_plugin_rule_with_options(self, tmp_path: Path) -> None:
        """Should be able to pass options to plugin rule."""
        plugin_file = tmp_path / "rules.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class OptionRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "option-rule"

    @property
    def description(self) -> str:
        return "Has options"

    def check(self, graph) -> Iterator[Violation]:
        threshold = self.get_option("threshold", 10)
        return iter(())
""")
        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        # Should be able to pass options
        rule_instance = rules[0](options={"threshold": 20})
        assert rule_instance.get_option("threshold") == 20


class TestPluginLoaderIsRuleClass:
    """Tests for _is_rule_class helper method."""

    def test_is_rule_class_true_for_subclass(self) -> None:
        """Should return True for LintRule subclasses."""
        loader = PluginLoader()
        assert loader._is_rule_class(SamplePluginRule)

    def test_is_rule_class_false_for_lint_rule(self) -> None:
        """Should return False for LintRule itself."""
        loader = PluginLoader()
        assert not loader._is_rule_class(LintRule)

    def test_is_rule_class_false_for_non_class(self) -> None:
        """Should return False for non-class objects."""
        loader = PluginLoader()
        assert not loader._is_rule_class("not a class")
        assert not loader._is_rule_class(42)
        assert not loader._is_rule_class(lambda: None)

    def test_is_rule_class_false_for_unrelated_class(self) -> None:
        """Should return False for unrelated classes."""
        loader = PluginLoader()

        class NotARule:
            pass

        assert not loader._is_rule_class(NotARule)
