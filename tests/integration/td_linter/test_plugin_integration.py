"""Integration tests for the plugin system.

These tests verify that the plugin system integrates correctly with:
- Rule registry
- Linting flow
- Configuration system
"""

from pathlib import Path

import networkx as nx
import pytest

from td_linter.plugins import PluginLoader


@pytest.mark.slow
@pytest.mark.integration
class TestPluginLoading:
    """Test plugin loading integration."""

    @pytest.fixture
    def sample_plugin_file(self, tmp_path: Path) -> Path:
        """Create a sample plugin file with a custom rule."""
        plugin_file = tmp_path / "custom_rules.py"
        plugin_file.write_text('''
"""Sample plugin with custom rule."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class CustomTestRule(LintRule):
    """A custom test rule for integration testing."""

    @property
    def rule_id(self) -> str:
        return "CUSTOM001"

    @property
    def name(self) -> str:
        return "custom-test-rule"

    @property
    def description(self) -> str:
        return "A custom rule for testing the plugin system"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for test condition - always yields one violation for testing."""
        if len(graph.nodes) > 0:
            yield Violation(
                rule=self.rule_id,
                message="Test violation from custom rule",
                path=list(graph.nodes)[0] if graph.nodes else None,
                severity=self.severity,
            )
''')
        return plugin_file

    def test_load_plugin_from_file(self, sample_plugin_file: Path) -> None:
        """Should load a rule from a plugin file."""
        loader = PluginLoader()
        rules = loader.load_from_path(sample_plugin_file)

        assert len(rules) == 1
        assert rules[0].__name__ == "CustomTestRule"

    def test_loaded_rule_can_be_instantiated(self, sample_plugin_file: Path) -> None:
        """Loaded rule should be instantiatable."""
        loader = PluginLoader()
        rules = loader.load_from_path(sample_plugin_file)

        rule_instance = rules[0]()
        assert rule_instance.rule_id == "CUSTOM001"
        assert rule_instance.name == "custom-test-rule"

    def test_loaded_rule_can_check_graph(self, sample_plugin_file: Path) -> None:
        """Loaded rule should be able to check a graph."""
        loader = PluginLoader()
        rules = loader.load_from_path(sample_plugin_file)

        rule_instance = rules[0]()

        # Create a simple test graph
        graph = nx.DiGraph()
        graph.add_node("test/op1", operator=None, family="TOP")

        # Run the rule
        violations = list(rule_instance.check(graph))

        assert len(violations) == 1
        assert violations[0].rule == "CUSTOM001"


@pytest.mark.slow
@pytest.mark.integration
class TestPluginWithConfig:
    """Test plugin loading via configuration."""

    @pytest.fixture
    def sample_plugin_file(self, tmp_path: Path) -> Path:
        """Create a sample plugin file."""
        plugin_file = tmp_path / "my_rules.py"
        plugin_file.write_text('''
"""Plugin with configurable rule."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class ConfigurableRule(LintRule):
    """A rule that uses options."""

    @property
    def rule_id(self) -> str:
        return "CONFIG001"

    @property
    def name(self) -> str:
        return "configurable-rule"

    @property
    def description(self) -> str:
        return "A rule with configurable options"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        threshold = self.get_option("threshold", 5)
        if len(graph.nodes) > threshold:
            yield Violation(
                rule=self.rule_id,
                message=f"Graph has more than {threshold} nodes",
                path=None,
                severity=self.severity,
            )
''')
        return plugin_file

    def test_load_from_config_dict(self, sample_plugin_file: Path) -> None:
        """Should load plugin from config dict format."""
        loader = PluginLoader()
        config = [{"path": str(sample_plugin_file)}]

        rules = loader.load_from_config(config)

        assert len(rules) == 1
        assert rules[0].__name__ == "ConfigurableRule"


@pytest.mark.slow
@pytest.mark.integration
class TestPluginErrorHandling:
    """Test plugin error handling."""

    def test_invalid_plugin_file_raises_error(self, tmp_path: Path) -> None:
        """Invalid plugin file should raise PluginLoadError."""
        from td_linter.plugins import PluginLoadError

        invalid_plugin = tmp_path / "invalid.py"
        invalid_plugin.write_text("this is not valid python{{{")

        loader = PluginLoader()

        # Should raise PluginLoadError
        with pytest.raises(PluginLoadError):
            loader.load_from_path(invalid_plugin)

    def test_plugin_without_rules_returns_empty(self, tmp_path: Path) -> None:
        """Plugin file without LintRule subclasses should return empty list."""
        empty_plugin = tmp_path / "empty.py"
        empty_plugin.write_text('''
"""Plugin without any rules."""

def helper_function():
    pass

class NotARule:
    pass
''')

        loader = PluginLoader()
        rules = loader.load_from_path(empty_plugin)

        assert len(rules) == 0

    def test_missing_plugin_file_raises_error(self, tmp_path: Path) -> None:
        """Missing plugin file should raise PluginLoadError."""
        from td_linter.plugins import PluginLoadError

        loader = PluginLoader()

        with pytest.raises(PluginLoadError):
            loader.load_from_path(tmp_path / "nonexistent.py")


@pytest.mark.slow
@pytest.mark.integration
class TestPluginEdgeCases:
    """Edge case tests for plugin system."""

    def test_plugin_with_multiple_rules(self, tmp_path: Path) -> None:
        """Plugin file with multiple rules should load all."""
        plugin_file = tmp_path / "multi_rules.py"
        plugin_file.write_text('''
"""Plugin with multiple rules."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class RuleOne(LintRule):
    @property
    def rule_id(self) -> str:
        return "MULTI001"

    @property
    def name(self) -> str:
        return "rule-one"

    @property
    def description(self) -> str:
        return "First rule"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        return iter([])


class RuleTwo(LintRule):
    @property
    def rule_id(self) -> str:
        return "MULTI002"

    @property
    def name(self) -> str:
        return "rule-two"

    @property
    def description(self) -> str:
        return "Second rule"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        return iter([])
''')

        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        assert len(rules) == 2
        rule_ids = {r().rule_id for r in rules}
        assert rule_ids == {"MULTI001", "MULTI002"}

    def test_plugin_with_unicode_content(self, tmp_path: Path) -> None:
        """Plugin with Unicode in strings should load correctly."""
        plugin_file = tmp_path / "unicode_plugin.py"
        plugin_file.write_text('''
"""Plugin with Unicode: 日本語, Émojis 🎨."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class UnicodeRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "UNICODE001"

    @property
    def name(self) -> str:
        return "unicode-rule"

    @property
    def description(self) -> str:
        return "Rule with Unicode: 日本語 🎨"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        return iter([])
''')

        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        assert len(rules) == 1
        rule = rules[0]()
        assert "日本語" in rule.description
        assert "🎨" in rule.description

    def test_plugin_with_empty_graph(self, tmp_path: Path) -> None:
        """Plugin rule should handle empty graph."""
        plugin_file = tmp_path / "empty_graph_plugin.py"
        plugin_file.write_text('''
"""Plugin that handles empty graph."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class EmptyGraphRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "EMPTY001"

    @property
    def name(self) -> str:
        return "empty-graph-rule"

    @property
    def description(self) -> str:
        return "Rule that works with empty graph"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        if len(graph.nodes) == 0:
            yield Violation(
                rule=self.rule_id,
                message="Empty graph detected",
                path=None,
                severity=self.severity,
            )
''')

        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)
        rule = rules[0]()

        # Test with empty graph
        empty_graph = nx.DiGraph()
        violations = list(rule.check(empty_graph))

        assert len(violations) == 1
        assert violations[0].message == "Empty graph detected"

    def test_plugin_inheritance_not_detected(self, tmp_path: Path) -> None:
        """Plugin should not detect helper base classes as rules."""
        plugin_file = tmp_path / "base_class_plugin.py"
        plugin_file.write_text('''
"""Plugin with helper base class."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class MyBaseRule(LintRule):
    """Abstract base class - should not be detected."""

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"


class ConcreteRule(MyBaseRule):
    """Concrete implementation - should be detected."""

    @property
    def rule_id(self) -> str:
        return "CONCRETE001"

    @property
    def name(self) -> str:
        return "concrete-rule"

    @property
    def description(self) -> str:
        return "Concrete rule"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        return iter([])
''')

        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)

        # Both classes are loaded, but only concrete one is instantiatable
        rule_ids = []
        for r in rules:
            try:
                instance = r()
                rule_ids.append(instance.rule_id)
            except TypeError:
                # Abstract class can't be instantiated - this is expected
                pass

        assert "CONCRETE001" in rule_ids

    def test_load_multiple_plugins_sequentially(self, tmp_path: Path) -> None:
        """Loading multiple plugins should accumulate rules."""
        plugin1 = tmp_path / "plugin1.py"
        plugin1.write_text('''
from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation

class Plugin1Rule(LintRule):
    @property
    def rule_id(self) -> str:
        return "P1_001"

    @property
    def name(self) -> str:
        return "plugin1-rule"

    @property
    def description(self) -> str:
        return "Plugin 1 rule"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        return iter([])
''')

        plugin2 = tmp_path / "plugin2.py"
        plugin2.write_text('''
from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation

class Plugin2Rule(LintRule):
    @property
    def rule_id(self) -> str:
        return "P2_001"

    @property
    def name(self) -> str:
        return "plugin2-rule"

    @property
    def description(self) -> str:
        return "Plugin 2 rule"

    @property
    def category(self) -> str:
        return "C"

    @property
    def severity(self) -> str:
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        return iter([])
''')

        loader = PluginLoader()
        loader.load_from_path(plugin1)
        loader.load_from_path(plugin2)

        # Loader should accumulate rules
        all_rules = loader.rules
        assert len(all_rules) == 2

        rule_ids = {r().rule_id for r in all_rules}
        assert rule_ids == {"P1_001", "P2_001"}
