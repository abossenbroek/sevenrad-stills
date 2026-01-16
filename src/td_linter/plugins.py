"""Plugin loading for custom lint rules."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td_linter.rules.base import LintRule


class PluginLoadError(Exception):
    """Error loading a plugin."""

    pass


class PluginLoader:
    """Load custom lint rules from plugins."""

    def __init__(self) -> None:
        """Initialize the plugin loader."""
        self._loaded_rules: list[type[LintRule]] = []
        self._errors: list[tuple[str, Exception]] = []

    @property
    def rules(self) -> list[type[LintRule]]:
        """Return all loaded rule classes."""
        return list(self._loaded_rules)

    @property
    def errors(self) -> list[tuple[str, Exception]]:
        """Return any errors encountered during loading."""
        return list(self._errors)

    def load_from_path(self, path: Path) -> list[type[LintRule]]:
        """Load rules from a Python file.

        Args:
            path: Path to a Python file containing LintRule subclasses.

        Returns:
            List of rule classes found in the file.

        Raises:
            PluginLoadError: If the file cannot be loaded.
        """
        path = path.resolve()
        if not path.exists():
            raise PluginLoadError(f"Plugin file not found: {path}")

        if not path.suffix == ".py":
            raise PluginLoadError(f"Plugin must be a .py file: {path}")

        try:
            # Create a unique module name based on path
            module_name = f"td_linter_plugin_{path.stem}_{hash(str(path))}"

            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot create module spec for: {path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            rules = self._extract_rules(module, str(path))
            self._loaded_rules.extend(rules)
            return rules

        except PluginLoadError:
            raise
        except Exception as e:
            raise PluginLoadError(f"Error loading plugin {path}: {e}") from e

    def load_from_module(self, module_name: str) -> list[type[LintRule]]:
        """Load rules from an installed Python module.

        Args:
            module_name: Fully qualified module name (e.g., 'my_company.td_rules').

        Returns:
            List of rule classes found in the module.

        Raises:
            PluginLoadError: If the module cannot be loaded.
        """
        try:
            module = importlib.import_module(module_name)
            rules = self._extract_rules(module, module_name)
            self._loaded_rules.extend(rules)
            return rules

        except ImportError as e:
            raise PluginLoadError(f"Cannot import module '{module_name}': {e}") from e
        except Exception as e:
            raise PluginLoadError(f"Error loading module '{module_name}': {e}") from e

    def load_from_entry_points(self, group: str = "td_linter.rules") -> list[type[LintRule]]:
        """Load rules from entry points.

        Args:
            group: Entry point group name.

        Returns:
            List of rule classes found via entry points.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return []

        rules: list[type[LintRule]] = []

        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python 3.9 compatibility
            all_eps = entry_points()
            eps = all_eps.get(group, [])

        for ep in eps:
            try:
                obj = ep.load()
                if self._is_rule_class(obj):
                    rules.append(obj)
                    self._loaded_rules.append(obj)
                elif hasattr(obj, "__iter__"):
                    # Entry point can return a list of rules
                    for item in obj:
                        if self._is_rule_class(item):
                            rules.append(item)
                            self._loaded_rules.append(item)
            except Exception as e:
                self._errors.append((ep.name, e))

        return rules

    def load_from_config(self, plugins_config: list[dict[str, str]]) -> list[type[LintRule]]:
        """Load rules from configuration.

        Args:
            plugins_config: List of plugin configs, each with either 'path' or 'module' key.

        Returns:
            List of all loaded rule classes.
        """
        rules: list[type[LintRule]] = []

        for plugin_spec in plugins_config:
            try:
                if "path" in plugin_spec:
                    path = Path(plugin_spec["path"])
                    rules.extend(self.load_from_path(path))
                elif "module" in plugin_spec:
                    module_name = plugin_spec["module"]
                    rules.extend(self.load_from_module(module_name))
                else:
                    self._errors.append(
                        ("unknown", PluginLoadError(f"Invalid plugin spec: {plugin_spec}"))
                    )
            except PluginLoadError as e:
                source = plugin_spec.get("path") or plugin_spec.get("module") or "unknown"
                self._errors.append((source, e))

        return rules

    def _extract_rules(self, module: object, source: str) -> list[type[LintRule]]:
        """Extract LintRule subclasses from a module.

        Args:
            module: The loaded module.
            source: Source description for error messages.

        Returns:
            List of LintRule subclasses found in the module.
        """
        # Import here to avoid circular imports
        from td_linter.rules.base import LintRule

        rules: list[type[LintRule]] = []

        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)
            if self._is_rule_class(obj):
                rules.append(obj)

        return rules

    def _is_rule_class(self, obj: object) -> bool:
        """Check if an object is a LintRule subclass (not LintRule itself)."""
        # Import here to avoid circular imports
        from td_linter.rules.base import LintRule

        return (
            isinstance(obj, type)
            and issubclass(obj, LintRule)
            and obj is not LintRule
        )


def load_plugins(
    plugins_config: list[dict[str, str]] | None = None,
    load_entry_points: bool = True,
) -> tuple[list[type[LintRule]], list[tuple[str, Exception]]]:
    """Load all plugins and return rules.

    Args:
        plugins_config: Optional list of plugin configurations.
        load_entry_points: Whether to load from entry points.

    Returns:
        Tuple of (loaded_rules, errors).
    """
    loader = PluginLoader()

    if plugins_config:
        loader.load_from_config(plugins_config)

    if load_entry_points:
        loader.load_from_entry_points()

    return loader.rules, loader.errors
