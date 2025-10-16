"""
Configuration loader for sevenrad-stills.

Loads and merges configuration from YAML files with defaults.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sevenrad_stills.settings.models import AppSettings
from sevenrad_stills.utils.exceptions import ConfigError, InvalidConfigError


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary

    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If file cannot be loaded

    """
    try:
        with path.open() as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except FileNotFoundError as e:
        msg = f"Configuration file not found: {path}"
        raise ConfigError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Invalid YAML in configuration file {path}: {e}"
        raise ConfigError(msg) from e
    except Exception as e:
        msg = f"Error loading configuration file {path}: {e}"
        raise ConfigError(msg) from e


def get_default_config_path() -> Path:
    """
    Get path to default configuration file.

    Returns:
        Path to config/default.yaml

    """
    # Get the project root (4 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "config" / "default.yaml"


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppSettings:
    """
    Load configuration from YAML file(s).

    Loads default configuration and optionally merges with user config and overrides.

    Args:
        config_path: Optional path to user configuration file
        overrides: Optional dictionary of configuration overrides

    Returns:
        AppSettings instance

    Raises:
        ConfigError: If configuration cannot be loaded
        InvalidConfigError: If configuration validation fails

    """
    # Load default configuration
    default_path = get_default_config_path()
    if not default_path.exists():
        msg = f"Default configuration not found: {default_path}"
        raise ConfigError(msg)

    config_dict = load_yaml(default_path)

    # Merge with user configuration if provided
    if config_path is not None:
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise ConfigError(msg)
        user_config = load_yaml(config_path)
        config_dict = _deep_merge(config_dict, user_config)

    # Apply overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Validate and create settings
    try:
        return AppSettings(**config_dict)
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        raise InvalidConfigError(msg) from e
