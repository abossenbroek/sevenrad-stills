"""Tests for settings.loader module."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from sevenrad_stills.settings.loader import (
    _deep_merge,
    get_default_config_path,
    load_config,
    load_yaml,
)
from sevenrad_stills.settings.models import AppSettings
from sevenrad_stills.utils.exceptions import ConfigError, InvalidConfigError


class TestDeepMerge:
    """Tests for _deep_merge function."""

    def test_simple_merge(self) -> None:
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Test merging nested dictionaries."""
        base = {"cache": {"max_size_gb": 10, "retention_days": 7}}
        override = {"cache": {"max_size_gb": 20}}
        result = _deep_merge(base, override)
        assert result == {"cache": {"max_size_gb": 20, "retention_days": 7}}

    def test_override_with_non_dict(self) -> None:
        """Test that non-dict values replace entirely."""
        base = {"cache": {"max_size_gb": 10}}
        override = {"cache": "disabled"}
        result = _deep_merge(base, override)
        assert result == {"cache": "disabled"}

    def test_empty_override(self) -> None:
        """Test merging with empty override."""
        base = {"a": 1, "b": 2}
        override: dict[str, Any] = {}
        result = _deep_merge(base, override)
        assert result == base

    def test_empty_base(self) -> None:
        """Test merging with empty base."""
        base: dict[str, Any] = {}
        override = {"a": 1, "b": 2}
        result = _deep_merge(base, override)
        assert result == override


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cache:\n  max_size_gb: 20\n")

        result = load_yaml(config_file)
        assert result == {"cache": {"max_size_gb": 20}}

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading empty YAML file returns empty dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        result = load_yaml(config_file)
        assert result == {}

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(nonexistent)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error with invalid YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content:")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_yaml(config_file)


class TestGetDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_returns_path_to_default_yaml(self) -> None:
        """Test that function returns path to config/default.yaml."""
        path = get_default_config_path()
        assert path.name == "default.yaml"
        assert path.parent.name == "config"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self) -> None:
        """Test loading default configuration."""
        settings = load_config()
        assert isinstance(settings, AppSettings)
        # Verify some default values
        assert settings.cache.max_size_gb == 10.0
        assert settings.download.format == "mp4"

    def test_load_with_user_config(self, tmp_path: Path) -> None:
        """Test loading with user configuration override."""
        user_config = tmp_path / "user.yaml"
        user_config.write_text(
            """
cache:
  max_size_gb: 20
download:
  quality: 720p
"""
        )

        settings = load_config(config_path=user_config)
        assert settings.cache.max_size_gb == 20.0
        assert settings.download.quality == "720p"
        # Default values should still be present
        assert settings.cache.retention_days == 7

    def test_load_with_overrides(self) -> None:
        """Test loading with runtime overrides."""
        overrides = {
            "cache": {"max_size_gb": 30},
            "logging": {"level": "DEBUG"},
        }

        settings = load_config(overrides=overrides)
        assert settings.cache.max_size_gb == 30.0
        assert settings.logging.level == "DEBUG"

    def test_load_with_user_config_and_overrides(self, tmp_path: Path) -> None:
        """Test precedence: default < user config < overrides."""
        user_config = tmp_path / "user.yaml"
        user_config.write_text("cache:\n  max_size_gb: 20\n")

        overrides = {"cache": {"max_size_gb": 30}}

        settings = load_config(config_path=user_config, overrides=overrides)
        # Override should win
        assert settings.cache.max_size_gb == 30.0

    def test_nonexistent_user_config(self, tmp_path: Path) -> None:
        """Test error when user config doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigError, match="not found"):
            load_config(config_path=nonexistent)

    def test_invalid_configuration(self, tmp_path: Path) -> None:
        """Test error with invalid configuration values."""
        user_config = tmp_path / "invalid.yaml"
        user_config.write_text(
            """
cache:
  max_size_gb: -10
"""
        )

        with pytest.raises(InvalidConfigError):
            load_config(config_path=user_config)

    def test_extraction_validation(self, tmp_path: Path) -> None:
        """Test extraction settings validation in full config."""
        user_config = tmp_path / "invalid_extraction.yaml"
        user_config.write_text(
            """
extraction:
  fps: 1.0
  frame_interval: 30
"""
        )

        with pytest.raises(InvalidConfigError, match="Cannot specify both"):
            load_config(config_path=user_config)
