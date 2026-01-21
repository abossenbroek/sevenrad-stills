"""Pydantic models for td-linter configuration validation.

This module provides type-safe configuration loading with comprehensive
validation using Pydantic v2 models. All configuration is validated
before being processed, ensuring early detection of configuration errors.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Maximum config file size (1MB)
MAX_CONFIG_FILE_SIZE = 1024 * 1024


class Severity(str, Enum):
    """Severity level for lint rules."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class PresetName(str, Enum):
    """Available preset configuration names."""

    RECOMMENDED = "recommended"
    STRICT = "strict"
    MINIMAL = "minimal"
    PEDANTIC = "pedantic"


# Pattern for rule IDs (category + number, e.g., S001, C002)
RULE_ID_PATTERN = re.compile(r"^[SCTRPGF]\d{3}$")

# Pattern for select/ignore patterns (category code or full rule ID)
SELECT_PATTERN = re.compile(r"^[SCTRPGF](\d{3})?$")


class RuleConfigModel(BaseModel):
    """Configuration for a single lint rule."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    severity: Severity = Severity.ERROR
    options: dict[str, Any] = Field(default_factory=dict)


class PluginPathConfig(BaseModel):
    """Plugin configuration for loading from file path."""

    model_config = ConfigDict(extra="forbid")

    path: str

    @field_validator("path")
    @classmethod
    def validate_path_extension(cls, v: str) -> str:
        """Validate that path ends with .py."""
        if not v.endswith(".py"):
            msg = f"Plugin path must end with .py, got: {v}"
            raise ValueError(msg)
        return v


class PluginModuleConfig(BaseModel):
    """Plugin configuration for loading from installed module."""

    model_config = ConfigDict(extra="forbid")

    module: str

    @field_validator("module")
    @classmethod
    def validate_module_name(cls, v: str) -> str:
        """Validate module name format."""
        if not v or not all(
            part.isidentifier() for part in v.split(".")
        ):
            msg = f"Invalid module name: {v}"
            raise ValueError(msg)
        return v


# Union type for plugin configs
PluginConfig = PluginPathConfig | PluginModuleConfig


class LintConfigModel(BaseModel):
    """Complete linting configuration model.

    This model validates configuration files before they are processed.
    It ensures all values are of the correct type and within valid ranges.
    """

    model_config = ConfigDict(extra="forbid")

    version: Annotated[
        str,
        Field(pattern=r"^\d+\.\d+\.\d+$", default="1.0.0"),
    ]
    extends: PresetName | list[PresetName] | None = None
    select: list[str] | None = None
    ignore: list[str] | None = None
    rules: dict[str, RuleConfigModel] = Field(default_factory=dict)
    plugins: list[dict[str, str]] | None = None

    @field_validator("select", "ignore", mode="before")
    @classmethod
    def validate_patterns(cls, v: list[str] | None) -> list[str] | None:
        """Validate select/ignore patterns match expected format."""
        if v is None:
            return v
        for pattern in v:
            if not SELECT_PATTERN.match(pattern):
                msg = (
                    f"Invalid pattern '{pattern}'. "
                    f"Must be category code (S, C, T, R, P, G, F) "
                    f"or rule ID (e.g., S001)"
                )
                raise ValueError(msg)
        return v

    @field_validator("plugins", mode="before")
    @classmethod
    def validate_plugins(
        cls, v: list[dict[str, str]] | None
    ) -> list[dict[str, str]] | None:
        """Validate plugin configurations have required keys."""
        if v is None:
            return v
        for i, plugin in enumerate(v):
            if not isinstance(plugin, dict):
                msg = f"Plugin {i} must be a dictionary"
                raise ValueError(msg)
            if "path" not in plugin and "module" not in plugin:
                msg = f"Plugin {i} must have either 'path' or 'module' key"
                raise ValueError(msg)
            if "path" in plugin and "module" in plugin:
                msg = f"Plugin {i} cannot have both 'path' and 'module'"
                raise ValueError(msg)
            # Validate path extension
            if "path" in plugin and not plugin["path"].endswith(".py"):
                msg = f"Plugin {i} path must end with .py"
                raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_rule_ids(self) -> "LintConfigModel":
        """Validate that all rule IDs in rules dict are valid."""
        for rule_id in self.rules:
            if not RULE_ID_PATTERN.match(rule_id):
                msg = (
                    f"Invalid rule ID '{rule_id}'. "
                    f"Must be category letter + 3 digits (e.g., S001)"
                )
                raise ValueError(msg)
        return self


class ConfigFileSizeError(Exception):
    """Raised when config file exceeds maximum size."""

    def __init__(self, path: Path, size: int, max_size: int) -> None:
        self.path = path
        self.size = size
        self.max_size = max_size
        super().__init__(
            f"Configuration file '{path}' is {size} bytes, "
            f"exceeds maximum of {max_size} bytes"
        )


def validate_config_file_size(path: Path) -> None:
    """Check that config file doesn't exceed maximum size.

    This is a security measure to prevent DoS via large config files.

    Args:
        path: Path to the config file.

    Raises:
        ConfigFileSizeError: If file exceeds MAX_CONFIG_FILE_SIZE.
    """
    size = path.stat().st_size
    if size > MAX_CONFIG_FILE_SIZE:
        raise ConfigFileSizeError(path, size, MAX_CONFIG_FILE_SIZE)
